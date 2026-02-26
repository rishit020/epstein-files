# calibration/calibration_manager.py — Startup Calibration Orchestrator
# PRD §23 — Per-Vehicle Mounting Calibration Protocol
#
# Manages the calibration sequence that runs on first startup or VIN change.
# Collects head pose (yaw, pitch) and EAR samples from the signal pipeline,
# computes neutral pose offsets, and persists the result to session_state.json.
#
# State machine (per PRD §23.2):
#
#   IDLE ──startup()──► LOADED  (valid session_state.json, VIN match)
#        └─────────────► COLLECTING  (needs calibration)
#
#   COLLECTING ──enough frames, std OK──► COMPLETE
#              ──enough frames, std bad, retries left──► COLLECTING (reset)
#              ──enough frames, std bad, no retries──► FAILED (fallback 0.0)
#              ──not enough frames, first miss──► EXTENDING
#              ──not enough frames, extension miss──► FAILED (fallback 0.0)
#
#   EXTENDING ──enough frames, std OK──► COMPLETE
#             ──enough frames, std bad, retries left──► COLLECTING (reset)
#             ──enough frames, std bad, no retries──► FAILED (fallback 0.0)
#             ──still not enough frames──► FAILED (fallback 0.0)
#
# Audio hooks:
#   3 short beeps on calibration start (PRD §23.2 Step 2)
#   2 long beeps on success (PRD §23.2 Step 5)
# On Mac the default audio_fn is a no-op (audio_handler is Phase 6).

from __future__ import annotations

import logging
from enum import Enum, auto
from typing import Callable

import numpy as np

import config
from calibration.session_state import (
    SessionState,
    load_session_state,
    make_session_state,
    save_session_state,
    vin_matches,
)

logger = logging.getLogger(__name__)


class CalibrationStatus(Enum):
    """Current state of the calibration state machine."""
    IDLE       = auto()   # Not yet started
    LOADED     = auto()   # Valid state loaded — no calibration needed
    COLLECTING = auto()   # Accumulating frames in primary window
    EXTENDING  = auto()   # Primary window short on frames — extended 5s
    COMPLETE   = auto()   # Calibration succeeded; offsets written to disk
    FAILED     = auto()   # Fallback 0.0 offsets used; logged CALIBRATION_FAILED


def _noop_audio(count: int, kind: str) -> None:
    """Default no-op audio beep (Mac dev — audio is Phase 6)."""


class CalibrationManager:
    """Startup calibration orchestrator (PRD §23).

    Usage pattern (streaming)::

        manager = CalibrationManager(signal_processor)
        status = manager.startup(current_vin)

        if status in (CalibrationStatus.LOADED,
                      CalibrationStatus.COMPLETE,
                      CalibrationStatus.FAILED):
            pass  # ready — offsets already applied to signal_processor

        elif status == CalibrationStatus.COLLECTING:
            # Feed frames until terminal state reached
            for yaw, pitch, ear, pose_valid in frame_source:
                status = manager.feed_frame(yaw, pitch, ear, pose_valid)
                if status in (CalibrationStatus.COMPLETE, CalibrationStatus.FAILED):
                    break

    Args:
        signal_processor: Layer 2 SignalProcessor instance. Offsets are applied
            to it automatically on LOADED or COMPLETE.
        audio_fn: Callable(count, kind) that plays audio beeps.
            ``kind`` is ``"short"`` or ``"long"``.
            Defaults to no-op (Mac dev — audio is Phase 6).
        state_file: Path to session_state.json. Defaults to config value.
    """

    def __init__(
        self,
        signal_processor,                          # SignalProcessor (avoid circular import)
        audio_fn: Callable[[int, str], None] = _noop_audio,
        state_file: str = config.NEUTRAL_POSE_FILE,
    ) -> None:
        self._sp         = signal_processor
        self._audio_fn   = audio_fn
        self._state_file = state_file

        self._status: CalibrationStatus = CalibrationStatus.IDLE

        # Collected samples for the current attempt
        self._yaw_samples:  list[float] = []
        self._pitch_samples: list[float] = []
        self._ear_samples:  list[float] = []

        # Attempt bookkeeping
        self._attempt:         int = 0       # 0-based attempt index
        self._extended:        bool = False  # Whether we are in the extension window
        self._primary_target:  int = 0       # Frame target for primary window
        self._extended_target: int = 0       # Frame target after extension

        # Output — set on COMPLETE or FAILED
        self._result: SessionState | None = None

        # VIN for this session (set in startup)
        self._current_vin: str = ""

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def status(self) -> CalibrationStatus:
        return self._status

    @property
    def result(self) -> SessionState | None:
        """SessionState written/loaded, or None if not yet terminal."""
        return self._result

    def startup(self, current_vin: str = "") -> CalibrationStatus:
        """Check session_state.json and either load or initiate calibration.

        Follows PRD §24.2 startup load behavior.

        Returns:
            LOADED   — valid persisted state applied to signal_processor.
            COLLECTING — calibration is required; begin feeding frames.
        """
        self._current_vin = current_vin

        if not config.FORCE_RECALIBRATION:
            persisted = load_session_state(self._state_file)
            if (
                persisted is not None
                and persisted.calibration_complete
                and vin_matches(persisted, current_vin)
            ):
                self._apply_to_signal_processor(persisted)
                self._result = persisted
                self._status = CalibrationStatus.LOADED
                logger.info(
                    "CALIBRATION_LOADED: vin=%s yaw_offset=%.2f pitch_offset=%.2f",
                    persisted.vehicle_vin,
                    persisted.neutral_yaw_offset,
                    persisted.neutral_pitch_offset,
                )
                return self._status

        logger.info("CALIBRATION_REQUIRED: starting sequence for vin=%s", current_vin)
        self._begin_attempt()
        return self._status

    def feed_frame(
        self,
        yaw_deg: float,
        pitch_deg: float,
        ear: float,
        pose_valid: bool,
    ) -> CalibrationStatus:
        """Feed one frame of data into the calibration state machine.

        Only frames with ``pose_valid=True`` contribute to the sample pool.
        Invalid frames still count against the window frame budget so that
        the duration guarantee (10s wall time) is honoured.

        Args:
            yaw_deg:    Kalman-filtered head yaw in degrees (before pose correction).
            pitch_deg:  Kalman-filtered head pitch in degrees (before pose correction).
            ear:        Mean EAR for this frame.
            pose_valid: True if head pose PnP solve was successful this frame.

        Returns:
            Current CalibrationStatus after processing this frame.
        """
        if self._status not in (CalibrationStatus.COLLECTING, CalibrationStatus.EXTENDING):
            return self._status

        # Only valid frames contribute to samples
        if pose_valid:
            self._yaw_samples.append(yaw_deg)
            self._pitch_samples.append(pitch_deg)
            self._ear_samples.append(ear)

        # Count all frames (valid or not) against the window budget
        total_frames_seen = self._frames_seen
        self._frames_seen += 1
        total_frames_seen = self._frames_seen  # post-increment

        # Choose the frame target based on whether we're in the extension window
        target = self._extended_target if self._extended else self._primary_target

        if total_frames_seen < target:
            # Still collecting
            return self._status

        # Window complete — evaluate
        valid_count = len(self._yaw_samples)
        if valid_count < config.CALIBRATION_MIN_VALID_FRAMES:
            # Not enough valid frames
            if not self._extended:
                # Enter extension window (PRD §23.2 Step 2: "Extend window by 5s, retry once")
                self._extended = True
                self._status = CalibrationStatus.EXTENDING
                logger.info(
                    "CALIBRATION_EXTENDING: only %d valid frames in primary window",
                    valid_count,
                )
                return self._status
            else:
                # Extension also failed
                logger.warning(
                    "CALIBRATION_FAILED: only %d valid frames after extension",
                    valid_count,
                )
                return self._finalize_fallback()

        # Enough frames — validate std dev (PRD §23.2 Step 3)
        std_yaw   = float(np.std(self._yaw_samples))
        std_pitch = float(np.std(self._pitch_samples))

        if std_yaw >= config.CALIBRATION_MAX_POSE_STD_DEG or std_pitch >= config.CALIBRATION_MAX_POSE_STD_DEG:
            self._attempt += 1
            if self._attempt < config.CALIBRATION_MAX_ATTEMPTS:
                logger.info(
                    "CALIBRATION_RETRY: std_yaw=%.2f std_pitch=%.2f (attempt %d/%d)",
                    std_yaw, std_pitch, self._attempt, config.CALIBRATION_MAX_ATTEMPTS,
                )
                self._begin_attempt()
                return self._status
            else:
                logger.warning(
                    "CALIBRATION_FAILED: pose unstable after %d attempts "
                    "(std_yaw=%.2f std_pitch=%.2f)",
                    config.CALIBRATION_MAX_ATTEMPTS, std_yaw, std_pitch,
                )
                return self._finalize_fallback()

        # Validation passed — compute offsets and persist (PRD §23.2 Step 4)
        return self._finalize_success(std_yaw, std_pitch)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _begin_attempt(self) -> None:
        """Reset sample buffers and set frame targets for a new collection attempt."""
        self._yaw_samples   = []
        self._pitch_samples = []
        self._ear_samples   = []
        self._extended      = False
        self._frames_seen   = 0

        fps = config.CAPTURE_FPS
        self._primary_target  = int(config.CALIBRATION_DURATION_S * fps)
        self._extended_target = int(
            (config.CALIBRATION_DURATION_S + config.CALIBRATION_EXTENSION_S) * fps
        )

        self._status = CalibrationStatus.COLLECTING

        # PRD §23.2 Step 2: 3 short audio beeps at calibration start
        self._audio_fn(3, "short")
        logger.info(
            "CALIBRATION_COLLECTING: attempt %d/%d target=%d frames",
            self._attempt + 1, config.CALIBRATION_MAX_ATTEMPTS, self._primary_target,
        )

    def _finalize_success(self, std_yaw: float, std_pitch: float) -> CalibrationStatus:
        """Compute offsets, persist, apply to signal_processor. (PRD §23.2 Step 4+5)"""
        neutral_yaw   = float(np.mean(self._yaw_samples))
        neutral_pitch = float(np.mean(self._pitch_samples))
        baseline_ear  = float(np.mean(self._ear_samples)) if self._ear_samples else (
            config.EAR_DEFAULT_CLOSE_THRESHOLD / config.EAR_CALIBRATION_MULTIPLIER
        )
        close_threshold = baseline_ear * config.EAR_CALIBRATION_MULTIPLIER

        state = make_session_state(
            vehicle_vin=self._current_vin,
            neutral_yaw_offset=neutral_yaw,
            neutral_pitch_offset=neutral_pitch,
            baseline_ear=baseline_ear,
            close_threshold=close_threshold,
            frames_collected=len(self._yaw_samples),
            pose_std_yaw=std_yaw,
            pose_std_pitch=std_pitch,
        )

        saved = save_session_state(state, self._state_file)
        if not saved:
            logger.error("CALIBRATION_DISK_ERROR: using in-memory values only")

        self._apply_to_signal_processor(state)
        self._result = state
        self._status = CalibrationStatus.COMPLETE

        # PRD §23.2 Step 5: 2 long audio beeps on success
        self._audio_fn(2, "long")
        logger.info(
            "CALIBRATION_COMPLETE: yaw_offset=%.2f pitch_offset=%.2f "
            "baseline_ear=%.3f frames=%d",
            neutral_yaw, neutral_pitch, baseline_ear, len(self._yaw_samples),
        )
        return self._status

    def _finalize_fallback(self) -> CalibrationStatus:
        """Use 0.0 offsets and default EAR. (PRD §23.2 Step 2/3 failure path)"""
        default_baseline = config.EAR_DEFAULT_CLOSE_THRESHOLD / config.EAR_CALIBRATION_MULTIPLIER
        state = make_session_state(
            vehicle_vin=self._current_vin,
            neutral_yaw_offset=0.0,
            neutral_pitch_offset=0.0,
            baseline_ear=default_baseline,
            close_threshold=config.EAR_DEFAULT_CLOSE_THRESHOLD,
            frames_collected=len(self._yaw_samples),
            pose_std_yaw=0.0,
            pose_std_pitch=0.0,
        )
        # Mark as incomplete so next startup triggers calibration again
        state = SessionState(
            schema_version=state.schema_version,
            calibrated_at=state.calibrated_at,
            vehicle_vin=state.vehicle_vin,
            neutral_yaw_offset=state.neutral_yaw_offset,
            neutral_pitch_offset=state.neutral_pitch_offset,
            baseline_ear=state.baseline_ear,
            close_threshold=state.close_threshold,
            calibration_complete=False,
            frames_collected=state.frames_collected,
            pose_std_yaw=state.pose_std_yaw,
            pose_std_pitch=state.pose_std_pitch,
        )

        self._apply_to_signal_processor(state)
        self._result = state
        self._status = CalibrationStatus.FAILED
        logger.warning("CALIBRATION_FAILED: using 0.0 fallback offsets")
        return self._status

    def _apply_to_signal_processor(self, state: SessionState) -> None:
        """Push calibration values into the SignalProcessor (PRD §24.2)."""
        self._sp.set_neutral_pose(state.neutral_yaw_offset, state.neutral_pitch_offset)
        self._sp.set_ear_baseline(state.baseline_ear, state.close_threshold)

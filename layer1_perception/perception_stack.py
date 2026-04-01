# layer1_perception/perception_stack.py — Perception stack orchestrator
# PRD §3.2, §FR-1.1 through §FR-1.6
#
# Orchestrates: FaceDetector → LandmarkModel + GazeModel ‖ PhoneDetector (every frame)
#
# Mac dev phase note: runs single-threaded. T-1/T-2 parallelism is added in Phase 6
# when threading and queue wiring are implemented in main.py.
#
# Confidence gates:
#   face.confidence < FACE_CONFIDENCE_GATE     → landmarks=None, gaze=None   (PRD §FR-1.2)
#   landmarks.confidence < LANDMARK_CONFIDENCE_GATE → gaze=None, lstm reset  (PRD §FR-1.3)
#
# LSTM reset: hidden state cleared when face absent > LSTM_RESET_ABSENT_FRAMES  (PRD §FR-1.3)
# Exception handling: all exceptions caught; returns safe PerceptionBundle        (PRD §FR-1.6)

from __future__ import annotations

import logging
import time

import numpy as np

import config
from layer1_perception.face_detector import FaceDetector
from layer1_perception.gaze_model import GazeModel
from layer1_perception.landmark_model import LandmarkModel
from layer1_perception.messages import (
    FaceDetection,
    PerceptionBundle,
    PhoneDetectionOutput,
)
from layer1_perception.phone_detector import PhoneDetector

_log = logging.getLogger(__name__)


def _safe_face() -> FaceDetection:
    return FaceDetection(present=False, confidence=0.0, bbox_norm=None, face_size_px=0)


def _safe_phone() -> PhoneDetectionOutput:
    return PhoneDetectionOutput(detected=False, max_confidence=0.0, bbox_norm=None)


class PerceptionStack:
    """Orchestrates all four perception models for one frame.

    Applies confidence gates, manages LSTM hidden-state lifecycle, and
    catches all model exceptions per PRD §FR-1.6.
    """

    def __init__(
        self,
        face_detector: FaceDetector,
        landmark_model: LandmarkModel,
        gaze_model: GazeModel,
        phone_detector: PhoneDetector,
    ) -> None:
        self._face  = face_detector
        self._lm    = landmark_model
        self._gaze  = gaze_model
        self._phone = phone_detector
        self._face_absent_count: int = 0

    def infer(
        self,
        frame: np.ndarray,
        frame_id: int,
        hidden_state: tuple | None = None,
    ) -> PerceptionBundle:
        """Run all perception models on one frame.

        Args:
            frame:        BGR frame (H, W, 3).
            frame_id:     Monotonic frame counter.
            hidden_state: LSTM hidden state (h_t, c_t) from previous bundle's
                          ``lstm_hidden_state``. None on first frame or after reset.

        Returns:
            PerceptionBundle with all model outputs merged.
            On exception: bundle with ``face.present=False`` and
            ``lstm_reset_occurred=True`` (PRD §FR-1.6).
        """
        t0 = time.perf_counter()
        try:
            return self._run(frame, frame_id, hidden_state, t0)
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            _log.error(
                "PerceptionStack.infer() exception frame=%d: %s",
                frame_id, exc, exc_info=True,
            )
            self._face_absent_count += 1
            return PerceptionBundle(
                timestamp_ns=time.time_ns(),
                frame_id=frame_id,
                face=_safe_face(),
                landmarks=None,
                gaze=None,
                phone=_safe_phone(),
                phone_result_stale=False,
                inference_ms=elapsed_ms,
                lstm_hidden_state=None,
                lstm_reset_occurred=True,
            )

    # ── Private ───────────────────────────────────────────────────────────────

    def _run(
        self,
        frame: np.ndarray,
        frame_id: int,
        hidden_state: tuple | None,
        t0: float,
    ) -> PerceptionBundle:
        timestamp_ns = time.time_ns()

        face  = self._face.infer(frame)
        phone = self._phone.infer(frame)

        face_valid = face.present and face.confidence >= config.FACE_CONFIDENCE_GATE

        if not face_valid:
            return self._handle_face_absent(
                face, phone, frame_id, timestamp_ns, hidden_state, t0
            )

        return self._handle_face_present(
            frame, face, phone, frame_id, timestamp_ns, hidden_state, t0
        )

    def _handle_face_absent(
        self,
        face: FaceDetection,
        phone: PhoneDetectionOutput,
        frame_id: int,
        timestamp_ns: int,
        hidden_state: tuple | None,
        t0: float,
    ) -> PerceptionBundle:
        """Handle a frame where face detection failed or was below confidence gate."""
        self._face_absent_count += 1
        threshold_exceeded = self._face_absent_count > config.LSTM_RESET_ABSENT_FRAMES
        # lstm_reset_occurred=True on the single frame where threshold is first crossed
        first_threshold_cross = (
            threshold_exceeded
            and self._face_absent_count == config.LSTM_RESET_ABSENT_FRAMES + 1
        )
        carry_state = None if threshold_exceeded else hidden_state

        if first_threshold_cross:
            _log.debug(
                "LSTM reset: face absent %d consecutive frames (threshold=%d) frame=%d",
                self._face_absent_count, config.LSTM_RESET_ABSENT_FRAMES, frame_id,
            )

        return PerceptionBundle(
            timestamp_ns=timestamp_ns,
            frame_id=frame_id,
            face=face,
            landmarks=None,
            gaze=None,
            phone=phone,
            phone_result_stale=False,
            inference_ms=(time.perf_counter() - t0) * 1000.0,
            lstm_hidden_state=carry_state,
            lstm_reset_occurred=first_threshold_cross,
        )

    def _handle_face_present(
        self,
        frame: np.ndarray,
        face: FaceDetection,
        phone: PhoneDetectionOutput,
        frame_id: int,
        timestamp_ns: int,
        hidden_state: tuple | None,
        t0: float,
    ) -> PerceptionBundle:
        """Handle a frame where face detection passed the confidence gate."""
        # If face was absent long enough, hidden state should already be None
        # (cleared during the absence), but enforce it explicitly for safety.
        if self._face_absent_count > config.LSTM_RESET_ABSENT_FRAMES:
            hidden_state = None
        self._face_absent_count = 0

        face_crop = _extract_face_crop(frame, face.bbox_norm)
        landmarks = self._lm.infer(face_crop)

        gaze = None
        new_hidden: tuple | None = None
        lstm_reset = False

        if landmarks.confidence < config.LANDMARK_CONFIDENCE_GATE:
            _log.debug(
                "Gaze gate skipped: landmark confidence %.3f < %.3f frame=%d",
                landmarks.confidence, config.LANDMARK_CONFIDENCE_GATE, frame_id,
            )
            if hidden_state is not None:
                lstm_reset = True
        else:
            gaze, new_hidden = self._gaze.infer(face_crop, hidden_state)

        return PerceptionBundle(
            timestamp_ns=timestamp_ns,
            frame_id=frame_id,
            face=face,
            landmarks=landmarks,
            gaze=gaze,
            phone=phone,
            phone_result_stale=False,
            inference_ms=(time.perf_counter() - t0) * 1000.0,
            lstm_hidden_state=new_hidden,
            lstm_reset_occurred=lstm_reset,
        )


def _extract_face_crop(frame: np.ndarray, bbox_norm: tuple) -> np.ndarray:
    """Extract face crop from full frame using normalised bbox (x, y, w, h).

    Coordinates are clamped to frame bounds. Returns a 1×1 crop on degenerate input
    (zero-area bbox) so downstream models always receive a valid array.
    """
    h, w = frame.shape[:2]
    bx, by, bw, bh = bbox_norm
    x1 = max(0, int(bx * w))
    y1 = max(0, int(by * h))
    x2 = min(w, int((bx + bw) * w))
    y2 = min(h, int((by + bh) * h))
    if x2 <= x1 or y2 <= y1:
        return frame[:1, :1]
    return frame[y1:y2, x1:x2]

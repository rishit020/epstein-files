# layer2_signals/messages.py — Layer 2 output messages
# PRD §4.3 — SignalFrame: Layer 2 → Layer 3
#
# Note: All angle values in HeadPose and GazeWorld are Kalman-filtered and
# neutral-pose-corrected before being placed in SignalFrame (PRD CHANGE-06, CHANGE-08).

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HeadPose:
    """Head pose angles — Kalman-filtered and neutral-pose-corrected.

    PRD §4.3
    """
    yaw_deg: float    # Kalman-filtered + neutral-pose-corrected. Positive = right.
    pitch_deg: float  # Kalman-filtered + neutral-pose-corrected. Positive = up.
    roll_deg: float   # Kalman-filtered. Range: [-45, 45].
    valid: bool       # False if reprojection error > PNP_REPROJECTION_ERR_MAX (8.0 px)
    # Debug fields — raw (unfiltered, uncorrected) values for logging/tuning only
    raw_yaw_deg: float
    raw_pitch_deg: float
    raw_roll_deg: float


@dataclass
class EyeSignals:
    """Eye Aspect Ratio signals and calibration state.

    PRD §4.3
    """
    left_EAR: float             # Eye Aspect Ratio [0.0, ~0.4]
    right_EAR: float
    mean_EAR: float
    baseline_EAR: float         # Per-session calibrated open-eye baseline
    close_threshold: float      # baseline_EAR * EAR_CALIBRATION_MULTIPLIER (0.75)
    valid: bool
    calibration_complete: bool  # True after 30s of valid driving data collected


@dataclass
class GazeWorld:
    """World-space gaze angles — Kalman-filtered and neutral-pose-corrected.

    PRD §4.3
    """
    yaw_deg: float    # World-space gaze yaw — Kalman-filtered + neutral-pose-corrected
    pitch_deg: float  # World-space gaze pitch — same
    on_road: bool     # True if within ROAD_ZONE (compared against corrected angles)
    valid: bool       # False if gaze OR head_pose is invalid


@dataclass
class PhoneSignal:
    """Phone detection signal for downstream layers.

    PRD §4.3
    """
    detected: bool
    confidence: float  # [0.0, 1.0]
    stale: bool        # True if using T-2 timeout fallback result


@dataclass
class SignalFrame:
    """Full processed signal output — Layer 2 → Layer 3.

    PRD §4.3
    """
    timestamp_ns: int
    frame_id: int
    face_present: bool
    head_pose: HeadPose | None      # None if face not present
    eye_signals: EyeSignals | None  # None if face not present
    gaze_world: GazeWorld | None    # None if face not present
    phone_signal: PhoneSignal
    speed_mps: float      # From SpeedSource (PRD §22). 0.0 if unavailable.
    speed_stale: bool     # True if speed reading is older than SPEED_STALE_THRESHOLD_S
    signals_valid: bool   # False if any critical signal is unavailable

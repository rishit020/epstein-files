# layer1_perception/messages.py — Layer 1 output messages
# PRD §4.2 — PerceptionBundle: Layer 1 → Layer 2

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class FaceDetection:
    """Face detector output — one per frame.

    PRD §4.2
    """
    present: bool
    confidence: float    # [0.0, 1.0]
    bbox_norm: tuple     # (x, y, w, h) normalized to [0, 1]; None if not present
    face_size_px: int    # Width of bounding box in pixels; 0 if not present


@dataclass
class LandmarkOutput:
    """PFLD 68-point landmark output (iBUG convention). PRD specifies 98-point; accepted deviation.

    PRD §4.2
    """
    landmarks: np.ndarray   # Shape: (68, 2), normalized [0, 1]
    confidence: float       # [0.0, 1.0]
    pose_valid: bool        # False if face rotation exceeds reliable limit


@dataclass
class GazeOutput:
    """MobileNetV3+LSTM gaze estimation output (camera space, pre-transform).

    PRD §4.2
    """
    left_eye_yaw: float      # Degrees (camera space)
    left_eye_pitch: float
    right_eye_yaw: float
    right_eye_pitch: float
    combined_yaw: float      # Weighted mean of left/right
    combined_pitch: float
    confidence: float        # [0.0, 1.0]
    valid: bool


@dataclass
class PhoneDetectionOutput:
    """YOLOv8-nano phone detection output.

    PRD §4.2
    """
    detected: bool
    max_confidence: float    # [0.0, 1.0]; 0.0 if not detected
    bbox_norm: tuple | None  # (x, y, w, h) normalized; None if not detected


@dataclass
class PerceptionBundle:
    """Full output of the parallel perception stack — Layer 1 → Layer 2.

    PRD §4.2 (v2.0.0: includes lstm_hidden_state and phone_result_stale).
    """
    timestamp_ns: int
    frame_id: int
    face: FaceDetection
    landmarks: LandmarkOutput | None   # None if face not present
    gaze: GazeOutput | None            # None if face not present
    phone: PhoneDetectionOutput
    phone_result_stale: bool           # True if T-2 timed out; last result used
    inference_ms: float                # Total perception inference time (ms)
    # LSTM state — passed back to Layer 1 on next frame call (PRD CHANGE-03)
    lstm_hidden_state: Any | None      # Opaque (h_t, c_t) tuple; None on first frame
                                       # or after face-absent reset (> 10 frame gap)
    lstm_reset_occurred: bool          # True if hidden state was reset this frame

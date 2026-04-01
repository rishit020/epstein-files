# layer1_perception/gaze_model.py — MobileNetV3+LSTM gaze estimation wrapper (ONNX)
# PRD §FR-1.3, §FR-1.5
#
# Interface: infer(face_crop, hidden_state) -> (GazeOutput, hidden_state)
#
# Model:  gaze_mobilenetv3_lstm.onnx
# Input:  face_crop BGR (H, W, 3) → resized to 448×448, ImageNet-normalised, NCHW float32
# Output: yaw (1, 90), pitch (1, 90) — 90-bin logits decoded via soft-argmax
#         angle = sum(softmax(logits) × arange(90)) × 4 − 180   → nominal range ≈ [−180, 176]°
#
# Known deviations (Mac dev phase):
#   - ONNX model has no LSTM hidden-state I/O (single-frame model at 448×448 input).
#     hidden_state is accepted/returned for PerceptionStack API compatibility only.
#     Replace with true MobileNetV3+LSTM model before Phase 7.
#   - config.GAZE_INPUT_RESOLUTION = 112 is stale — actual model requires 448.
#     _INPUT_SIZE overrides config for this module.

from __future__ import annotations

import cv2
import numpy as np
import onnxruntime as ort

import config
from layer1_perception.messages import GazeOutput

# ─── Model constants ──────────────────────────────────────────────────────────
_INPUT_SIZE = 448             # Actual ONNX model requirement (overrides stale config value of 112)
_NUM_BINS = 90                # 90-bin softmax classification head
_IDX = np.arange(_NUM_BINS, dtype=np.float32)   # [0.0, 1.0, ..., 89.0]
_ANGLE_SCALE = 4.0            # Degrees per bin index unit
_ANGLE_OFFSET = -180.0        # Degrees at bin index 0

# ─── ImageNet normalisation (RGB order) ───────────────────────────────────────
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ─── Confidence ───────────────────────────────────────────────────────────────
_LOG_NUM_BINS = float(np.log(_NUM_BINS))  # Max entropy for a 90-bin uniform distribution (nats)
_EPS = 1e-9


class GazeModel:
    """MobileNetV3+LSTM gaze estimation wrapper (ONNX runtime).

    PRD §FR-1.5: exposes ``infer(face_crop, hidden_state) -> (GazeOutput, hidden_state)``.

    The current ONNX model has no LSTM state I/O.  hidden_state is accepted and
    returned unchanged for PerceptionStack API compatibility (always None out).
    """

    def __init__(self, model_path: str | None = None) -> None:
        path = model_path if model_path is not None else config.GAZE_MODEL_PATH
        self._session = ort.InferenceSession(path)

    # ── Public API ────────────────────────────────────────────────────────────

    def infer(
        self,
        face_crop: np.ndarray,
        hidden_state: tuple | None = None,
    ) -> tuple[GazeOutput, tuple | None]:
        """Run gaze inference on a BGR face crop.

        Args:
            face_crop:    BGR image array, shape (H, W, 3).
            hidden_state: LSTM hidden state (h_t, c_t) from the previous frame.
                          Passthrough only — current ONNX model has no LSTM state I/O.

        Returns:
            (GazeOutput, new_hidden_state)
            new_hidden_state is always None (no LSTM in current model).

        Raises:
            ValueError: If face_crop is not a 3-channel array.
        """
        self._validate_input(face_crop)
        tensor = self._preprocess(face_crop)

        outputs = self._session.run(None, {'input': tensor})
        yaw_logits   = outputs[0][0]   # (90,)
        pitch_logits = outputs[1][0]   # (90,)

        yaw_probs   = _softmax(yaw_logits)
        pitch_probs = _softmax(pitch_logits)

        yaw_deg   = float(np.dot(yaw_probs,   _IDX)) * _ANGLE_SCALE + _ANGLE_OFFSET
        pitch_deg = float(np.dot(pitch_probs, _IDX)) * _ANGLE_SCALE + _ANGLE_OFFSET

        confidence = _entropy_confidence(yaw_probs, pitch_probs)

        gaze = GazeOutput(
            left_eye_yaw=yaw_deg,
            left_eye_pitch=pitch_deg,
            right_eye_yaw=yaw_deg,
            right_eye_pitch=pitch_deg,
            combined_yaw=yaw_deg,
            combined_pitch=pitch_deg,
            confidence=confidence,
            valid=True,
        )
        return gaze, None  # hidden_state passthrough — no LSTM state in current ONNX model

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _validate_input(face_crop: np.ndarray) -> None:
        if not isinstance(face_crop, np.ndarray):
            raise ValueError(f"Expected np.ndarray, got {type(face_crop).__name__}")
        if face_crop.ndim != 3:
            raise ValueError(
                f"Expected 3D array (H, W, 3), got shape {face_crop.shape}"
            )
        if face_crop.shape[2] != 3:
            raise ValueError(
                f"Expected 3 channels, got {face_crop.shape[2]}"
            )

    @staticmethod
    def _preprocess(face_crop: np.ndarray) -> np.ndarray:
        """BGR (H, W, 3) -> ONNX input (1, 3, 448, 448) float32, ImageNet-normalised."""
        if face_crop.dtype != np.uint8:
            face_crop = np.clip(face_crop, 0, 255).astype(np.uint8)
        resized = cv2.resize(face_crop, (_INPUT_SIZE, _INPUT_SIZE))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - _IMAGENET_MEAN) / _IMAGENET_STD
        return rgb.transpose(2, 0, 1)[np.newaxis]  # (1, 3, 448, 448)


# ─── Module-level pure helpers (testable independently) ───────────────────────

def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over a 1D array."""
    shifted = x - x.max()
    exp_x = np.exp(shifted)
    return exp_x / exp_x.sum()


def _entropy_confidence(yaw_probs: np.ndarray, pitch_probs: np.ndarray) -> float:
    """Confidence from softmax distribution sharpness (entropy-based).

    Returns a value in [0, 1]:
      - 0.0 → maximally uncertain (uniform distribution over 90 bins)
      - 1.0 → maximally certain (all probability on one bin)

    Combined as the mean of independent yaw and pitch confidence scores.
    """
    def _single(probs: np.ndarray) -> float:
        p = np.clip(probs, _EPS, 1.0)
        entropy = float(-np.sum(p * np.log(p)))
        return float(np.clip(1.0 - entropy / _LOG_NUM_BINS, 0.0, 1.0))

    return (_single(yaw_probs) + _single(pitch_probs)) / 2.0

# layer1_perception/landmark_model.py — PFLD 68-point landmark wrapper (ONNX)
# PRD §FR-1.2, §FR-1.5 — standard infer() interface
#
# Input:  BGR face crop (H, W, 3) uint8 or float
# Output: LandmarkOutput with (68, 2) normalized [0,1] iBUG coords
#
# Accepted PRD deviation: 68-point iBUG convention instead of 98-point.
# Documented in config.py and PFLD_HANDOFF.md.

from __future__ import annotations

import cv2
import numpy as np
import onnxruntime as ort

import config
from layer1_perception.messages import LandmarkOutput

# ─── Model constants ─────────────────────────────────────────────────────────
_INPUT_SIZE = 112
_NUM_LANDMARKS = 68
_OUTPUT_FLAT_SIZE = _NUM_LANDMARKS * 2  # 136

# ─── Confidence heuristic thresholds ─────────────────────────────────────────
_BOUNDARY_MARGIN = 0.01
_BOUNDARY_FRACTION_WARN = 0.20
_IOD_MIN_SEVERE = 0.05
_IOD_MIN_WARN = 0.10

# ─── Pose validity thresholds ────────────────────────────────────────────────
_IOD_FACE_WIDTH_RATIO_MIN = 0.15
_FACE_WIDTH_MIN = 0.01

# ─── iBUG 68-point indices used for quality checks ──────────────────────────
_LEFT_EYE_OUTER = 36
_RIGHT_EYE_OUTER = 45
_NOSE_TIP = 30
_CHIN = 8
_LEFT_JAW = 0
_RIGHT_JAW = 16


class LandmarkModel:
    """PFLD 68-point landmark model wrapper (ONNX runtime).

    PRD §FR-1.5: exposes ``infer(frame) -> LandmarkOutput``.
    PRD §FR-1.2: confidence gating is handled by PerceptionStack, not here.
    """

    def __init__(self, model_path: str | None = None) -> None:
        path = model_path if model_path is not None else config.PFLD_MODEL_PATH
        self._session = ort.InferenceSession(path)
        self._input_name = self._session.get_inputs()[0].name

    # ── Public API ───────────────────────────────────────────────────────────

    def infer(self, face_crop: np.ndarray) -> LandmarkOutput:
        """Run PFLD inference on a BGR face crop.

        Args:
            face_crop: BGR image array, shape (H, W, 3).

        Returns:
            LandmarkOutput with (68, 2) normalised landmarks in [0, 1],
            a quality-derived confidence score, and a pose_valid flag.

        Raises:
            ValueError: If face_crop is not a 3-channel image.
        """
        self._validate_input(face_crop)
        tensor = self._preprocess(face_crop)

        raw = self._session.run(None, {self._input_name: tensor})[0]  # (1, 136)
        landmarks = raw[0].reshape(_NUM_LANDMARKS, 2)                 # (68, 2)
        landmarks = np.clip(landmarks, 0.0, 1.0).astype(np.float32)

        confidence = self._compute_confidence(landmarks)
        pose_valid = self._compute_pose_valid(landmarks)

        return LandmarkOutput(
            landmarks=landmarks,
            confidence=confidence,
            pose_valid=pose_valid,
        )

    # ── Private helpers ──────────────────────────────────────────────────────

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
        """BGR (H, W, 3) -> ONNX input (1, 3, 112, 112) float32 in [0, 1]."""
        if face_crop.dtype != np.uint8:
            face_crop = np.clip(face_crop, 0, 255).astype(np.uint8)
        resized = cv2.resize(face_crop, (_INPUT_SIZE, _INPUT_SIZE))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return rgb.transpose(2, 0, 1)[np.newaxis]  # (1, 3, 112, 112)

    @staticmethod
    def _compute_confidence(landmarks: np.ndarray) -> float:
        """Derive a quality-based confidence from landmark geometry.

        The PFLD model has no confidence output — this heuristic checks:
        1. Boundary landmarks (face partially out of crop)
        2. Interocular distance (degenerate geometry / extreme yaw)
        3. Face structure (chin above nose = physically impossible)
        """
        confidence = 1.0

        # 1. Boundary check — landmarks piled at crop edges
        near_boundary = (
            (landmarks[:, 0] < _BOUNDARY_MARGIN)
            | (landmarks[:, 0] > 1.0 - _BOUNDARY_MARGIN)
            | (landmarks[:, 1] < _BOUNDARY_MARGIN)
            | (landmarks[:, 1] > 1.0 - _BOUNDARY_MARGIN)
        )
        if near_boundary.mean() > _BOUNDARY_FRACTION_WARN:
            confidence -= 0.3

        # 2. Interocular distance check
        iod = float(np.linalg.norm(
            landmarks[_LEFT_EYE_OUTER] - landmarks[_RIGHT_EYE_OUTER]
        ))
        if iod < _IOD_MIN_SEVERE:
            confidence = min(confidence, 0.3)
        elif iod < _IOD_MIN_WARN:
            confidence -= 0.2

        # 3. Structure check — chin should be below nose in image coords (y↓)
        if landmarks[_CHIN, 1] <= landmarks[_NOSE_TIP, 1]:
            confidence -= 0.3

        return float(np.clip(confidence, 0.0, 1.0))

    @staticmethod
    def _compute_pose_valid(landmarks: np.ndarray) -> bool:
        """Determine if face pose is within reliable range for PnP / EAR.

        Returns False for extreme yaw or degenerate geometry where downstream
        consumers (head_pose_solver, ear_calculator) would produce unreliable results.
        """
        left_jaw_x = landmarks[_LEFT_JAW, 0]
        right_jaw_x = landmarks[_RIGHT_JAW, 0]
        face_width = abs(right_jaw_x - left_jaw_x)

        if face_width < _FACE_WIDTH_MIN:
            return False

        iod = float(np.linalg.norm(
            landmarks[_LEFT_EYE_OUTER] - landmarks[_RIGHT_EYE_OUTER]
        ))
        if iod / face_width < _IOD_FACE_WIDTH_RATIO_MIN:
            return False

        nose_x = landmarks[_NOSE_TIP, 0]
        jaw_min_x = min(left_jaw_x, right_jaw_x)
        jaw_max_x = max(left_jaw_x, right_jaw_x)
        if nose_x < jaw_min_x or nose_x > jaw_max_x:
            return False

        return True

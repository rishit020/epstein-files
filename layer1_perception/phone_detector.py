# layer1_perception/phone_detector.py — YOLOv8n phone detector wrapper (ONNX)
# PRD §FR-1.4 — standard infer() interface, runs every frame regardless of face

from __future__ import annotations

import cv2
import numpy as np
import onnxruntime as ort

import config
from layer1_perception.messages import PhoneDetectionOutput

_INPUT_SIZE = 640
_OUTPUT_DETECTIONS = 8400


class PhoneDetector:
    """YOLOv8n phone detector wrapper (ONNX runtime).

    Runs on full frame. Output: (1, 5, 8400) where each of 8400 anchors has
    [x, y, w, h, conf_phone] (no class dim — model is single-class).
    """

    def __init__(self, model_path: str | None = None) -> None:
        path = model_path if model_path is not None else config.YOLO_MODEL_PATH
        self._session = ort.InferenceSession(path)

    def infer(self, frame: np.ndarray) -> PhoneDetectionOutput:
        """Run YOLOv8n on full frame.

        Args:
            frame: BGR frame (H, W, 3) uint8

        Returns:
            PhoneDetectionOutput with detected=True/False, max_confidence, bbox_norm.
        """
        self._validate_input(frame)
        h, w = frame.shape[:2]
        tensor = self._preprocess(frame)

        raw = self._session.run(None, {'images': tensor})[0]  # (1, 5, 8400)
        return self._parse_output(raw, frame_w=w, frame_h=h)

    @staticmethod
    def _validate_input(frame: np.ndarray) -> None:
        if not isinstance(frame, np.ndarray):
            raise ValueError(f"Expected np.ndarray, got {type(frame).__name__}")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Expected (H, W, 3), got {frame.shape}")

    @staticmethod
    def _preprocess(frame: np.ndarray) -> np.ndarray:
        """BGR (H, W, 3) -> ONNX (1, 3, 640, 640) float32 [0, 1]."""
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        resized = cv2.resize(frame, (_INPUT_SIZE, _INPUT_SIZE))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return rgb.transpose(2, 0, 1)[np.newaxis]

    @staticmethod
    def _parse_output(raw: np.ndarray, frame_w: int, frame_h: int) -> PhoneDetectionOutput:
        """Parse (1, 5, 8400) YOLOv8 output.

        Format: [x, y, w, h, conf_phone] for each of 8400 anchors.
        Returns highest-confidence detection meeting threshold.
        """
        detections = raw[0]  # (5, 8400)
        confs = detections[4, :]  # (8400,)

        max_conf_idx = int(np.argmax(confs))
        max_conf = float(confs[max_conf_idx])

        if max_conf < config.PHONE_CONFIDENCE_THRESHOLD:
            return PhoneDetectionOutput(detected=False, max_confidence=0.0, bbox_norm=None)

        x = float(detections[0, max_conf_idx])
        y = float(detections[1, max_conf_idx])
        w = float(detections[2, max_conf_idx])
        h = float(detections[3, max_conf_idx])

        # Clamp to [0, 1]
        x = max(0.0, min(1.0, x))
        y = max(0.0, min(1.0, y))
        w = max(0.0, min(1.0, w))
        h = max(0.0, min(1.0, h))

        return PhoneDetectionOutput(
            detected=True,
            max_confidence=max_conf,
            bbox_norm=(x, y, w, h),
        )

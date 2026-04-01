# layer1_perception/face_detector.py — BlazeFace wrapper (ONNX)
# PRD §FR-1.1, §FR-1.5 — standard infer() interface
#
# Input:  BGR frame (H, W, 3) uint8
# Output: FaceDetection with bbox_norm (x, y, w, h) normalized to [0, 1]
#
# The ONNX model performs NMS internally via conf/iou threshold inputs.
# Output shape: (1, N, 16) — [x1, y1, x2, y2, conf, kp0_x, kp0_y, ..., kp5_x, kp5_y]

from __future__ import annotations

import cv2
import numpy as np
import onnxruntime as ort

import config
from layer1_perception.messages import FaceDetection

# ─── Model constants ─────────────────────────────────────────────────────────
_INPUT_SIZE = 128
_DET_X1 = 0
_DET_Y1 = 1
_DET_X2 = 2
_DET_Y2 = 3
_DET_CONF = 4

# NMS parameters — passed as model inputs
# NOTE: conf_threshold passed to model does NOT act as a reliable minimum-confidence filter
# for this BlazeFace ONNX export. We pass a low value to get all candidate detections,
# then apply config.FACE_CONFIDENCE_GATE manually in _parse_detections.
_NMS_CONF_THRESHOLD = 0.001   # Low — get all anchors, filter manually by FACE_CONFIDENCE_GATE
_NMS_IOU_THRESHOLD  = 0.3
_MAX_DETECTIONS     = 100     # High enough to capture all candidates before manual filter

# Geometry filters applied after confidence gate.
# BlazeFace on close-up Mac webcam produces false positives that:
#   (a) extend significantly outside the image boundary (chin/partial-face regions, y2>1)
#   (b) cover an unrealistically large fraction of the frame (whole-scene detections)
# Real face detections at laptop webcam distance score 0.43–0.63 confidence and
# fit within the image with area ≈ 5–15% of frame.
_MAX_OOB_FRACTION = 0.08   # Reject if any bbox edge is > 8% outside [0, 1]
_MAX_DET_AREA     = 0.18   # Reject if bbox area > 18% of frame
_MIN_DET_AREA     = 0.004  # Reject if bbox area < 0.4% of frame (sub-pixel noise)


class FaceDetector:
    """BlazeFace face detector wrapper (ONNX runtime).

    PRD §FR-1.5: exposes ``infer(frame) -> FaceDetection``.
    Returns the highest-confidence face detection per frame.
    """

    def __init__(self, model_path: str | None = None) -> None:
        path = model_path if model_path is not None else config.BLAZEFACE_MODEL_PATH
        opts = ort.SessionOptions()
        opts.log_severity_level = 3  # suppress VerifyOutputSizes shape warnings (NMS dynamic output)
        self._session = ort.InferenceSession(path, sess_options=opts)
        self._conf_threshold = np.array([_NMS_CONF_THRESHOLD], dtype=np.float32)
        self._iou_threshold  = np.array([_NMS_IOU_THRESHOLD],  dtype=np.float32)
        self._max_detections = np.array([_MAX_DETECTIONS],     dtype=np.int64)

    # ── Public API ───────────────────────────────────────────────────────────

    def infer(self, frame: np.ndarray) -> FaceDetection:
        """Run BlazeFace on a full BGR frame.

        Args:
            frame: BGR image array, shape (H, W, 3).

        Returns:
            FaceDetection. If no face meets the confidence gate, returns
            FaceDetection(present=False, confidence=0.0, bbox_norm=None, face_size_px=0).

        Raises:
            ValueError: If frame is not a 3-channel image.
        """
        self._validate_input(frame)
        h, w = frame.shape[:2]
        tensor = self._preprocess(frame)

        detections = self._session.run(None, {
            'image': tensor,
            'conf_threshold': self._conf_threshold,
            'max_detections': self._max_detections,
            'iou_threshold': self._iou_threshold,
        })[0]  # (1, N, 16)

        return self._parse_detections(detections, frame_w=w, frame_h=h)

    # ── Private helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _validate_input(frame: np.ndarray) -> None:
        if not isinstance(frame, np.ndarray):
            raise ValueError(f"Expected np.ndarray, got {type(frame).__name__}")
        if frame.ndim != 3:
            raise ValueError(
                f"Expected 3D array (H, W, 3), got shape {frame.shape}"
            )
        if frame.shape[2] != 3:
            raise ValueError(
                f"Expected 3 channels, got {frame.shape[2]}"
            )

    @staticmethod
    def _preprocess(frame: np.ndarray) -> np.ndarray:
        """BGR (H, W, 3) -> ONNX input (1, 3, 128, 128) float32 in [0, 1]."""
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        resized = cv2.resize(frame, (_INPUT_SIZE, _INPUT_SIZE))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return rgb.transpose(2, 0, 1)[np.newaxis]  # (1, 3, 128, 128)

    @staticmethod
    def _parse_detections(
        detections: np.ndarray,
        frame_w: int,
        frame_h: int,
    ) -> FaceDetection:
        """Select highest-confidence detection and convert to FaceDetection.

        Args:
            detections: Shape (1, N, 16) normally; model squeezes N=1 → (1, 16).
                        Each row: [x1, y1, x2, y2, conf, kp0_x, kp0_y, ...].
                        Coordinates are normalized to [0, 1].
            frame_w, frame_h: Original frame dimensions for face_size_px calculation.
        """
        # Normalize: model emits (1, 16) when exactly one detection passes NMS.
        if detections.ndim == 2:
            detections = detections[np.newaxis]  # (1, 16) → (1, 1, 16)

        n = detections.shape[1]
        if n == 0:
            return FaceDetection(
                present=False, confidence=0.0, bbox_norm=None, face_size_px=0
            )

        rows = detections[0]  # (N, 16)

        # Manual confidence gate — filter to only rows meeting FACE_CONFIDENCE_GATE.
        # The model's built-in conf_threshold param does not reliably act as a minimum
        # confidence filter for this ONNX export; we apply the gate here instead.
        confs = rows[:, _DET_CONF]
        qualifying = np.where(confs >= config.FACE_CONFIDENCE_GATE)[0]
        if qualifying.size == 0:
            return FaceDetection(
                present=False, confidence=0.0, bbox_norm=None, face_size_px=0
            )

        # ── Geometry filter ───────────────────────────────────────────────────
        # Remove false positives by rejecting detections that are:
        #   (a) significantly outside the image boundary (e.g. chin/partial-face)
        #   (b) unrealistically large (whole-scene spurious detections)
        #   (c) sub-pixel noise
        edge_lo = -_MAX_OOB_FRACTION
        edge_hi = 1.0 + _MAX_OOB_FRACTION
        bw_all = rows[:, _DET_X2] - rows[:, _DET_X1]
        bh_all = rows[:, _DET_Y2] - rows[:, _DET_Y1]
        area_all = bw_all * bh_all

        geo_mask = (
            (rows[:, _DET_X1] >= edge_lo) & (rows[:, _DET_Y1] >= edge_lo)
            & (rows[:, _DET_X2] <= edge_hi) & (rows[:, _DET_Y2] <= edge_hi)
            & (area_all <= _MAX_DET_AREA)
            & (area_all >= _MIN_DET_AREA)
        )
        geo_qualifying = qualifying[geo_mask[qualifying]]

        # Fall back to original qualifying set if geometry filter removes everything
        selection_pool = geo_qualifying if geo_qualifying.size > 0 else qualifying

        # Among the surviving detections, pick the one with the LARGEST area.
        # At close webcam distance the correct whole-face detection is larger than
        # chin/partial-face false positives that survived confidence gating.
        pool_areas = area_all[selection_pool]
        best_idx = int(selection_pool[np.argmax(pool_areas)])
        det = rows[best_idx]

        x1, y1, x2, y2 = (
            float(det[_DET_X1]),
            float(det[_DET_Y1]),
            float(det[_DET_X2]),
            float(det[_DET_Y2]),
        )
        confidence = float(det[_DET_CONF])

        # Clamp to [0, 1]
        x1 = max(0.0, min(1.0, x1))
        y1 = max(0.0, min(1.0, y1))
        x2 = max(0.0, min(1.0, x2))
        y2 = max(0.0, min(1.0, y2))

        bw = x2 - x1
        bh = y2 - y1
        face_size_px = int(bw * frame_w)

        return FaceDetection(
            present=True,
            confidence=confidence,
            bbox_norm=(x1, y1, bw, bh),
            face_size_px=face_size_px,
        )

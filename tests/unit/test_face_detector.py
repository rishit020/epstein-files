"""test_face_detector.py — Tests for BlazeFace face detector wrapper.

Tests cover: construction, preprocessing, output contract, detection parsing,
no-detection case, error handling, and edge cases.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from unittest.mock import patch
from dataclasses import dataclass

import numpy as np
import pytest

import config
from layer1_perception.messages import FaceDetection
from layer1_perception.face_detector import FaceDetector, _INPUT_SIZE


# ═══════════════════════════════════════════════════════════════════════════════
# Mock infrastructure
# ═══════════════════════════════════════════════════════════════════════════════

def _make_detection(x1=0.2, y1=0.2, x2=0.6, y2=0.8, conf=0.95) -> np.ndarray:
    """Create a single (16,) detection row."""
    row = np.zeros(16, dtype=np.float32)
    row[0], row[1], row[2], row[3], row[4] = x1, y1, x2, y2, conf
    return row


class _MockONNXSession:
    _override_output: np.ndarray | None = None

    def __init__(self, path, **kwargs):
        self._path = path

    def get_inputs(self):
        return []

    def get_outputs(self):
        return []

    def run(self, output_names, feed_dict):
        if _MockONNXSession._override_output is not None:
            return [_MockONNXSession._override_output.copy()]
        # Default: one strong detection
        det = _make_detection()
        return [det.reshape(1, 1, 16)]


class _FailingSession:
    def __init__(self, path, **kwargs):
        raise FileNotFoundError(f"Not found: {path}")


def _build_detector() -> FaceDetector:
    with patch('layer1_perception.face_detector.ort.InferenceSession', _MockONNXSession):
        return FaceDetector()


def _make_frame(h=480, w=640, dtype=np.uint8) -> np.ndarray:
    return np.random.default_rng(7).integers(0, 256, (h, w, 3), dtype=np.uint8)


# ═══════════════════════════════════════════════════════════════════════════════
# A. Construction (3 tests)
# ═══════════════════════════════════════════════════════════════════════════════

class TestConstruction:

    def test_construction_default_path(self):
        with patch('layer1_perception.face_detector.ort.InferenceSession', _MockONNXSession):
            det = FaceDetector()
            assert det._session._path == config.BLAZEFACE_MODEL_PATH

    def test_construction_custom_path(self):
        with patch('layer1_perception.face_detector.ort.InferenceSession', _MockONNXSession):
            det = FaceDetector(model_path='custom.onnx')
            assert det._session._path == 'custom.onnx'

    def test_construction_missing_model_raises(self):
        with patch('layer1_perception.face_detector.ort.InferenceSession', _FailingSession):
            with pytest.raises(FileNotFoundError):
                FaceDetector(model_path='/nonexistent.onnx')


# ═══════════════════════════════════════════════════════════════════════════════
# B. Preprocessing (4 tests)
# ═══════════════════════════════════════════════════════════════════════════════

class TestPreprocessing:

    def test_preprocess_output_shape(self):
        result = FaceDetector._preprocess(np.zeros((480, 640, 3), dtype=np.uint8))
        assert result.shape == (1, 3, _INPUT_SIZE, _INPUT_SIZE)

    def test_preprocess_dtype_float32(self):
        result = FaceDetector._preprocess(np.zeros((128, 128, 3), dtype=np.uint8))
        assert result.dtype == np.float32

    def test_preprocess_value_range_0_1(self):
        frame = np.random.default_rng(0).integers(0, 256, (200, 300, 3), dtype=np.uint8)
        result = FaceDetector._preprocess(frame)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_preprocess_bgr_to_rgb(self):
        # Pure red in BGR = (0, 0, 255) -> in RGB channel 0 is 1.0
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        frame[:, :, 2] = 255  # R channel in BGR
        result = FaceDetector._preprocess(frame)
        assert result[0, 0, 5, 5] == pytest.approx(1.0)  # R channel in RGB
        assert result[0, 2, 5, 5] == pytest.approx(0.0)  # B channel in RGB


# ═══════════════════════════════════════════════════════════════════════════════
# C. Output Contract (5 tests)
# ═══════════════════════════════════════════════════════════════════════════════

class TestOutputContract:

    def test_infer_returns_face_detection(self):
        det = _build_detector()
        assert isinstance(det.infer(_make_frame()), FaceDetection)

    def test_present_true_when_detection_found(self):
        det = _build_detector()
        result = det.infer(_make_frame())
        assert result.present is True

    def test_confidence_in_range(self):
        det = _build_detector()
        result = det.infer(_make_frame())
        assert 0.0 <= result.confidence <= 1.0

    def test_bbox_norm_is_xywh_tuple(self):
        det = _build_detector()
        result = det.infer(_make_frame())
        assert result.bbox_norm is not None
        assert len(result.bbox_norm) == 4

    def test_face_size_px_positive(self):
        det = _build_detector()
        result = det.infer(_make_frame(w=640))
        assert result.face_size_px > 0


# ═══════════════════════════════════════════════════════════════════════════════
# D. Detection Parsing (5 tests)
# ═══════════════════════════════════════════════════════════════════════════════

class TestDetectionParsing:

    def test_no_detections_returns_not_present(self):
        empty = np.zeros((1, 0, 16), dtype=np.float32)
        result = FaceDetector._parse_detections(empty, frame_w=640, frame_h=480)
        assert result.present is False
        assert result.confidence == 0.0
        assert result.bbox_norm is None
        assert result.face_size_px == 0

    def test_single_detection_parsed_correctly(self):
        det_row = _make_detection(x1=0.1, y1=0.2, x2=0.5, y2=0.8, conf=0.9)
        dets = det_row.reshape(1, 1, 16)
        result = FaceDetector._parse_detections(dets, frame_w=100, frame_h=100)
        assert result.present is True
        assert result.confidence == pytest.approx(0.9)
        x, y, w, h = result.bbox_norm
        assert x == pytest.approx(0.1)
        assert y == pytest.approx(0.2)
        assert w == pytest.approx(0.4, abs=1e-5)
        assert h == pytest.approx(0.6, abs=1e-5)

    def test_highest_confidence_selected_from_multiple(self):
        det_low = _make_detection(x1=0.0, y1=0.0, x2=0.2, y2=0.2, conf=0.6)
        det_high = _make_detection(x1=0.3, y1=0.3, x2=0.7, y2=0.8, conf=0.95)
        dets = np.stack([det_low, det_high])[np.newaxis]  # (1, 2, 16)
        result = FaceDetector._parse_detections(dets, frame_w=640, frame_h=480)
        assert result.confidence == pytest.approx(0.95)
        x, y, w, h = result.bbox_norm
        assert x == pytest.approx(0.3)

    def test_face_size_px_uses_frame_width(self):
        det_row = _make_detection(x1=0.1, y1=0.1, x2=0.5, y2=0.9, conf=0.9)
        dets = det_row.reshape(1, 1, 16)
        result = FaceDetector._parse_detections(dets, frame_w=1000, frame_h=1000)
        # bbox width ~0.4, frame_w = 1000 -> face_size_px ~400 (float32 rounding)
        assert 395 <= result.face_size_px <= 405

    def test_bbox_clamped_to_0_1(self):
        det_row = _make_detection(x1=-0.1, y1=-0.1, x2=1.2, y2=1.3, conf=0.8)
        dets = det_row.reshape(1, 1, 16)
        result = FaceDetector._parse_detections(dets, frame_w=640, frame_h=480)
        x, y, w, h = result.bbox_norm
        assert x >= 0.0
        assert y >= 0.0
        assert x + w <= 1.0
        assert y + h <= 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# E. Error Handling (4 tests)
# ═══════════════════════════════════════════════════════════════════════════════

class TestErrorHandling:

    def test_none_input_raises(self):
        det = _build_detector()
        with pytest.raises(ValueError):
            det.infer(None)

    def test_2d_input_raises(self):
        det = _build_detector()
        with pytest.raises(ValueError):
            det.infer(np.zeros((480, 640)))

    def test_4_channel_raises(self):
        det = _build_detector()
        with pytest.raises(ValueError):
            det.infer(np.zeros((480, 640, 4), dtype=np.uint8))

    def test_1d_input_raises(self):
        det = _build_detector()
        with pytest.raises(ValueError):
            det.infer(np.zeros((100,)))


# ═══════════════════════════════════════════════════════════════════════════════
# F. Edge Cases (4 tests)
# ═══════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:

    def test_tiny_frame_1x1(self):
        det = _build_detector()
        result = det.infer(np.zeros((1, 1, 3), dtype=np.uint8))
        assert isinstance(result, FaceDetection)

    def test_large_frame_2k(self):
        det = _build_detector()
        result = det.infer(np.zeros((1080, 1920, 3), dtype=np.uint8))
        assert isinstance(result, FaceDetection)

    def test_float_input_handled(self):
        det = _build_detector()
        frame = np.random.default_rng(1).uniform(0, 255, (480, 640, 3)).astype(np.float64)
        result = det.infer(frame)
        assert isinstance(result, FaceDetection)

    def test_no_detection_mock(self):
        det = _build_detector()
        _MockONNXSession._override_output = np.zeros((1, 0, 16), dtype=np.float32)
        try:
            result = det.infer(_make_frame())
            assert result.present is False
            assert result.bbox_norm is None
        finally:
            _MockONNXSession._override_output = None

    def test_single_detection_squeezed_shape(self):
        """Model emits (1, 16) when exactly N=1 detection passes NMS — must be handled."""
        det = _build_detector()
        row = _make_detection(x1=0.1, y1=0.1, x2=0.5, y2=0.7, conf=0.88)
        # Squeezed shape: (1, 16) instead of (1, 1, 16)
        _MockONNXSession._override_output = row.reshape(1, 16)
        try:
            result = det.infer(_make_frame())
            assert result.present is True
            assert result.confidence == pytest.approx(0.88, abs=1e-4)
        finally:
            _MockONNXSession._override_output = None

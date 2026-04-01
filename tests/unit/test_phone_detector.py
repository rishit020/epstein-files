"""test_phone_detector.py — Tests for YOLOv8n phone detector wrapper."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from unittest.mock import patch
import numpy as np
import pytest
import config
from layer1_perception.messages import PhoneDetectionOutput
from layer1_perception.phone_detector import PhoneDetector


class _MockSession:
    _override = None
    def __init__(self, path, **kw):
        self._path = path
    def run(self, _, feed):
        if _MockSession._override is not None:
            return [_MockSession._override.copy()]
        det = np.zeros((1, 5, 8400), dtype=np.float32)
        det[0, 4, 1000] = 0.85  # conf at index 1000
        return [det]


def _build() -> PhoneDetector:
    with patch('layer1_perception.phone_detector.ort.InferenceSession', _MockSession):
        return PhoneDetector()


class TestPhoneDetector:
    def test_construction(self):
        with patch('layer1_perception.phone_detector.ort.InferenceSession', _MockSession):
            d = PhoneDetector()
            assert d._session._path == config.YOLO_MODEL_PATH

    def test_preprocess_shape(self):
        result = PhoneDetector._preprocess(np.zeros((480, 640, 3), dtype=np.uint8))
        assert result.shape == (1, 3, 640, 640)

    def test_infer_returns_phone_detection_output(self):
        d = _build()
        result = d.infer(np.zeros((480, 640, 3), dtype=np.uint8))
        assert isinstance(result, PhoneDetectionOutput)

    def test_detected_true_above_threshold(self):
        d = _build()
        result = d.infer(np.zeros((480, 640, 3), dtype=np.uint8))
        assert result.detected is True

    def test_confidence_in_range(self):
        d = _build()
        result = d.infer(np.zeros((480, 640, 3), dtype=np.uint8))
        assert 0.0 <= result.max_confidence <= 1.0

    def test_no_detection_below_threshold(self):
        d = _build()
        det = np.zeros((1, 5, 8400), dtype=np.float32)
        det[0, 4, 100] = 0.3  # below threshold
        _MockSession._override = det
        try:
            result = d.infer(np.zeros((480, 640, 3), dtype=np.uint8))
            assert result.detected is False
        finally:
            _MockSession._override = None

    def test_invalid_input_none_raises(self):
        d = _build()
        with pytest.raises(ValueError):
            d.infer(None)

    def test_invalid_input_2d_raises(self):
        d = _build()
        with pytest.raises(ValueError):
            d.infer(np.zeros((480, 640)))

    def test_float_input_handled(self):
        d = _build()
        frame = np.random.rand(480, 640, 3).astype(np.float32)
        result = d.infer(frame)
        assert isinstance(result, PhoneDetectionOutput)

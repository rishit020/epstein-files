"""test_gaze_model.py — Comprehensive tests for MobileNetV3+LSTM gaze model wrapper.

Tests cover: construction, preprocessing, output contract, angle decoding,
confidence, hidden-state passthrough, per-eye field consistency, determinism,
error handling, edge cases.

All tests use a mock ONNX session — no real model file required.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from unittest.mock import patch
from dataclasses import dataclass

import numpy as np
import pytest

from layer1_perception.messages import GazeOutput
from layer1_perception.gaze_model import (
    GazeModel,
    _softmax,
    _entropy_confidence,
    _INPUT_SIZE,
    _NUM_BINS,
    _IDX,
    _ANGLE_SCALE,
    _ANGLE_OFFSET,
    _IMAGENET_MEAN,
    _IMAGENET_STD,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Mock infrastructure
# ═══════════════════════════════════════════════════════════════════════════════

def _uniform_logits() -> np.ndarray:
    """90-bin logits for a near-uniform distribution (low confidence)."""
    return np.zeros(_NUM_BINS, dtype=np.float32)


def _peaked_logits(peak_bin: int = 45, scale: float = 10.0) -> np.ndarray:
    """90-bin logits with most probability mass at peak_bin (high confidence)."""
    logits = np.zeros(_NUM_BINS, dtype=np.float32)
    logits[peak_bin] = scale
    return logits


@dataclass
class _MockInput:
    name: str
    shape: list


class _MockONNXSession:
    """Deterministic mock for onnxruntime.InferenceSession.

    Class-level overrides let individual tests inject specific logits.
    """
    _override_yaw: np.ndarray | None = None
    _override_pitch: np.ndarray | None = None

    def __init__(self, path, **kwargs):
        self._path = path

    def get_inputs(self):
        return [_MockInput(name='input', shape=[1, 3, _INPUT_SIZE, _INPUT_SIZE])]

    def get_outputs(self):
        return [
            _MockInput(name='yaw',   shape=[1, _NUM_BINS]),
            _MockInput(name='pitch', shape=[1, _NUM_BINS]),
        ]

    def run(self, output_names, feed_dict):
        yaw   = _MockONNXSession._override_yaw   if _MockONNXSession._override_yaw   is not None else _peaked_logits(45)
        pitch = _MockONNXSession._override_pitch if _MockONNXSession._override_pitch is not None else _peaked_logits(45)
        return [
            yaw[np.newaxis].astype(np.float32),    # (1, 90)
            pitch[np.newaxis].astype(np.float32),  # (1, 90)
        ]


class _FailingMockSession:
    def __init__(self, path, **kwargs):
        raise FileNotFoundError(f"Model not found: {path}")


def _build_model() -> GazeModel:
    with patch('layer1_perception.gaze_model.ort.InferenceSession', _MockONNXSession):
        return GazeModel()


def _make_crop(h: int = 200, w: int = 300, dtype=np.uint8) -> np.ndarray:
    if dtype == np.uint8:
        return np.random.default_rng(7).integers(0, 256, (h, w, 3), dtype=np.uint8)
    return np.random.default_rng(7).uniform(0, 255, (h, w, 3)).astype(dtype)


# ═══════════════════════════════════════════════════════════════════════════════
# A. Construction (3 tests)
# ═══════════════════════════════════════════════════════════════════════════════

class TestConstruction:

    def test_construction_default_path(self):
        """Constructor accepts no arguments and uses config.GAZE_MODEL_PATH."""
        with patch('layer1_perception.gaze_model.ort.InferenceSession', _MockONNXSession):
            model = GazeModel()
            assert model._session is not None

    def test_construction_custom_path(self):
        """Constructor accepts a custom model path."""
        with patch('layer1_perception.gaze_model.ort.InferenceSession', _MockONNXSession):
            model = GazeModel(model_path='custom/gaze.onnx')
            assert model._session._path == 'custom/gaze.onnx'

    def test_construction_missing_model_raises(self):
        """Constructor propagates error when model file is missing."""
        with patch('layer1_perception.gaze_model.ort.InferenceSession', _FailingMockSession):
            with pytest.raises(FileNotFoundError):
                GazeModel(model_path='/nonexistent/model.onnx')


# ═══════════════════════════════════════════════════════════════════════════════
# B. Preprocessing (6 tests)
# ═══════════════════════════════════════════════════════════════════════════════

class TestPreprocessing:

    def test_preprocess_output_shape(self):
        """_preprocess produces (1, 3, 448, 448) from arbitrary input."""
        result = GazeModel._preprocess(np.zeros((200, 300, 3), dtype=np.uint8))
        assert result.shape == (1, 3, _INPUT_SIZE, _INPUT_SIZE)

    def test_preprocess_dtype_float32(self):
        """Output dtype is float32."""
        result = GazeModel._preprocess(np.zeros((100, 100, 3), dtype=np.uint8))
        assert result.dtype == np.float32

    def test_preprocess_imagenet_normalised(self):
        """A pure-white pixel (255, 255, 255) normalises to known ImageNet-corrected value."""
        crop = np.full((10, 10, 3), 255, dtype=np.uint8)
        result = GazeModel._preprocess(crop)
        # White pixel: raw RGB = (1.0, 1.0, 1.0) -> (1.0 - mean) / std for each channel
        expected_r = (1.0 - float(_IMAGENET_MEAN[0])) / float(_IMAGENET_STD[0])
        assert result[0, 0, 5, 5] == pytest.approx(expected_r, abs=1e-4)

    def test_preprocess_bgr_to_rgb(self):
        """Pure-blue BGR pixel ends up in the blue (index 2) channel after normalisation."""
        crop = np.zeros((10, 10, 3), dtype=np.uint8)
        crop[:, :, 0] = 255   # B in BGR
        result = GazeModel._preprocess(crop)
        # After BGR->RGB: R=0, G=0, B=1.0 → channel 2 should be normalised blue
        expected_b = (1.0 - float(_IMAGENET_MEAN[2])) / float(_IMAGENET_STD[2])
        assert result[0, 2, 5, 5] == pytest.approx(expected_b, abs=1e-4)
        # R and G channels should reflect their normalised zero-pixel values
        expected_zero_r = (0.0 - float(_IMAGENET_MEAN[0])) / float(_IMAGENET_STD[0])
        assert result[0, 0, 5, 5] == pytest.approx(expected_zero_r, abs=1e-4)

    def test_preprocess_resize_various_sizes(self):
        """Various input sizes all produce (1, 3, 448, 448)."""
        for h, w in [(50, 50), (640, 480), (1, 1), (1920, 1080)]:
            result = GazeModel._preprocess(np.zeros((h, w, 3), dtype=np.uint8))
            assert result.shape == (1, 3, _INPUT_SIZE, _INPUT_SIZE)

    def test_preprocess_float_input_clamped(self):
        """Float64 input is clamped to uint8 before normalisation."""
        crop = np.full((10, 10, 3), 300.0, dtype=np.float64)   # out-of-range floats
        result = GazeModel._preprocess(crop)
        assert result.shape == (1, 3, _INPUT_SIZE, _INPUT_SIZE)
        assert not np.any(np.isnan(result))


# ═══════════════════════════════════════════════════════════════════════════════
# C. Output Contract (7 tests)
# ═══════════════════════════════════════════════════════════════════════════════

class TestOutputContract:

    def test_infer_returns_two_tuple(self):
        """infer() returns a 2-tuple."""
        model = _build_model()
        result = model.infer(_make_crop())
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_first_element_is_gaze_output(self):
        """First element of return tuple is GazeOutput."""
        model = _build_model()
        gaze, _ = model.infer(_make_crop())
        assert isinstance(gaze, GazeOutput)

    def test_output_angles_are_floats(self):
        """All angle fields are Python floats."""
        model = _build_model()
        gaze, _ = model.infer(_make_crop())
        assert isinstance(gaze.combined_yaw,   float)
        assert isinstance(gaze.combined_pitch, float)
        assert isinstance(gaze.left_eye_yaw,   float)
        assert isinstance(gaze.left_eye_pitch, float)
        assert isinstance(gaze.right_eye_yaw,  float)
        assert isinstance(gaze.right_eye_pitch, float)

    def test_output_confidence_range(self):
        """confidence is in [0.0, 1.0]."""
        model = _build_model()
        gaze, _ = model.infer(_make_crop())
        assert 0.0 <= gaze.confidence <= 1.0

    def test_output_valid_is_bool(self):
        """valid is a bool."""
        model = _build_model()
        gaze, _ = model.infer(_make_crop())
        assert isinstance(gaze.valid, bool)

    def test_output_valid_true_on_normal_input(self):
        """valid=True for a normal face crop (model ran successfully)."""
        model = _build_model()
        gaze, _ = model.infer(_make_crop())
        assert gaze.valid is True

    def test_hidden_state_is_none(self):
        """Second element (new_hidden_state) is None — no LSTM state in current model."""
        model = _build_model()
        _, hidden = model.infer(_make_crop())
        assert hidden is None


# ═══════════════════════════════════════════════════════════════════════════════
# D. Angle Decoding (5 tests)
# ═══════════════════════════════════════════════════════════════════════════════

class TestAngleDecoding:

    def test_peaked_at_bin_45_gives_zero_degrees(self):
        """All probability at bin 45 → angle = 45×4−180 = 0°."""
        _MockONNXSession._override_yaw   = _peaked_logits(45, scale=50.0)
        _MockONNXSession._override_pitch = _peaked_logits(45, scale=50.0)
        try:
            model = _build_model()
            gaze, _ = model.infer(_make_crop())
            assert gaze.combined_yaw   == pytest.approx(0.0, abs=0.5)
            assert gaze.combined_pitch == pytest.approx(0.0, abs=0.5)
        finally:
            _MockONNXSession._override_yaw   = None
            _MockONNXSession._override_pitch = None

    def test_peaked_at_bin_0_gives_negative_180(self):
        """All probability at bin 0 → angle = 0×4−180 = −180°."""
        _MockONNXSession._override_yaw   = _peaked_logits(0, scale=50.0)
        _MockONNXSession._override_pitch = _peaked_logits(0, scale=50.0)
        try:
            model = _build_model()
            gaze, _ = model.infer(_make_crop())
            assert gaze.combined_yaw   == pytest.approx(-180.0, abs=0.5)
            assert gaze.combined_pitch == pytest.approx(-180.0, abs=0.5)
        finally:
            _MockONNXSession._override_yaw   = None
            _MockONNXSession._override_pitch = None

    def test_peaked_at_bin_67_gives_88_degrees(self):
        """All probability at bin 67 → angle = 67×4−180 = 88°."""
        _MockONNXSession._override_yaw   = _peaked_logits(67, scale=50.0)
        _MockONNXSession._override_pitch = _peaked_logits(67, scale=50.0)
        try:
            model = _build_model()
            gaze, _ = model.infer(_make_crop())
            assert gaze.combined_yaw   == pytest.approx(88.0, abs=0.5)
            assert gaze.combined_pitch == pytest.approx(88.0, abs=0.5)
        finally:
            _MockONNXSession._override_yaw   = None
            _MockONNXSession._override_pitch = None

    def test_uniform_logits_give_midpoint_angle(self):
        """Uniform logits → soft-argmax at index 44.5 → angle = 44.5×4−180 = −2°."""
        _MockONNXSession._override_yaw   = _uniform_logits()
        _MockONNXSession._override_pitch = _uniform_logits()
        try:
            model = _build_model()
            gaze, _ = model.infer(_make_crop())
            expected = 44.5 * _ANGLE_SCALE + _ANGLE_OFFSET  # -2.0
            assert gaze.combined_yaw   == pytest.approx(expected, abs=0.1)
            assert gaze.combined_pitch == pytest.approx(expected, abs=0.1)
        finally:
            _MockONNXSession._override_yaw   = None
            _MockONNXSession._override_pitch = None

    def test_yaw_pitch_decoded_independently(self):
        """Different logits for yaw and pitch produce different decoded angles."""
        _MockONNXSession._override_yaw   = _peaked_logits(30, scale=50.0)
        _MockONNXSession._override_pitch = _peaked_logits(60, scale=50.0)
        try:
            model = _build_model()
            gaze, _ = model.infer(_make_crop())
            expected_yaw   = 30 * _ANGLE_SCALE + _ANGLE_OFFSET   # -60°
            expected_pitch = 60 * _ANGLE_SCALE + _ANGLE_OFFSET   # 60°
            assert gaze.combined_yaw   == pytest.approx(expected_yaw,   abs=0.5)
            assert gaze.combined_pitch == pytest.approx(expected_pitch, abs=0.5)
        finally:
            _MockONNXSession._override_yaw   = None
            _MockONNXSession._override_pitch = None


# ═══════════════════════════════════════════════════════════════════════════════
# E. Confidence (4 tests)
# ═══════════════════════════════════════════════════════════════════════════════

class TestConfidence:

    def test_peaked_distribution_gives_high_confidence(self):
        """Strongly peaked logits → confidence close to 1.0."""
        _MockONNXSession._override_yaw   = _peaked_logits(45, scale=50.0)
        _MockONNXSession._override_pitch = _peaked_logits(45, scale=50.0)
        try:
            model = _build_model()
            gaze, _ = model.infer(_make_crop())
            assert gaze.confidence > 0.9
        finally:
            _MockONNXSession._override_yaw   = None
            _MockONNXSession._override_pitch = None

    def test_uniform_distribution_gives_low_confidence(self):
        """Uniform logits → near-zero confidence."""
        _MockONNXSession._override_yaw   = _uniform_logits()
        _MockONNXSession._override_pitch = _uniform_logits()
        try:
            model = _build_model()
            gaze, _ = model.infer(_make_crop())
            assert gaze.confidence < 0.05
        finally:
            _MockONNXSession._override_yaw   = None
            _MockONNXSession._override_pitch = None

    def test_confidence_higher_for_peaked_than_flat(self):
        """Peaked distribution always produces higher confidence than a flat one."""
        model = _build_model()

        _MockONNXSession._override_yaw   = _peaked_logits(45, scale=10.0)
        _MockONNXSession._override_pitch = _peaked_logits(45, scale=10.0)
        peaked_gaze, _ = model.infer(_make_crop())

        _MockONNXSession._override_yaw   = _uniform_logits()
        _MockONNXSession._override_pitch = _uniform_logits()
        uniform_gaze, _ = model.infer(_make_crop())

        _MockONNXSession._override_yaw   = None
        _MockONNXSession._override_pitch = None

        assert peaked_gaze.confidence > uniform_gaze.confidence

    def test_confidence_in_range_for_arbitrary_logits(self):
        """Random logits always produce confidence in [0, 1]."""
        model = _build_model()
        rng = np.random.default_rng(42)
        for _ in range(10):
            _MockONNXSession._override_yaw   = rng.uniform(-5, 5, _NUM_BINS).astype(np.float32)
            _MockONNXSession._override_pitch = rng.uniform(-5, 5, _NUM_BINS).astype(np.float32)
            gaze, _ = model.infer(_make_crop())
            assert 0.0 <= gaze.confidence <= 1.0
        _MockONNXSession._override_yaw   = None
        _MockONNXSession._override_pitch = None


# ═══════════════════════════════════════════════════════════════════════════════
# F. Per-eye Field Consistency (3 tests)
# ═══════════════════════════════════════════════════════════════════════════════

class TestPerEyeConsistency:
    """Current model produces a single gaze estimate; left/right mirror combined."""

    def test_left_right_equal_combined_yaw(self):
        """left_eye_yaw == right_eye_yaw == combined_yaw."""
        model = _build_model()
        gaze, _ = model.infer(_make_crop())
        assert gaze.left_eye_yaw  == gaze.combined_yaw
        assert gaze.right_eye_yaw == gaze.combined_yaw

    def test_left_right_equal_combined_pitch(self):
        """left_eye_pitch == right_eye_pitch == combined_pitch."""
        model = _build_model()
        gaze, _ = model.infer(_make_crop())
        assert gaze.left_eye_pitch  == gaze.combined_pitch
        assert gaze.right_eye_pitch == gaze.combined_pitch

    def test_per_eye_fields_are_floats(self):
        """All individual eye fields are Python floats."""
        model = _build_model()
        gaze, _ = model.infer(_make_crop())
        for field in (gaze.left_eye_yaw, gaze.left_eye_pitch,
                      gaze.right_eye_yaw, gaze.right_eye_pitch):
            assert isinstance(field, float)


# ═══════════════════════════════════════════════════════════════════════════════
# G. Hidden State Passthrough (3 tests)
# ═══════════════════════════════════════════════════════════════════════════════

class TestHiddenState:

    def test_hidden_state_none_input_accepted(self):
        """hidden_state=None is the default and is accepted without error."""
        model = _build_model()
        gaze, _ = model.infer(_make_crop(), hidden_state=None)
        assert isinstance(gaze, GazeOutput)

    def test_hidden_state_tuple_input_accepted(self):
        """Passing a tuple as hidden_state is accepted (passthrough interface)."""
        model = _build_model()
        fake_state = (np.zeros((1, 64), dtype=np.float32), np.zeros((1, 64), dtype=np.float32))
        gaze, new_state = model.infer(_make_crop(), hidden_state=fake_state)
        assert isinstance(gaze, GazeOutput)
        assert new_state is None  # always None — no LSTM state in current model

    def test_output_identical_regardless_of_hidden_state_input(self):
        """Gaze output is the same whether hidden_state is None or a tuple."""
        model = _build_model()
        crop = _make_crop()
        gaze_no_state, _ = model.infer(crop.copy(), hidden_state=None)
        fake_state = (np.ones((1, 64), dtype=np.float32), np.ones((1, 64), dtype=np.float32))
        gaze_with_state, _ = model.infer(crop.copy(), hidden_state=fake_state)
        assert gaze_no_state.combined_yaw   == gaze_with_state.combined_yaw
        assert gaze_no_state.combined_pitch == gaze_with_state.combined_pitch


# ═══════════════════════════════════════════════════════════════════════════════
# H. Determinism (2 tests)
# ═══════════════════════════════════════════════════════════════════════════════

class TestDeterminism:

    def test_same_input_same_output(self):
        """Two calls with identical input produce identical GazeOutput."""
        model = _build_model()
        crop = _make_crop()
        g1, _ = model.infer(crop.copy())
        g2, _ = model.infer(crop.copy())
        assert g1.combined_yaw   == g2.combined_yaw
        assert g1.combined_pitch == g2.combined_pitch
        assert g1.confidence     == g2.confidence

    def test_deterministic_with_seeded_input(self):
        """Seeded input is reproducible across independent calls."""
        model = _build_model()
        crop = np.random.default_rng(42).integers(0, 256, (112, 112, 3), dtype=np.uint8)
        g1, _ = model.infer(crop)
        g2, _ = model.infer(crop)
        assert g1.combined_yaw == g2.combined_yaw


# ═══════════════════════════════════════════════════════════════════════════════
# I. Error Handling (4 tests)
# ═══════════════════════════════════════════════════════════════════════════════

class TestErrorHandling:

    def test_none_input_raises(self):
        model = _build_model()
        with pytest.raises(ValueError):
            model.infer(None)

    def test_1d_input_raises(self):
        model = _build_model()
        with pytest.raises(ValueError):
            model.infer(np.zeros((100,)))

    def test_2d_input_raises(self):
        model = _build_model()
        with pytest.raises(ValueError):
            model.infer(np.zeros((100, 100)))

    def test_4_channel_input_raises(self):
        model = _build_model()
        with pytest.raises(ValueError):
            model.infer(np.zeros((100, 100, 4), dtype=np.uint8))


# ═══════════════════════════════════════════════════════════════════════════════
# J. Edge Cases (4 tests)
# ═══════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:

    def test_tiny_crop_1x1(self):
        """1×1 pixel crop processes without error."""
        model = _build_model()
        gaze, _ = model.infer(np.zeros((1, 1, 3), dtype=np.uint8))
        assert isinstance(gaze, GazeOutput)

    def test_black_image(self):
        """All-black frame produces valid GazeOutput."""
        model = _build_model()
        gaze, _ = model.infer(np.zeros((112, 112, 3), dtype=np.uint8))
        assert isinstance(gaze, GazeOutput)
        assert 0.0 <= gaze.confidence <= 1.0

    def test_white_image(self):
        """All-white frame produces valid GazeOutput."""
        model = _build_model()
        gaze, _ = model.infer(np.full((112, 112, 3), 255, dtype=np.uint8))
        assert isinstance(gaze, GazeOutput)
        assert 0.0 <= gaze.confidence <= 1.0

    def test_float_input_handled(self):
        """Float32 input is clamped and processed correctly."""
        model = _build_model()
        crop = np.random.default_rng(42).uniform(0, 255, (100, 100, 3)).astype(np.float32)
        gaze, _ = model.infer(crop)
        assert isinstance(gaze, GazeOutput)


# ═══════════════════════════════════════════════════════════════════════════════
# K. Pure-function unit tests (_softmax, _entropy_confidence)
# ═══════════════════════════════════════════════════════════════════════════════

class TestPureFunctions:

    def test_softmax_sums_to_one(self):
        """softmax output always sums to 1.0."""
        for _ in range(5):
            x = np.random.default_rng(0).uniform(-5, 5, 90).astype(np.float32)
            probs = _softmax(x)
            assert probs.sum() == pytest.approx(1.0, abs=1e-5)

    def test_softmax_all_positive(self):
        """All softmax output values are > 0."""
        x = np.zeros(90, dtype=np.float32)
        probs = _softmax(x)
        assert np.all(probs > 0)

    def test_softmax_peak_at_highest_logit(self):
        """Highest logit bin has highest probability after softmax."""
        x = np.zeros(90, dtype=np.float32)
        x[33] = 5.0
        probs = _softmax(x)
        assert int(np.argmax(probs)) == 33

    def test_entropy_confidence_uniform_near_zero(self):
        """Uniform distribution → confidence close to 0."""
        probs = np.ones(90, dtype=np.float32) / 90.0
        conf = _entropy_confidence(probs, probs)
        assert conf < 0.05

    def test_entropy_confidence_onehot_is_one(self):
        """One-hot distribution → confidence = 1.0."""
        probs = np.zeros(90, dtype=np.float32)
        probs[45] = 1.0
        conf = _entropy_confidence(probs, probs)
        assert conf == pytest.approx(1.0, abs=0.01)

    def test_entropy_confidence_in_range(self):
        """confidence is always in [0, 1] for any valid probability distribution."""
        rng = np.random.default_rng(99)
        for _ in range(20):
            logits = rng.uniform(-3, 3, 90).astype(np.float32)
            probs = _softmax(logits)
            conf = _entropy_confidence(probs, probs)
            assert 0.0 <= conf <= 1.0

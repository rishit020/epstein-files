"""test_landmark_model.py — Comprehensive tests for PFLD 68-point landmark wrapper.

Tests cover: construction, preprocessing, output contract, confidence derivation,
pose_valid derivation, iBUG index correctness, determinism, error handling, edge cases.

All tests use a mock ONNX session — no real model file required.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from unittest.mock import patch, MagicMock
from dataclasses import dataclass

import numpy as np
import pytest

import config
from layer1_perception.messages import LandmarkOutput
from layer1_perception.landmark_model import (
    LandmarkModel,
    _INPUT_SIZE,
    _NUM_LANDMARKS,
    _OUTPUT_FLAT_SIZE,
    _BOUNDARY_MARGIN,
    _IOD_MIN_SEVERE,
    _IOD_MIN_WARN,
    _IOD_FACE_WIDTH_RATIO_MIN,
    _FACE_WIDTH_MIN,
    _LEFT_EYE_OUTER,
    _RIGHT_EYE_OUTER,
    _NOSE_TIP,
    _CHIN,
    _LEFT_JAW,
    _RIGHT_JAW,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Mock infrastructure
# ═══════════════════════════════════════════════════════════════════════════════

def _frontal_face_landmarks_flat() -> np.ndarray:
    """Return a deterministic (1, 136) array representing a plausible frontal face.

    Key properties:
    - All coords in [0.15, 0.85] (well within bounds)
    - Left eye outer (idx 36) at x~0.30, right eye outer (idx 45) at x~0.70  (IOD ~0.40)
    - Nose tip (idx 30) at x~0.50, y~0.50
    - Chin (idx 8) at x~0.50, y~0.80 (below nose)
    - Left jaw (idx 0) at x~0.10, right jaw (idx 16) at x~0.90 (face_width ~0.80)
    - Mouth corners (48, 54) below nose
    """
    rng = np.random.default_rng(42)
    flat = rng.uniform(0.25, 0.75, size=(1, 136)).astype(np.float32)
    lm = flat[0].reshape(68, 2)

    # Set key anatomical landmarks for a frontal face
    lm[_LEFT_JAW] = [0.10, 0.50]         # left jaw
    lm[_CHIN] = [0.50, 0.85]             # chin — below nose
    lm[_RIGHT_JAW] = [0.90, 0.50]        # right jaw
    lm[_NOSE_TIP] = [0.50, 0.55]         # nose tip
    lm[_LEFT_EYE_OUTER] = [0.30, 0.40]   # left eye outer
    lm[_RIGHT_EYE_OUTER] = [0.70, 0.40]  # right eye outer
    lm[48] = [0.35, 0.70]                # left mouth corner
    lm[54] = [0.65, 0.70]                # right mouth corner

    return lm.reshape(1, 136).astype(np.float32)


@dataclass
class _MockInput:
    name: str
    shape: list


class _MockONNXSession:
    """Deterministic mock for onnxruntime.InferenceSession."""

    # Class-level override: set this to control what run() returns
    _override_output: np.ndarray | None = None

    def __init__(self, path, **kwargs):
        self._path = path

    def get_inputs(self):
        return [_MockInput(name="input", shape=[1, 3, 112, 112])]

    def get_outputs(self):
        return [_MockInput(name="output", shape=[1, 136])]

    def run(self, output_names, feed_dict):
        if _MockONNXSession._override_output is not None:
            return [_MockONNXSession._override_output.copy()]
        return [_frontal_face_landmarks_flat()]


class _FailingMockSession:
    """Mock that raises on construction (simulates missing model)."""

    def __init__(self, path, **kwargs):
        raise FileNotFoundError(f"Model not found: {path}")


def _make_crop(h: int = 200, w: int = 300, dtype=np.uint8) -> np.ndarray:
    """Create a synthetic BGR face crop."""
    if dtype == np.uint8:
        return np.random.default_rng(99).integers(0, 256, (h, w, 3), dtype=np.uint8)
    return np.random.default_rng(99).uniform(0, 1, (h, w, 3)).astype(dtype)


def _build_model() -> LandmarkModel:
    """Create a LandmarkModel with mocked ONNX session."""
    with patch('layer1_perception.landmark_model.ort.InferenceSession', _MockONNXSession):
        return LandmarkModel()


def _custom_landmarks(modifier_fn) -> np.ndarray:
    """Create (1, 136) landmarks from frontal base, then apply modifier_fn."""
    base = _frontal_face_landmarks_flat().copy()
    lm = base[0].reshape(68, 2)
    modifier_fn(lm)
    return lm.reshape(1, 136).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# A. Construction (3 tests)
# ═══════════════════════════════════════════════════════════════════════════════

class TestConstruction:

    def test_construction_default_path(self):
        """Constructor uses config.PFLD_MODEL_PATH by default."""
        with patch('layer1_perception.landmark_model.ort.InferenceSession', _MockONNXSession) as mock_cls:
            model = LandmarkModel()
            assert model._input_name == "input"

    def test_construction_custom_path(self):
        """Constructor accepts a custom model path."""
        with patch('layer1_perception.landmark_model.ort.InferenceSession', _MockONNXSession):
            model = LandmarkModel(model_path='custom/path.onnx')
            assert model._session._path == 'custom/path.onnx'

    def test_construction_missing_model_raises(self):
        """Constructor propagates error when model file is missing."""
        with patch('layer1_perception.landmark_model.ort.InferenceSession', _FailingMockSession):
            with pytest.raises(FileNotFoundError):
                LandmarkModel(model_path='/nonexistent/model.onnx')


# ═══════════════════════════════════════════════════════════════════════════════
# B. Preprocessing (5 tests)
# ═══════════════════════════════════════════════════════════════════════════════

class TestPreprocessing:

    def test_preprocess_output_shape(self):
        """_preprocess produces (1, 3, 112, 112) from arbitrary input size."""
        result = LandmarkModel._preprocess(np.zeros((200, 300, 3), dtype=np.uint8))
        assert result.shape == (1, 3, _INPUT_SIZE, _INPUT_SIZE)

    def test_preprocess_dtype_float32(self):
        """Output dtype is float32."""
        result = LandmarkModel._preprocess(np.zeros((100, 100, 3), dtype=np.uint8))
        assert result.dtype == np.float32

    def test_preprocess_value_range_0_1(self):
        """All values in [0.0, 1.0] for uint8 input."""
        crop = np.random.default_rng(42).integers(0, 256, (100, 100, 3), dtype=np.uint8)
        result = LandmarkModel._preprocess(crop)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_preprocess_bgr_to_rgb_conversion(self):
        """BGR->RGB channel reorder: pure blue BGR pixel -> channel 2 in output is 1.0."""
        # Pure blue in BGR = (255, 0, 0) -> in RGB = (0, 0, 255) -> channel 2 = blue = 1.0
        crop = np.zeros((10, 10, 3), dtype=np.uint8)
        crop[:, :, 0] = 255  # B channel in BGR
        result = LandmarkModel._preprocess(crop)
        # After BGR->RGB: R=0, G=0, B=255 -> channel ordering is [R, G, B]
        assert result[0, 2, 5, 5] == pytest.approx(1.0)  # B channel in RGB
        assert result[0, 0, 5, 5] == pytest.approx(0.0)  # R channel in RGB

    def test_preprocess_resize_various_sizes(self):
        """Various input sizes all produce (1, 3, 112, 112)."""
        for h, w in [(50, 50), (640, 480), (1, 1), (1920, 1080)]:
            result = LandmarkModel._preprocess(np.zeros((h, w, 3), dtype=np.uint8))
            assert result.shape == (1, 3, _INPUT_SIZE, _INPUT_SIZE)


# ═══════════════════════════════════════════════════════════════════════════════
# C. Output Contract (6 tests)
# ═══════════════════════════════════════════════════════════════════════════════

class TestOutputContract:

    def test_infer_returns_landmark_output(self):
        """infer() returns a LandmarkOutput instance."""
        model = _build_model()
        result = model.infer(_make_crop())
        assert isinstance(result, LandmarkOutput)

    def test_output_landmarks_shape(self):
        """Landmarks shape is (68, 2)."""
        model = _build_model()
        result = model.infer(_make_crop())
        assert result.landmarks.shape == (_NUM_LANDMARKS, 2)

    def test_output_landmarks_dtype_float32(self):
        """Landmarks dtype is float32."""
        model = _build_model()
        result = model.infer(_make_crop())
        assert result.landmarks.dtype == np.float32

    def test_output_landmarks_range_0_1(self):
        """All landmark coordinates are in [0.0, 1.0]."""
        model = _build_model()
        result = model.infer(_make_crop())
        assert result.landmarks.min() >= 0.0
        assert result.landmarks.max() <= 1.0

    def test_output_confidence_range(self):
        """Confidence is in [0.0, 1.0]."""
        model = _build_model()
        result = model.infer(_make_crop())
        assert 0.0 <= result.confidence <= 1.0

    def test_output_pose_valid_is_bool(self):
        """pose_valid is a bool."""
        model = _build_model()
        result = model.infer(_make_crop())
        assert isinstance(result.pose_valid, bool)


# ═══════════════════════════════════════════════════════════════════════════════
# D. Confidence Derivation (5 tests)
# ═══════════════════════════════════════════════════════════════════════════════

class TestConfidence:

    def test_confidence_high_for_valid_face(self):
        """Well-formed frontal face landmarks produce confidence >= 0.9."""
        model = _build_model()
        result = model.infer(_make_crop())
        assert result.confidence >= 0.9

    def test_confidence_degrades_with_boundary_landmarks(self):
        """When >20% of landmarks are at crop edges, confidence drops."""
        def push_to_boundary(lm):
            # Push 30% of landmarks to x=0.0 (boundary)
            n_boundary = int(0.30 * _NUM_LANDMARKS)
            lm[:n_boundary, 0] = 0.0

        output = _custom_landmarks(push_to_boundary)
        model = _build_model()
        _MockONNXSession._override_output = output
        try:
            result = model.infer(_make_crop())
            assert result.confidence < 0.8
        finally:
            _MockONNXSession._override_output = None

    def test_confidence_low_for_collapsed_iod(self):
        """Collapsed interocular distance (eyes at same point) -> confidence <= 0.3."""
        def collapse_eyes(lm):
            lm[_LEFT_EYE_OUTER] = [0.50, 0.40]
            lm[_RIGHT_EYE_OUTER] = [0.51, 0.40]  # IOD ~0.01

        output = _custom_landmarks(collapse_eyes)
        model = _build_model()
        _MockONNXSession._override_output = output
        try:
            result = model.infer(_make_crop())
            assert result.confidence <= 0.3
        finally:
            _MockONNXSession._override_output = None

    def test_confidence_degrades_for_small_iod(self):
        """IOD between severe and warning thresholds degrades confidence."""
        def small_iod(lm):
            lm[_LEFT_EYE_OUTER] = [0.45, 0.40]
            lm[_RIGHT_EYE_OUTER] = [0.52, 0.40]  # IOD ~0.07

        output = _custom_landmarks(small_iod)
        model = _build_model()
        _MockONNXSession._override_output = output
        try:
            result = model.infer(_make_crop())
            assert result.confidence < 1.0
            assert result.confidence > 0.3
        finally:
            _MockONNXSession._override_output = None

    def test_confidence_degrades_chin_above_nose(self):
        """Chin above nose (physically impossible) degrades confidence."""
        def flip_chin_nose(lm):
            lm[_CHIN] = [0.50, 0.30]       # chin at y=0.30
            lm[_NOSE_TIP] = [0.50, 0.55]   # nose at y=0.55 (below chin)

        output = _custom_landmarks(flip_chin_nose)
        model = _build_model()
        _MockONNXSession._override_output = output
        try:
            result = model.infer(_make_crop())
            assert result.confidence < 0.8
        finally:
            _MockONNXSession._override_output = None


# ═══════════════════════════════════════════════════════════════════════════════
# E. pose_valid Derivation (5 tests)
# ═══════════════════════════════════════════════════════════════════════════════

class TestPoseValid:

    def test_pose_valid_true_for_frontal_face(self):
        """Frontal face with reasonable geometry -> pose_valid=True."""
        model = _build_model()
        result = model.infer(_make_crop())
        assert result.pose_valid is True

    def test_pose_valid_false_for_extreme_yaw(self):
        """IOD/face_width ratio < 0.15 -> pose_valid=False (extreme yaw)."""
        def extreme_yaw(lm):
            lm[_LEFT_JAW] = [0.10, 0.50]
            lm[_RIGHT_JAW] = [0.90, 0.50]    # face_width = 0.80
            lm[_LEFT_EYE_OUTER] = [0.49, 0.40]
            lm[_RIGHT_EYE_OUTER] = [0.51, 0.40]  # IOD = 0.02, ratio = 0.025

        output = _custom_landmarks(extreme_yaw)
        model = _build_model()
        _MockONNXSession._override_output = output
        try:
            result = model.infer(_make_crop())
            assert result.pose_valid is False
        finally:
            _MockONNXSession._override_output = None

    def test_pose_valid_false_for_degenerate_face_width(self):
        """Jaw points nearly identical -> pose_valid=False."""
        def degenerate_width(lm):
            lm[_LEFT_JAW] = [0.500, 0.50]
            lm[_RIGHT_JAW] = [0.505, 0.50]  # face_width = 0.005

        output = _custom_landmarks(degenerate_width)
        model = _build_model()
        _MockONNXSession._override_output = output
        try:
            result = model.infer(_make_crop())
            assert result.pose_valid is False
        finally:
            _MockONNXSession._override_output = None

    def test_pose_valid_false_nose_outside_contour(self):
        """Nose x-coord outside jaw contour -> pose_valid=False."""
        def nose_outside(lm):
            lm[_LEFT_JAW] = [0.30, 0.50]
            lm[_RIGHT_JAW] = [0.70, 0.50]
            lm[_NOSE_TIP] = [0.25, 0.55]   # nose left of left jaw

        output = _custom_landmarks(nose_outside)
        model = _build_model()
        _MockONNXSession._override_output = output
        try:
            result = model.infer(_make_crop())
            assert result.pose_valid is False
        finally:
            _MockONNXSession._override_output = None

    def test_pose_valid_true_for_moderate_yaw(self):
        """Moderate yaw (IOD/face_width > 0.15) -> pose_valid=True."""
        def moderate_yaw(lm):
            lm[_LEFT_JAW] = [0.10, 0.50]
            lm[_RIGHT_JAW] = [0.90, 0.50]    # face_width = 0.80
            lm[_LEFT_EYE_OUTER] = [0.35, 0.40]
            lm[_RIGHT_EYE_OUTER] = [0.55, 0.40]  # IOD = 0.20, ratio = 0.25
            lm[_NOSE_TIP] = [0.45, 0.55]     # nose within jaw contour

        output = _custom_landmarks(moderate_yaw)
        model = _build_model()
        _MockONNXSession._override_output = output
        try:
            result = model.infer(_make_crop())
            assert result.pose_valid is True
        finally:
            _MockONNXSession._override_output = None


# ═══════════════════════════════════════════════════════════════════════════════
# F. iBUG Index Correctness (3 tests)
# ═══════════════════════════════════════════════════════════════════════════════

class TestIBUGIndices:
    """Verify that the mock frontal face landmarks have correct spatial ordering
    consistent with the iBUG 68-point convention used by downstream consumers."""

    def test_eye_landmarks_horizontal_order(self):
        """Left eye outer (36) is left of right eye outer (45)."""
        model = _build_model()
        result = model.infer(_make_crop())
        assert result.landmarks[_LEFT_EYE_OUTER, 0] < result.landmarks[_RIGHT_EYE_OUTER, 0]

    def test_nose_between_eyes(self):
        """Nose tip (30) x-coord is between left and right eye outer corners."""
        model = _build_model()
        result = model.infer(_make_crop())
        nose_x = result.landmarks[_NOSE_TIP, 0]
        left_x = result.landmarks[_LEFT_EYE_OUTER, 0]
        right_x = result.landmarks[_RIGHT_EYE_OUTER, 0]
        assert left_x < nose_x < right_x

    def test_mouth_below_nose(self):
        """Mouth corners (48, 54) have y > nose tip (30) y (image coords, y increases downward)."""
        model = _build_model()
        result = model.infer(_make_crop())
        nose_y = result.landmarks[_NOSE_TIP, 1]
        assert result.landmarks[48, 1] > nose_y
        assert result.landmarks[54, 1] > nose_y


# ═══════════════════════════════════════════════════════════════════════════════
# G. Determinism (2 tests)
# ═══════════════════════════════════════════════════════════════════════════════

class TestDeterminism:

    def test_same_input_same_output(self):
        """Two calls with identical input produce identical results."""
        model = _build_model()
        crop = _make_crop()
        r1 = model.infer(crop.copy())
        r2 = model.infer(crop.copy())
        np.testing.assert_array_equal(r1.landmarks, r2.landmarks)
        assert r1.confidence == r2.confidence
        assert r1.pose_valid == r2.pose_valid

    def test_deterministic_with_seeded_input(self):
        """Result from seeded input is reproducible across calls."""
        model = _build_model()
        crop = np.random.default_rng(42).integers(0, 256, (112, 112, 3), dtype=np.uint8)
        r1 = model.infer(crop)
        r2 = model.infer(crop)
        np.testing.assert_array_equal(r1.landmarks, r2.landmarks)


# ═══════════════════════════════════════════════════════════════════════════════
# H. Error Handling (4 tests)
# ═══════════════════════════════════════════════════════════════════════════════

class TestErrorHandling:

    def test_invalid_input_none_raises(self):
        """infer(None) raises ValueError."""
        model = _build_model()
        with pytest.raises(ValueError):
            model.infer(None)

    def test_invalid_input_wrong_shape_raises(self):
        """1D array raises ValueError."""
        model = _build_model()
        with pytest.raises(ValueError):
            model.infer(np.zeros((10,)))

    def test_invalid_input_2d_raises(self):
        """2D array (missing channel dim) raises ValueError."""
        model = _build_model()
        with pytest.raises(ValueError):
            model.infer(np.zeros((100, 100)))

    def test_invalid_input_4_channel_raises(self):
        """4-channel image raises ValueError."""
        model = _build_model()
        with pytest.raises(ValueError):
            model.infer(np.zeros((100, 100, 4), dtype=np.uint8))


# ═══════════════════════════════════════════════════════════════════════════════
# I. Edge Cases (5 tests)
# ═══════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:

    def test_tiny_crop_1x1(self):
        """1x1 pixel crop succeeds without crashing."""
        model = _build_model()
        result = model.infer(np.zeros((1, 1, 3), dtype=np.uint8))
        assert isinstance(result, LandmarkOutput)

    def test_black_image(self):
        """All-black image returns valid LandmarkOutput."""
        model = _build_model()
        result = model.infer(np.zeros((112, 112, 3), dtype=np.uint8))
        assert isinstance(result, LandmarkOutput)
        assert 0.0 <= result.confidence <= 1.0

    def test_white_image(self):
        """All-white image returns valid LandmarkOutput."""
        model = _build_model()
        result = model.infer(np.full((112, 112, 3), 255, dtype=np.uint8))
        assert isinstance(result, LandmarkOutput)
        assert 0.0 <= result.confidence <= 1.0

    def test_extreme_aspect_ratio(self):
        """Extreme aspect ratio (10x500) doesn't crash."""
        model = _build_model()
        result = model.infer(np.zeros((10, 500, 3), dtype=np.uint8))
        assert isinstance(result, LandmarkOutput)

    def test_float_input_handled(self):
        """Float64 input is accepted and processed correctly."""
        model = _build_model()
        crop = np.random.default_rng(42).uniform(0, 255, (100, 100, 3)).astype(np.float64)
        result = model.infer(crop)
        assert isinstance(result, LandmarkOutput)
        assert result.landmarks.shape == (_NUM_LANDMARKS, 2)

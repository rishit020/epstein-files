"""test_perception_stack.py — Tests for PerceptionStack orchestrator.

Covers: confidence gates (face/landmark), LSTM hidden-state lifecycle,
LSTM reset on absence threshold, exception handling (FR-1.6), phone
detector independence, PerceptionBundle contract, _extract_face_crop.

All models are fully mocked — no real ONNX files required.
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from unittest.mock import MagicMock, patch
import numpy as np
import pytest

import config
from layer1_perception.messages import (
    FaceDetection,
    GazeOutput,
    LandmarkOutput,
    PerceptionBundle,
    PhoneDetectionOutput,
)
from layer1_perception.perception_stack import PerceptionStack, _extract_face_crop


# ═══════════════════════════════════════════════════════════════════════════════
# Mock factories
# ═══════════════════════════════════════════════════════════════════════════════

def _make_frame(h: int = 480, w: int = 640) -> np.ndarray:
    return np.random.default_rng(0).integers(0, 256, (h, w, 3), dtype=np.uint8)


def _face(present: bool = True, confidence: float = 0.95) -> FaceDetection:
    if not present:
        return FaceDetection(present=False, confidence=0.0, bbox_norm=None, face_size_px=0)
    return FaceDetection(
        present=True,
        confidence=confidence,
        bbox_norm=(0.2, 0.1, 0.4, 0.5),
        face_size_px=256,
    )


def _landmarks(confidence: float = 0.90) -> LandmarkOutput:
    lm = np.full((68, 2), 0.5, dtype=np.float32)
    return LandmarkOutput(landmarks=lm, confidence=confidence, pose_valid=True)


def _gaze(yaw: float = 5.0, pitch: float = -3.0) -> GazeOutput:
    return GazeOutput(
        left_eye_yaw=yaw, left_eye_pitch=pitch,
        right_eye_yaw=yaw, right_eye_pitch=pitch,
        combined_yaw=yaw, combined_pitch=pitch,
        confidence=0.85, valid=True,
    )


def _phone(detected: bool = False) -> PhoneDetectionOutput:
    return PhoneDetectionOutput(detected=detected, max_confidence=0.0, bbox_norm=None)


_FAKE_HIDDEN = (np.zeros((1, 64), dtype=np.float32), np.ones((1, 64), dtype=np.float32))


def _build_stack(
    face_conf: float = 0.95,
    face_present: bool = True,
    lm_conf: float = 0.90,
    gaze_yaw: float = 5.0,
    phone_detected: bool = False,
    gaze_hidden_out: tuple | None = None,
) -> PerceptionStack:
    """Create a PerceptionStack with fully mocked models."""
    face_det = MagicMock()
    face_det.infer.return_value = _face(present=face_present, confidence=face_conf)

    lm_model = MagicMock()
    lm_model.infer.return_value = _landmarks(confidence=lm_conf)

    gaze_model = MagicMock()
    gaze_model.infer.return_value = (_gaze(yaw=gaze_yaw), gaze_hidden_out)

    phone_det = MagicMock()
    phone_det.infer.return_value = _phone(detected=phone_detected)

    return PerceptionStack(face_det, lm_model, gaze_model, phone_det)


# ═══════════════════════════════════════════════════════════════════════════════
# A. PerceptionBundle Contract (6 tests)
# ═══════════════════════════════════════════════════════════════════════════════

class TestBundleContract:

    def test_infer_returns_perception_bundle(self):
        stack = _build_stack()
        result = stack.infer(_make_frame(), frame_id=1)
        assert isinstance(result, PerceptionBundle)

    def test_frame_id_preserved(self):
        stack = _build_stack()
        result = stack.infer(_make_frame(), frame_id=42)
        assert result.frame_id == 42

    def test_inference_ms_positive(self):
        stack = _build_stack()
        result = stack.infer(_make_frame(), frame_id=1)
        assert result.inference_ms >= 0.0

    def test_timestamp_ns_positive(self):
        stack = _build_stack()
        result = stack.infer(_make_frame(), frame_id=1)
        assert result.timestamp_ns > 0

    def test_phone_result_stale_false_single_threaded(self):
        """phone_result_stale is always False in single-threaded Mac dev path."""
        stack = _build_stack()
        result = stack.infer(_make_frame(), frame_id=1)
        assert result.phone_result_stale is False

    def test_valid_face_produces_landmarks_and_gaze(self):
        stack = _build_stack(face_conf=0.95, lm_conf=0.90)
        result = stack.infer(_make_frame(), frame_id=1)
        assert result.landmarks is not None
        assert result.gaze is not None


# ═══════════════════════════════════════════════════════════════════════════════
# B. Face Confidence Gate — FR-1.2 (5 tests)
# ═══════════════════════════════════════════════════════════════════════════════

class TestFaceConfidenceGate:

    def test_face_absent_produces_none_landmarks_and_gaze(self):
        stack = _build_stack(face_present=False)
        result = stack.infer(_make_frame(), frame_id=1)
        assert result.landmarks is None
        assert result.gaze is None

    def test_face_below_gate_produces_none_landmarks_and_gaze(self):
        """face.confidence just below FACE_CONFIDENCE_GATE → skip landmarks + gaze."""
        gate = config.FACE_CONFIDENCE_GATE
        stack = _build_stack(face_conf=gate - 0.01)
        result = stack.infer(_make_frame(), frame_id=1)
        assert result.landmarks is None
        assert result.gaze is None

    def test_face_at_gate_produces_none_landmarks_and_gaze(self):
        """face.confidence exactly at gate is below threshold (strict <)."""
        gate = config.FACE_CONFIDENCE_GATE
        stack = _build_stack(face_conf=gate - 1e-6)
        result = stack.infer(_make_frame(), frame_id=1)
        assert result.landmarks is None
        assert result.gaze is None

    def test_face_above_gate_runs_landmark_model(self):
        gate = config.FACE_CONFIDENCE_GATE
        face_det = MagicMock()
        face_det.infer.return_value = _face(present=True, confidence=gate + 0.01)
        lm_model = MagicMock()
        lm_model.infer.return_value = _landmarks(confidence=0.90)
        gaze_model = MagicMock()
        gaze_model.infer.return_value = (_gaze(), None)
        phone_det = MagicMock()
        phone_det.infer.return_value = _phone()

        stack = PerceptionStack(face_det, lm_model, gaze_model, phone_det)
        stack.infer(_make_frame(), frame_id=1)
        lm_model.infer.assert_called_once()

    def test_landmark_model_not_called_when_face_absent(self):
        face_det = MagicMock()
        face_det.infer.return_value = _face(present=False)
        lm_model = MagicMock()
        gaze_model = MagicMock()
        phone_det = MagicMock()
        phone_det.infer.return_value = _phone()

        stack = PerceptionStack(face_det, lm_model, gaze_model, phone_det)
        stack.infer(_make_frame(), frame_id=1)
        lm_model.infer.assert_not_called()
        gaze_model.infer.assert_not_called()


# ═══════════════════════════════════════════════════════════════════════════════
# C. Landmark Confidence Gate — FR-1.3 (4 tests)
# ═══════════════════════════════════════════════════════════════════════════════

class TestLandmarkConfidenceGate:

    def test_low_landmark_conf_skips_gaze(self):
        gate = config.LANDMARK_CONFIDENCE_GATE
        stack = _build_stack(face_conf=0.95, lm_conf=gate - 0.01)
        result = stack.infer(_make_frame(), frame_id=1)
        assert result.gaze is None

    def test_low_landmark_conf_has_landmarks_but_no_gaze(self):
        gate = config.LANDMARK_CONFIDENCE_GATE
        stack = _build_stack(face_conf=0.95, lm_conf=gate - 0.01)
        result = stack.infer(_make_frame(), frame_id=1)
        assert result.landmarks is not None
        assert result.gaze is None

    def test_landmark_gate_fail_with_existing_hidden_state_sets_lstm_reset(self):
        gate = config.LANDMARK_CONFIDENCE_GATE
        stack = _build_stack(face_conf=0.95, lm_conf=gate - 0.01)
        result = stack.infer(_make_frame(), frame_id=1, hidden_state=_FAKE_HIDDEN)
        assert result.lstm_reset_occurred is True

    def test_landmark_gate_fail_without_hidden_state_no_lstm_reset(self):
        """No hidden state to reset → lstm_reset_occurred=False even if gate fails."""
        gate = config.LANDMARK_CONFIDENCE_GATE
        stack = _build_stack(face_conf=0.95, lm_conf=gate - 0.01)
        result = stack.infer(_make_frame(), frame_id=1, hidden_state=None)
        assert result.lstm_reset_occurred is False

    def test_gaze_model_not_called_on_low_landmark_conf(self):
        gate = config.LANDMARK_CONFIDENCE_GATE
        face_det = MagicMock()
        face_det.infer.return_value = _face(present=True, confidence=0.95)
        lm_model = MagicMock()
        lm_model.infer.return_value = _landmarks(confidence=gate - 0.01)
        gaze_model = MagicMock()
        phone_det = MagicMock()
        phone_det.infer.return_value = _phone()

        stack = PerceptionStack(face_det, lm_model, gaze_model, phone_det)
        stack.infer(_make_frame(), frame_id=1)
        gaze_model.infer.assert_not_called()


# ═══════════════════════════════════════════════════════════════════════════════
# D. LSTM Hidden State Lifecycle (8 tests)
# ═══════════════════════════════════════════════════════════════════════════════

class TestLSTMLifecycle:

    def test_hidden_state_passed_through_to_gaze_model(self):
        """hidden_state argument is forwarded to gaze_model.infer()."""
        face_det = MagicMock()
        face_det.infer.return_value = _face()
        lm_model = MagicMock()
        lm_model.infer.return_value = _landmarks(confidence=0.90)
        gaze_model = MagicMock()
        gaze_model.infer.return_value = (_gaze(), None)
        phone_det = MagicMock()
        phone_det.infer.return_value = _phone()

        stack = PerceptionStack(face_det, lm_model, gaze_model, phone_det)
        stack.infer(_make_frame(), frame_id=1, hidden_state=_FAKE_HIDDEN)
        gaze_model.infer.assert_called_once()
        _, call_kwargs = gaze_model.infer.call_args
        # gaze_model.infer(face_crop, hidden_state) — check positional args
        call_args = gaze_model.infer.call_args[0]
        assert call_args[1] is _FAKE_HIDDEN  # hidden_state forwarded

    def test_gaze_hidden_state_returned_in_bundle(self):
        """lstm_hidden_state in bundle equals what gaze_model returned."""
        returned_hidden = _FAKE_HIDDEN
        stack = _build_stack(face_conf=0.95, lm_conf=0.90, gaze_hidden_out=returned_hidden)
        result = stack.infer(_make_frame(), frame_id=1)
        assert result.lstm_hidden_state is returned_hidden

    def test_hidden_state_none_when_gaze_returns_none(self):
        stack = _build_stack(face_conf=0.95, lm_conf=0.90, gaze_hidden_out=None)
        result = stack.infer(_make_frame(), frame_id=1)
        assert result.lstm_hidden_state is None

    def test_hidden_state_carried_during_short_absence(self):
        """hidden_state preserved when face absent < threshold."""
        stack = _build_stack(face_present=False)
        # Simulate short absence (1 frame, well below threshold=10)
        result = stack.infer(_make_frame(), frame_id=1, hidden_state=_FAKE_HIDDEN)
        assert result.lstm_hidden_state is _FAKE_HIDDEN
        assert result.lstm_reset_occurred is False

    def test_hidden_state_dropped_after_threshold_absent_frames(self):
        """hidden_state becomes None once absence exceeds LSTM_RESET_ABSENT_FRAMES."""
        threshold = config.LSTM_RESET_ABSENT_FRAMES
        stack = _build_stack(face_present=False)

        hidden = _FAKE_HIDDEN
        result = None
        for i in range(threshold + 2):
            result = stack.infer(_make_frame(), frame_id=i, hidden_state=hidden)
            hidden = result.lstm_hidden_state  # feed back each frame

        assert result.lstm_hidden_state is None

    def test_lstm_reset_occurs_at_exact_threshold_plus_one(self):
        """lstm_reset_occurred=True on the (threshold+1)-th consecutive absent frame only."""
        threshold = config.LSTM_RESET_ABSENT_FRAMES
        stack = _build_stack(face_present=False)

        reset_frames = []
        hidden = _FAKE_HIDDEN
        for i in range(threshold + 3):
            result = stack.infer(_make_frame(), frame_id=i, hidden_state=hidden)
            if result.lstm_reset_occurred:
                reset_frames.append(i)
            hidden = result.lstm_hidden_state

        # Reset should happen exactly once, at frame (threshold + 1) - 1 = threshold
        assert len(reset_frames) == 1
        assert reset_frames[0] == threshold  # 0-indexed: frame index == threshold

    def test_absent_count_resets_when_face_returns(self):
        """After face returns, short absence no longer triggers LSTM reset."""
        threshold = config.LSTM_RESET_ABSENT_FRAMES
        face_det = MagicMock()
        lm_model = MagicMock()
        lm_model.infer.return_value = _landmarks(confidence=0.90)
        gaze_model = MagicMock()
        gaze_model.infer.return_value = (_gaze(), None)
        phone_det = MagicMock()
        phone_det.infer.return_value = _phone()

        stack = PerceptionStack(face_det, lm_model, gaze_model, phone_det)

        # First: run below threshold absent frames
        face_det.infer.return_value = _face(present=False)
        for i in range(threshold - 1):
            stack.infer(_make_frame(), frame_id=i)

        # Face returns — count should reset
        face_det.infer.return_value = _face(present=True, confidence=0.95)
        stack.infer(_make_frame(), frame_id=threshold)

        # One more absent frame — should NOT trigger reset (count restarted)
        face_det.infer.return_value = _face(present=False)
        result = stack.infer(_make_frame(), frame_id=threshold + 1, hidden_state=_FAKE_HIDDEN)
        assert result.lstm_reset_occurred is False
        assert result.lstm_hidden_state is _FAKE_HIDDEN

    def test_lstm_reset_occurred_false_on_normal_frame(self):
        stack = _build_stack(face_conf=0.95, lm_conf=0.90)
        result = stack.infer(_make_frame(), frame_id=1, hidden_state=_FAKE_HIDDEN)
        assert result.lstm_reset_occurred is False


# ═══════════════════════════════════════════════════════════════════════════════
# E. Exception Handling — FR-1.6 (4 tests)
# ═══════════════════════════════════════════════════════════════════════════════

class TestExceptionHandling:

    def test_face_detector_exception_returns_safe_bundle(self):
        face_det = MagicMock()
        face_det.infer.side_effect = RuntimeError("ONNX crash")
        lm_model = MagicMock()
        gaze_model = MagicMock()
        phone_det = MagicMock()
        phone_det.infer.return_value = _phone()

        stack = PerceptionStack(face_det, lm_model, gaze_model, phone_det)
        result = stack.infer(_make_frame(), frame_id=5)

        assert result.face.present is False
        assert result.landmarks is None
        assert result.gaze is None
        assert result.lstm_reset_occurred is True
        assert result.lstm_hidden_state is None

    def test_landmark_exception_returns_safe_bundle(self):
        face_det = MagicMock()
        face_det.infer.return_value = _face(present=True, confidence=0.95)
        lm_model = MagicMock()
        lm_model.infer.side_effect = MemoryError("OOM")
        gaze_model = MagicMock()
        phone_det = MagicMock()
        phone_det.infer.return_value = _phone()

        stack = PerceptionStack(face_det, lm_model, gaze_model, phone_det)
        result = stack.infer(_make_frame(), frame_id=5)

        assert result.face.present is False
        assert result.lstm_reset_occurred is True

    def test_gaze_exception_returns_safe_bundle(self):
        face_det = MagicMock()
        face_det.infer.return_value = _face(present=True, confidence=0.95)
        lm_model = MagicMock()
        lm_model.infer.return_value = _landmarks(confidence=0.90)
        gaze_model = MagicMock()
        gaze_model.infer.side_effect = ValueError("bad input")
        phone_det = MagicMock()
        phone_det.infer.return_value = _phone()

        stack = PerceptionStack(face_det, lm_model, gaze_model, phone_det)
        result = stack.infer(_make_frame(), frame_id=5)

        assert result.face.present is False
        assert result.lstm_reset_occurred is True

    def test_exception_bundle_frame_id_preserved(self):
        face_det = MagicMock()
        face_det.infer.side_effect = RuntimeError("crash")
        lm_model = MagicMock()
        gaze_model = MagicMock()
        phone_det = MagicMock()
        phone_det.infer.return_value = _phone()

        stack = PerceptionStack(face_det, lm_model, gaze_model, phone_det)
        result = stack.infer(_make_frame(), frame_id=77)
        assert result.frame_id == 77


# ═══════════════════════════════════════════════════════════════════════════════
# F. Phone Detector Independence — FR-1.4 (3 tests)
# ═══════════════════════════════════════════════════════════════════════════════

class TestPhoneIndependence:

    def test_phone_detected_even_when_face_absent(self):
        """Phone detector runs regardless of face detection outcome."""
        face_det = MagicMock()
        face_det.infer.return_value = _face(present=False)
        lm_model = MagicMock()
        gaze_model = MagicMock()
        phone_det = MagicMock()
        phone_det.infer.return_value = PhoneDetectionOutput(
            detected=True, max_confidence=0.92, bbox_norm=(0.1, 0.2, 0.3, 0.4)
        )

        stack = PerceptionStack(face_det, lm_model, gaze_model, phone_det)
        result = stack.infer(_make_frame(), frame_id=1)

        assert result.phone.detected is True
        assert result.phone.max_confidence == pytest.approx(0.92)

    def test_phone_runs_on_every_frame(self):
        """phone_detector.infer is called exactly once per PerceptionStack.infer() call."""
        face_det = MagicMock()
        face_det.infer.return_value = _face(present=True, confidence=0.95)
        lm_model = MagicMock()
        lm_model.infer.return_value = _landmarks()
        gaze_model = MagicMock()
        gaze_model.infer.return_value = (_gaze(), None)
        phone_det = MagicMock()
        phone_det.infer.return_value = _phone()

        stack = PerceptionStack(face_det, lm_model, gaze_model, phone_det)
        for i in range(5):
            stack.infer(_make_frame(), frame_id=i)

        assert phone_det.infer.call_count == 5

    def test_phone_output_preserved_in_bundle(self):
        phone_out = PhoneDetectionOutput(detected=True, max_confidence=0.80, bbox_norm=(0.0, 0.0, 0.5, 0.5))
        stack = _build_stack(phone_detected=True)
        # Override the phone mock return directly
        stack._phone.infer.return_value = phone_out
        result = stack.infer(_make_frame(), frame_id=1)
        assert result.phone is phone_out


# ═══════════════════════════════════════════════════════════════════════════════
# G. _extract_face_crop (5 tests)
# ═══════════════════════════════════════════════════════════════════════════════

class TestExtractFaceCrop:

    def test_crop_shape_correct(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # bbox covers centre 50% of frame
        crop = _extract_face_crop(frame, (0.25, 0.25, 0.50, 0.50))
        assert crop.shape[2] == 3
        assert crop.shape[0] > 0
        assert crop.shape[1] > 0

    def test_full_frame_bbox(self):
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        crop = _extract_face_crop(frame, (0.0, 0.0, 1.0, 1.0))
        assert crop.shape == (100, 200, 3)

    def test_degenerate_zero_width_returns_1x1(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        crop = _extract_face_crop(frame, (0.5, 0.5, 0.0, 0.5))
        assert crop.shape[0] >= 1
        assert crop.shape[1] >= 1

    def test_degenerate_zero_height_returns_1x1(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        crop = _extract_face_crop(frame, (0.5, 0.5, 0.5, 0.0))
        assert crop.shape[0] >= 1
        assert crop.shape[1] >= 1

    def test_out_of_bounds_bbox_clamped(self):
        """bbox extending beyond frame edges is clamped without error."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        crop = _extract_face_crop(frame, (-0.1, -0.1, 1.5, 1.5))
        assert crop.shape[0] <= 100
        assert crop.shape[1] <= 100
        assert crop.shape[2] == 3


# ═══════════════════════════════════════════════════════════════════════════════
# H. Multi-frame sequence (3 tests)
# ═══════════════════════════════════════════════════════════════════════════════

class TestMultiFrameSequence:

    def test_state_propagates_across_frames(self):
        """Returned lstm_hidden_state from frame N can be passed to frame N+1."""
        returned_hidden = _FAKE_HIDDEN
        stack = _build_stack(face_conf=0.95, lm_conf=0.90, gaze_hidden_out=returned_hidden)

        r1 = stack.infer(_make_frame(), frame_id=0, hidden_state=None)
        r2 = stack.infer(_make_frame(), frame_id=1, hidden_state=r1.lstm_hidden_state)

        assert r1.lstm_hidden_state is returned_hidden
        assert r2.lstm_hidden_state is returned_hidden  # passed through → returned

    def test_consecutive_valid_frames_no_reset(self):
        """10 consecutive valid frames: lstm_reset_occurred is never True."""
        stack = _build_stack(face_conf=0.95, lm_conf=0.90)
        for i in range(10):
            r = stack.infer(_make_frame(), frame_id=i)
            assert r.lstm_reset_occurred is False

    def test_absent_then_present_gaze_runs_after_return(self):
        """After short absence (below threshold), gaze resumes when face returns."""
        threshold = config.LSTM_RESET_ABSENT_FRAMES
        face_det = MagicMock()
        lm_model = MagicMock()
        lm_model.infer.return_value = _landmarks(confidence=0.90)
        gaze_model = MagicMock()
        gaze_model.infer.return_value = (_gaze(), None)
        phone_det = MagicMock()
        phone_det.infer.return_value = _phone()

        stack = PerceptionStack(face_det, lm_model, gaze_model, phone_det)

        # Short absence (below threshold)
        face_det.infer.return_value = _face(present=False)
        for i in range(threshold - 1):
            stack.infer(_make_frame(), frame_id=i)

        # Face returns
        face_det.infer.return_value = _face(present=True, confidence=0.95)
        result = stack.infer(_make_frame(), frame_id=threshold)

        assert result.gaze is not None
        gaze_model.infer.assert_called_once()

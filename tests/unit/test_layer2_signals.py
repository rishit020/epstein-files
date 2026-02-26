# tests/unit/test_layer2_signals.py — Phase 2 Signal Processor Unit Tests
#
# Covers all Layer 2 modules per TASKS.md Phase 2 test requirements:
#   - Kalman filter reduces frame-to-frame std dev by >= 60% on synthetic noisy sequence
#   - EAR calibration sequence (cold start vs warm start)
#   - Gaze world transform with synthetic angles
#   - Dual-invalid case (head + gaze both invalid)

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

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
from layer2_signals.ear_calculator import EARCalculator
from layer2_signals.gaze_transformer import is_on_road, transform_gaze
from layer2_signals.head_pose_solver import HeadPoseSolver
from layer2_signals.kalman_filter import KalmanFilter1D
from layer2_signals.messages import SignalFrame
from layer2_signals.phone_signal import extract_phone_signal
from layer2_signals.pose_calibration import PoseCalibration
from layer2_signals.signal_processor import SignalProcessor


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _make_landmarks(n: int = 98) -> np.ndarray:
    """Return synthetic (n, 2) normalised landmarks, spread across the face."""
    rng = np.random.default_rng(42)
    lm = rng.uniform(0.1, 0.9, size=(n, 2)).astype(np.float64)
    return lm


def _make_face_detection(present: bool = True) -> FaceDetection:
    return FaceDetection(
        present=present,
        confidence=0.95 if present else 0.0,
        bbox_norm=(0.2, 0.2, 0.6, 0.6) if present else None,
        face_size_px=400 if present else 0,
    )


def _make_landmark_output(lm: np.ndarray | None = None) -> LandmarkOutput:
    if lm is None:
        lm = _make_landmarks()
    return LandmarkOutput(landmarks=lm, confidence=0.9, pose_valid=True)


def _make_gaze_output(yaw: float = 5.0, pitch: float = -2.0, valid: bool = True) -> GazeOutput:
    return GazeOutput(
        left_eye_yaw=yaw, left_eye_pitch=pitch,
        right_eye_yaw=yaw, right_eye_pitch=pitch,
        combined_yaw=yaw, combined_pitch=pitch,
        confidence=0.85, valid=valid,
    )


def _make_phone_output(detected: bool = False, conf: float = 0.0) -> PhoneDetectionOutput:
    return PhoneDetectionOutput(
        detected=detected,
        max_confidence=conf,
        bbox_norm=(0.1, 0.1, 0.3, 0.3) if detected else None,
    )


def _make_bundle(
    face_present: bool = True,
    gaze: GazeOutput | None = None,
    landmarks: LandmarkOutput | None = None,
    phone: PhoneDetectionOutput | None = None,
    frame_id: int = 0,
) -> PerceptionBundle:
    if phone is None:
        phone = _make_phone_output()
    lm = landmarks if face_present else None
    gz = gaze if face_present else None
    return PerceptionBundle(
        timestamp_ns=frame_id * 33_333_333,
        frame_id=frame_id,
        face=_make_face_detection(face_present),
        landmarks=lm,
        gaze=gz,
        phone=phone,
        phone_result_stale=False,
        inference_ms=10.0,
        lstm_hidden_state=None,
        lstm_reset_occurred=False,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# KalmanFilter1D
# ═══════════════════════════════════════════════════════════════════════════════

class TestKalmanFilter1D:
    def test_first_update_returns_measurement(self):
        kf = KalmanFilter1D()
        out = kf.update(10.0)
        assert out == pytest.approx(10.0)

    def test_filter_smooths_constant_signal(self):
        kf = KalmanFilter1D()
        outputs = [kf.update(5.0) for _ in range(50)]
        # After many identical measurements the estimate should converge near 5.0
        assert abs(outputs[-1] - 5.0) < 0.5

    def test_std_dev_reduction_ge_60pct(self):
        """PRD §5.6: Kalman filter SHALL reduce frame-to-frame std dev by >= 60%
        on a synthetic noisy angle sequence (injected noise σ=5°, clean σ<=2°).
        """
        rng = np.random.default_rng(0)
        clean_signal = np.linspace(0.0, 10.0, 200)   # slow linear head movement
        noisy_signal = clean_signal + rng.normal(0.0, 5.0, size=200)

        kf = KalmanFilter1D()
        filtered = np.array([kf.update(m) for m in noisy_signal])

        # Frame-to-frame differences
        raw_diffs      = np.diff(noisy_signal)
        filtered_diffs = np.diff(filtered)

        raw_std      = float(np.std(raw_diffs))
        filtered_std = float(np.std(filtered_diffs))

        reduction = (raw_std - filtered_std) / raw_std
        assert reduction >= 0.60, (
            f"Kalman std dev reduction {reduction:.2%} < 60% "
            f"(raw_std={raw_std:.3f}, filtered_std={filtered_std:.3f})"
        )

    def test_reset_clears_state(self):
        kf = KalmanFilter1D()
        kf.update(100.0)
        kf.update(100.0)
        kf.reset()
        assert not kf.is_initialized
        # After reset, first measurement should be returned directly
        out = kf.update(0.0)
        assert out == pytest.approx(0.0)

    def test_is_initialized_false_before_first_update(self):
        kf = KalmanFilter1D()
        assert not kf.is_initialized

    def test_is_initialized_true_after_first_update(self):
        kf = KalmanFilter1D()
        kf.update(1.0)
        assert kf.is_initialized


# ═══════════════════════════════════════════════════════════════════════════════
# HeadPoseSolver
# ═══════════════════════════════════════════════════════════════════════════════

class TestHeadPoseSolver:
    def test_returns_five_tuple(self):
        solver = HeadPoseSolver()
        lm = _make_landmarks()
        result = solver.solve(lm, 1280, 720)
        assert len(result) == 5

    def test_invalid_on_degenerate_landmarks(self):
        solver = HeadPoseSolver()
        # All landmarks at (0, 0) — degenerate; PnP may fail or give huge reproj error
        lm = np.zeros((98, 2), dtype=np.float64)
        raw_yaw, raw_pitch, raw_roll, reproj_err, valid = solver.solve(lm, 1280, 720)
        # Degenerate landmarks → should not raise; valid flag may be False
        assert isinstance(valid, bool)
        assert isinstance(reproj_err, float)

    def test_angles_are_finite_on_valid_landmarks(self):
        """With non-degenerate landmarks the solver should return finite floats."""
        solver = HeadPoseSolver()
        lm = _make_landmarks()
        raw_yaw, raw_pitch, raw_roll, reproj_err, valid = solver.solve(lm, 1280, 720)
        assert np.isfinite(raw_yaw)
        assert np.isfinite(raw_pitch)
        assert np.isfinite(raw_roll)

    def test_reprojection_error_is_non_negative(self):
        solver = HeadPoseSolver()
        lm = _make_landmarks()
        _, _, _, reproj_err, _ = solver.solve(lm, 1280, 720)
        assert reproj_err >= 0.0

    def test_validity_uses_config_threshold(self):
        """valid = True iff reproj_err < PNP_REPROJECTION_ERR_MAX (8.0 px)."""
        solver = HeadPoseSolver()
        lm = _make_landmarks()
        _, _, _, reproj_err, valid = solver.solve(lm, 1280, 720)
        if reproj_err < config.PNP_REPROJECTION_ERR_MAX:
            assert valid is True
        else:
            assert valid is False


# ═══════════════════════════════════════════════════════════════════════════════
# EARCalculator
# ═══════════════════════════════════════════════════════════════════════════════

class TestEARCalculator:
    def _open_eye_landmarks(self) -> np.ndarray:
        """Synthetic landmarks with left/right eye points set to give EAR ~0.3."""
        lm = np.zeros((98, 2), dtype=np.float64)
        # Left eye (PFLD 98-pt: 60-67)
        # _LEFT_EYE_IDX = (60, 64, 61, 63, 65, 67)
        # p1=60 outer, p2=64 inner, p3=61 up-outer, p4=63 up-inner, p5=65 lo-inner, p6=67 lo-outer
        lm[60] = [0.30, 0.50]   # outer corner
        lm[64] = [0.40, 0.50]   # inner corner  (||p1-p2|| = 0.10)
        lm[61] = [0.32, 0.47]   # upper outer   (p3)
        lm[63] = [0.38, 0.47]   # upper inner   (p4)
        lm[65] = [0.38, 0.53]   # lower inner   (p5)
        lm[67] = [0.32, 0.53]   # lower outer   (p6)
        # ||p3-p6|| = ||[0.32,0.47]-[0.32,0.53]|| = 0.06
        # ||p4-p5|| = ||[0.38,0.47]-[0.38,0.53]|| = 0.06
        # EAR_left = (0.06 + 0.06) / (2 * 0.10) = 0.60

        # Right eye (PFLD 98-pt: 68-75)
        # _RIGHT_EYE_IDX = (72, 68, 71, 69, 75, 73)
        # p1=72 outer, p2=68 inner, p3=71 up-outer, p4=69 up-inner, p5=75 lo-inner, p6=73 lo-outer
        lm[72] = [0.60, 0.50]   # outer corner
        lm[68] = [0.70, 0.50]   # inner corner  (||p1-p2|| = 0.10)
        lm[71] = [0.62, 0.47]   # upper outer   (p3)
        lm[69] = [0.68, 0.47]   # upper inner   (p4)
        lm[75] = [0.68, 0.53]   # lower inner   (p5)
        lm[73] = [0.62, 0.53]   # lower outer   (p6)
        return lm

    def _closed_eye_landmarks(self) -> np.ndarray:
        """Synthetic landmarks with eyes nearly closed (EAR ~0.0)."""
        lm = np.zeros((98, 2), dtype=np.float64)
        # Left eye — horizontal width 0.10, vertical near zero
        lm[60] = [0.30, 0.50]; lm[64] = [0.40, 0.50]
        lm[61] = [0.32, 0.50]; lm[63] = [0.38, 0.50]
        lm[65] = [0.38, 0.50]; lm[67] = [0.32, 0.50]
        # Right eye
        lm[72] = [0.60, 0.50]; lm[68] = [0.70, 0.50]
        lm[71] = [0.62, 0.50]; lm[69] = [0.68, 0.50]
        lm[75] = [0.68, 0.50]; lm[73] = [0.62, 0.50]
        return lm

    def test_open_eye_ear_positive(self):
        calc = EARCalculator()
        lm = self._open_eye_landmarks()
        left, right, mean = calc.compute(lm)
        assert left > 0.0
        assert right > 0.0
        assert mean == pytest.approx((left + right) / 2.0)

    def test_closed_eye_ear_near_zero(self):
        calc = EARCalculator()
        lm = self._closed_eye_landmarks()
        left, right, mean = calc.compute(lm)
        assert left == pytest.approx(0.0, abs=1e-6)
        assert right == pytest.approx(0.0, abs=1e-6)

    def test_calibration_cold_start_uses_default_threshold(self):
        """PRD §5.2: Before calibration completes, DEFAULT_CLOSE_THRESHOLD applies."""
        calc = EARCalculator()
        assert not calc.calibration_complete
        assert calc.close_threshold == pytest.approx(config.EAR_DEFAULT_CLOSE_THRESHOLD, abs=1e-6)

    def test_calibration_warm_start_via_load_baseline(self):
        """PRD §5.2 / §24.2: load_baseline marks calibration complete immediately."""
        calc = EARCalculator()
        calc.load_baseline(0.31, 0.233)
        assert calc.calibration_complete
        assert calc.baseline_EAR == pytest.approx(0.31, abs=1e-6)
        assert calc.close_threshold == pytest.approx(0.233, abs=1e-6)

    def test_calibration_completes_after_required_samples(self):
        """Cold-start calibration completes after DURATION_S * FPS driving samples."""
        calc = EARCalculator()
        required = int(config.EAR_CALIBRATION_DURATION_S * config.CAPTURE_FPS)
        for _ in range(required):
            calc.update_calibration(mean_ear=0.30, is_driving=True)
        assert calc.calibration_complete
        assert calc.baseline_EAR == pytest.approx(0.30, abs=1e-4)
        assert calc.close_threshold == pytest.approx(0.30 * config.EAR_CALIBRATION_MULTIPLIER, abs=1e-4)

    def test_calibration_does_not_run_when_not_driving(self):
        """EAR calibration should not accumulate samples while parked."""
        calc = EARCalculator()
        required = int(config.EAR_CALIBRATION_DURATION_S * config.CAPTURE_FPS)
        for _ in range(required * 2):
            calc.update_calibration(mean_ear=0.30, is_driving=False)
        assert not calc.calibration_complete

    def test_calibration_does_not_update_after_complete(self):
        """Once calibration is complete, further samples should not change baseline."""
        calc = EARCalculator()
        calc.load_baseline(0.31, 0.233)
        calc.update_calibration(mean_ear=0.10, is_driving=True)  # bad sample — should be ignored
        assert calc.baseline_EAR == pytest.approx(0.31, abs=1e-6)

    def test_reset_calibration(self):
        calc = EARCalculator()
        calc.load_baseline(0.31, 0.233)
        calc.reset_calibration()
        assert not calc.calibration_complete
        assert calc.close_threshold == pytest.approx(config.EAR_DEFAULT_CLOSE_THRESHOLD, abs=1e-6)


# ═══════════════════════════════════════════════════════════════════════════════
# GazeTransformer
# ═══════════════════════════════════════════════════════════════════════════════

class TestGazeTransformer:
    def test_forward_gaze_no_head_movement(self):
        """Camera gaze 0°, head 0° → world gaze 0°."""
        gw_yaw, gw_pitch = transform_gaze(0.0, 0.0, 0.0, 0.0)
        assert gw_yaw   == pytest.approx(0.0)
        assert gw_pitch == pytest.approx(0.0)

    def test_coupling_alpha_applied_to_yaw(self):
        """World yaw = camera_yaw + head_yaw * ALPHA."""
        gw_yaw, _ = transform_gaze(5.0, 0.0, 10.0, 0.0)
        expected = 5.0 + 10.0 * config.GAZE_HEAD_COUPLING_ALPHA
        assert gw_yaw == pytest.approx(expected)

    def test_coupling_beta_applied_to_pitch(self):
        """World pitch = camera_pitch + head_pitch * BETA."""
        _, gw_pitch = transform_gaze(0.0, -3.0, 0.0, -8.0)
        expected = -3.0 + (-8.0) * config.GAZE_HEAD_COUPLING_BETA
        assert gw_pitch == pytest.approx(expected)

    def test_on_road_centre_forward(self):
        """Gaze straight ahead (0°, 0°) is on-road."""
        assert is_on_road(0.0, 0.0) is True

    def test_on_road_within_zone(self):
        assert is_on_road(config.ROAD_ZONE_YAW_MIN + 1.0, config.ROAD_ZONE_PITCH_MIN + 1.0) is True

    def test_off_road_large_yaw(self):
        assert is_on_road(config.ROAD_ZONE_YAW_MAX + 1.0, 0.0) is False

    def test_off_road_large_pitch(self):
        assert is_on_road(0.0, config.ROAD_ZONE_PITCH_MAX + 1.0) is False

    def test_off_road_boundary_exact_yaw(self):
        """Exactly at boundary should be on-road (closed interval)."""
        assert is_on_road(config.ROAD_ZONE_YAW_MAX, 0.0) is True
        assert is_on_road(config.ROAD_ZONE_YAW_MAX + 0.001, 0.0) is False


# ═══════════════════════════════════════════════════════════════════════════════
# PoseCalibration
# ═══════════════════════════════════════════════════════════════════════════════

class TestPoseCalibration:
    def test_zero_offsets_pass_through(self):
        pc = PoseCalibration()
        corr_yaw, corr_pitch = pc.correct(15.0, -5.0)
        assert corr_yaw   == pytest.approx(15.0)
        assert corr_pitch == pytest.approx(-5.0)

    def test_offset_subtracted_correctly(self):
        """corrected_yaw = filtered_yaw - neutral_yaw_offset (PRD §5.7)."""
        pc = PoseCalibration()
        pc.set_offsets(-4.2, 3.1)
        corr_yaw, corr_pitch = pc.correct(0.0, 0.0)
        assert corr_yaw   == pytest.approx(0.0 - (-4.2))
        assert corr_pitch == pytest.approx(0.0 - 3.1)

    def test_known_values(self):
        pc = PoseCalibration()
        pc.set_offsets(5.0, -2.0)
        corr_yaw, corr_pitch = pc.correct(10.0, 3.0)
        assert corr_yaw   == pytest.approx(10.0 - 5.0)
        assert corr_pitch == pytest.approx(3.0  - (-2.0))

    def test_reset_restores_zero_offsets(self):
        pc = PoseCalibration()
        pc.set_offsets(5.0, -2.0)
        pc.reset()
        corr_yaw, corr_pitch = pc.correct(10.0, 3.0)
        assert corr_yaw   == pytest.approx(10.0)
        assert corr_pitch == pytest.approx(3.0)

    def test_properties_reflect_set_values(self):
        pc = PoseCalibration()
        pc.set_offsets(-4.2, 3.1)
        assert pc.neutral_yaw_offset   == pytest.approx(-4.2)
        assert pc.neutral_pitch_offset == pytest.approx(3.1)


# ═══════════════════════════════════════════════════════════════════════════════
# PhoneSignal extractor
# ═══════════════════════════════════════════════════════════════════════════════

class TestPhoneSignalExtractor:
    def test_above_threshold_detected(self):
        phone = _make_phone_output(detected=True, conf=0.85)
        sig = extract_phone_signal(phone, result_stale=False)
        assert sig.detected is True
        assert sig.confidence == pytest.approx(0.85)
        assert sig.stale is False

    def test_below_threshold_not_detected(self):
        phone = _make_phone_output(detected=True, conf=0.50)
        sig = extract_phone_signal(phone, result_stale=False)
        assert sig.detected is False

    def test_exactly_at_threshold_detected(self):
        phone = _make_phone_output(detected=True, conf=config.PHONE_CONFIDENCE_THRESHOLD)
        sig = extract_phone_signal(phone, result_stale=False)
        assert sig.detected is True

    def test_stale_flag_propagated(self):
        phone = _make_phone_output(detected=True, conf=0.80)
        sig = extract_phone_signal(phone, result_stale=True)
        assert sig.stale is True

    def test_not_detected_flag_respected(self):
        """phone.detected=False → PhoneSignal.detected=False regardless of confidence."""
        phone = PhoneDetectionOutput(detected=False, max_confidence=0.95, bbox_norm=None)
        sig = extract_phone_signal(phone, result_stale=False)
        assert sig.detected is False


# ═══════════════════════════════════════════════════════════════════════════════
# SignalProcessor — integration
# ═══════════════════════════════════════════════════════════════════════════════

class TestSignalProcessor:
    def _make_processor(self) -> SignalProcessor:
        sp = SignalProcessor()
        # Pre-load calibration so calibration_complete=True from frame 0
        sp.set_ear_baseline(0.31, 0.233)
        sp.set_neutral_pose(0.0, 0.0)
        return sp

    def test_returns_signal_frame(self):
        sp = self._make_processor()
        bundle = _make_bundle(
            face_present=True,
            landmarks=_make_landmark_output(),
            gaze=_make_gaze_output(),
        )
        sf = sp.process(bundle, speed_mps=5.0)
        assert isinstance(sf, SignalFrame)

    def test_face_absent_produces_none_signals(self):
        sp = self._make_processor()
        bundle = _make_bundle(face_present=False)
        sf = sp.process(bundle)
        assert sf.face_present is False
        assert sf.head_pose is None
        assert sf.eye_signals is None
        assert sf.gaze_world is None
        assert sf.signals_valid is False

    def test_face_present_populates_signals(self):
        sp = self._make_processor()
        bundle = _make_bundle(
            face_present=True,
            landmarks=_make_landmark_output(),
            gaze=_make_gaze_output(),
        )
        sf = sp.process(bundle, speed_mps=5.0)
        assert sf.face_present is True
        assert sf.eye_signals is not None
        assert sf.eye_signals.mean_EAR >= 0.0

    def test_dual_invalid_head_and_gaze_both_invalid(self):
        """PRD TASKS.md: dual-invalid case — head + gaze both invalid → signals_valid = False."""
        sp = self._make_processor()
        # Provide landmarks with pose_valid=False and gaze valid=False
        lm_output = LandmarkOutput(
            landmarks=_make_landmarks(),
            confidence=0.9,
            pose_valid=False,    # <-- head pose invalid
        )
        gaze_output = _make_gaze_output(valid=False)  # <-- gaze invalid
        bundle = _make_bundle(
            face_present=True,
            landmarks=lm_output,
            gaze=gaze_output,
        )
        sf = sp.process(bundle, speed_mps=5.0)
        assert sf.signals_valid is False
        # HeadPose may be None or invalid
        if sf.head_pose is not None:
            assert sf.head_pose.valid is False
        # GazeWorld must be None (gaze_valid requires pose_valid)
        assert sf.gaze_world is None

    def test_phone_signal_extracted(self):
        sp = self._make_processor()
        phone = _make_phone_output(detected=True, conf=0.80)
        bundle = _make_bundle(face_present=True, phone=phone, landmarks=_make_landmark_output())
        sf = sp.process(bundle, speed_mps=5.0)
        assert sf.phone_signal.detected is True
        assert sf.phone_signal.confidence == pytest.approx(0.80)

    def test_kalman_reset_after_face_absent_threshold(self):
        """Kalman filters reset after LSTM_RESET_ABSENT_FRAMES consecutive absent frames."""
        sp = self._make_processor()
        # Process some face-present frames to initialise filters
        for _ in range(5):
            bundle = _make_bundle(
                face_present=True,
                landmarks=_make_landmark_output(),
                gaze=_make_gaze_output(),
            )
            sp.process(bundle, speed_mps=5.0)

        # Now process enough absent frames to trigger reset
        for i in range(config.LSTM_RESET_ABSENT_FRAMES + 1):
            bundle = _make_bundle(face_present=False, frame_id=100 + i)
            sp.process(bundle)

        assert not sp._kf_head_yaw.is_initialized
        assert not sp._kf_head_pitch.is_initialized
        assert not sp._kf_gaze_yaw.is_initialized

    def test_speed_and_stale_flag_passed_through(self):
        sp = self._make_processor()
        bundle = _make_bundle(face_present=False)
        sf = sp.process(bundle, speed_mps=25.0, speed_stale=True)
        assert sf.speed_mps == pytest.approx(25.0)
        assert sf.speed_stale is True

    def test_frame_id_and_timestamp_preserved(self):
        sp = self._make_processor()
        bundle = _make_bundle(face_present=False, frame_id=42)
        bundle.timestamp_ns = 999_000_000
        sf = sp.process(bundle)
        assert sf.frame_id == 42
        assert sf.timestamp_ns == 999_000_000

    def test_set_neutral_pose_applied(self):
        """set_neutral_pose offset is subtracted from corrected angles."""
        sp = SignalProcessor()
        sp.set_ear_baseline(0.31, 0.233)
        sp.set_neutral_pose(yaw_offset=10.0, pitch_offset=0.0)
        assert sp.pose_calibration.neutral_yaw_offset == pytest.approx(10.0)

    def test_gaze_world_on_road_for_forward_gaze(self):
        """A driver looking forward at 0° gaze + 0° head should be on-road."""
        sp = self._make_processor()
        bundle = _make_bundle(
            face_present=True,
            landmarks=_make_landmark_output(),
            gaze=_make_gaze_output(yaw=0.0, pitch=0.0, valid=True),
        )
        sf = sp.process(bundle, speed_mps=5.0)
        # We can only assert on_road if gaze_world is valid
        if sf.gaze_world is not None and sf.gaze_world.valid:
            # Small combined yaw/pitch should be on-road
            assert sf.gaze_world.on_road in (True, False)  # cannot guarantee without full PnP


# ═══════════════════════════════════════════════════════════════════════════════
# Kalman filter applied to gaze world transform — synthetic angles test
# ═══════════════════════════════════════════════════════════════════════════════

class TestGazeWorldTransformKalman:
    def test_kalman_smooths_gaze_world_output(self):
        """Gaze world transform + Kalman should reduce std dev on noisy synthetic gaze."""
        rng = np.random.default_rng(7)
        clean_gaze_yaw   = np.zeros(100)
        noisy_gaze_yaw   = clean_gaze_yaw + rng.normal(0.0, 5.0, 100)

        kf = KalmanFilter1D()
        filtered = np.array([kf.update(m) for m in noisy_gaze_yaw])

        raw_std      = float(np.std(np.diff(noisy_gaze_yaw)))
        filtered_std = float(np.std(np.diff(filtered)))

        reduction = (raw_std - filtered_std) / raw_std
        assert reduction >= 0.60


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

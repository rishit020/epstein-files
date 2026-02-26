# tests/unit/test_dataclasses.py — Unit tests for all Phase 1 dataclass schemas
# PRD §4.1 – §4.6

import sys
import os

import numpy as np
import pytest

# Make the project root importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


# ─── Layer 0: RawFrame ────────────────────────────────────────────────────────

from layer0_video.messages import RawFrame


class TestRawFrame:
    def _make(self, **kwargs):
        defaults = dict(
            timestamp_ns=1_000_000_000,
            frame_id=0,
            width=1280,
            height=720,
            channels=3,
            data=np.zeros((720, 1280, 3), dtype=np.uint8),
            source_type='webcam',
        )
        defaults.update(kwargs)
        return RawFrame(**defaults)

    def test_construction(self):
        frame = self._make()
        assert frame.frame_id == 0
        assert frame.width == 1280
        assert frame.height == 720
        assert frame.channels == 3
        assert frame.source_type == 'webcam'

    def test_data_shape(self):
        frame = self._make()
        assert frame.data.shape == (720, 1280, 3)
        assert frame.data.dtype == np.uint8

    def test_source_types(self):
        for src in ('imx219_v4l2', 'webcam', 'file'):
            frame = self._make(source_type=src)
            assert frame.source_type == src

    def test_timestamp_positive(self):
        frame = self._make(timestamp_ns=999_999_999_999)
        assert frame.timestamp_ns > 0

    def test_frame_id_zero(self):
        frame = self._make(frame_id=0)
        assert frame.frame_id == 0

    def test_frame_id_increments(self):
        f1 = self._make(frame_id=41)
        f2 = self._make(frame_id=42)
        assert f2.frame_id == f1.frame_id + 1

    def test_missing_required_field_raises(self):
        with pytest.raises(TypeError):
            RawFrame(timestamp_ns=0, frame_id=0)  # missing required fields


# ─── Layer 1: Perception types ────────────────────────────────────────────────

from layer1_perception.messages import (
    FaceDetection,
    GazeOutput,
    LandmarkOutput,
    PerceptionBundle,
    PhoneDetectionOutput,
)


class TestFaceDetection:
    def _make(self, **kwargs):
        defaults = dict(
            present=True,
            confidence=0.95,
            bbox_norm=(0.3, 0.2, 0.4, 0.5),
            face_size_px=200,
        )
        defaults.update(kwargs)
        return FaceDetection(**defaults)

    def test_construction_present(self):
        fd = self._make()
        assert fd.present is True
        assert fd.confidence == 0.95

    def test_construction_absent(self):
        fd = self._make(present=False, confidence=0.0, bbox_norm=None, face_size_px=0)
        assert fd.present is False
        assert fd.confidence == 0.0
        assert fd.bbox_norm is None
        assert fd.face_size_px == 0

    def test_confidence_range(self):
        for conf in (0.0, 0.5, 1.0):
            fd = self._make(confidence=conf)
            assert 0.0 <= fd.confidence <= 1.0


class TestLandmarkOutput:
    def _make(self, **kwargs):
        defaults = dict(
            landmarks=np.zeros((98, 2), dtype=np.float32),
            confidence=0.80,
            pose_valid=True,
        )
        defaults.update(kwargs)
        return LandmarkOutput(**defaults)

    def test_construction(self):
        lo = self._make()
        assert lo.landmarks.shape == (98, 2)
        assert lo.confidence == 0.80
        assert lo.pose_valid is True

    def test_landmark_shape(self):
        lo = self._make(landmarks=np.random.rand(98, 2).astype(np.float32))
        assert lo.landmarks.shape == (98, 2)

    def test_pose_valid_false(self):
        lo = self._make(pose_valid=False)
        assert lo.pose_valid is False


class TestGazeOutput:
    def _make(self, **kwargs):
        defaults = dict(
            left_eye_yaw=5.0,
            left_eye_pitch=-2.0,
            right_eye_yaw=4.5,
            right_eye_pitch=-2.5,
            combined_yaw=4.75,
            combined_pitch=-2.25,
            confidence=0.88,
            valid=True,
        )
        defaults.update(kwargs)
        return GazeOutput(**defaults)

    def test_construction(self):
        go = self._make()
        assert go.combined_yaw == 4.75
        assert go.valid is True

    def test_invalid_gaze(self):
        go = self._make(valid=False, confidence=0.0)
        assert go.valid is False

    def test_confidence_range(self):
        for conf in (0.0, 0.65, 1.0):
            go = self._make(confidence=conf)
            assert 0.0 <= go.confidence <= 1.0


class TestPhoneDetectionOutput:
    def _make(self, **kwargs):
        defaults = dict(
            detected=True,
            max_confidence=0.85,
            bbox_norm=(0.1, 0.2, 0.3, 0.4),
        )
        defaults.update(kwargs)
        return PhoneDetectionOutput(**defaults)

    def test_detected(self):
        pdo = self._make()
        assert pdo.detected is True
        assert pdo.max_confidence == 0.85
        assert pdo.bbox_norm is not None

    def test_not_detected(self):
        pdo = self._make(detected=False, max_confidence=0.0, bbox_norm=None)
        assert pdo.detected is False
        assert pdo.max_confidence == 0.0
        assert pdo.bbox_norm is None


class TestPerceptionBundle:
    def _make_face(self):
        return FaceDetection(present=True, confidence=0.95,
                             bbox_norm=(0.3, 0.2, 0.4, 0.5), face_size_px=200)

    def _make_landmarks(self):
        return LandmarkOutput(landmarks=np.zeros((98, 2), dtype=np.float32),
                              confidence=0.80, pose_valid=True)

    def _make_gaze(self):
        return GazeOutput(left_eye_yaw=5.0, left_eye_pitch=-2.0,
                          right_eye_yaw=4.5, right_eye_pitch=-2.5,
                          combined_yaw=4.75, combined_pitch=-2.25,
                          confidence=0.88, valid=True)

    def _make_phone(self):
        return PhoneDetectionOutput(detected=False, max_confidence=0.0, bbox_norm=None)

    def _make(self, **kwargs):
        defaults = dict(
            timestamp_ns=1_000_000_000,
            frame_id=10,
            face=self._make_face(),
            landmarks=self._make_landmarks(),
            gaze=self._make_gaze(),
            phone=self._make_phone(),
            phone_result_stale=False,
            inference_ms=45.0,
            lstm_hidden_state=None,
            lstm_reset_occurred=False,
        )
        defaults.update(kwargs)
        return PerceptionBundle(**defaults)

    def test_construction_full(self):
        pb = self._make()
        assert pb.frame_id == 10
        assert pb.face.present is True
        assert pb.landmarks is not None
        assert pb.gaze is not None
        assert pb.phone_result_stale is False
        assert pb.lstm_hidden_state is None
        assert pb.lstm_reset_occurred is False

    def test_face_absent_fields_none(self):
        pb = self._make(
            face=FaceDetection(present=False, confidence=0.0,
                               bbox_norm=None, face_size_px=0),
            landmarks=None,
            gaze=None,
            lstm_hidden_state=None,
            lstm_reset_occurred=True,
        )
        assert pb.face.present is False
        assert pb.landmarks is None
        assert pb.gaze is None
        assert pb.lstm_reset_occurred is True

    def test_phone_result_stale_flag(self):
        pb = self._make(phone_result_stale=True)
        assert pb.phone_result_stale is True

    def test_lstm_hidden_state_opaque(self):
        # Hidden state is opaque — any tuple is acceptable
        fake_state = (np.zeros((1, 8)), np.zeros((1, 8)))
        pb = self._make(lstm_hidden_state=fake_state)
        assert pb.lstm_hidden_state is fake_state


# ─── Layer 2: Signal types ────────────────────────────────────────────────────

from layer2_signals.messages import (
    EyeSignals,
    GazeWorld,
    HeadPose,
    PhoneSignal,
    SignalFrame,
)


class TestHeadPose:
    def _make(self, **kwargs):
        defaults = dict(
            yaw_deg=5.0,
            pitch_deg=-3.0,
            roll_deg=1.0,
            valid=True,
            raw_yaw_deg=5.5,
            raw_pitch_deg=-3.3,
            raw_roll_deg=1.1,
        )
        defaults.update(kwargs)
        return HeadPose(**defaults)

    def test_construction(self):
        hp = self._make()
        assert hp.yaw_deg == 5.0
        assert hp.valid is True

    def test_invalid_pose(self):
        hp = self._make(valid=False)
        assert hp.valid is False

    def test_debug_fields_preserved(self):
        hp = self._make(raw_yaw_deg=12.3, raw_pitch_deg=-5.6, raw_roll_deg=2.1)
        assert hp.raw_yaw_deg == 12.3
        assert hp.raw_pitch_deg == -5.6
        assert hp.raw_roll_deg == 2.1

    def test_corrected_differs_from_raw(self):
        hp = self._make(yaw_deg=3.0, raw_yaw_deg=7.2)
        # Corrected (filtered + offset) can differ from raw
        assert hp.yaw_deg != hp.raw_yaw_deg


class TestEyeSignals:
    def _make(self, **kwargs):
        defaults = dict(
            left_EAR=0.31,
            right_EAR=0.29,
            mean_EAR=0.30,
            baseline_EAR=0.31,
            close_threshold=0.2325,   # 0.31 * 0.75
            valid=True,
            calibration_complete=True,
        )
        defaults.update(kwargs)
        return EyeSignals(**defaults)

    def test_construction(self):
        es = self._make()
        assert es.mean_EAR == 0.30
        assert es.calibration_complete is True

    def test_close_threshold_formula(self):
        baseline = 0.32
        es = self._make(baseline_EAR=baseline, close_threshold=baseline * 0.75)
        assert abs(es.close_threshold - baseline * 0.75) < 1e-9

    def test_uncalibrated_state(self):
        es = self._make(calibration_complete=False,
                        baseline_EAR=0.21, close_threshold=0.21)
        assert es.calibration_complete is False

    def test_ear_fully_closed(self):
        es = self._make(left_EAR=0.0, right_EAR=0.0, mean_EAR=0.0)
        assert es.mean_EAR == 0.0


class TestGazeWorld:
    def _make(self, **kwargs):
        defaults = dict(
            yaw_deg=3.0,
            pitch_deg=-1.5,
            on_road=True,
            valid=True,
        )
        defaults.update(kwargs)
        return GazeWorld(**defaults)

    def test_on_road(self):
        gw = self._make()
        assert gw.on_road is True
        assert gw.valid is True

    def test_off_road(self):
        gw = self._make(yaw_deg=25.0, on_road=False)
        assert gw.on_road is False

    def test_invalid(self):
        gw = self._make(valid=False)
        assert gw.valid is False


class TestPhoneSignal:
    def test_not_detected(self):
        ps = PhoneSignal(detected=False, confidence=0.0, stale=False)
        assert ps.detected is False
        assert ps.stale is False

    def test_detected(self):
        ps = PhoneSignal(detected=True, confidence=0.85, stale=False)
        assert ps.detected is True
        assert ps.confidence == 0.85

    def test_stale(self):
        ps = PhoneSignal(detected=True, confidence=0.72, stale=True)
        assert ps.stale is True


class TestSignalFrame:
    def _make_head_pose(self, valid=True):
        return HeadPose(yaw_deg=2.0, pitch_deg=-1.0, roll_deg=0.5,
                        valid=valid,
                        raw_yaw_deg=2.5, raw_pitch_deg=-1.3, raw_roll_deg=0.6)

    def _make_eye_signals(self):
        return EyeSignals(left_EAR=0.30, right_EAR=0.28, mean_EAR=0.29,
                          baseline_EAR=0.31, close_threshold=0.2325,
                          valid=True, calibration_complete=True)

    def _make_gaze_world(self, valid=True):
        return GazeWorld(yaw_deg=3.0, pitch_deg=-1.5, on_road=True, valid=valid)

    def _make_phone(self):
        return PhoneSignal(detected=False, confidence=0.0, stale=False)

    def _make(self, **kwargs):
        defaults = dict(
            timestamp_ns=2_000_000_000,
            frame_id=100,
            face_present=True,
            head_pose=self._make_head_pose(),
            eye_signals=self._make_eye_signals(),
            gaze_world=self._make_gaze_world(),
            phone_signal=self._make_phone(),
            speed_mps=10.0,
            speed_stale=False,
            signals_valid=True,
        )
        defaults.update(kwargs)
        return SignalFrame(**defaults)

    def test_construction_full(self):
        sf = self._make()
        assert sf.face_present is True
        assert sf.signals_valid is True
        assert sf.speed_mps == 10.0

    def test_face_absent(self):
        sf = self._make(face_present=False, head_pose=None,
                        eye_signals=None, gaze_world=None,
                        signals_valid=False)
        assert sf.face_present is False
        assert sf.head_pose is None
        assert sf.eye_signals is None
        assert sf.gaze_world is None
        assert sf.signals_valid is False

    def test_dual_invalid(self):
        # PRD FR-2.4: signals_valid=False when both head_pose and gaze_world invalid
        sf = self._make(
            head_pose=self._make_head_pose(valid=False),
            gaze_world=self._make_gaze_world(valid=False),
            signals_valid=False,
        )
        assert sf.head_pose.valid is False
        assert sf.gaze_world.valid is False
        assert sf.signals_valid is False

    def test_speed_stale(self):
        sf = self._make(speed_stale=True)
        assert sf.speed_stale is True

    def test_speed_zero(self):
        sf = self._make(speed_mps=0.0)
        assert sf.speed_mps == 0.0


# ─── Layer 3: TemporalFeatures ────────────────────────────────────────────────

from layer3_temporal.messages import TemporalFeatures


class TestTemporalFeatures:
    def _make(self, **kwargs):
        defaults = dict(
            timestamp_ns=3_000_000_000,
            gaze_off_road_fraction=0.0,
            gaze_continuous_secs=0.0,
            head_deviation_mean_deg=5.0,
            head_continuous_secs=0.0,
            perclos=0.05,
            blink_rate_score=0.0,
            phone_confidence_mean=0.0,
            phone_continuous_secs=0.0,
            speed_zone='URBAN',
            speed_modifier=1.0,
            frames_valid_in_window=60,
            thermal_throttle_active=False,
        )
        defaults.update(kwargs)
        return TemporalFeatures(**defaults)

    def test_construction_nominal(self):
        tf = self._make()
        assert tf.speed_zone == 'URBAN'
        assert tf.speed_modifier == 1.0
        assert tf.thermal_throttle_active is False

    def test_speed_zones(self):
        for zone, mod in [('PARKED', 0.0), ('URBAN', 1.0), ('HIGHWAY', 1.4)]:
            tf = self._make(speed_zone=zone, speed_modifier=mod)
            assert tf.speed_zone == zone
            assert tf.speed_modifier == mod

    def test_perclos_range(self):
        for p in (0.0, 0.15, 0.5, 1.0):
            tf = self._make(perclos=p)
            assert 0.0 <= tf.perclos <= 1.0

    def test_gaze_off_road_fraction_range(self):
        tf = self._make(gaze_off_road_fraction=0.75)
        assert 0.0 <= tf.gaze_off_road_fraction <= 1.0

    def test_blink_rate_score_range(self):
        tf = self._make(blink_rate_score=1.0)
        assert 0.0 <= tf.blink_rate_score <= 1.0

    def test_thermal_throttle(self):
        tf = self._make(thermal_throttle_active=True)
        assert tf.thermal_throttle_active is True

    def test_frames_valid_in_window(self):
        tf = self._make(frames_valid_in_window=45)
        assert tf.frames_valid_in_window == 45


# ─── Layer 4: DistractionScore ───────────────────────────────────────────────

from layer4_scoring.messages import DistractionScore


class TestDistractionScore:
    def _make(self, **kwargs):
        defaults = dict(
            timestamp_ns=4_000_000_000,
            composite_score=0.30,
            component_gaze=0.135,
            component_head=0.09,
            component_perclos=0.06,
            component_blink=0.015,
            gaze_threshold_breached=False,
            head_threshold_breached=False,
            perclos_threshold_breached=False,
            phone_threshold_breached=False,
            active_classes=[],
        )
        defaults.update(kwargs)
        return DistractionScore(**defaults)

    def test_construction_nominal(self):
        ds = self._make()
        assert ds.composite_score == 0.30
        assert ds.active_classes == []

    def test_all_threshold_breached(self):
        ds = self._make(
            composite_score=0.70,
            gaze_threshold_breached=True,
            head_threshold_breached=True,
            perclos_threshold_breached=True,
            phone_threshold_breached=True,
            active_classes=['D-A', 'D-B', 'D-C', 'D-D'],
        )
        assert ds.gaze_threshold_breached is True
        assert ds.head_threshold_breached is True
        assert ds.perclos_threshold_breached is True
        assert ds.phone_threshold_breached is True
        assert set(ds.active_classes) == {'D-A', 'D-B', 'D-C', 'D-D'}

    def test_active_classes_default_empty(self):
        ds = DistractionScore(
            timestamp_ns=0,
            composite_score=0.0,
            component_gaze=0.0,
            component_head=0.0,
            component_perclos=0.0,
            component_blink=0.0,
            gaze_threshold_breached=False,
            head_threshold_breached=False,
            perclos_threshold_breached=False,
            phone_threshold_breached=False,
        )
        assert ds.active_classes == []

    def test_component_sum_approximates_raw_score(self):
        # Pre-modifier components should approximately sum to raw composite
        ds = self._make(
            component_gaze=0.225,   # W1=0.45 * F1=0.5
            component_head=0.15,    # W2=0.30 * F2=0.5
            component_perclos=0.0,
            component_blink=0.0,
            composite_score=0.375,  # speed_modifier=1.0
        )
        raw = ds.component_gaze + ds.component_head + ds.component_perclos + ds.component_blink
        assert abs(raw - 0.375) < 1e-6

    def test_composite_score_above_threshold(self):
        ds = self._make(composite_score=0.60)
        assert ds.composite_score >= 0.55  # COMPOSITE_ALERT_THRESHOLD


# ─── Layer 5: Alert types ────────────────────────────────────────────────────

from layer5_alert.alert_types import AlertLevel, AlertType
from layer5_alert.messages import AlertCommand


class TestAlertLevel:
    def test_values(self):
        assert AlertLevel.LOW.value == 1
        assert AlertLevel.HIGH.value == 2
        assert AlertLevel.URGENT.value == 3

    def test_ordering(self):
        assert AlertLevel.LOW.value < AlertLevel.HIGH.value < AlertLevel.URGENT.value

    def test_all_levels_defined(self):
        levels = {level.name for level in AlertLevel}
        assert levels == {'LOW', 'HIGH', 'URGENT'}


class TestAlertType:
    def test_values(self):
        assert AlertType.VISUAL_INATTENTION.value == 'D-A'
        assert AlertType.HEAD_INATTENTION.value   == 'D-B'
        assert AlertType.DROWSINESS.value         == 'D-C'
        assert AlertType.PHONE_USE.value          == 'D-D'
        assert AlertType.FACE_ABSENT.value        == 'FACE'

    def test_all_types_defined(self):
        names = {t.name for t in AlertType}
        assert names == {
            'VISUAL_INATTENTION',
            'HEAD_INATTENTION',
            'DROWSINESS',
            'PHONE_USE',
            'FACE_ABSENT',
        }


class TestAlertCommand:
    def _make(self, **kwargs):
        defaults = dict(
            alert_id='550e8400-e29b-41d4-a716-446655440000',
            timestamp_ns=5_000_000_000,
            level=AlertLevel.HIGH,
            alert_type=AlertType.VISUAL_INATTENTION,
            composite_score=0.65,
            suppress_until_ns=5_008_000_000_000,
        )
        defaults.update(kwargs)
        return AlertCommand(**defaults)

    def test_construction(self):
        ac = self._make()
        assert ac.level == AlertLevel.HIGH
        assert ac.alert_type == AlertType.VISUAL_INATTENTION
        assert ac.composite_score == 0.65

    def test_phone_urgent(self):
        ac = self._make(level=AlertLevel.URGENT, alert_type=AlertType.PHONE_USE)
        assert ac.level == AlertLevel.URGENT
        assert ac.alert_type == AlertType.PHONE_USE

    def test_suppress_until_after_timestamp(self):
        ac = self._make(timestamp_ns=1_000_000, suppress_until_ns=9_000_000)
        assert ac.suppress_until_ns > ac.timestamp_ns

    def test_alert_id_is_string(self):
        ac = self._make()
        assert isinstance(ac.alert_id, str)
        assert len(ac.alert_id) > 0

    def test_all_alert_types_constructable(self):
        for alert_type in AlertType:
            level = AlertLevel.URGENT if alert_type == AlertType.PHONE_USE else AlertLevel.HIGH
            ac = self._make(level=level, alert_type=alert_type)
            assert ac.alert_type == alert_type


# ─── Config integrity ─────────────────────────────────────────────────────────

import config


class TestConfig:
    def test_weights_sum_to_one(self):
        total = config.WEIGHT_GAZE + config.WEIGHT_HEAD + config.WEIGHT_PERCLOS + config.WEIGHT_BLINK
        assert abs(total - 1.0) < 1e-6, f"Weights sum to {total}, expected 1.0"

    def test_speed_zones_ordered(self):
        assert config.V_MIN_MPS < config.V_HIGHWAY_MPS

    def test_highway_modifier_greater_than_one(self):
        assert config.HIGHWAY_SCORE_MODIFIER > 1.0

    def test_composite_alert_threshold_in_range(self):
        assert 0.0 < config.COMPOSITE_ALERT_THRESHOLD < 1.0

    def test_perclos_threshold_in_range(self):
        assert 0.0 < config.PERCLOS_ALERT_THRESHOLD < 1.0

    def test_degraded_recovery_less_than_trigger(self):
        assert config.DEGRADED_RECOVERY_FRAMES < config.DEGRADED_TRIGGER_FRAMES

    def test_blink_rate_low_less_than_high(self):
        assert config.BLINK_RATE_NORMAL_LOW_HZ < config.BLINK_RATE_NORMAL_HIGH_HZ

    def test_circular_buffer_larger_than_feature_window(self):
        assert config.CIRCULAR_BUFFER_SIZE >= config.FEATURE_WINDOW_FRAMES

    def test_perclos_window_equals_feature_window(self):
        assert config.PERCLOS_WINDOW_FRAMES == config.FEATURE_WINDOW_FRAMES

    def test_perclos_min_valid_frames_less_than_window(self):
        assert config.PERCLOS_MIN_VALID_FRAMES <= config.PERCLOS_WINDOW_FRAMES

    def test_ear_calibration_multiplier_lt_one(self):
        assert 0.0 < config.EAR_CALIBRATION_MULTIPLIER < 1.0

    def test_cooldowns_positive(self):
        for name in ('COOLDOWN_VISUAL', 'COOLDOWN_HEAD', 'COOLDOWN_DROWSINESS',
                     'COOLDOWN_PHONE', 'COOLDOWN_FACE_ABSENT', 'COOLDOWN_COMPOSITE'):
            assert getattr(config, name) > 0.0, f"{name} must be positive"

    def test_thermal_warn_less_than_critical(self):
        assert config.THERMAL_WARN_TEMP_C < config.THERMAL_CRITICAL_TEMP_C

    def test_phone_confidence_gate_in_range(self):
        assert 0.0 < config.PHONE_CONFIDENCE_THRESHOLD < 1.0

    def test_face_gate_lt_landmark_gate(self):
        # Face detection gate must pass before landmark gate is meaningful
        assert config.FACE_CONFIDENCE_GATE <= config.LANDMARK_CONFIDENCE_GATE

    def test_log_max_bytes_positive(self):
        assert config.LOG_MAX_BYTES > 0
        assert config.LOG_BACKUP_COUNT > 0

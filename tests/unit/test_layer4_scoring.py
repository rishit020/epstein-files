# tests/unit/test_layer4_scoring.py — Phase 4 Scoring Engine Unit Tests
#
# Covers all Layer 4 test requirements from TASKS.md:
#   - Weight validation (sum = 1.0 ± 1e-6)
#   - PARKED zone suppression
#   - Composite formula hand-verification
#   - All 6 threshold breach flags (ALT-01 through ALT-06)

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pytest

import config
from layer3_temporal.messages import TemporalFeatures
from layer3_temporal.speed_context import ZONE_HIGHWAY, ZONE_PARKED, ZONE_URBAN
from layer4_scoring.feature_weights import DEFAULT_WEIGHTS, FeatureWeights
from layer4_scoring.messages import DistractionScore
from layer4_scoring.scoring_engine import ScoringEngine


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

_TS = 5_000_000_000  # arbitrary base timestamp (5s in ns)


def _make_features(**kwargs) -> TemporalFeatures:
    """Build a TemporalFeatures with safe defaults (all nominal, URBAN zone)."""
    defaults = dict(
        timestamp_ns=_TS,
        gaze_off_road_fraction=0.0,
        gaze_continuous_secs=0.0,
        head_deviation_mean_deg=0.0,
        head_continuous_secs=0.0,
        perclos=0.0,
        blink_rate_score=0.0,
        phone_confidence_mean=0.0,
        phone_continuous_secs=0.0,
        speed_zone=ZONE_URBAN,
        speed_modifier=1.0,
        frames_valid_in_window=60,
        face_absent_continuous_secs=0.0,
        thermal_throttle_active=False,
    )
    defaults.update(kwargs)
    return TemporalFeatures(**defaults)


_ENGINE = ScoringEngine()


# ═══════════════════════════════════════════════════════════════════════════════
# FeatureWeights validation
# ═══════════════════════════════════════════════════════════════════════════════

class TestFeatureWeights:
    def test_default_weights_sum_to_one(self):
        total = DEFAULT_WEIGHTS.gaze + DEFAULT_WEIGHTS.head + DEFAULT_WEIGHTS.perclos + DEFAULT_WEIGHTS.blink
        assert abs(total - 1.0) < 1e-6

    def test_default_weight_values(self):
        assert DEFAULT_WEIGHTS.gaze    == pytest.approx(config.WEIGHT_GAZE)
        assert DEFAULT_WEIGHTS.head    == pytest.approx(config.WEIGHT_HEAD)
        assert DEFAULT_WEIGHTS.perclos == pytest.approx(config.WEIGHT_PERCLOS)
        assert DEFAULT_WEIGHTS.blink   == pytest.approx(config.WEIGHT_BLINK)

    def test_invalid_weights_raise_value_error(self):
        with pytest.raises(ValueError, match="sum to 1.0"):
            FeatureWeights(gaze=0.5, head=0.5, perclos=0.1, blink=0.1)

    def test_all_zero_weights_raise_value_error(self):
        with pytest.raises(ValueError):
            FeatureWeights(gaze=0.0, head=0.0, perclos=0.0, blink=0.0)

    def test_custom_valid_weights(self):
        # Should not raise — exact 1.0 sum
        w = FeatureWeights(gaze=0.25, head=0.25, perclos=0.25, blink=0.25)
        assert abs(w.gaze + w.head + w.perclos + w.blink - 1.0) < 1e-6

    def test_weights_are_immutable(self):
        with pytest.raises(Exception):
            DEFAULT_WEIGHTS.gaze = 0.99  # type: ignore[misc]


# ═══════════════════════════════════════════════════════════════════════════════
# Composite score formula (PRD §6.1)
# ═══════════════════════════════════════════════════════════════════════════════

class TestCompositeFormula:
    def test_all_zero_features_give_zero_score(self):
        score = _ENGINE.score(_make_features())
        assert score.composite_score == pytest.approx(0.0)

    def test_composite_formula_hand_computed(self):
        # F1=0.6, F2_norm=clamp(15/30,0,1)=0.5, F3=0.1, F4=0.2
        # D_raw = 0.45*0.6 + 0.30*0.5 + 0.20*0.1 + 0.05*0.2
        #       = 0.270 + 0.150 + 0.020 + 0.010 = 0.450
        # D = 0.450 * 1.0 (URBAN) = 0.450
        features = _make_features(
            gaze_off_road_fraction=0.6,
            head_deviation_mean_deg=15.0,
            perclos=0.1,
            blink_rate_score=0.2,
            speed_zone=ZONE_URBAN,
            speed_modifier=1.0,
        )
        score = _ENGINE.score(features)
        assert score.composite_score == pytest.approx(0.450, abs=1e-9)

    def test_component_scores_are_pre_modifier(self):
        # With HIGHWAY modifier 1.4, component scores should NOT be scaled
        features = _make_features(
            gaze_off_road_fraction=1.0,
            speed_zone=ZONE_HIGHWAY,
            speed_modifier=config.HIGHWAY_SCORE_MODIFIER,
        )
        score = _ENGINE.score(features)
        expected_component_gaze = config.WEIGHT_GAZE * 1.0  # 0.45 * 1.0 = 0.45
        assert score.component_gaze == pytest.approx(expected_component_gaze)
        # But composite should be scaled
        expected_d_raw = config.WEIGHT_GAZE * 1.0  # other features = 0
        expected_composite = expected_d_raw * config.HIGHWAY_SCORE_MODIFIER
        assert score.composite_score == pytest.approx(expected_composite)

    def test_highway_modifier_applied_to_composite_only(self):
        # Gaze=1.0, all others 0 → D_raw = 0.45, D = 0.45 * 1.4 = 0.63
        features = _make_features(
            gaze_off_road_fraction=1.0,
            speed_zone=ZONE_HIGHWAY,
            speed_modifier=config.HIGHWAY_SCORE_MODIFIER,
        )
        score = _ENGINE.score(features)
        assert score.composite_score == pytest.approx(0.45 * 1.4, abs=1e-9)

    def test_parked_modifier_zeros_composite(self):
        # Speed modifier 0.0 → composite = 0.0 regardless of features
        features = _make_features(
            gaze_off_road_fraction=1.0,
            head_deviation_mean_deg=60.0,
            perclos=1.0,
            blink_rate_score=1.0,
            speed_zone=ZONE_PARKED,
            speed_modifier=0.0,
        )
        score = _ENGINE.score(features)
        assert score.composite_score == pytest.approx(0.0)

    def test_head_deviation_normalized_by_30_degrees(self):
        # head_deviation = 15° → F2_norm = 15/30 = 0.5
        score_15 = _ENGINE.score(_make_features(head_deviation_mean_deg=15.0))
        # head_deviation = 30° → F2_norm = 1.0 (clamped)
        score_30 = _ENGINE.score(_make_features(head_deviation_mean_deg=30.0))
        # score_30 component_head should be 2x score_15 component_head
        assert score_30.component_head == pytest.approx(score_15.component_head * 2.0)

    def test_head_deviation_clamped_above_30_degrees(self):
        # 60° should clamp to 1.0 — same component as 30°
        score_30 = _ENGINE.score(_make_features(head_deviation_mean_deg=30.0))
        score_60 = _ENGINE.score(_make_features(head_deviation_mean_deg=60.0))
        assert score_60.component_head == pytest.approx(score_30.component_head)

    def test_timestamp_passes_through(self):
        ts = 999_000_000_000
        score = _ENGINE.score(_make_features(timestamp_ns=ts))
        assert score.timestamp_ns == ts


# ═══════════════════════════════════════════════════════════════════════════════
# ALT-01: Gaze threshold (PRD §6.2)
# ═══════════════════════════════════════════════════════════════════════════════

class TestALT01GazeThreshold:
    def test_below_threshold_not_breached(self):
        features = _make_features(
            gaze_continuous_secs=config.T_GAZE_SECONDS - 0.001
        )
        score = _ENGINE.score(features)
        assert score.gaze_threshold_breached is False
        assert 'D-A' not in score.active_classes

    def test_at_threshold_breached(self):
        features = _make_features(
            gaze_continuous_secs=config.T_GAZE_SECONDS
        )
        score = _ENGINE.score(features)
        assert score.gaze_threshold_breached is True
        assert 'D-A' in score.active_classes

    def test_above_threshold_breached(self):
        features = _make_features(
            gaze_continuous_secs=config.T_GAZE_SECONDS + 1.0
        )
        score = _ENGINE.score(features)
        assert score.gaze_threshold_breached is True


# ═══════════════════════════════════════════════════════════════════════════════
# ALT-02: Head threshold (PRD §6.2)
# ═══════════════════════════════════════════════════════════════════════════════

class TestALT02HeadThreshold:
    def test_below_threshold_not_breached(self):
        features = _make_features(
            head_continuous_secs=config.T_HEAD_SECONDS - 0.001
        )
        score = _ENGINE.score(features)
        assert score.head_threshold_breached is False
        assert 'D-B' not in score.active_classes

    def test_at_threshold_breached(self):
        features = _make_features(
            head_continuous_secs=config.T_HEAD_SECONDS
        )
        score = _ENGINE.score(features)
        assert score.head_threshold_breached is True
        assert 'D-B' in score.active_classes

    def test_above_threshold_breached(self):
        features = _make_features(
            head_continuous_secs=config.T_HEAD_SECONDS + 0.5
        )
        score = _ENGINE.score(features)
        assert score.head_threshold_breached is True


# ═══════════════════════════════════════════════════════════════════════════════
# ALT-03: PERCLOS threshold (PRD §6.2)
# ═══════════════════════════════════════════════════════════════════════════════

class TestALT03PERCLOSThreshold:
    def test_below_threshold_not_breached(self):
        features = _make_features(perclos=config.PERCLOS_ALERT_THRESHOLD - 0.001)
        score = _ENGINE.score(features)
        assert score.perclos_threshold_breached is False
        assert 'D-C' not in score.active_classes

    def test_at_threshold_breached(self):
        features = _make_features(perclos=config.PERCLOS_ALERT_THRESHOLD)
        score = _ENGINE.score(features)
        assert score.perclos_threshold_breached is True
        assert 'D-C' in score.active_classes

    def test_above_threshold_breached(self):
        features = _make_features(perclos=config.PERCLOS_ALERT_THRESHOLD + 0.1)
        score = _ENGINE.score(features)
        assert score.perclos_threshold_breached is True


# ═══════════════════════════════════════════════════════════════════════════════
# ALT-04: Phone threshold (PRD §6.2)
# ═══════════════════════════════════════════════════════════════════════════════

class TestALT04PhoneThreshold:
    def test_below_threshold_not_breached(self):
        features = _make_features(
            phone_continuous_secs=config.T_PHONE_SECONDS - 0.001
        )
        score = _ENGINE.score(features)
        assert score.phone_threshold_breached is False
        assert 'D-D' not in score.active_classes

    def test_at_threshold_breached(self):
        features = _make_features(
            phone_continuous_secs=config.T_PHONE_SECONDS
        )
        score = _ENGINE.score(features)
        assert score.phone_threshold_breached is True
        assert 'D-D' in score.active_classes

    def test_above_threshold_breached(self):
        features = _make_features(
            phone_continuous_secs=config.T_PHONE_SECONDS + 2.0
        )
        score = _ENGINE.score(features)
        assert score.phone_threshold_breached is True


# ═══════════════════════════════════════════════════════════════════════════════
# ALT-05: Composite score threshold (PRD §6.2)
# ═══════════════════════════════════════════════════════════════════════════════

class TestALT05CompositeThreshold:
    def test_composite_above_threshold(self):
        # Max features → composite definitely >= 0.55
        features = _make_features(
            gaze_off_road_fraction=1.0,
            head_deviation_mean_deg=60.0,
            perclos=1.0,
            blink_rate_score=1.0,
            speed_zone=ZONE_URBAN,
            speed_modifier=1.0,
        )
        score = _ENGINE.score(features)
        assert score.composite_score >= config.COMPOSITE_ALERT_THRESHOLD

    def test_composite_below_threshold_with_minimal_features(self):
        # All zeros → 0.0 < 0.55
        score = _ENGINE.score(_make_features())
        assert score.composite_score < config.COMPOSITE_ALERT_THRESHOLD


# ═══════════════════════════════════════════════════════════════════════════════
# ALT-06: Face-absent threshold (PRD §6.2)
# ═══════════════════════════════════════════════════════════════════════════════

class TestALT06FaceAbsentThreshold:
    def test_below_duration_not_breached(self):
        features = _make_features(
            face_absent_continuous_secs=config.T_FACE_ABSENT_SECONDS - 0.001,
            speed_zone=ZONE_URBAN,
        )
        score = _ENGINE.score(features)
        assert score.face_absent_threshold_breached is False
        assert 'FACE' not in score.active_classes

    def test_at_duration_urban_zone_breached(self):
        features = _make_features(
            face_absent_continuous_secs=config.T_FACE_ABSENT_SECONDS,
            speed_zone=ZONE_URBAN,
            speed_modifier=1.0,
        )
        score = _ENGINE.score(features)
        assert score.face_absent_threshold_breached is True
        assert 'FACE' in score.active_classes

    def test_at_duration_highway_zone_breached(self):
        features = _make_features(
            face_absent_continuous_secs=config.T_FACE_ABSENT_SECONDS,
            speed_zone=ZONE_HIGHWAY,
            speed_modifier=config.HIGHWAY_SCORE_MODIFIER,
        )
        score = _ENGINE.score(features)
        assert score.face_absent_threshold_breached is True

    def test_at_duration_but_parked_not_breached(self):
        # ALT-06 requires speed > V_MIN — PARKED means not moving, so no fire
        features = _make_features(
            face_absent_continuous_secs=config.T_FACE_ABSENT_SECONDS + 10.0,
            speed_zone=ZONE_PARKED,
            speed_modifier=0.0,
        )
        score = _ENGINE.score(features)
        assert score.face_absent_threshold_breached is False
        assert 'FACE' not in score.active_classes

    def test_above_duration_urban_breached(self):
        features = _make_features(
            face_absent_continuous_secs=config.T_FACE_ABSENT_SECONDS + 5.0,
            speed_zone=ZONE_URBAN,
            speed_modifier=1.0,
        )
        score = _ENGINE.score(features)
        assert score.face_absent_threshold_breached is True


# ═══════════════════════════════════════════════════════════════════════════════
# PARKED zone suppression (TASKS.md test requirement)
# ═══════════════════════════════════════════════════════════════════════════════

class TestParkedZoneSuppression:
    def test_parked_zeros_composite_score(self):
        # PARKED speed_modifier = 0.0, so D = D_raw * 0.0 = 0.0
        features = _make_features(
            gaze_off_road_fraction=1.0,
            head_deviation_mean_deg=60.0,
            perclos=1.0,
            blink_rate_score=1.0,
            speed_zone=ZONE_PARKED,
            speed_modifier=0.0,
        )
        score = _ENGINE.score(features)
        assert score.composite_score == pytest.approx(0.0)

    def test_parked_individual_flags_still_evaluated(self):
        # Threshold flags are evaluated independent of speed modifier
        # (Alert state machine will suppress based on zone, not scoring engine)
        features = _make_features(
            gaze_continuous_secs=config.T_GAZE_SECONDS + 1.0,
            head_continuous_secs=config.T_HEAD_SECONDS + 1.0,
            perclos=config.PERCLOS_ALERT_THRESHOLD + 0.1,
            phone_continuous_secs=config.T_PHONE_SECONDS + 1.0,
            speed_zone=ZONE_PARKED,
            speed_modifier=0.0,
        )
        score = _ENGINE.score(features)
        # Flags are still set — suppression is the alert state machine's job
        assert score.gaze_threshold_breached is True
        assert score.head_threshold_breached is True
        assert score.perclos_threshold_breached is True
        assert score.phone_threshold_breached is True

    def test_parked_face_absent_not_breached(self):
        # ALT-06 requires NOT parked
        features = _make_features(
            face_absent_continuous_secs=999.0,
            speed_zone=ZONE_PARKED,
            speed_modifier=0.0,
        )
        score = _ENGINE.score(features)
        assert score.face_absent_threshold_breached is False


# ═══════════════════════════════════════════════════════════════════════════════
# Perception validity pass-through
# ═══════════════════════════════════════════════════════════════════════════════

class TestPerceptionValidPassThrough:
    def test_valid_window_is_valid(self):
        score = _ENGINE.score(_make_features(frames_valid_in_window=30))
        assert score.perception_valid is True

    def test_zero_valid_frames_is_invalid(self):
        score = _ENGINE.score(_make_features(frames_valid_in_window=0))
        assert score.perception_valid is False

    def test_thermal_throttle_passes_through(self):
        score_off = _ENGINE.score(_make_features(thermal_throttle_active=False))
        assert score_off.thermal_throttle_active is False

        score_on = _ENGINE.score(_make_features(thermal_throttle_active=True))
        assert score_on.thermal_throttle_active is True


# ═══════════════════════════════════════════════════════════════════════════════
# Multiple simultaneous breaches
# ═══════════════════════════════════════════════════════════════════════════════

class TestMultipleSimultaneousBreaches:
    def test_all_conditions_breach_simultaneously(self):
        features = _make_features(
            gaze_continuous_secs=config.T_GAZE_SECONDS + 1.0,
            head_continuous_secs=config.T_HEAD_SECONDS + 1.0,
            perclos=config.PERCLOS_ALERT_THRESHOLD + 0.1,
            phone_continuous_secs=config.T_PHONE_SECONDS + 1.0,
            face_absent_continuous_secs=config.T_FACE_ABSENT_SECONDS + 1.0,
            speed_zone=ZONE_URBAN,
            speed_modifier=1.0,
        )
        score = _ENGINE.score(features)
        assert score.gaze_threshold_breached is True
        assert score.head_threshold_breached is True
        assert score.perclos_threshold_breached is True
        assert score.phone_threshold_breached is True
        assert score.face_absent_threshold_breached is True
        assert set(score.active_classes) == {'D-A', 'D-B', 'D-C', 'D-D', 'FACE'}

    def test_active_classes_empty_when_no_breach(self):
        score = _ENGINE.score(_make_features())
        assert score.active_classes == []

    def test_custom_weights_change_composite(self):
        # Use equal weights → each feature contributes equally
        equal = FeatureWeights(gaze=0.25, head=0.25, perclos=0.25, blink=0.25)
        engine = ScoringEngine(weights=equal)
        # gaze=1.0, rest=0 → D_raw = 0.25 * 1.0 = 0.25
        features = _make_features(gaze_off_road_fraction=1.0)
        score = engine.score(features)
        assert score.composite_score == pytest.approx(0.25)

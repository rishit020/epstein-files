# layer4_scoring/scoring_engine.py — Composite Distraction Scorer
# PRD §6
#
# Stateless per-frame transformer:
#   TemporalFeatures (Layer 3) → DistractionScore (Layer 5)
#
# Formula (PRD §6.1):
#   F1 = gaze_off_road_fraction                       [0, 1]
#   F2_norm = clamp(head_deviation_mean_deg / 30°, 0, 1)
#   F3 = perclos                                      [0, 1]
#   F4 = blink_rate_score                             [0, 1]
#   D_raw = W1*F1 + W2*F2_norm + W3*F3 + W4*F4
#   D     = D_raw * speed_modifier
#
# Individual thresholds (PRD §6.2) are evaluated independently.

import config
from layer3_temporal.messages import TemporalFeatures
from layer3_temporal.speed_context import ZONE_PARKED
from layer4_scoring.feature_weights import DEFAULT_WEIGHTS, FeatureWeights
from layer4_scoring.messages import DistractionScore


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


class ScoringEngine:
    """Computes composite distraction score and threshold breach flags.

    PRD §6 — Distraction Scoring Engine.
    Stateless: each call to score() is independent.
    """

    def __init__(self, weights: FeatureWeights = DEFAULT_WEIGHTS) -> None:
        self._weights = weights

    def score(self, features: TemporalFeatures) -> DistractionScore:
        """Compute DistractionScore from TemporalFeatures.

        Args:
            features: Layer 3 output for this frame.

        Returns:
            DistractionScore with composite score, component scores,
            and threshold breach flags.
        """
        w = self._weights

        # ── Feature normalization (PRD §6.1) ──────────────────────────────
        f1 = features.gaze_off_road_fraction                                   # [0, 1]
        f2_norm = _clamp(
            features.head_deviation_mean_deg / config.HEAD_YAW_THRESHOLD_DEG, # /30.0
            0.0, 1.0,
        )
        f3 = features.perclos                                                  # [0, 1]
        f4 = features.blink_rate_score                                         # [0, 1]

        # ── Component scores (pre-modifier, for logging) ───────────────────
        component_gaze    = w.gaze    * f1
        component_head    = w.head    * f2_norm
        component_perclos = w.perclos * f3
        component_blink   = w.blink   * f4

        # ── Composite score (PRD §6.1) ─────────────────────────────────────
        d_raw = component_gaze + component_head + component_perclos + component_blink
        composite_score = d_raw * features.speed_modifier

        # ── Individual threshold breach flags (PRD §6.2) ───────────────────
        # ALT-01: Gaze continuous off-road >= 2.0s
        gaze_threshold_breached = (
            features.gaze_continuous_secs >= config.T_GAZE_SECONDS
        )

        # ALT-02: Head pose continuous breach >= 1.5s
        head_threshold_breached = (
            features.head_continuous_secs >= config.T_HEAD_SECONDS
        )

        # ALT-03: PERCLOS >= 0.15
        perclos_threshold_breached = (
            features.perclos >= config.PERCLOS_ALERT_THRESHOLD
        )

        # ALT-04: Phone detected continuously >= 1.0s
        phone_threshold_breached = (
            features.phone_continuous_secs >= config.T_PHONE_SECONDS
        )

        # ALT-06: Face absent >= 5.0s AND not parked (speed > V_MIN)
        face_absent_threshold_breached = (
            features.face_absent_continuous_secs >= config.T_FACE_ABSENT_SECONDS
            and features.speed_zone != ZONE_PARKED
        )

        # ── Active distraction class codes ────────────────────────────────
        active_classes: list[str] = []
        if gaze_threshold_breached:
            active_classes.append('D-A')
        if head_threshold_breached:
            active_classes.append('D-B')
        if perclos_threshold_breached:
            active_classes.append('D-C')
        if phone_threshold_breached:
            active_classes.append('D-D')
        if face_absent_threshold_breached:
            active_classes.append('FACE')

        # ── Pass-through context for alert state machine ───────────────────
        perception_valid = features.frames_valid_in_window > 0

        return DistractionScore(
            timestamp_ns=features.timestamp_ns,
            composite_score=composite_score,
            component_gaze=component_gaze,
            component_head=component_head,
            component_perclos=component_perclos,
            component_blink=component_blink,
            gaze_threshold_breached=gaze_threshold_breached,
            head_threshold_breached=head_threshold_breached,
            perclos_threshold_breached=perclos_threshold_breached,
            phone_threshold_breached=phone_threshold_breached,
            active_classes=active_classes,
            face_absent_threshold_breached=face_absent_threshold_breached,
            perception_valid=perception_valid,
            thermal_throttle_active=features.thermal_throttle_active,
            speed_zone=features.speed_zone,
        )

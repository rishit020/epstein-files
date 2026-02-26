# tests/unit/test_layer5_alert.py — Phase 4/5 Alert State Machine Unit Tests
#
# Covers all Layer 5 test requirements from TASKS.md:
#   - PARKED zone suppression (P-05)
#   - Cooldown isolation per alert type
#   - Phone URGENT overrides all suppression (P-01)
#   - DEGRADED state suppresses all alerts (P-04)
#   - DEGRADED recovery after 30 consecutive valid frames
#   - Failure mode tests: FM-02, FM-04, FM-08

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import uuid

import pytest

import config
from layer3_temporal.speed_context import ZONE_HIGHWAY, ZONE_PARKED, ZONE_URBAN
from layer4_scoring.messages import DistractionScore
from layer5_alert.alert_state_machine import AlertStateMachine
from layer5_alert.alert_types import AlertLevel, AlertType
from layer5_alert.messages import AlertCommand


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

_TS_BASE = 10_000_000_000        # 10 seconds in ns
_DT_NS   = int(1e9 / config.CAPTURE_FPS)  # ~33ms per frame


def _make_score(
    ts_ns: int = _TS_BASE,
    composite_score: float = 0.0,
    gaze_breached: bool = False,
    head_breached: bool = False,
    perclos_breached: bool = False,
    phone_breached: bool = False,
    face_absent_breached: bool = False,
    active_classes: list | None = None,
    perception_valid: bool = True,
    thermal_throttle_active: bool = False,
    speed_zone: str = ZONE_URBAN,
) -> DistractionScore:
    if active_classes is None:
        # Auto-populate from breach flags
        active_classes = []
        if gaze_breached:
            active_classes.append('D-A')
        if head_breached:
            active_classes.append('D-B')
        if perclos_breached:
            active_classes.append('D-C')
        if phone_breached:
            active_classes.append('D-D')
        if face_absent_breached:
            active_classes.append('FACE')

    return DistractionScore(
        timestamp_ns=ts_ns,
        composite_score=composite_score,
        component_gaze=0.0,
        component_head=0.0,
        component_perclos=0.0,
        component_blink=0.0,
        gaze_threshold_breached=gaze_breached,
        head_threshold_breached=head_breached,
        perclos_threshold_breached=perclos_breached,
        phone_threshold_breached=phone_breached,
        active_classes=active_classes,
        face_absent_threshold_breached=face_absent_breached,
        perception_valid=perception_valid,
        thermal_throttle_active=thermal_throttle_active,
        speed_zone=speed_zone,
    )


def _nominal_score(ts_ns: int = _TS_BASE) -> DistractionScore:
    """A score with no breaches and all signals valid."""
    return _make_score(ts_ns=ts_ns)


def _invalid_score(ts_ns: int = _TS_BASE) -> DistractionScore:
    """A score produced from a frame with no valid perception signals."""
    return _make_score(ts_ns=ts_ns, perception_valid=False)


def _make_machine() -> AlertStateMachine:
    return AlertStateMachine()


# ═══════════════════════════════════════════════════════════════════════════════
# Nominal operation — no alerts when nothing breaches
# ═══════════════════════════════════════════════════════════════════════════════

class TestNominalOperation:
    def test_no_alert_when_nothing_breaches(self):
        machine = _make_machine()
        result = machine.process(_nominal_score())
        assert result is None

    def test_initial_state_is_nominal(self):
        machine = _make_machine()
        assert machine.state == 'NOMINAL'

    def test_alert_command_has_valid_uuid(self):
        machine = _make_machine()
        result = machine.process(_make_score(gaze_breached=True))
        assert result is not None
        # Should be parseable as UUID
        parsed = uuid.UUID(result.alert_id)
        assert str(parsed) == result.alert_id

    def test_alert_command_has_correct_level_for_high(self):
        machine = _make_machine()
        result = machine.process(_make_score(gaze_breached=True))
        assert result is not None
        assert result.level == AlertLevel.HIGH

    def test_alert_command_suppress_until_after_timestamp(self):
        machine = _make_machine()
        result = machine.process(_make_score(ts_ns=_TS_BASE, gaze_breached=True))
        assert result is not None
        assert result.suppress_until_ns > _TS_BASE

    def test_gaze_breach_fires_visual_inattention(self):
        machine = _make_machine()
        result = machine.process(_make_score(gaze_breached=True))
        assert result is not None
        assert result.alert_type == AlertType.VISUAL_INATTENTION

    def test_head_breach_fires_head_inattention(self):
        machine = _make_machine()
        result = machine.process(_make_score(head_breached=True))
        assert result is not None
        assert result.alert_type == AlertType.HEAD_INATTENTION

    def test_perclos_breach_fires_drowsiness(self):
        machine = _make_machine()
        result = machine.process(_make_score(perclos_breached=True))
        assert result is not None
        assert result.alert_type == AlertType.DROWSINESS

    def test_phone_breach_fires_phone_use_urgent(self):
        machine = _make_machine()
        result = machine.process(_make_score(phone_breached=True))
        assert result is not None
        assert result.alert_type == AlertType.PHONE_USE
        assert result.level == AlertLevel.URGENT

    def test_face_absent_breach_fires_face_absent(self):
        machine = _make_machine()
        result = machine.process(_make_score(face_absent_breached=True))
        assert result is not None
        assert result.alert_type == AlertType.FACE_ABSENT


# ═══════════════════════════════════════════════════════════════════════════════
# Cooldown suppression
# ═══════════════════════════════════════════════════════════════════════════════

class TestCooldownSuppression:
    def test_same_type_suppressed_during_cooldown(self):
        machine = _make_machine()
        ts = _TS_BASE

        # First gaze alert fires
        r1 = machine.process(_make_score(ts_ns=ts, gaze_breached=True))
        assert r1 is not None

        # Second gaze alert immediately after — should be suppressed
        ts2 = ts + _DT_NS
        r2 = machine.process(_make_score(ts_ns=ts2, gaze_breached=True))
        assert r2 is None

    def test_same_type_fires_after_cooldown_expires(self):
        machine = _make_machine()
        ts = _TS_BASE

        # First gaze alert fires
        r1 = machine.process(_make_score(ts_ns=ts, gaze_breached=True))
        assert r1 is not None

        # Jump past cooldown window (8.0s)
        ts_after = ts + int(config.COOLDOWN_VISUAL * 1e9) + 1
        r2 = machine.process(_make_score(ts_ns=ts_after, gaze_breached=True))
        assert r2 is not None
        assert r2.alert_type == AlertType.VISUAL_INATTENTION

    def test_suppress_until_ns_matches_cooldown(self):
        machine = _make_machine()
        ts = _TS_BASE
        r = machine.process(_make_score(ts_ns=ts, gaze_breached=True))
        assert r is not None
        expected = ts + int(config.COOLDOWN_VISUAL * 1e9)
        assert r.suppress_until_ns == expected

    def test_head_cooldown_matches_config(self):
        machine = _make_machine()
        r = machine.process(_make_score(ts_ns=_TS_BASE, head_breached=True))
        assert r is not None
        assert r.suppress_until_ns == _TS_BASE + int(config.COOLDOWN_HEAD * 1e9)

    def test_drowsiness_cooldown_matches_config(self):
        machine = _make_machine()
        r = machine.process(_make_score(ts_ns=_TS_BASE, perclos_breached=True))
        assert r is not None
        assert r.suppress_until_ns == _TS_BASE + int(config.COOLDOWN_DROWSINESS * 1e9)

    def test_phone_cooldown_matches_config(self):
        machine = _make_machine()
        r = machine.process(_make_score(ts_ns=_TS_BASE, phone_breached=True))
        assert r is not None
        assert r.suppress_until_ns == _TS_BASE + int(config.COOLDOWN_PHONE * 1e9)

    def test_face_absent_cooldown_matches_config(self):
        machine = _make_machine()
        r = machine.process(_make_score(ts_ns=_TS_BASE, face_absent_breached=True))
        assert r is not None
        assert r.suppress_until_ns == _TS_BASE + int(config.COOLDOWN_FACE_ABSENT * 1e9)


# ═══════════════════════════════════════════════════════════════════════════════
# Cooldown isolation per alert type (TASKS.md test requirement)
# ═══════════════════════════════════════════════════════════════════════════════

class TestCooldownIsolation:
    def test_gaze_cooldown_does_not_suppress_head(self):
        machine = _make_machine()
        ts = _TS_BASE

        # Fire gaze alert
        r1 = machine.process(_make_score(ts_ns=ts, gaze_breached=True))
        assert r1 is not None
        assert r1.alert_type == AlertType.VISUAL_INATTENTION

        # Same timestamp — head breach. Gaze is in cooldown but head is NOT.
        r2 = machine.process(_make_score(ts_ns=ts + _DT_NS, head_breached=True))
        assert r2 is not None
        assert r2.alert_type == AlertType.HEAD_INATTENTION

    def test_gaze_cooldown_does_not_suppress_perclos(self):
        machine = _make_machine()
        ts = _TS_BASE
        machine.process(_make_score(ts_ns=ts, gaze_breached=True))
        r2 = machine.process(_make_score(ts_ns=ts + _DT_NS, perclos_breached=True))
        assert r2 is not None
        assert r2.alert_type == AlertType.DROWSINESS

    def test_head_cooldown_does_not_suppress_gaze(self):
        machine = _make_machine()
        ts = _TS_BASE
        machine.process(_make_score(ts_ns=ts, head_breached=True))
        r2 = machine.process(_make_score(ts_ns=ts + _DT_NS, gaze_breached=True))
        assert r2 is not None
        assert r2.alert_type == AlertType.VISUAL_INATTENTION

    def test_drowsiness_cooldown_does_not_suppress_gaze(self):
        machine = _make_machine()
        ts = _TS_BASE
        machine.process(_make_score(ts_ns=ts, perclos_breached=True))
        r2 = machine.process(_make_score(ts_ns=ts + _DT_NS, gaze_breached=True))
        assert r2 is not None
        assert r2.alert_type == AlertType.VISUAL_INATTENTION


# ═══════════════════════════════════════════════════════════════════════════════
# P-01: Phone URGENT overrides all suppression (TASKS.md test requirement)
# ═══════════════════════════════════════════════════════════════════════════════

class TestPhoneOverrideP01:
    def test_phone_fires_while_gaze_in_cooldown(self):
        machine = _make_machine()
        ts = _TS_BASE

        # Trigger gaze alert (in cooldown now)
        machine.process(_make_score(ts_ns=ts, gaze_breached=True))

        # Phone fires independently — not blocked by gaze cooldown (P-01)
        r = machine.process(_make_score(ts_ns=ts + _DT_NS, phone_breached=True))
        assert r is not None
        assert r.alert_type == AlertType.PHONE_USE
        assert r.level == AlertLevel.URGENT

    def test_phone_fires_while_head_in_cooldown(self):
        machine = _make_machine()
        ts = _TS_BASE
        machine.process(_make_score(ts_ns=ts, head_breached=True))
        r = machine.process(_make_score(ts_ns=ts + _DT_NS, phone_breached=True))
        assert r is not None
        assert r.alert_type == AlertType.PHONE_USE

    def test_phone_fires_while_perclos_in_cooldown(self):
        machine = _make_machine()
        ts = _TS_BASE
        machine.process(_make_score(ts_ns=ts, perclos_breached=True))
        r = machine.process(_make_score(ts_ns=ts + _DT_NS, phone_breached=True))
        assert r is not None
        assert r.alert_type == AlertType.PHONE_USE

    def test_phone_fires_while_face_absent_in_cooldown(self):
        machine = _make_machine()
        ts = _TS_BASE
        machine.process(_make_score(ts_ns=ts, face_absent_breached=True))
        r = machine.process(_make_score(ts_ns=ts + _DT_NS, phone_breached=True))
        assert r is not None
        assert r.alert_type == AlertType.PHONE_USE

    def test_phone_has_own_cooldown(self):
        machine = _make_machine()
        ts = _TS_BASE
        # First phone alert fires
        r1 = machine.process(_make_score(ts_ns=ts, phone_breached=True))
        assert r1 is not None
        # Second phone immediately after — suppressed by phone's own cooldown
        r2 = machine.process(_make_score(ts_ns=ts + _DT_NS, phone_breached=True))
        assert r2 is None


# ═══════════════════════════════════════════════════════════════════════════════
# P-02: Multiple simultaneous breaches → ONE alert
# ═══════════════════════════════════════════════════════════════════════════════

class TestMultipleBreachesP02:
    def test_gaze_and_head_simultaneous_fires_once(self):
        machine = _make_machine()
        r = machine.process(_make_score(gaze_breached=True, head_breached=True))
        assert r is not None
        # One alert, not two
        assert isinstance(r, AlertCommand)

    def test_gaze_and_head_simultaneous_picks_higher_priority(self):
        # DROWSINESS > VISUAL_INATTENTION per priority table
        machine = _make_machine()
        r = machine.process(_make_score(gaze_breached=True, perclos_breached=True))
        assert r is not None
        assert r.alert_type == AlertType.DROWSINESS

    def test_gaze_head_perclos_simultaneous_picks_highest(self):
        machine = _make_machine()
        r = machine.process(_make_score(
            gaze_breached=True, head_breached=True, perclos_breached=True
        ))
        assert r is not None
        assert r.alert_type == AlertType.DROWSINESS

    def test_phone_with_gaze_fires_phone(self):
        # Phone is always highest priority
        machine = _make_machine()
        r = machine.process(_make_score(gaze_breached=True, phone_breached=True))
        assert r is not None
        assert r.alert_type == AlertType.PHONE_USE


# ═══════════════════════════════════════════════════════════════════════════════
# P-03: FACE_ABSENT independent suppression
# ═══════════════════════════════════════════════════════════════════════════════

class TestFaceAbsentIndependentP03:
    def test_face_absent_fires_while_gaze_in_cooldown(self):
        machine = _make_machine()
        ts = _TS_BASE
        machine.process(_make_score(ts_ns=ts, gaze_breached=True))
        r = machine.process(_make_score(ts_ns=ts + _DT_NS, face_absent_breached=True))
        assert r is not None
        assert r.alert_type == AlertType.FACE_ABSENT

    def test_face_absent_has_own_cooldown(self):
        machine = _make_machine()
        ts = _TS_BASE
        r1 = machine.process(_make_score(ts_ns=ts, face_absent_breached=True))
        assert r1 is not None
        r2 = machine.process(_make_score(ts_ns=ts + _DT_NS, face_absent_breached=True))
        assert r2 is None


# ═══════════════════════════════════════════════════════════════════════════════
# P-04 / P-05: PARKED zone suppression (TASKS.md test requirement)
# ═══════════════════════════════════════════════════════════════════════════════

class TestParkedZoneSuppression:
    def test_parked_gaze_breach_suppressed(self):
        machine = _make_machine()
        r = machine.process(_make_score(
            gaze_breached=True, speed_zone=ZONE_PARKED
        ))
        assert r is None

    def test_parked_head_breach_suppressed(self):
        machine = _make_machine()
        r = machine.process(_make_score(
            head_breached=True, speed_zone=ZONE_PARKED
        ))
        assert r is None

    def test_parked_perclos_breach_suppressed(self):
        machine = _make_machine()
        r = machine.process(_make_score(
            perclos_breached=True, speed_zone=ZONE_PARKED
        ))
        assert r is None

    def test_parked_face_absent_suppressed(self):
        machine = _make_machine()
        r = machine.process(_make_score(
            face_absent_breached=True, speed_zone=ZONE_PARKED
        ))
        assert r is None

    def test_parked_phone_fires(self):
        # P-05: only phone fires in PARKED zone
        machine = _make_machine()
        r = machine.process(_make_score(
            phone_breached=True, speed_zone=ZONE_PARKED
        ))
        assert r is not None
        assert r.alert_type == AlertType.PHONE_USE

    def test_parked_all_breaches_except_phone_suppressed(self):
        machine = _make_machine()
        r = machine.process(_make_score(
            gaze_breached=True,
            head_breached=True,
            perclos_breached=True,
            phone_breached=False,
            face_absent_breached=True,
            speed_zone=ZONE_PARKED,
        ))
        assert r is None

    def test_parked_composite_score_suppressed(self):
        machine = _make_machine()
        r = machine.process(_make_score(
            composite_score=config.COMPOSITE_ALERT_THRESHOLD + 0.1,
            speed_zone=ZONE_PARKED,
        ))
        assert r is None


# ═══════════════════════════════════════════════════════════════════════════════
# P-04: DEGRADED state suppresses all alerts (TASKS.md test requirement)
# ═══════════════════════════════════════════════════════════════════════════════

class TestDegradedState:
    def _drive_to_degraded(self, machine: AlertStateMachine, ts_start: int) -> int:
        """Feed DEGRADED_TRIGGER_FRAMES invalid frames. Returns next timestamp."""
        ts = ts_start
        for _ in range(config.DEGRADED_TRIGGER_FRAMES):
            machine.process(_invalid_score(ts_ns=ts))
            ts += _DT_NS
        return ts

    def test_degraded_entered_after_60_invalid_frames(self):
        machine = _make_machine()
        self._drive_to_degraded(machine, _TS_BASE)
        assert machine.state == 'DEGRADED'

    def test_degraded_suppresses_all_alerts(self):
        machine = _make_machine()
        ts = self._drive_to_degraded(machine, _TS_BASE)

        # All conditions breaching but DEGRADED — no alert should fire
        score = _make_score(
            ts_ns=ts,
            gaze_breached=True,
            head_breached=True,
            perclos_breached=True,
            phone_breached=True,
            face_absent_breached=True,
            perception_valid=False,  # still invalid
        )
        r = machine.process(score)
        assert r is None

    def test_degraded_suppresses_phone_too(self):
        # Even URGENT phone is suppressed in DEGRADED (P-04)
        machine = _make_machine()
        ts = self._drive_to_degraded(machine, _TS_BASE)
        r = machine.process(_make_score(
            ts_ns=ts, phone_breached=True, perception_valid=False
        ))
        assert r is None

    def test_degraded_not_entered_at_59_invalid_frames(self):
        machine = _make_machine()
        ts = _TS_BASE
        for _ in range(config.DEGRADED_TRIGGER_FRAMES - 1):
            machine.process(_invalid_score(ts_ns=ts))
            ts += _DT_NS
        assert machine.state != 'DEGRADED'

    def test_degraded_exactly_at_60_frames(self):
        machine = _make_machine()
        ts = _TS_BASE
        for _ in range(config.DEGRADED_TRIGGER_FRAMES):
            machine.process(_invalid_score(ts_ns=ts))
            ts += _DT_NS
        assert machine.state == 'DEGRADED'


# ═══════════════════════════════════════════════════════════════════════════════
# DEGRADED recovery (TASKS.md test requirement)
# ═══════════════════════════════════════════════════════════════════════════════

class TestDegradedRecovery:
    def _drive_to_degraded(self, machine: AlertStateMachine, ts_start: int) -> int:
        ts = ts_start
        for _ in range(config.DEGRADED_TRIGGER_FRAMES):
            machine.process(_invalid_score(ts_ns=ts))
            ts += _DT_NS
        return ts

    def test_recovery_after_30_consecutive_valid_frames(self):
        machine = _make_machine()
        ts = self._drive_to_degraded(machine, _TS_BASE)
        assert machine.state == 'DEGRADED'

        for _ in range(config.DEGRADED_RECOVERY_FRAMES):
            machine.process(_nominal_score(ts_ns=ts))
            ts += _DT_NS

        # After 30 valid frames, should recover to NOMINAL
        assert machine.state == 'NOMINAL'

    def test_partial_recovery_resets_on_invalid_frame(self):
        machine = _make_machine()
        ts = self._drive_to_degraded(machine, _TS_BASE)

        # Feed 29 valid frames (one short of recovery)
        for _ in range(config.DEGRADED_RECOVERY_FRAMES - 1):
            machine.process(_nominal_score(ts_ns=ts))
            ts += _DT_NS
        assert machine.state == 'DEGRADED'

        # One invalid frame — resets consecutive_valid counter
        machine.process(_invalid_score(ts_ns=ts))
        ts += _DT_NS
        assert machine.state == 'DEGRADED'

        # Must complete another full 30-frame run to recover
        for _ in range(config.DEGRADED_RECOVERY_FRAMES):
            machine.process(_nominal_score(ts_ns=ts))
            ts += _DT_NS
        assert machine.state == 'NOMINAL'

    def test_alerts_resume_after_recovery(self):
        machine = _make_machine()
        ts = self._drive_to_degraded(machine, _TS_BASE)

        # Recover
        for _ in range(config.DEGRADED_RECOVERY_FRAMES):
            machine.process(_nominal_score(ts_ns=ts))
            ts += _DT_NS
        assert machine.state == 'NOMINAL'

        # Alert should now fire
        r = machine.process(_make_score(ts_ns=ts, gaze_breached=True))
        assert r is not None
        assert r.alert_type == AlertType.VISUAL_INATTENTION

    def test_29_valid_not_enough_for_recovery(self):
        machine = _make_machine()
        ts = self._drive_to_degraded(machine, _TS_BASE)

        for _ in range(config.DEGRADED_RECOVERY_FRAMES - 1):
            machine.process(_nominal_score(ts_ns=ts))
            ts += _DT_NS
        assert machine.state == 'DEGRADED'


# ═══════════════════════════════════════════════════════════════════════════════
# FM-02: Face absent while moving → ALT-06 fires
# ═══════════════════════════════════════════════════════════════════════════════

class TestFM02FaceAbsent:
    def test_face_absent_at_urban_speed_fires(self):
        # Scoring engine sets face_absent_threshold_breached
        # State machine receives it and fires ALT-06
        machine = _make_machine()
        r = machine.process(_make_score(
            face_absent_breached=True,
            speed_zone=ZONE_URBAN,
        ))
        assert r is not None
        assert r.alert_type == AlertType.FACE_ABSENT
        assert r.level == AlertLevel.HIGH

    def test_face_absent_at_highway_speed_fires(self):
        machine = _make_machine()
        r = machine.process(_make_score(
            face_absent_breached=True,
            speed_zone=ZONE_HIGHWAY,
        ))
        assert r is not None
        assert r.alert_type == AlertType.FACE_ABSENT

    def test_face_absent_at_parked_suppressed(self):
        machine = _make_machine()
        # Scoring engine sets face_absent_threshold_breached=False when parked,
        # but even if True here, state machine P-05 suppresses it
        r = machine.process(_make_score(
            face_absent_breached=True,
            speed_zone=ZONE_PARKED,
        ))
        assert r is None


# ═══════════════════════════════════════════════════════════════════════════════
# FM-04: Model exception → DEGRADED → recovery
# ═══════════════════════════════════════════════════════════════════════════════

class TestFM04ModelException:
    def test_60_invalid_frames_enters_degraded(self):
        # FM-04: model exception → PerceptionBundle with face.present=False
        # → perception_valid=False after 60 frames
        machine = _make_machine()
        ts = _TS_BASE
        for i in range(config.DEGRADED_TRIGGER_FRAMES):
            machine.process(_invalid_score(ts_ns=ts))
            ts += _DT_NS
        assert machine.state == 'DEGRADED'

    def test_recovery_after_model_restored(self):
        machine = _make_machine()
        ts = _TS_BASE
        for _ in range(config.DEGRADED_TRIGGER_FRAMES):
            machine.process(_invalid_score(ts_ns=ts))
            ts += _DT_NS
        for _ in range(config.DEGRADED_RECOVERY_FRAMES):
            machine.process(_nominal_score(ts_ns=ts))
            ts += _DT_NS
        assert machine.state == 'NOMINAL'


# ═══════════════════════════════════════════════════════════════════════════════
# FM-08: Thermal throttle → DEGRADED (P-06)
# ═══════════════════════════════════════════════════════════════════════════════

class TestFM08ThermalDegraded:
    def test_thermal_throttle_enters_degraded_immediately(self):
        machine = _make_machine()
        r = machine.process(_make_score(thermal_throttle_active=True))
        # P-06: thermal → DEGRADED immediately, no alert fires
        assert r is None
        assert machine.state == 'DEGRADED'

    def test_thermal_degraded_suppresses_phone(self):
        machine = _make_machine()
        r = machine.process(_make_score(
            phone_breached=True, thermal_throttle_active=True
        ))
        assert r is None

    def test_thermal_degraded_clears_when_thermal_drops(self):
        machine = _make_machine()
        ts = _TS_BASE

        # Enter thermal DEGRADED
        machine.process(_make_score(ts_ns=ts, thermal_throttle_active=True))
        assert machine.state == 'DEGRADED'
        ts += _DT_NS

        # Thermal clears but valid frames not yet accumulated — still DEGRADED
        machine.process(_make_score(ts_ns=ts, thermal_throttle_active=False))
        assert machine.state == 'DEGRADED'
        ts += _DT_NS

        # Feed DEGRADED_RECOVERY_FRAMES valid frames without thermal
        for _ in range(config.DEGRADED_RECOVERY_FRAMES - 1):
            machine.process(_nominal_score(ts_ns=ts))
            ts += _DT_NS

        # Complete the recovery
        machine.process(_nominal_score(ts_ns=ts))
        assert machine.state == 'NOMINAL'

    def test_thermal_degraded_does_not_recover_while_thermal_hot(self):
        machine = _make_machine()
        ts = _TS_BASE

        # Enter thermal DEGRADED
        machine.process(_make_score(ts_ns=ts, thermal_throttle_active=True))
        ts += _DT_NS

        # Feed 30 valid frames BUT thermal still active → no recovery
        for _ in range(config.DEGRADED_RECOVERY_FRAMES + 5):
            machine.process(_make_score(ts_ns=ts, thermal_throttle_active=True))
            ts += _DT_NS

        # Should still be DEGRADED because thermal never cleared
        assert machine.state == 'DEGRADED'


# ═══════════════════════════════════════════════════════════════════════════════
# Composite alert (ALT-05)
# ═══════════════════════════════════════════════════════════════════════════════

class TestALT05CompositeAlert:
    def test_composite_threshold_fires_alert(self):
        machine = _make_machine()
        # Composite fires when no individual threshold is breached but score is high
        r = machine.process(_make_score(
            composite_score=config.COMPOSITE_ALERT_THRESHOLD + 0.1,
            gaze_breached=False,
            head_breached=False,
            perclos_breached=False,
            phone_breached=False,
            active_classes=['D-A'],  # gaze is dominant
        ))
        assert r is not None

    def test_composite_with_individual_does_not_double_fire(self):
        # When individual thresholds also breach, composite should NOT fire separately
        machine = _make_machine()
        r = machine.process(_make_score(
            composite_score=config.COMPOSITE_ALERT_THRESHOLD + 0.1,
            gaze_breached=True,  # ALT-01 will fire
        ))
        assert r is not None
        assert r.alert_type == AlertType.VISUAL_INATTENTION

    def test_composite_below_threshold_no_alert(self):
        machine = _make_machine()
        r = machine.process(_make_score(
            composite_score=config.COMPOSITE_ALERT_THRESHOLD - 0.01,
        ))
        assert r is None

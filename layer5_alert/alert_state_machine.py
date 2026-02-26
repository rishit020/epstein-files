# layer5_alert/alert_state_machine.py — Alert State Machine
# PRD §7
#
# Consumes DistractionScore (Layer 4) and emits AlertCommand | None.
#
# States: NOMINAL → PRE_ALERT → ALERTING → COOLDOWN → (back to NOMINAL)
#         Any state → DEGRADED (on invalid frames or thermal throttle)
#         DEGRADED → NOMINAL (on recovery)
#
# Priority arbitration rules (PRD §7.3):
#   P-01: Phone (URGENT) fires independently of all other cooldowns
#   P-02: Multiple simultaneous non-phone breaches → ONE AlertCommand
#   P-03: FACE_ABSENT fires independently of all other cooldowns
#   P-04: DEGRADED state — no alerts fire
#   P-05: PARKED zone — only phone alert can fire
#   P-06: Thermal DEGRADED — no alerts, log THERMAL_DEGRADED event
#
# Cooldown tracking: per-type, frame-timestamp-based (not wall-clock) — NFR-R4.

import uuid
from enum import Enum
from typing import Optional

import config
from layer4_scoring.messages import DistractionScore
from layer3_temporal.speed_context import ZONE_PARKED
from layer5_alert.alert_types import AlertLevel, AlertType
from layer5_alert.messages import AlertCommand


class _MachineState(Enum):
    NOMINAL    = 'NOMINAL'
    PRE_ALERT  = 'PRE_ALERT'
    ALERTING   = 'ALERTING'
    COOLDOWN   = 'COOLDOWN'
    DEGRADED   = 'DEGRADED'


# Per-type cooldown durations in nanoseconds (sourced from config)
_COOLDOWN_NS: dict[AlertType, int] = {
    AlertType.VISUAL_INATTENTION: int(config.COOLDOWN_VISUAL      * 1e9),
    AlertType.HEAD_INATTENTION:   int(config.COOLDOWN_HEAD        * 1e9),
    AlertType.DROWSINESS:         int(config.COOLDOWN_DROWSINESS  * 1e9),
    AlertType.PHONE_USE:          int(config.COOLDOWN_PHONE       * 1e9),
    AlertType.FACE_ABSENT:        int(config.COOLDOWN_FACE_ABSENT * 1e9),
}

# Composite alert (ALT-05) uses the cooldown of whichever type is chosen as primary
_COOLDOWN_COMPOSITE_NS: int = int(config.COOLDOWN_COMPOSITE * 1e9)

# Alert type priority order for P-02 (highest priority first)
# Phone is handled separately (P-01); this list covers non-phone types.
_NON_PHONE_PRIORITY: list[AlertType] = [
    AlertType.DROWSINESS,
    AlertType.VISUAL_INATTENTION,
    AlertType.HEAD_INATTENTION,
    AlertType.FACE_ABSENT,
]


class AlertStateMachine:
    """Converts DistractionScore into AlertCommand output.

    PRD §7 — Alert State Machine.
    One instance per pipeline run. Call process() once per frame.
    """

    def __init__(self) -> None:
        self._state: _MachineState = _MachineState.NOMINAL

        # Consecutive frame counters for DEGRADED entry/exit
        self._consecutive_invalid: int = 0
        self._consecutive_valid: int   = 0

        # Tracks whether DEGRADED was triggered by thermal (FM-08)
        # Thermal DEGRADED can only exit when thermal clears AND valid frames recovered
        self._thermal_degraded: bool = False

        # Per-type suppress_until_ns — 0 means not suppressed
        self._suppress_until_ns: dict[AlertType, int] = {t: 0 for t in AlertType}

    # ── Public API ─────────────────────────────────────────────────────────────

    @property
    def state(self) -> str:
        """Current state name (for logging/testing)."""
        return self._state.value

    def process(self, score: DistractionScore) -> Optional[AlertCommand]:
        """Process one DistractionScore and return an AlertCommand or None.

        Args:
            score: Layer 4 output for this frame.

        Returns:
            AlertCommand if an alert fires this frame, otherwise None.
        """
        now_ns = score.timestamp_ns

        # ── Step 1: Update validity counters ──────────────────────────────
        if score.perception_valid:
            self._consecutive_valid   += 1
            self._consecutive_invalid  = 0
        else:
            self._consecutive_invalid += 1
            self._consecutive_valid    = 0

        # ── Step 2: DEGRADED entry checks ─────────────────────────────────
        # P-06 / FM-08: thermal throttle → DEGRADED immediately
        if score.thermal_throttle_active and self._state != _MachineState.DEGRADED:
            self._state = _MachineState.DEGRADED
            self._thermal_degraded = True

        # FM-04: consecutive invalid frames threshold → DEGRADED
        if (
            self._consecutive_invalid >= config.DEGRADED_TRIGGER_FRAMES
            and self._state != _MachineState.DEGRADED
        ):
            self._state = _MachineState.DEGRADED

        # ── Step 3: DEGRADED — check recovery or stay suppressed (P-04) ───
        if self._state == _MachineState.DEGRADED:
            # Thermal DEGRADED can only exit when thermal is no longer active
            if self._thermal_degraded and not score.thermal_throttle_active:
                self._thermal_degraded = False
                # Still need consecutive_valid to satisfy before full recovery

            can_recover = (
                not self._thermal_degraded
                and self._consecutive_valid >= config.DEGRADED_RECOVERY_FRAMES
            )
            if can_recover:
                self._state = _MachineState.NOMINAL
                self._consecutive_invalid = 0
                # Do not return — fall through and evaluate conditions this frame
            else:
                # P-04: no alerts fire while DEGRADED
                return None

        # ── Step 4: Collect non-suppressed, breaching conditions ──────────
        # Each condition maps to (AlertType, AlertLevel)
        # Suppression is checked per-type independently (P-01, P-03)

        candidates: list[tuple[AlertType, AlertLevel]] = []

        # ALT-01: Gaze off-road
        if score.gaze_threshold_breached:
            candidates.append((AlertType.VISUAL_INATTENTION, AlertLevel.HIGH))

        # ALT-02: Head pose
        if score.head_threshold_breached:
            candidates.append((AlertType.HEAD_INATTENTION, AlertLevel.HIGH))

        # ALT-03: PERCLOS drowsiness
        if score.perclos_threshold_breached:
            candidates.append((AlertType.DROWSINESS, AlertLevel.HIGH))

        # ALT-04: Phone use — URGENT
        if score.phone_threshold_breached:
            candidates.append((AlertType.PHONE_USE, AlertLevel.URGENT))

        # ALT-05: Composite score threshold
        if (
            score.composite_score >= config.COMPOSITE_ALERT_THRESHOLD
            and not score.gaze_threshold_breached    # avoid double-firing with ALT-01
            and not score.head_threshold_breached    # avoid double-firing with ALT-02
            and not score.perclos_threshold_breached # avoid double-firing with ALT-03
            and not score.phone_threshold_breached   # phone handled separately
        ):
            # Composite fires: pick the dominant type from active_classes,
            # or fall back to VISUAL_INATTENTION (highest weight: W1=0.45)
            primary = _dominant_type(score.active_classes)
            candidates.append((primary, AlertLevel.HIGH))

        # ALT-06: Face absent while moving
        if score.face_absent_threshold_breached:
            candidates.append((AlertType.FACE_ABSENT, AlertLevel.HIGH))

        # ── Step 5: Apply priority arbitration ────────────────────────────

        # P-05: PARKED zone — only phone can fire
        if score.speed_zone == ZONE_PARKED:
            candidates = [
                (t, lvl) for (t, lvl) in candidates
                if t == AlertType.PHONE_USE
            ]

        if not candidates:
            self._state = _MachineState.NOMINAL
            return None

        # Separate phone from non-phone candidates
        phone_candidates = [(t, lvl) for (t, lvl) in candidates if t == AlertType.PHONE_USE]
        other_candidates = [(t, lvl) for (t, lvl) in candidates if t != AlertType.PHONE_USE]

        # P-01: Phone fires independently — check its own cooldown only
        if phone_candidates:
            phone_type, phone_level = phone_candidates[0]
            if now_ns >= self._suppress_until_ns[phone_type]:
                return self._fire(score, phone_type, phone_level, now_ns)
            # Phone is suppressed — continue to check other candidates

        # Filter non-phone candidates through their own cooldowns
        # P-03: FACE_ABSENT is also independent — checked per its own key
        non_suppressed = [
            (t, lvl) for (t, lvl) in other_candidates
            if now_ns >= self._suppress_until_ns[t]
        ]

        if not non_suppressed:
            # All active conditions are in cooldown
            if self._state not in (_MachineState.NOMINAL,):
                self._state = _MachineState.COOLDOWN
            return None

        # P-02: Multiple non-phone conditions → fire ONE alert with highest priority type
        primary_type, primary_level = _select_primary(non_suppressed)

        return self._fire(score, primary_type, primary_level, now_ns)

    # ── Private ────────────────────────────────────────────────────────────────

    def _fire(
        self,
        score: DistractionScore,
        alert_type: AlertType,
        level: AlertLevel,
        now_ns: int,
    ) -> AlertCommand:
        """Emit an AlertCommand and update suppression + state."""
        cooldown_ns = _COOLDOWN_NS.get(alert_type, _COOLDOWN_COMPOSITE_NS)
        suppress_until = now_ns + cooldown_ns

        self._suppress_until_ns[alert_type] = suppress_until
        self._state = _MachineState.ALERTING

        return AlertCommand(
            alert_id=str(uuid.uuid4()),
            timestamp_ns=now_ns,
            level=level,
            alert_type=alert_type,
            composite_score=score.composite_score,
            suppress_until_ns=suppress_until,
        )


# ── Module-level helpers ───────────────────────────────────────────────────────

def _dominant_type(active_classes: list[str]) -> AlertType:
    """Pick the highest-priority AlertType from the active_classes code list.

    Priority (by weight): D-A > D-B > D-C > D-D > FACE.
    Falls back to VISUAL_INATTENTION if active_classes is empty.
    """
    _CODE_TO_TYPE = {
        'D-A':  AlertType.VISUAL_INATTENTION,
        'D-B':  AlertType.HEAD_INATTENTION,
        'D-C':  AlertType.DROWSINESS,
        'D-D':  AlertType.PHONE_USE,
        'FACE': AlertType.FACE_ABSENT,
    }
    _PRIORITY = {
        AlertType.VISUAL_INATTENTION: 0,
        AlertType.HEAD_INATTENTION:   1,
        AlertType.DROWSINESS:         2,
        AlertType.PHONE_USE:          3,
        AlertType.FACE_ABSENT:        4,
    }
    types = [_CODE_TO_TYPE[c] for c in active_classes if c in _CODE_TO_TYPE]
    if not types:
        return AlertType.VISUAL_INATTENTION
    return min(types, key=lambda t: _PRIORITY[t])


def _select_primary(
    candidates: list[tuple[AlertType, AlertLevel]],
) -> tuple[AlertType, AlertLevel]:
    """Select the highest-priority (alert_type, level) from a non-empty list.

    Priority order: PHONE_USE > DROWSINESS > VISUAL_INATTENTION
                    > HEAD_INATTENTION > FACE_ABSENT
    """
    _PRIORITY = {
        AlertType.PHONE_USE:          0,
        AlertType.DROWSINESS:         1,
        AlertType.VISUAL_INATTENTION: 2,
        AlertType.HEAD_INATTENTION:   3,
        AlertType.FACE_ABSENT:        4,
    }
    return min(candidates, key=lambda pair: _PRIORITY.get(pair[0], 99))

# layer4_scoring/messages.py — Layer 4 output messages
# PRD §4.5 — DistractionScore: Layer 4 → Layer 5

from dataclasses import dataclass, field


@dataclass
class DistractionScore:
    """Composite distraction score and threshold breach flags — Layer 4 → Layer 5.

    PRD §4.5
    """
    timestamp_ns: int

    # Composite score AFTER speed modifier applied
    composite_score: float            # [0.0, 1.0+]

    # Component scores (pre-modifier, for logging/debugging)
    component_gaze: float             # W1 * F1
    component_head: float             # W2 * F2_norm
    component_perclos: float          # W3 * F3
    component_blink: float            # W4 * F4

    # Threshold breach flags — evaluated independently per ALT-01 through ALT-06
    gaze_threshold_breached: bool     # continuous_secs >= T_GAZE_SECONDS
    head_threshold_breached: bool     # continuous_secs >= T_HEAD_SECONDS
    perclos_threshold_breached: bool  # perclos >= PERCLOS_ALERT_THRESHOLD
    phone_threshold_breached: bool    # phone_continuous_secs >= T_PHONE_SECONDS

    # Which distraction classes are currently active
    active_classes: list = field(default_factory=list)  # e.g. ['D-A', 'D-C']

    # Pass-through context for alert state machine (Layer 5)
    face_absent_threshold_breached: bool = False  # ALT-06: face absent >= T_FACE_ABSENT_SECONDS while moving
    perception_valid: bool = True                 # False if frames_valid_in_window == 0 (FM-04 trigger)
    thermal_throttle_active: bool = False         # True if thermal throttle declared (FM-08 trigger)
    speed_zone: str = 'URBAN'                     # 'PARKED' | 'URBAN' | 'HIGHWAY' — for P-05 arbitration

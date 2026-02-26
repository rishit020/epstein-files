# layer5_alert/messages.py — Layer 5 output messages
# PRD §4.6 — AlertCommand: Layer 5 → Layer 6

from dataclasses import dataclass

from layer5_alert.alert_types import AlertLevel, AlertType


@dataclass
class AlertCommand:
    """Alert output — carries a single dispatched alert to the output layer.

    PRD §4.6
    """
    alert_id: str           # UUID
    timestamp_ns: int
    level: AlertLevel
    alert_type: AlertType
    composite_score: float  # Score that triggered this alert
    suppress_until_ns: int  # Do not repeat until this time (per-type cooldown)

# layer5_alert/alert_types.py — Alert enum definitions
# PRD §4.6

from enum import Enum


class AlertLevel(Enum):
    """Severity level of an alert — determines audio behaviour.

    PRD §4.6
    """
    LOW    = 1  # Advisory — no current use in MVP
    HIGH   = 2  # Standard distraction alert
    URGENT = 3  # Phone use alert (highest priority; overrides all suppression)


class AlertType(Enum):
    """Distraction class that triggered the alert.

    PRD §4.6
    """
    VISUAL_INATTENTION = 'D-A'
    HEAD_INATTENTION   = 'D-B'
    DROWSINESS         = 'D-C'
    PHONE_USE          = 'D-D'
    FACE_ABSENT        = 'FACE'

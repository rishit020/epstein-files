# layer3_temporal/messages.py — Layer 3 output messages
# PRD §4.4 — TemporalFeatures: Layer 3 → Layer 4

from dataclasses import dataclass


@dataclass
class TemporalFeatures:
    """Aggregated temporal features over the circular buffer — Layer 3 → Layer 4.

    PRD §4.4
    """
    timestamp_ns: int

    # F1: Gaze off-road features
    gaze_off_road_fraction: float   # [0.0, 1.0] — fraction of window gaze was OFF_ROAD
    gaze_continuous_secs: float     # Duration of current continuous off-road event (s)

    # F2: Head pose features
    head_deviation_mean_deg: float  # Euclidean norm of yaw+pitch (corrected angles) over window
    head_continuous_secs: float     # Duration of current continuous head pose breach (s)

    # F3: PERCLOS
    perclos: float                  # [0.0, 1.0] — fraction of 60-frame window with eyes >= 80% closed

    # F4: Blink rate anomaly score (PRD §5.5)
    blink_rate_score: float         # [0.0, 1.0]; 1.0 = maximally anomalous

    # F5: Phone features
    phone_confidence_mean: float    # [0.0, 1.0] — mean phone confidence in current window
    phone_continuous_secs: float    # Duration of current continuous phone detection (s)

    # Context
    speed_zone: str                 # 'PARKED' | 'URBAN' | 'HIGHWAY'
    speed_modifier: float           # 0.0 | 1.0 | 1.4
    frames_valid_in_window: int     # Count of valid frames in the circular buffer window

    # F6: Face-absent duration (PRD §6.2 ALT-06)
    face_absent_continuous_secs: float = 0.0  # Duration of current continuous face-absent event (s)

    # Thermal state (PRD CHANGE-09)
    thermal_throttle_active: bool = False  # True if ThermalMonitor has declared throttle state

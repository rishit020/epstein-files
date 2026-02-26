# layer2_signals/gaze_transformer.py — Camera-Space to World-Space Gaze Transform
# PRD §5.3 — Gaze World-Space Transform
#
# Combines camera-space gaze with (already Kalman-filtered, neutral-pose-corrected)
# head pose to produce a world-space gaze estimate.
#
# Formula (PRD §5.3):
#   gaze_world_yaw   = gaze_camera_yaw   + head_yaw   * ALPHA
#   gaze_world_pitch = gaze_camera_pitch + head_pitch  * BETA
#
# ALPHA = BETA = 0.7 (head-to-gaze coupling factor, PRD §5.3)
#
# Validity: gaze_world is invalid if EITHER gaze OR head_pose is invalid.
# On-road check: corrected world angles compared against ROAD_ZONE config values.

import config


def transform_gaze(
    gaze_camera_yaw: float,
    gaze_camera_pitch: float,
    head_yaw_corrected: float,
    head_pitch_corrected: float,
) -> tuple[float, float]:
    """Combine camera-space gaze with corrected head pose to get world-space gaze.

    Args:
        gaze_camera_yaw:      Gaze yaw in camera space (degrees).
        gaze_camera_pitch:    Gaze pitch in camera space (degrees).
        head_yaw_corrected:   Head yaw — Kalman-filtered + neutral-pose-corrected (degrees).
        head_pitch_corrected: Head pitch — same (degrees).

    Returns:
        (gaze_world_yaw_deg, gaze_world_pitch_deg) — before Kalman filtering.
        Caller (signal_processor.py) applies Kalman filter to these values.
    """
    gaze_world_yaw   = gaze_camera_yaw   + head_yaw_corrected   * config.GAZE_HEAD_COUPLING_ALPHA
    gaze_world_pitch = gaze_camera_pitch + head_pitch_corrected  * config.GAZE_HEAD_COUPLING_BETA
    return gaze_world_yaw, gaze_world_pitch


def is_on_road(gaze_world_yaw: float, gaze_world_pitch: float) -> bool:
    """Return True if world-space gaze falls within the forward road zone.

    Compares corrected, Kalman-filtered world angles against ROAD_ZONE config.
    PRD §2.2
    """
    yaw_ok   = config.ROAD_ZONE_YAW_MIN   <= gaze_world_yaw   <= config.ROAD_ZONE_YAW_MAX
    pitch_ok = config.ROAD_ZONE_PITCH_MIN <= gaze_world_pitch <= config.ROAD_ZONE_PITCH_MAX
    return yaw_ok and pitch_ok

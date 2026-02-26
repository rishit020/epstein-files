# layer3_temporal/speed_context.py — Speed zone resolver
# PRD §2.3, §FR-3.4
#
# Resolves the current speed zone from a speed reading.
# Returns (zone_name, speed_modifier).
#
# Zones:
#   PARKED  — speed < V_MIN_MPS          — modifier 0.0
#   URBAN   — V_MIN_MPS <= speed < V_HWY — modifier 1.0
#   HIGHWAY — speed >= V_HIGHWAY_MPS     — modifier 1.4
#
# None or stale speed defaults to URBAN (FM-05 fallback).

import config

# Zone name constants
ZONE_PARKED  = 'PARKED'
ZONE_URBAN   = 'URBAN'
ZONE_HIGHWAY = 'HIGHWAY'

# Speed clamp range (PRD §FR-3.4)
_SPEED_MIN_CLAMP = 0.0
_SPEED_MAX_CLAMP = 100.0


def resolve_speed_zone(
    speed_mps: float | None,
    speed_stale: bool,
) -> tuple[str, float]:
    """Classify the current speed into a driving zone.

    Args:
        speed_mps:   Vehicle speed in m/s, or None if unavailable.
        speed_stale: True if the speed reading is older than SPEED_STALE_THRESHOLD_S.

    Returns:
        (zone_name, modifier) where zone_name is one of 'PARKED', 'URBAN', 'HIGHWAY'
        and modifier is the composite-score multiplier for that zone.
    """
    if speed_mps is None or speed_stale:
        # FM-05: no speed source available — default to URBAN
        return ZONE_URBAN, 1.0

    # Clamp to valid physical range before classification
    speed = max(_SPEED_MIN_CLAMP, min(_SPEED_MAX_CLAMP, speed_mps))

    if speed < config.V_MIN_MPS:
        return ZONE_PARKED, 0.0
    if speed < config.V_HIGHWAY_MPS:
        return ZONE_URBAN, 1.0
    return ZONE_HIGHWAY, config.HIGHWAY_SCORE_MODIFIER

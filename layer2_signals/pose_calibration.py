# layer2_signals/pose_calibration.py — Neutral Pose Offset Correction
# PRD §5.7 — Neutral Pose Offset Correction
#
# Corrects head pose angles for per-vehicle camera mounting angle variation.
# Applied every frame, after Kalman filtering, before threshold comparison.
#
# Formula (PRD §5.7):
#   corrected_yaw   = filtered_yaw   - neutral_yaw_offset
#   corrected_pitch = filtered_pitch - neutral_pitch_offset
#
# Offsets are loaded from calibration/session_state.json on startup.
# If no valid calibration exists, offsets default to 0.0 (no correction).


class PoseCalibration:
    """Applies neutral pose offset correction to Kalman-filtered head angles.

    PRD §5.7
    """

    def __init__(self) -> None:
        self._neutral_yaw_offset: float = 0.0
        self._neutral_pitch_offset: float = 0.0

    # ── Public API ───────────────────────────────────────────────────────────

    def correct(self, filtered_yaw: float, filtered_pitch: float) -> tuple[float, float]:
        """Return (corrected_yaw, corrected_pitch) after subtracting neutral offsets.

        Args:
            filtered_yaw:   Kalman-filtered head yaw (degrees).
            filtered_pitch: Kalman-filtered head pitch (degrees).

        Returns:
            (corrected_yaw_deg, corrected_pitch_deg)
        """
        corrected_yaw   = filtered_yaw   - self._neutral_yaw_offset
        corrected_pitch = filtered_pitch - self._neutral_pitch_offset
        return corrected_yaw, corrected_pitch

    def set_offsets(self, yaw_offset: float, pitch_offset: float) -> None:
        """Set neutral pose offsets (called from session_state load or calibration).

        Args:
            yaw_offset:   Mean head yaw recorded during calibration (degrees).
            pitch_offset: Mean head pitch recorded during calibration (degrees).
        """
        self._neutral_yaw_offset   = yaw_offset
        self._neutral_pitch_offset = pitch_offset

    def reset(self) -> None:
        """Reset offsets to zero (no correction)."""
        self._neutral_yaw_offset   = 0.0
        self._neutral_pitch_offset = 0.0

    @property
    def neutral_yaw_offset(self) -> float:
        return self._neutral_yaw_offset

    @property
    def neutral_pitch_offset(self) -> float:
        return self._neutral_pitch_offset

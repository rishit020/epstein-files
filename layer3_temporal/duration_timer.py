# layer3_temporal/duration_timer.py — Continuous condition duration timer
# PRD §FR-3.2
#
# Increments by frame_delta_seconds each frame the condition is True.
# Resets to 0.0 on the first frame where the condition is False.
# All timers use frame-derived dt (not wall-clock) per NFR-R4.


class DurationTimer:
    """Tracks how many seconds a condition has been continuously True.

    PRD §FR-3.2
    """

    def __init__(self) -> None:
        self._value: float = 0.0

    def update(self, condition_met: bool, dt_seconds: float) -> float:
        """Update timer for one frame.

        Args:
            condition_met: True if the monitored condition holds this frame.
            dt_seconds:    Frame duration in seconds (from consecutive timestamps).

        Returns:
            Current accumulated duration in seconds.
        """
        if condition_met:
            self._value += dt_seconds
        else:
            self._value = 0.0
        return self._value

    def reset(self) -> None:
        """Force-reset the timer to 0.0."""
        self._value = 0.0

    @property
    def value(self) -> float:
        """Current accumulated duration in seconds."""
        return self._value

# layer3_temporal/blink_detector.py — Blink detection + anomaly score
# PRD §5.5, §FR-3.5
#
# State machine detects blinks by tracking consecutive frames where
# mean_EAR < close_threshold. A blink is valid if it lasts between
# BLINK_MIN_FRAMES and BLINK_MAX_FRAMES (67–333ms at 30fps).
#
# Blink rate anomaly score formula (PRD §5.5):
#   LOW  = BLINK_RATE_NORMAL_LOW_HZ  (0.13 Hz = 8 blinks/min)
#   HIGH = BLINK_RATE_NORMAL_HIGH_HZ (0.50 Hz = 30 blinks/min)
#
#   if rate < LOW:   score = 1.0 - (rate / LOW)
#   elif rate > HIGH: score = min(1.0, (rate - HIGH) / 0.5)
#   else:             score = 0.0
#   blink_rate_score = clamp(score, 0.0, 1.0)

from collections import deque
from typing import Optional

import config

# Window over which blink rate is computed (same as PERCLOS window)
_WINDOW_DURATION_S: float = config.FEATURE_WINDOW_FRAMES / config.CAPTURE_FPS


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


class BlinkDetector:
    """Detects blink events and computes blink rate anomaly score.

    PRD §5.5 / §FR-3.5
    """

    def __init__(self) -> None:
        # State machine
        self._consecutive_closed: int = 0
        self._currently_closed: bool = False

        # Cumulative time (seconds) — used as blink event timestamp axis
        self._elapsed_s: float = 0.0

        # Ring of blink event timestamps (cumulative seconds at event fire)
        # Unbounded in size; old events pruned in blink_rate_hz property.
        self._blink_event_times: deque[float] = deque()

    def update(
        self,
        mean_ear: Optional[float],
        close_threshold: float,
        dt_seconds: float,
    ) -> None:
        """Process one frame.

        Args:
            mean_ear:        Current mean EAR, or None if unavailable.
            close_threshold: EAR threshold below which eyes are considered closed.
            dt_seconds:      Frame duration in seconds (from consecutive timestamps).
        """
        self._elapsed_s += dt_seconds

        if mean_ear is None:
            # Treat missing EAR as eye open — reset closed counter
            if self._currently_closed:
                self._currently_closed = False
                self._consecutive_closed = 0
            return

        eye_closed = mean_ear < close_threshold

        if eye_closed:
            self._currently_closed = True
            self._consecutive_closed += 1
        else:
            if self._currently_closed:
                # Eyes just opened — check if closed run qualifies as a blink
                if config.BLINK_MIN_FRAMES <= self._consecutive_closed <= config.BLINK_MAX_FRAMES:
                    self._blink_event_times.append(self._elapsed_s)
            self._currently_closed = False
            self._consecutive_closed = 0

    def _prune_old_events(self) -> None:
        """Remove blink events older than the rate window."""
        cutoff = self._elapsed_s - _WINDOW_DURATION_S
        while self._blink_event_times and self._blink_event_times[0] <= cutoff:
            self._blink_event_times.popleft()

    @property
    def blink_rate_hz(self) -> float:
        """Blink rate in Hz over the current feature window."""
        self._prune_old_events()
        if _WINDOW_DURATION_S <= 0.0:
            return 0.0
        return len(self._blink_event_times) / _WINDOW_DURATION_S

    @property
    def blink_rate_score(self) -> float:
        """Blink rate anomaly score in [0.0, 1.0] per PRD §5.5."""
        rate = self.blink_rate_hz
        low  = config.BLINK_RATE_NORMAL_LOW_HZ
        high = config.BLINK_RATE_NORMAL_HIGH_HZ

        if rate < low:
            score = 1.0 - (rate / low) if low > 0.0 else 1.0
        elif rate > high:
            score = min(1.0, (rate - high) / 0.5)
        else:
            score = 0.0

        return _clamp(score, 0.0, 1.0)

# layer3_temporal/perclos_window.py — PERCLOS sliding window
# PRD §5.4, §FR-3.3
#
# Maintains a rolling window of PERCLOS_WINDOW_FRAMES eye-closure booleans.
# Invalid frames (signals_valid=False or mean_ear=None) are excluded from
# both numerator and denominator (PRD §FR-3.3).
#
# Eye closure criterion (P80 definition):
#   eyes_closed = mean_EAR <= baseline_EAR * (1.0 - PERCLOS_CLOSURE_FRACTION)
#               = mean_EAR <= baseline_EAR * 0.20   (at default 0.80 fraction)
#
# PERCLOS is marked invalid if fewer than PERCLOS_MIN_VALID_FRAMES valid frames
# exist in the window.

from collections import deque
from typing import Optional

import config


class PERCLOSWindow:
    """Sliding window PERCLOS calculator.

    PRD §5.4 / §FR-3.3
    """

    def __init__(self, window_size: int = config.PERCLOS_WINDOW_FRAMES) -> None:
        self._window_size = window_size
        # Each slot stores (is_valid: bool, is_closed: bool)
        self._window: deque[tuple[bool, bool]] = deque(maxlen=window_size)

    def update(
        self,
        mean_ear: Optional[float],
        baseline_ear: float,
        is_valid: bool,
    ) -> None:
        """Feed one frame into the PERCLOS window.

        Args:
            mean_ear:     Mean EAR for this frame, or None if unavailable.
            baseline_ear: Current calibrated baseline EAR.
            is_valid:     True if the frame has valid eye signals.
        """
        if not is_valid or mean_ear is None:
            self._window.append((False, False))
            return

        threshold = baseline_ear * (1.0 - config.PERCLOS_CLOSURE_FRACTION)
        is_closed = mean_ear <= threshold
        self._window.append((True, is_closed))

    @property
    def perclos(self) -> float:
        """PERCLOS value over the current window.

        Returns 0.0 if fewer than PERCLOS_MIN_VALID_FRAMES valid frames exist.
        """
        if not self.valid:
            return 0.0
        closed_count = sum(1 for valid, closed in self._window if valid and closed)
        return closed_count / self.frames_valid

    @property
    def valid(self) -> bool:
        """True if there are enough valid frames to compute a meaningful PERCLOS."""
        return self.frames_valid >= config.PERCLOS_MIN_VALID_FRAMES

    @property
    def frames_valid(self) -> int:
        """Number of valid frames in the current window."""
        return sum(1 for valid, _ in self._window if valid)

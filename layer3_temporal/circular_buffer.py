# layer3_temporal/circular_buffer.py — Fixed-size circular frame buffer
# PRD §FR-3.1
#
# Maintains exactly CIRCULAR_BUFFER_SIZE SignalFrame entries.
# On overflow the oldest entry is evicted (FIFO).

from collections import deque

import config
from layer2_signals.messages import SignalFrame


class CircularBuffer:
    """Fixed-size FIFO buffer of SignalFrame objects.

    PRD §FR-3.1
    """

    def __init__(self, maxsize: int = config.CIRCULAR_BUFFER_SIZE) -> None:
        self._maxsize = maxsize
        self._buf: deque[SignalFrame] = deque(maxlen=maxsize)

    def push(self, frame: SignalFrame) -> None:
        """Append a frame; oldest is evicted automatically when full."""
        self._buf.append(frame)

    def get_window(self, n: int) -> list[SignalFrame]:
        """Return the last *n* frames, oldest first.

        If fewer than *n* frames have been pushed, returns all stored frames.
        """
        n = min(n, len(self._buf))
        if n <= 0:
            return []
        return list(self._buf)[-n:]

    def __len__(self) -> int:
        return len(self._buf)

    @property
    def is_full(self) -> bool:
        return len(self._buf) == self._maxsize

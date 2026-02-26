# layer3_temporal/watchdog.py — WatchdogManager
# PRD §FR-3.6
#
# Detects hung threads by requiring a kick() call at least every
# WATCHDOG_TIMEOUT_S (2.0s). Background thread checks every WATCHDOG_HEARTBEAT_S.
#
# Design for testability: check() accepts an injectable `now` timestamp so
# unit tests can verify timeout behaviour without sleeping.

import logging
import threading
import time
from typing import Callable, Optional

import config

logger = logging.getLogger(__name__)


class WatchdogManager:
    """Detects hung processing threads by monitoring kick frequency.

    PRD §FR-3.6
    """

    def __init__(self, timeout_s: float = config.WATCHDOG_TIMEOUT_S) -> None:
        self._timeout_s  = timeout_s
        self._last_kick_time: float = time.monotonic()
        self._last_frame_id: int = -1
        self._timed_out: bool = False
        self._callback: Optional[Callable[[], None]] = None
        self._lock = threading.Lock()

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    # ── Public API ────────────────────────────────────────────────────────────

    def kick(self, frame_id: int) -> None:
        """Record a heartbeat for the given frame_id."""
        with self._lock:
            self._last_kick_time = time.monotonic()
            self._last_frame_id  = frame_id
            self._timed_out      = False   # recover on kick

    def check(self, now: Optional[float] = None) -> bool:
        """Check whether the watchdog timeout has elapsed.

        Args:
            now: Injectable monotonic timestamp (seconds) for testing.
                 Uses time.monotonic() if not provided.

        Returns:
            True if timed out (no kick in WATCHDOG_TIMEOUT_S seconds).
        """
        if now is None:
            now = time.monotonic()

        with self._lock:
            elapsed = now - self._last_kick_time

        if elapsed >= self._timeout_s and not self._timed_out:
            with self._lock:
                self._timed_out = True
            self._on_timeout(elapsed)
            return True

        return self._timed_out

    def set_timeout_callback(self, cb: Callable[[], None]) -> None:
        """Register a callback to invoke when the watchdog fires."""
        self._callback = cb

    def start(self) -> None:
        """Start the background watchdog thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._watchdog_loop,
            name='WatchdogThread',
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the background watchdog thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=config.WATCHDOG_HEARTBEAT_S * 2)
        self._thread = None

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def timed_out(self) -> bool:
        """True if the watchdog has fired and not yet recovered via kick()."""
        return self._timed_out

    @property
    def last_frame_id(self) -> int:
        """Frame ID of the most recent kick."""
        return self._last_frame_id

    @property
    def secs_since_last_kick(self) -> float:
        """Seconds elapsed since the most recent kick()."""
        with self._lock:
            return time.monotonic() - self._last_kick_time

    # ── Private ───────────────────────────────────────────────────────────────

    def _watchdog_loop(self) -> None:
        while not self._stop_event.is_set():
            self.check()
            self._stop_event.wait(timeout=config.WATCHDOG_HEARTBEAT_S)

    def _on_timeout(self, elapsed_s: float) -> None:
        logger.error(
            'WATCHDOG_TIMEOUT: no kick for %.2fs (last frame_id=%d) — signalling DEGRADED',
            elapsed_s,
            self._last_frame_id,
        )
        if self._callback is not None:
            try:
                self._callback()
            except Exception as exc:  # noqa: BLE001
                logger.exception('WatchdogManager callback raised: %s', exc)

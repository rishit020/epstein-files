# layer6_output/audio_handler.py — Audio alert dispatcher (Mac: afplay)
# PRD §FR-6.1 — dispatch latency <= 50ms (NFR-P4)

from __future__ import annotations

import logging
import subprocess
import time

import config
from layer5_alert.alert_types import AlertLevel
from layer5_alert.messages import AlertCommand

_log = logging.getLogger(__name__)


class AudioAlertHandler:
    """Plays audio alerts via platform sound command.

    Mac implementation uses afplay (fire-and-forget via subprocess.Popen).
    PRD §FR-6.1, NFR-P4: dispatch latency <= 50ms.
    """

    def play(self, command: AlertCommand) -> None:
        """Dispatch audio alert for the given AlertCommand.

        Args:
            command: AlertCommand from AlertStateMachine.

        Selects sound file based on command.level:
            - URGENT -> config.AUDIO_ALERT_SOUND_URGENT
            - HIGH (or any other) -> config.AUDIO_ALERT_SOUND

        Fire-and-forget: does NOT wait for playback to complete.
        """
        sound_file = (
            config.AUDIO_ALERT_SOUND_URGENT
            if command.level == AlertLevel.URGENT
            else config.AUDIO_ALERT_SOUND
        )
        t0 = time.perf_counter()
        try:
            subprocess.Popen(['afplay', sound_file])
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            if elapsed_ms > 50.0:
                _log.warning(
                    "Audio dispatch latency %.1fms exceeds 50ms NFR-P4 limit "
                    "(alert_type=%s, level=%s)",
                    elapsed_ms, command.alert_type.value, command.level.value,
                )
            else:
                _log.debug(
                    "Audio dispatched in %.1fms (alert_type=%s, level=%s)",
                    elapsed_ms, command.alert_type.value, command.level.value,
                )
        except Exception as exc:
            _log.error(
                "AudioAlertHandler.play() failed: %s (sound_file=%s)",
                exc, sound_file,
            )

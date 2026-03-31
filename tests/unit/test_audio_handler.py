"""test_audio_handler.py — Unit tests for AudioAlertHandler.

PRD §FR-6.1: Audio alert dispatch via afplay on Mac.
PRD §NFR-P4: Dispatch latency <= 50ms P95.
Decision D-03 (06-CONTEXT.md): fire-and-forget via subprocess.Popen.
"""

from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import uuid
import pytest
from unittest.mock import MagicMock, patch

from layer6_output.audio_handler import AudioAlertHandler
from layer5_alert.alert_types import AlertLevel, AlertType
from layer5_alert.messages import AlertCommand
import config


# ═══════════════════════════════════════════════════════════════════════════════
# Helper factories
# ═══════════════════════════════════════════════════════════════════════════════

def _make_alert(
    level: AlertLevel = AlertLevel.HIGH,
    alert_type: AlertType = AlertType.VISUAL_INATTENTION,
) -> AlertCommand:
    """Return a valid AlertCommand with synthetic values."""
    return AlertCommand(
        alert_id=str(uuid.uuid4()),
        timestamp_ns=1_000_000_000,
        level=level,
        alert_type=alert_type,
        composite_score=0.7,
        suppress_until_ns=2_000_000_000,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TestAudioAlertHandler
# FR-6.1: afplay-backed audio alert dispatcher
# NFR-P4: Dispatch latency <= 50ms from AlertCommand receipt to audio start
# ═══════════════════════════════════════════════════════════════════════════════

class TestAudioAlertHandler:

    @patch('layer6_output.audio_handler.subprocess.Popen')
    def test_play_high_calls_afplay_with_ping(self, mock_popen):
        """FR-6.1: HIGH level alert dispatches afplay with config.AUDIO_ALERT_SOUND (Ping.aiff)."""
        mock_popen.return_value = MagicMock()
        handler = AudioAlertHandler()
        alert = _make_alert(level=AlertLevel.HIGH)
        handler.play(alert)
        mock_popen.assert_called_once_with(['afplay', config.AUDIO_ALERT_SOUND])

    @patch('layer6_output.audio_handler.subprocess.Popen')
    def test_play_urgent_calls_afplay_with_sosumi(self, mock_popen):
        """FR-6.1: URGENT level alert dispatches afplay with config.AUDIO_ALERT_SOUND_URGENT (Sosumi.aiff)."""
        mock_popen.return_value = MagicMock()
        handler = AudioAlertHandler()
        alert = _make_alert(level=AlertLevel.URGENT)
        handler.play(alert)
        mock_popen.assert_called_once_with(['afplay', config.AUDIO_ALERT_SOUND_URGENT])

    @patch('layer6_output.audio_handler.subprocess.Popen')
    def test_play_does_not_wait_on_process(self, mock_popen):
        """FR-6.1 / D-03: fire-and-forget — must NOT call .wait() or .communicate() on the process."""
        mock_proc = MagicMock()
        mock_popen.return_value = mock_proc
        handler = AudioAlertHandler()
        alert = _make_alert(level=AlertLevel.HIGH)
        handler.play(alert)
        mock_proc.wait.assert_not_called()
        mock_proc.communicate.assert_not_called()

    @patch('layer6_output.audio_handler.subprocess.Popen')
    @patch('layer6_output.audio_handler.time.perf_counter')
    @patch('layer6_output.audio_handler._log')
    def test_play_logs_warning_on_slow_dispatch(self, mock_log, mock_perf_counter, mock_popen):
        """NFR-P4: Logs a warning if dispatch takes >50ms."""
        mock_popen.return_value = MagicMock()
        # First call returns 0.0 (t0), second call returns 0.06 (60ms elapsed)
        mock_perf_counter.side_effect = [0.0, 0.06]
        handler = AudioAlertHandler()
        alert = _make_alert(level=AlertLevel.HIGH)
        handler.play(alert)
        mock_log.warning.assert_called_once()
        warn_args = mock_log.warning.call_args[0]
        # Check the format string contains '50ms' or 'NFR-P4'
        assert '50ms' in warn_args[0] or 'NFR-P4' in warn_args[0]

    def test_play_catches_exceptions(self):
        """D-03: OSError from subprocess must be caught and not propagate to caller."""
        with patch('layer6_output.audio_handler.subprocess.Popen', side_effect=OSError("afplay not found")):
            handler = AudioAlertHandler()
            alert = _make_alert(level=AlertLevel.HIGH)
            # Must not raise
            handler.play(alert)

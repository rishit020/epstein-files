"""test_audio_handler.py — TDD stub tests for AudioAlertHandler.

PRD §FR-6.1: Audio alert dispatch via afplay on Mac.
PRD §NFR-P4: Dispatch latency <= 50ms P95.
Decision D-03 (06-CONTEXT.md): fire-and-forget via subprocess.Popen.

All tests skip until layer6_output.audio_handler is implemented (Plan 03).
"""

from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import uuid
import pytest
from unittest.mock import MagicMock, patch

_IMPL_MISSING = False
try:
    from layer6_output.audio_handler import AudioAlertHandler
except ImportError:
    _IMPL_MISSING = True

from layer5_alert.alert_types import AlertLevel, AlertType
from layer5_alert.messages import AlertCommand
import config

pytestmark = pytest.mark.skipif(
    _IMPL_MISSING,
    reason="AudioAlertHandler not yet implemented — will be implemented in Plan 03",
)


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

    def test_play_high_calls_afplay_with_ping(self):
        """FR-6.1: HIGH level alert dispatches afplay with config.AUDIO_ALERT_SOUND (Ping.aiff)."""
        pytest.skip("Stub — implementation in Plan 03")
        with patch('subprocess.Popen') as mock_popen:
            mock_popen.return_value = MagicMock()
            handler = AudioAlertHandler()
            alert = _make_alert(level=AlertLevel.HIGH)
            handler.play(alert)
            mock_popen.assert_called_once()
            cmd = mock_popen.call_args[0][0]
            assert cmd == ['afplay', config.AUDIO_ALERT_SOUND]

    def test_play_urgent_calls_afplay_with_sosumi(self):
        """FR-6.1: URGENT level alert dispatches afplay with config.AUDIO_ALERT_SOUND_URGENT (Sosumi.aiff)."""
        pytest.skip("Stub — implementation in Plan 03")
        with patch('subprocess.Popen') as mock_popen:
            mock_popen.return_value = MagicMock()
            handler = AudioAlertHandler()
            alert = _make_alert(level=AlertLevel.URGENT)
            handler.play(alert)
            mock_popen.assert_called_once()
            cmd = mock_popen.call_args[0][0]
            assert cmd == ['afplay', config.AUDIO_ALERT_SOUND_URGENT]

    def test_play_does_not_wait_on_process(self):
        """FR-6.1 / D-03: fire-and-forget — must NOT call .wait() or .communicate() on the process."""
        pytest.skip("Stub — implementation in Plan 03")
        with patch('subprocess.Popen') as mock_popen:
            mock_proc = MagicMock()
            mock_popen.return_value = mock_proc
            handler = AudioAlertHandler()
            alert = _make_alert(level=AlertLevel.HIGH)
            handler.play(alert)
            mock_proc.wait.assert_not_called()
            mock_proc.communicate.assert_not_called()

    def test_play_logs_warning_on_slow_dispatch(self):
        """NFR-P4: Logs a warning if dispatch takes >50ms."""
        pytest.skip("Stub — implementation in Plan 03")
        import time
        import logging
        _call_count = [0]
        _base = time.perf_counter()

        def _slow_perf_counter():
            _call_count[0] += 1
            if _call_count[0] == 1:
                return _base
            # Second call (after Popen): simulate 60ms elapsed
            return _base + 0.06

        with patch('subprocess.Popen') as mock_popen:
            mock_popen.return_value = MagicMock()
            with patch('time.perf_counter', side_effect=_slow_perf_counter):
                with patch('logging.warning') as mock_warn:
                    handler = AudioAlertHandler()
                    alert = _make_alert(level=AlertLevel.HIGH)
                    handler.play(alert)
                    mock_warn.assert_called_once()
                    warn_msg = mock_warn.call_args[0][0]
                    assert '50ms' in warn_msg or '50' in warn_msg

    def test_play_catches_exceptions(self):
        """D-03: OSError from subprocess must be caught and not propagate to caller."""
        pytest.skip("Stub — implementation in Plan 03")
        with patch('subprocess.Popen', side_effect=OSError("afplay not found")):
            handler = AudioAlertHandler()
            alert = _make_alert(level=AlertLevel.HIGH)
            # Must not raise
            handler.play(alert)

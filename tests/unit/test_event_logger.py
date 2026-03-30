"""test_event_logger.py — TDD stub tests for EventLogger.

PRD §FR-6.3: Rotating JSON log, 50MB max, 5 files retained.
PRD §9: Event log format (ALERT, STATE_TRANSITION, DEGRADED, WATCHDOG_TIMEOUT, CALIBRATION_COMPLETE).
Decision D-04 (06-CONTEXT.md): RotatingFileHandler-backed JSONL logger.

All tests skip until layer6_output.event_logger is implemented (Plan 04).
"""

from __future__ import annotations

import json
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import uuid
import pytest
from unittest.mock import patch

_IMPL_MISSING = False
try:
    from layer6_output.event_logger import EventLogger
except ImportError:
    _IMPL_MISSING = True

from layer5_alert.alert_types import AlertLevel, AlertType
from layer5_alert.messages import AlertCommand
from layer3_temporal.messages import TemporalFeatures
import config

pytestmark = pytest.mark.skipif(
    _IMPL_MISSING,
    reason="EventLogger not yet implemented — will be implemented in Plan 04",
)


# ═══════════════════════════════════════════════════════════════════════════════
# Helper factories
# ═══════════════════════════════════════════════════════════════════════════════

def _make_alert(
    level: AlertLevel = AlertLevel.HIGH,
    alert_type: AlertType = AlertType.VISUAL_INATTENTION,
) -> AlertCommand:
    return AlertCommand(
        alert_id=str(uuid.uuid4()),
        timestamp_ns=1_000_000_000,
        level=level,
        alert_type=alert_type,
        composite_score=0.72,
        suppress_until_ns=9_000_000_000,
    )


def _make_features(**overrides) -> TemporalFeatures:
    defaults = dict(
        timestamp_ns=1_000_000_000,
        gaze_off_road_fraction=0.3,
        gaze_continuous_secs=2.5,
        head_deviation_mean_deg=10.0,
        head_continuous_secs=1.2,
        perclos=0.08,
        blink_rate_score=0.2,
        phone_confidence_mean=0.1,
        phone_continuous_secs=0.0,
        speed_zone='URBAN',
        speed_modifier=1.0,
        frames_valid_in_window=55,
        face_absent_continuous_secs=0.0,
        thermal_throttle_active=False,
    )
    defaults.update(overrides)
    return TemporalFeatures(**defaults)


# ═══════════════════════════════════════════════════════════════════════════════
# TestEventLogger
# FR-6.3: Rotating JSON log, 50MB max, 5 files retained
# PRD §9: JSONL event format with required fields per event type
# ═══════════════════════════════════════════════════════════════════════════════

class TestEventLogger:

    def test_log_alert_writes_jsonl(self, tmp_path):
        """FR-6.3 / §9: log_alert writes a valid JSON line with event_type='ALERT'."""
        pytest.skip("Stub — implementation in Plan 04")
        logger = EventLogger(log_dir=str(tmp_path))
        alert = _make_alert()
        features = _make_features()
        logger.log_alert(alert, features)

        log_files = list(tmp_path.glob('*.jsonl'))
        assert len(log_files) == 1
        line = log_files[0].read_text().strip()
        assert line  # non-empty
        obj = json.loads(line)
        assert obj['event_type'] == 'ALERT'

    def test_log_alert_contains_required_fields(self, tmp_path):
        """PRD §9: ALERT entry must contain all required keys per spec."""
        pytest.skip("Stub — implementation in Plan 04")
        logger = EventLogger(log_dir=str(tmp_path))
        alert = _make_alert()
        features = _make_features()
        logger.log_alert(alert, features)

        log_file = next(tmp_path.glob('*.jsonl'))
        obj = json.loads(log_file.read_text().strip())
        required_keys = [
            'event_type',
            'timestamp_ns',
            'alert_id',
            'alert_type',
            'alert_level',
            'composite_score',
            'active_classes',
            'gaze_continuous_secs',
            'head_continuous_secs',
            'perclos',
            'phone_confidence',
            'suppress_until_ns',
        ]
        for key in required_keys:
            assert key in obj, f"Missing required field: {key}"

    def test_log_state_transition_writes_jsonl(self, tmp_path):
        """PRD §9: log_state_transition writes a JSON line with event_type='STATE_TRANSITION'."""
        pytest.skip("Stub — implementation in Plan 04")
        logger = EventLogger(log_dir=str(tmp_path))
        logger.log_state_transition(
            prev='NOMINAL',
            new='ALERTING',
            trigger='ALT-01',
            frame_id=100,
            ts_ns=1_000_000_000,
        )
        log_file = next(tmp_path.glob('*.jsonl'))
        obj = json.loads(log_file.read_text().strip())
        assert obj['event_type'] == 'STATE_TRANSITION'
        assert obj['previous_state'] == 'NOMINAL'
        assert obj['new_state'] == 'ALERTING'
        assert obj['trigger'] == 'ALT-01'
        assert 'frame_id' in obj

    def test_log_degraded_writes_jsonl(self, tmp_path):
        """PRD §9: log_degraded writes a JSON line with event_type='DEGRADED'."""
        pytest.skip("Stub — implementation in Plan 04")
        logger = EventLogger(log_dir=str(tmp_path))
        logger.log_degraded(
            reason='perception_invalid_60_frames',
            duration_secs=2.0,
            ts_ns=1_000_000_000,
        )
        log_file = next(tmp_path.glob('*.jsonl'))
        obj = json.loads(log_file.read_text().strip())
        assert obj['event_type'] == 'DEGRADED'
        assert obj['reason'] == 'perception_invalid_60_frames'
        assert 'duration_secs' in obj

    def test_log_watchdog_timeout_writes_jsonl(self, tmp_path):
        """PRD §9: log_watchdog_timeout writes a JSON line with event_type='WATCHDOG_TIMEOUT'."""
        pytest.skip("Stub — implementation in Plan 04")
        logger = EventLogger(log_dir=str(tmp_path))
        logger.log_watchdog_timeout(
            last_frame_id=10531,
            secs_since_last_frame=2.14,
            ts_ns=1_000_000_000,
        )
        log_file = next(tmp_path.glob('*.jsonl'))
        obj = json.loads(log_file.read_text().strip())
        assert obj['event_type'] == 'WATCHDOG_TIMEOUT'
        assert obj['last_frame_id'] == 10531
        assert 'secs_since_last_frame' in obj

    def test_log_calibration_complete_writes_jsonl(self, tmp_path):
        """PRD §9: log_calibration_complete writes a JSON line with event_type='CALIBRATION_COMPLETE'."""
        pytest.skip("Stub — implementation in Plan 04")
        logger = EventLogger(log_dir=str(tmp_path))
        calibration_event = {
            'session_id': 'sess-001',
            'yaw_offset_deg': 3.2,
            'pitch_offset_deg': -1.5,
            'frames_used': 295,
        }
        logger.log_calibration_complete(calibration_event)
        log_file = next(tmp_path.glob('*.jsonl'))
        obj = json.loads(log_file.read_text().strip())
        assert obj['event_type'] == 'CALIBRATION_COMPLETE'

    def test_enum_serialization_uses_value(self, tmp_path):
        """D-04: alert_type serialized as 'D-A' not 'AlertType.VISUAL_INATTENTION'; level as 'HIGH' not 'AlertLevel.HIGH'."""
        pytest.skip("Stub — implementation in Plan 04")
        logger = EventLogger(log_dir=str(tmp_path))
        alert = _make_alert(
            level=AlertLevel.HIGH,
            alert_type=AlertType.VISUAL_INATTENTION,
        )
        features = _make_features()
        logger.log_alert(alert, features)

        log_file = next(tmp_path.glob('*.jsonl'))
        obj = json.loads(log_file.read_text().strip())
        assert obj['alert_type'] == 'D-A', f"Expected 'D-A', got {obj['alert_type']!r}"
        assert obj['alert_level'] == 'HIGH', f"Expected 'HIGH', got {obj['alert_level']!r}"

    def test_rotating_file_handler_configured(self, tmp_path, monkeypatch):
        """FR-6.3: RotatingFileHandler configured with maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUP_COUNT."""
        pytest.skip("Stub — implementation in Plan 04")
        # Set a tiny rotation threshold so we can trigger rotation in test
        monkeypatch.setattr(config, 'LOG_MAX_BYTES', 100)
        monkeypatch.setattr(config, 'LOG_BACKUP_COUNT', 5)

        logger = EventLogger(log_dir=str(tmp_path))
        alert = _make_alert()
        features = _make_features()

        # Write enough entries to exceed 100-byte limit and trigger rotation
        for _ in range(20):
            logger.log_alert(alert, features)

        # After rotation, a backup file (e.g. attentia_events.jsonl.1) should exist
        backup_files = list(tmp_path.glob('*.jsonl.*'))
        assert len(backup_files) >= 1, "Expected at least one backup file after log rotation"

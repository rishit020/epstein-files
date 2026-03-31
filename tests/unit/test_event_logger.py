"""test_event_logger.py — TDD tests for EventLogger.

PRD §FR-6.3: Rotating JSON log, 50MB max, 5 files retained.
PRD §9: Event log format (ALERT, STATE_TRANSITION, DEGRADED, WATCHDOG_TIMEOUT,
        THERMAL_WARNING, CALIBRATION_COMPLETE).
Decision D-04 (06-CONTEXT.md): RotatingFileHandler-backed JSONL logger.
"""

from __future__ import annotations

import json
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pytest

from layer6_output.event_logger import EventLogger
from layer5_alert.messages import AlertCommand
from layer5_alert.alert_types import AlertLevel, AlertType
from layer3_temporal.messages import TemporalFeatures
from layer4_scoring.messages import DistractionScore
import config


# ═══════════════════════════════════════════════════════════════════════════════
# Helper factories
# ═══════════════════════════════════════════════════════════════════════════════

def _make_alert_command() -> AlertCommand:
    return AlertCommand(
        alert_id='test-uuid-1234',
        timestamp_ns=1_740_000_000_000_000_000,
        level=AlertLevel.HIGH,
        alert_type=AlertType.VISUAL_INATTENTION,
        composite_score=0.67,
        suppress_until_ns=1_740_000_008_000_000_000,
    )


def _make_temporal_features() -> TemporalFeatures:
    return TemporalFeatures(
        timestamp_ns=1_740_000_000_000_000_000,
        gaze_off_road_fraction=0.4,
        gaze_continuous_secs=2.1,
        head_deviation_mean_deg=8.0,
        head_continuous_secs=1.6,
        perclos=0.08,
        blink_rate_score=0.5,
        phone_confidence_mean=0.0,
        phone_continuous_secs=0.0,
        speed_zone='URBAN',
        speed_modifier=1.0,
        frames_valid_in_window=60,
    )


def _make_distraction_score() -> DistractionScore:
    return DistractionScore(
        timestamp_ns=1_740_000_000_000_000_000,
        composite_score=0.67,
        component_gaze=0.30,
        component_head=0.15,
        component_perclos=0.02,
        component_blink=0.00,
        gaze_threshold_breached=True,
        head_threshold_breached=False,
        perclos_threshold_breached=False,
        phone_threshold_breached=False,
        active_classes=['D-A'],
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TestEventLogger
# FR-6.3: Rotating JSON log, 50MB max, 5 files retained
# PRD §9: JSONL event format with required fields per event type
# ═══════════════════════════════════════════════════════════════════════════════

class TestEventLogger:

    def test_log_alert_writes_jsonl(self, tmp_path):
        """FR-6.3 / §9: log_alert writes a valid JSON line with event_type='ALERT'."""
        logger = EventLogger(log_dir=str(tmp_path))
        logger.log_alert(_make_alert_command(), _make_temporal_features(), _make_distraction_score())
        lines = (tmp_path / 'attentia_events.jsonl').read_text().strip().splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry['event_type'] == 'ALERT'

    def test_log_alert_contains_required_fields(self, tmp_path):
        """PRD §9: ALERT entry must contain all 14 required keys per spec."""
        logger = EventLogger(log_dir=str(tmp_path))
        logger.log_alert(_make_alert_command(), _make_temporal_features(), _make_distraction_score())
        entry = json.loads((tmp_path / 'attentia_events.jsonl').read_text().strip())
        required = [
            'event_type', 'timestamp_ns', 'alert_id', 'alert_type', 'alert_level',
            'composite_score', 'active_classes', 'speed_mps', 'speed_zone',
            'speed_source', 'gaze_continuous_secs', 'head_continuous_secs',
            'perclos', 'phone_confidence', 'suppress_until_ns',
        ]
        for field in required:
            assert field in entry, f"Missing field: {field}"

    def test_enum_serialization_uses_string(self, tmp_path):
        """D-04: alert_type serialized as 'D-A', alert_level as 'HIGH' (not int 2)."""
        logger = EventLogger(log_dir=str(tmp_path))
        logger.log_alert(_make_alert_command(), _make_temporal_features(), _make_distraction_score())
        entry = json.loads((tmp_path / 'attentia_events.jsonl').read_text().strip())
        assert entry['alert_type'] == 'D-A', f"Expected 'D-A', got {entry['alert_type']!r}"
        assert entry['alert_level'] == 'HIGH', f"Expected 'HIGH', got {entry['alert_level']!r}"

    def test_log_state_transition_writes_jsonl(self, tmp_path):
        """PRD §9: log_state_transition writes STATE_TRANSITION entry with required fields."""
        logger = EventLogger(log_dir=str(tmp_path))
        logger.log_state_transition('NOMINAL', 'ALERTING', 'ALT-01', 4521, 1_740_000_000_000_000_000)
        entry = json.loads((tmp_path / 'attentia_events.jsonl').read_text().strip())
        assert entry['event_type'] == 'STATE_TRANSITION'
        assert entry['previous_state'] == 'NOMINAL'
        assert entry['new_state'] == 'ALERTING'
        assert entry['trigger'] == 'ALT-01'
        assert entry['frame_id'] == 4521

    def test_log_degraded_writes_jsonl(self, tmp_path):
        """PRD §9: log_degraded writes DEGRADED entry with reason and duration_secs."""
        logger = EventLogger(log_dir=str(tmp_path))
        logger.log_degraded('perception_invalid_60_frames', 2.0, 1_740_000_000_000_000_000)
        entry = json.loads((tmp_path / 'attentia_events.jsonl').read_text().strip())
        assert entry['event_type'] == 'DEGRADED'
        assert entry['reason'] == 'perception_invalid_60_frames'
        assert entry['duration_secs'] == 2.0

    def test_log_watchdog_timeout_writes_jsonl(self, tmp_path):
        """PRD §9: log_watchdog_timeout writes WATCHDOG_TIMEOUT entry with required fields."""
        logger = EventLogger(log_dir=str(tmp_path))
        logger.log_watchdog_timeout(10531, 2.14, 1_740_000_000_000_000_000)
        entry = json.loads((tmp_path / 'attentia_events.jsonl').read_text().strip())
        assert entry['event_type'] == 'WATCHDOG_TIMEOUT'
        assert entry['last_frame_id'] == 10531
        assert abs(entry['secs_since_last_frame'] - 2.14) < 0.001
        assert 'recovery_action' in entry

    def test_log_thermal_warning_writes_jsonl(self, tmp_path):
        """PRD §9: log_thermal_warning writes THERMAL_WARNING entry with required fields."""
        logger = EventLogger(log_dir=str(tmp_path))
        logger.log_thermal_warning(82.0, 'reduced_yolo_resolution_to_256', 1_740_000_000_000_000_000)
        entry = json.loads((tmp_path / 'attentia_events.jsonl').read_text().strip())
        assert entry['event_type'] == 'THERMAL_WARNING'
        assert entry['temperature_c'] == 82.0
        assert entry['action'] == 'reduced_yolo_resolution_to_256'
        assert 'inference_ms_mean' in entry

    def test_log_calibration_complete_writes_jsonl(self, tmp_path):
        """PRD §9: log_calibration_complete writes CALIBRATION_COMPLETE preserving caller dict."""
        logger = EventLogger(log_dir=str(tmp_path))
        cal_event = {
            'baseline_ear': 0.31,
            'neutral_yaw_offset': -4.2,
            'neutral_pitch_offset': 3.1,
            'vehicle_vin': '1HGCM82633A004352',
            'frames_collected': 298,
        }
        logger.log_calibration_complete(cal_event)
        entry = json.loads((tmp_path / 'attentia_events.jsonl').read_text().strip())
        assert entry['event_type'] == 'CALIBRATION_COMPLETE'
        assert entry['baseline_ear'] == 0.31
        assert 'timestamp_ns' in entry

    def test_logger_does_not_propagate(self, tmp_path):
        """FR-6.3: logger.propagate=False — JSONL entries do not appear on stderr/root logger."""
        logger = EventLogger(log_dir=str(tmp_path))
        assert logger._logger.propagate is False

    def test_multiple_entries_each_on_own_line(self, tmp_path):
        """FR-6.3: each log call produces exactly one valid JSON line."""
        logger = EventLogger(log_dir=str(tmp_path))
        logger.log_degraded('reason_a', 1.0, 1000)
        logger.log_degraded('reason_b', 2.0, 2000)
        lines = (tmp_path / 'attentia_events.jsonl').read_text().strip().splitlines()
        assert len(lines) == 2
        for line in lines:
            json.loads(line)  # must be valid JSON

    def test_rotating_file_handler_configured(self, tmp_path, monkeypatch):
        """FR-6.3: RotatingFileHandler configured with maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUP_COUNT."""
        monkeypatch.setattr(config, 'LOG_MAX_BYTES', 100)
        monkeypatch.setattr(config, 'LOG_BACKUP_COUNT', 5)

        logger = EventLogger(log_dir=str(tmp_path))
        alert = _make_alert_command()
        features = _make_temporal_features()
        score = _make_distraction_score()

        # Write enough entries to exceed 100-byte limit and trigger rotation
        for _ in range(20):
            logger.log_alert(alert, features, score)

        # After rotation, a backup file (e.g. attentia_events.jsonl.1) should exist
        backup_files = list(tmp_path.glob('*.jsonl.*'))
        assert len(backup_files) >= 1, "Expected at least one backup file after log rotation"

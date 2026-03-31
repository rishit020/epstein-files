# layer6_output/event_logger.py — Structured JSONL event log
# PRD §FR-6.3, §9 — rotating newline-delimited JSON log
# Thread-safe: Python logging module uses internal GIL + handler locks

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import time
from typing import Optional

import config
from layer3_temporal.messages import TemporalFeatures
from layer4_scoring.messages import DistractionScore
from layer5_alert.messages import AlertCommand

_LOG_FILENAME = 'attentia_events.jsonl'
_RECOVERY_ACTION_DEFAULT = 'thread_restart_attempted'
_INFERENCE_MS_MEAN_DEFAULT = 0.0  # ThermalMonitor does not expose this on Mac dev

_module_log = logging.getLogger(__name__)


class EventLogger:
    """Writes structured event records to a rotating JSONL log file.

    PRD §FR-6.3 — rotating JSON log, 50MB max, 5 files retained.
    PRD §9 — exact JSON schemas for all 6 event types.

    Thread-safe: Python logging module is thread-safe by default.
    """

    def __init__(self, log_dir: Optional[str] = None) -> None:
        """Create EventLogger.

        Args:
            log_dir: Directory for log file. Defaults to config.LOG_DIR.
                     Created if it does not exist.
        """
        dir_path = log_dir if log_dir is not None else config.LOG_DIR
        os.makedirs(dir_path, exist_ok=True)
        log_path = os.path.join(dir_path, _LOG_FILENAME)

        handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=config.LOG_MAX_BYTES,
            backupCount=config.LOG_BACKUP_COUNT,
        )
        handler.setFormatter(logging.Formatter('%(message)s'))

        self._logger = logging.getLogger(f'attentia.events.{id(self)}')
        self._logger.addHandler(handler)
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False  # prevent double-logging to root/stderr (Pitfall 5)

        _module_log.info(
            "EventLogger initialised: path=%s, maxBytes=%d, backupCount=%d",
            log_path, config.LOG_MAX_BYTES, config.LOG_BACKUP_COUNT,
        )

    def _write(self, entry: dict) -> None:
        """Serialise entry as JSON and write one line to the log."""
        self._logger.info(json.dumps(entry))

    # ── Public log methods ────────────────────────────────────────────────────

    def log_alert(
        self,
        command: AlertCommand,
        features: TemporalFeatures,
        score: DistractionScore,
        speed_mps: float = 0.0,
    ) -> None:
        """Write ALERT entry per PRD §9.

        Args:
            command:    AlertCommand from AlertStateMachine.
            features:   TemporalFeatures for this frame (speed_zone, gaze/head/perclos fields).
            score:      DistractionScore for this frame (active_classes).
            speed_mps:  Raw speed in m/s. Defaults to 0.0 on Mac dev (no OBD-II).
        """
        entry = {
            "event_type": "ALERT",
            "timestamp_ns": command.timestamp_ns,
            "alert_id": command.alert_id,
            "alert_type": command.alert_type.value,       # 'D-A', 'D-D', etc.
            "alert_level": command.level.name,            # 'HIGH' or 'URGENT' (.name not .value)
            "composite_score": round(command.composite_score, 4),
            "active_classes": score.active_classes,       # list[str] already serialisable
            "speed_mps": round(speed_mps, 2),
            "speed_zone": features.speed_zone,
            "speed_source": "NONE",                       # Mac dev: no OBD-II
            "gaze_continuous_secs": round(features.gaze_continuous_secs, 3),
            "head_continuous_secs": round(features.head_continuous_secs, 3),
            "perclos": round(features.perclos, 4),
            "phone_confidence": round(features.phone_confidence_mean, 4),
            "suppress_until_ns": command.suppress_until_ns,
        }
        self._write(entry)

    def log_state_transition(
        self,
        prev: str,
        new: str,
        trigger: str,
        frame_id: int,
        ts_ns: int,
    ) -> None:
        """Write STATE_TRANSITION entry per PRD §9."""
        entry = {
            "event_type": "STATE_TRANSITION",
            "timestamp_ns": ts_ns,
            "previous_state": prev,
            "new_state": new,
            "trigger": trigger,
            "frame_id": frame_id,
        }
        self._write(entry)

    def log_degraded(
        self,
        reason: str,
        duration_secs: float,
        ts_ns: int,
    ) -> None:
        """Write DEGRADED entry per PRD §9."""
        entry = {
            "event_type": "DEGRADED",
            "timestamp_ns": ts_ns,
            "reason": reason,
            "duration_secs": round(duration_secs, 3),
        }
        self._write(entry)

    def log_watchdog_timeout(
        self,
        last_frame_id: int,
        secs_since: float,
        ts_ns: int,
        recovery_action: str = _RECOVERY_ACTION_DEFAULT,
    ) -> None:
        """Write WATCHDOG_TIMEOUT entry per PRD §9."""
        entry = {
            "event_type": "WATCHDOG_TIMEOUT",
            "timestamp_ns": ts_ns,
            "last_frame_id": last_frame_id,
            "secs_since_last_frame": round(secs_since, 3),
            "recovery_action": recovery_action,
        }
        self._write(entry)

    def log_thermal_warning(
        self,
        cpu_temp: float,
        action_taken: str,
        ts_ns: int,
        inference_ms_mean: float = _INFERENCE_MS_MEAN_DEFAULT,
    ) -> None:
        """Write THERMAL_WARNING entry per PRD §9."""
        entry = {
            "event_type": "THERMAL_WARNING",
            "timestamp_ns": ts_ns,
            "temperature_c": round(cpu_temp, 1),
            "action": action_taken,
            "inference_ms_mean": round(inference_ms_mean, 1),
        }
        self._write(entry)

    def log_calibration_complete(self, event: dict) -> None:
        """Write CALIBRATION_COMPLETE entry per PRD §9.

        Args:
            event: Dict with calibration fields (baseline_ear, neutral_yaw_offset,
                   neutral_pitch_offset, vehicle_vin, frames_collected, etc.).
                   event_type and timestamp_ns are added automatically.
        """
        entry = {
            "event_type": "CALIBRATION_COMPLETE",
            "timestamp_ns": time.monotonic_ns(),
        }
        entry.update(event)
        self._write(entry)

---
phase: 06-video-output-wiring
plan: 04
subsystem: layer6_output
tags: [event-logger, jsonl, rotating-log, prd-compliance]
dependency_graph:
  requires: ["06-01"]
  provides: ["EventLogger"]
  affects: ["layer6_output", "tests/unit/test_event_logger.py"]
tech_stack:
  added: ["logging.handlers.RotatingFileHandler"]
  patterns: ["JSONL structured logging", "TDD RED-GREEN"]
key_files:
  created:
    - layer6_output/event_logger.py
    - tests/unit/test_event_logger.py
  modified:
    - layer6_output/__init__.py
decisions:
  - "AlertLevel.name used for string serialisation (HIGH/URGENT) — AlertLevel.value is int (2/3) which violates PRD §9 schema"
  - "speed_source hardcoded to NONE on Mac dev (no OBD-II)"
  - "log_dir defaults to config.LOG_DIR; accepts override for testability via tmp_path"
  - "Logger instance uniqueness via id(self) prevents handler leaks across test instances"
metrics:
  duration: 8min
  completed_date: "2026-03-31"
  tasks_completed: 1
  files_changed: 3
---

# Phase 06 Plan 04: EventLogger Implementation Summary

## One-liner

EventLogger writes 6 event types as rotating JSONL to attentia_events.jsonl using RotatingFileHandler per PRD §FR-6.3 and §9 schemas.

## What Was Built

Implemented `layer6_output/event_logger.py` — a structured JSONL event log backed by Python's `RotatingFileHandler`.

### EventLogger class

- `__init__(log_dir=None)`: creates log directory, installs RotatingFileHandler at `{log_dir}/attentia_events.jsonl`, sets `propagate=False`
- `log_alert(command, features, score, speed_mps=0.0)`: writes 15-field ALERT entry per PRD §9
- `log_state_transition(prev, new, trigger, frame_id, ts_ns)`: writes 6-field STATE_TRANSITION entry
- `log_degraded(reason, duration_secs, ts_ns)`: writes 4-field DEGRADED entry
- `log_watchdog_timeout(last_frame_id, secs_since, ts_ns, recovery_action=...)`: writes 5-field WATCHDOG_TIMEOUT entry
- `log_thermal_warning(cpu_temp, action_taken, ts_ns, inference_ms_mean=0.0)`: writes 5-field THERMAL_WARNING entry
- `log_calibration_complete(event: dict)`: writes CALIBRATION_COMPLETE entry preserving all caller fields plus event_type and timestamp_ns

### Key correctness details

- `AlertLevel.name` (not `.value`) gives string `"HIGH"`/`"URGENT"` — `.value` is int 2/3
- `AlertType.value` gives string `"D-A"`, `"D-D"` etc. — correct for PRD §9
- `logger.propagate = False` — entries don't bleed to stderr/root logger
- RotatingFileHandler: `maxBytes=config.LOG_MAX_BYTES`, `backupCount=config.LOG_BACKUP_COUNT`
- No magic numbers — all thresholds and paths from `config.py`

## Tests

11 unit tests, all passing:

| Test | Coverage |
|------|----------|
| test_log_alert_writes_jsonl | event_type='ALERT', file created |
| test_log_alert_contains_required_fields | all 15 PRD §9 ALERT fields present |
| test_enum_serialization_uses_string | alert_type='D-A', alert_level='HIGH' |
| test_log_state_transition_writes_jsonl | STATE_TRANSITION with all 5 fields |
| test_log_degraded_writes_jsonl | DEGRADED with reason + duration_secs |
| test_log_watchdog_timeout_writes_jsonl | WATCHDOG_TIMEOUT with last_frame_id + secs_since |
| test_log_thermal_warning_writes_jsonl | THERMAL_WARNING with temperature_c + action |
| test_log_calibration_complete_writes_jsonl | CALIBRATION_COMPLETE preserving caller dict |
| test_logger_does_not_propagate | propagate=False confirmed |
| test_multiple_entries_each_on_own_line | 2 log calls produce 2 valid JSON lines |
| test_rotating_file_handler_configured | backup files created after rotation |

Full test suite: 322 passing, 1 pre-existing failure (test_open_eye_ear_positive — unrelated).

## Commits

| Hash | Message |
|------|---------|
| edc49d4 | feat(06-04): implement EventLogger with 6 JSONL log methods |

## Deviations from Plan

None — plan executed exactly as written.

The test bodies in the plan's `<action>` section were adapted slightly:
- Added `test_rotating_file_handler_configured` and `test_log_thermal_warning_writes_jsonl` from the plan's `<behavior>` section (both were in the plan's test list, just merged into the unified test class)
- All 11 tests specified in `<behavior>` are implemented and passing

## Known Stubs

None — all log methods are fully implemented and write real JSONL output.

## Self-Check: PASSED

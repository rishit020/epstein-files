---
plan: 06-05
phase: 06-video-output-wiring
status: complete
started: 2026-03-31
completed: 2026-03-31
tasks_total: 2
tasks_completed: 2
self_check: PASSED
---

# Plan 06-05 Summary: Pipeline Wiring

## What Was Built

Wired all previously-implemented layers into a runnable `main.py` entry point implementing the T-0/T-1/T-2/T-3 thread architecture from PRD §3.3.

## Key Files

### Created
- `main.py` — 442-line entry point with full thread architecture

### Modified
- `tests/integration/test_pipeline_wiring.py` — replaced 8 stub tests with real integration tests (all GREEN)

## Decisions Made

- `_NoOpPhoneDetector` duck-typed (not inheriting `PhoneDetector`) to avoid loading ONNX model twice in T-1
- `_put_dropping_oldest()` helper implements T-0's non-blocking queue policy (PRD §3.3 queue depth)
- T-2 results stored in shared `dict` keyed by `frame_id`; T-1 pops (not reads) after merge to prevent unbounded growth (Pitfall 6)
- `speed_mps=0.0, speed_stale=True` passed in T-3 for Mac dev (no OBD-II, URBAN fallback per CLAUDE.md)
- State transition trigger uses `alert_cmd.alert_type.value` when alert fired, `'internal'` otherwise

## Thread Start Order

T-3 → T-2 → T-1 → T-0 (consumers ready before source emits — D-05)

## Test Results

8 integration tests: **8 passed, 0 failed**

Tests cover:
- T-0 distributes frames to both q_t1 and q_t2
- T-1/T-2 phone result merge within timeout
- Stale flag set on T-2 timeout
- t2_results dict cleaned up (no memory growth)
- LSTM hidden_state carried forward across frames
- Full T-3 chain: signal → temporal → scoring → alert
- stop_event triggers clean shutdown
- EventLogger.log_alert() JSONL write on alert fire

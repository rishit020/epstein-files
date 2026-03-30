---
phase: 06-video-output-wiring
plan: 01
subsystem: testing
tags: [pytest, tdd, stubs, webcam, audio, logging, pipeline, threading]

# Dependency graph
requires:
  - phase: 05-model-integration
    provides: PerceptionStack, PhoneDetector, PerceptionBundle, PhoneDetectionOutput
  - phase: 04-scoring-alert
    provides: AlertStateMachine, AlertCommand, AlertLevel, AlertType
  - phase: 03-temporal
    provides: TemporalFeatures, TemporalEngine
  - phase: 02-signal-processor
    provides: SignalProcessor, SignalFrame

provides:
  - "TDD stub tests for WebcamSource (tests/unit/test_webcam_source.py)"
  - "TDD stub tests for AudioAlertHandler (tests/unit/test_audio_handler.py)"
  - "TDD stub tests for EventLogger (tests/unit/test_event_logger.py)"
  - "TDD stub tests for pipeline wiring / main.py (tests/integration/test_pipeline_wiring.py)"

affects:
  - 06-02-webcam-source
  - 06-03-audio-handler
  - 06-04-event-logger
  - 06-05-main-pipeline

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "_IMPL_MISSING = False / try: import / except ImportError: _IMPL_MISSING = True guard pattern for test stubs"
    - "pytestmark = pytest.mark.skipif(_IMPL_MISSING, ...) for module-level skip"
    - "pytest.skip() as first statement in each test body for individual test documentation"

key-files:
  created:
    - tests/unit/test_webcam_source.py
    - tests/unit/test_audio_handler.py
    - tests/unit/test_event_logger.py
    - tests/integration/test_pipeline_wiring.py
  modified: []

key-decisions:
  - "Tests use pytestmark skipif at module level: no import errors even when implementation is absent"
  - "Integration test stubs import real dataclasses but mock PerceptionStack and PhoneDetector, matching PRD D-01 architecture"
  - "Each test body starts with pytest.skip() to document stub status and plan reference"

patterns-established:
  - "Pattern 1: TDD skip guard — _IMPL_MISSING + pytestmark allows collection without ImportError when module absent"
  - "Pattern 2: Helper factories (_make_synthetic_frame, _make_mock_perception_stack, etc.) used for consistent mock inputs"
  - "Pattern 3: Test names encode PRD requirement IDs in docstrings (FR-0.x, FR-6.1, NFR-P4, PRD §9)"

requirements-completed:
  - FR-0.1
  - FR-0.2
  - FR-0.3
  - FR-0.4
  - FR-6.1
  - FR-6.3
  - PRD-9
  - PRD-3.3

# Metrics
duration: 4min
completed: 2026-03-30
---

# Phase 6 Plan 01: Video+Output+Wiring Test Stubs Summary

**30 TDD stub tests across 4 files covering WebcamSource (FR-0.x), AudioAlertHandler (FR-6.1/NFR-P4), EventLogger (PRD §9), and T-0/T-1/T-2/T-3 pipeline wiring (PRD §3.3) — all collected and skipped until Plans 02-05 implement the modules**

## Performance

- **Duration:** 4 min
- **Started:** 2026-03-30T13:16:39Z
- **Completed:** 2026-03-30T13:20:18Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments

- Created 8 unit test stubs for WebcamSource covering device init, resolution setting, timeout error, read() RawFrame output, monotonic frame IDs, failure handling, and release()
- Created 5 unit test stubs for AudioAlertHandler covering HIGH/URGENT sound dispatch, fire-and-forget (no .wait/.communicate), 50ms latency warning, and OSError exception handling
- Created 8 unit test stubs for EventLogger covering all 5 JSONL event types (ALERT, STATE_TRANSITION, DEGRADED, WATCHDOG_TIMEOUT, CALIBRATION_COMPLETE), enum serialization, and RotatingFileHandler configuration
- Created 9 integration test stubs for pipeline wiring covering T-0 dual-queue frame distribution, T-1/T-2 merge (fresh and stale), T-3 full chain, stop_event thread termination, and LSTM hidden state persistence across frames

## Task Commits

Each task was committed atomically:

1. **Task 1: Unit test stubs for WebcamSource, AudioAlertHandler, EventLogger** - `e771df9` (test)
2. **Task 2: Integration test stubs for pipeline wiring** - `591295f` (test)

**Plan metadata:** _(docs commit — see final_commit below)_

## Files Created/Modified

- `tests/unit/test_webcam_source.py` — 8 stubs for WebcamSource (FR-0.1–FR-0.4, D-02)
- `tests/unit/test_audio_handler.py` — 5 stubs for AudioAlertHandler (FR-6.1, NFR-P4, D-03)
- `tests/unit/test_event_logger.py` — 8 stubs for EventLogger (FR-6.3, PRD §9, D-04)
- `tests/integration/test_pipeline_wiring.py` — 9 stubs for pipeline wiring (PRD §3.3, D-01, D-05)

## Decisions Made

- Used `_IMPL_MISSING` guard pattern (try/except ImportError at module level + `pytestmark`) so tests are collectable without import errors when implementation modules don't exist yet. This is the same pattern already established in test_perception_stack.py.
- Integration tests import real dataclasses (RawFrame, PerceptionBundle, AlertCommand, etc.) from already-implemented layers to keep type contracts visible in the stubs — only `main.py` is guarded.
- Each test body starts with `pytest.skip("Stub — implementation in Plan XX")` to document which plan will implement the feature and provide a clear failure signal once stubs are converted to real tests.

## Deviations from Plan

None — plan executed exactly as written.

## Issues Encountered

None.

## Known Stubs

All 4 test files are intentionally stub files — this is the purpose of Plan 01 (Wave 0 TDD setup). The stubs will be converted to real tests as Plans 02-05 implement the corresponding modules:

- `test_webcam_source.py` → Plan 02 (WebcamSource)
- `test_audio_handler.py` → Plan 03 (AudioAlertHandler)
- `test_event_logger.py` → Plan 04 (EventLogger)
- `test_pipeline_wiring.py` → Plan 05 (main.py)

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- 30 pytest-discoverable stub tests exist for all Phase 6 modules
- Plans 02, 03, 04, 05 can each implement their module and immediately run the pre-written tests
- pytest --collect-only confirms 0 errors, 30 skipped
- No blockers for Plans 02-05 to begin

---
*Phase: 06-video-output-wiring*
*Completed: 2026-03-30*

## Self-Check: PASSED

- FOUND: tests/unit/test_webcam_source.py
- FOUND: tests/unit/test_audio_handler.py
- FOUND: tests/unit/test_event_logger.py
- FOUND: tests/integration/test_pipeline_wiring.py
- FOUND: .planning/phases/06-video-output-wiring/06-01-SUMMARY.md
- FOUND: commit e771df9 (Task 1)
- FOUND: commit 591295f (Task 2)
- pytest --collect-only: 30 tests collected, 0 errors

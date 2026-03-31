---
phase: 06-video-output-wiring
plan: "03"
subsystem: audio-output
tags: [subprocess, afplay, audio, layer6, alerting, mac]

requires:
  - phase: 05-model-wiring
    provides: AlertCommand dataclass via layer5_alert
  - phase: 06-01
    provides: TDD stub tests for AudioAlertHandler

provides:
  - AudioAlertHandler class with fire-and-forget afplay dispatch
  - layer6_output package exports AudioAlertHandler
  - NFR-P4 latency measurement with >50ms warning

affects:
  - 06-04 (EventLogger plan — same layer6_output package)
  - 06-05 (pipeline wiring — uses AudioAlertHandler)

tech-stack:
  added: []
  patterns:
    - "subprocess.Popen for fire-and-forget audio dispatch (no wait/communicate)"
    - "Module-level logger (_log) patched in tests for warning assertion"

key-files:
  created:
    - layer6_output/audio_handler.py
    - tests/unit/test_audio_handler.py
  modified:
    - layer6_output/__init__.py
    - config.py

key-decisions:
  - "subprocess.Popen used (not subprocess.run) to avoid blocking the caller thread"
  - "Latency threshold of 50ms from config (NFR-P4); warning logged with elapsed_ms and 50ms label"
  - "All exceptions caught with broad except — log error and return silently, never propagate"

patterns-established:
  - "Module-level _log logger patched via @patch('module._log') in tests requiring warning assertions"
  - "fire-and-forget subprocess pattern: Popen, no .wait(), no .communicate()"

requirements-completed: [FR-6.1, FR-6.2, NFR-P4]

duration: 8min
completed: "2026-03-31"
---

# Phase 06 Plan 03: AudioAlertHandler Summary

**afplay fire-and-forget audio dispatcher with NFR-P4 latency measurement — HIGH uses Ping.aiff, URGENT uses Sosumi.aiff**

## Performance

- **Duration:** 8 min
- **Started:** 2026-03-31T15:18:00Z
- **Completed:** 2026-03-31T15:26:03Z
- **Tasks:** 1 (TDD: RED + GREEN)
- **Files modified:** 4

## Accomplishments
- AudioAlertHandler.play() dispatches system sounds via subprocess.Popen (non-blocking)
- HIGH alerts play config.AUDIO_ALERT_SOUND; URGENT alerts play config.AUDIO_ALERT_SOUND_URGENT
- Dispatch latency measured; warning logged with elapsed_ms if exceeds 50ms (NFR-P4)
- All exceptions caught at module boundary — never propagates to caller
- 5 unit tests passing: correct sound selection, fire-and-forget, latency warning, exception handling

## Task Commits

Each task was committed atomically:

1. **TDD RED: Test stubs for AudioAlertHandler** - `7475c91` (test)
2. **TDD GREEN: AudioAlertHandler implementation** - `e350fa5` (feat)

_Note: TDD tasks have two commits — RED (failing tests) then GREEN (implementation)_

## Files Created/Modified
- `layer6_output/audio_handler.py` - AudioAlertHandler class with play() method
- `layer6_output/__init__.py` - Package exports AudioAlertHandler
- `tests/unit/test_audio_handler.py` - 5 unit tests covering all PRD-defined behaviours
- `config.py` - Added AUDIO_ALERT_SOUND and AUDIO_ALERT_SOUND_URGENT constants (missing from worktree branch)

## Decisions Made
- Used subprocess.Popen (not subprocess.run) to achieve fire-and-forget without blocking caller thread
- Module-level `_log = logging.getLogger(__name__)` used; tests patch `layer6_output.audio_handler._log` for warning assertions (not `logging.warning`, which would miss module-level loggers)
- Broad `except Exception` at module boundary per CLAUDE.md error handling rules

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Added AUDIO_ALERT_SOUND constants to worktree config.py**
- **Found during:** Task 1 (GREEN phase — running tests)
- **Issue:** Worktree branch `worktree-agent-ac16efab` was branched before the audio constants were added to config.py on main. `AttributeError: module 'config' has no attribute 'AUDIO_ALERT_SOUND'`
- **Fix:** Added `AUDIO_ALERT_SOUND` and `AUDIO_ALERT_SOUND_URGENT` to worktree's config.py (identical values to main branch)
- **Files modified:** config.py
- **Verification:** All 5 tests pass after adding constants
- **Committed in:** e350fa5 (Task 1 GREEN commit)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Required to make tests runnable. No scope creep — values identical to main branch.

## Issues Encountered
- Stub test used `patch('logging.warning')` but implementation uses `_log.warning` (module-level logger). Fixed test to `patch('layer6_output.audio_handler._log')` for correct patching.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- AudioAlertHandler ready for use in Plan 05 (pipeline wiring)
- layer6_output package now exports AudioAlertHandler for main.py import
- Plan 04 (EventLogger) can proceed independently in parallel

---
*Phase: 06-video-output-wiring*
*Completed: 2026-03-31*

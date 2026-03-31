---
phase: 06-video-output-wiring
plan: 02
subsystem: layer0_video
tags: [webcam, video-source, opencv, tdd]
dependency_graph:
  requires: ["06-01"]
  provides: ["WebcamSource", "SourceUnavailableError"]
  affects: ["layer0_video", "main.py (plan 05)"]
tech_stack:
  added: []
  patterns: ["cv2.VideoCapture wrapper", "TDD red-green cycle", "exception boundary at module edge"]
key_files:
  created:
    - layer0_video/webcam_source.py
    - tests/unit/test_webcam_source.py
  modified:
    - layer0_video/__init__.py
decisions:
  - WebcamSource polls isOpened() in a monotonic loop (not a blocking open) to allow fast-forward mocking in tests
  - Open timeout (5.0s) and poll interval (0.1s) stored as local constants (not config.py) since they are constructor internals, not tunable system parameters
  - frame_id incremented after building RawFrame so first frame always has id=0
key_decisions:
  - Timeout constants (5.0s, 0.1s) kept as local constants per D-02 spec — not added to config.py
metrics:
  duration: 2min
  completed: "2026-03-31"
  tasks_completed: 1
  files_created: 2
  files_modified: 1
---

# Phase 06 Plan 02: WebcamSource Implementation Summary

## One-liner

OpenCV-backed Mac webcam source wrapping cv2.VideoCapture that emits typed RawFrame objects with monotonic frame_id, 5s open timeout, and full exception isolation at module boundary.

## What Was Built

`layer0_video/webcam_source.py` — `WebcamSource` class and `SourceUnavailableError` exception for PRD FR-0.1 through FR-0.4 compliance.

### Key behaviours

- `WebcamSource(device_index=0)` opens the Mac webcam via `cv2.VideoCapture(int)`, sets width/height/fps from `config.py` (no magic numbers), then polls `isOpened()` in a monotonic loop up to 5.0 seconds — raises `SourceUnavailableError` if not ready.
- `read()` returns `RawFrame(source_type='webcam', timestamp_ns=time.monotonic_ns(), frame_id=N, ...)` with monotonically increasing `frame_id` starting at 0. Returns `None` on `cap.read()` failure. Catches all exceptions internally — never propagates.
- `release()` calls `cap.release()` and logs closure. No resource leaks.
- `layer0_video/__init__.py` exports `WebcamSource` and `SourceUnavailableError`.

## TDD Cycle

| Phase | Commit | Result |
|-------|--------|--------|
| RED   | 7ee4f20 | 8 tests collected, ImportError (as expected) |
| GREEN | 4e8333c | 8 tests PASSED, 0 regressions |

## Test Results

```
8 passed in 0.38s
```

All 8 unit tests pass covering: device open, resolution set, timeout/error, RawFrame fields, monotonic frame_id, None-on-failure, release, timestamp_ns.

Pre-existing failure `test_open_eye_ear_positive` (documented in MEMORY.md) is unaffected.

## Deviations from Plan

None — plan executed exactly as written.

## Known Stubs

None — WebcamSource is fully wired to real cv2.VideoCapture. No placeholder data.

## Self-Check

- [x] `layer0_video/webcam_source.py` — created, min_lines >= 60 (actual: 97 lines)
- [x] `layer0_video/__init__.py` — exports WebcamSource and SourceUnavailableError
- [x] `tests/unit/test_webcam_source.py` — 8 tests, all pass
- [x] Commits: 7ee4f20 (test RED), 4e8333c (feat GREEN)

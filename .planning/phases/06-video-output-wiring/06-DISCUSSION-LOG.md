# Phase 6: Video + Output + Full Wiring — Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-03-29
**Phase:** 06-video-output-wiring
**Mode:** auto (--auto flag — all decisions auto-selected with recommended defaults)
**Areas discussed:** Thread Architecture, Webcam Source, Audio Handler, Event Logger, main.py Lifecycle

---

## Thread Architecture

| Option | Description | Selected |
|--------|-------------|----------|
| Single queue to T-1, T-1 passes frame to T-2 | Sequential — simpler but T-2 waits behind T-1 | |
| Two separate queues (T-1 queue + T-2 queue) | Parallel — T-0 pushes to both, true simultaneous execution | ✓ |
| Shared reference via threading.Event | Complex synchronization, race-prone | |

Auto-selected: Two separate queues — matches PRD §3.3 "T-1 and T-2 consume the same RawFrame simultaneously"

### T-1/T-2 Merge Strategy

| Option | Description | Selected |
|--------|-------------|----------|
| threading.Event per frame + timeout | T-1 signals event, T-2 waited up to PHONE_THREAD_TIMEOUT_MS | ✓ |
| frame_id keyed dict + polling | More complex, higher CPU | |
| T-3 waits for both unconditionally | Blocks on T-2 always — violates 5ms requirement | |

Auto-selected: threading.Event per frame + PHONE_THREAD_TIMEOUT_MS timeout

---

## Webcam Source

| Option | Description | Selected |
|--------|-------------|----------|
| cv2.VideoCapture(device_index) | Integer index, Mac-native, simple | ✓ |
| cv2.VideoCapture("/dev/video0") | Linux path — not Mac compatible | |
| AVFoundation direct | Complex, bypasses OpenCV abstractions | |

Auto-selected: cv2.VideoCapture(device_index) with integer index

### Init Timeout Strategy

| Option | Description | Selected |
|--------|-------------|----------|
| Monotonic loop up to 5.0s | Poll isOpened(), raise SourceUnavailableError on timeout | ✓ |
| Single call, immediate fail | Doesn't meet PRD §FR-0.3 5s timeout requirement | |
| Retry indefinitely | Hangs — violates FR-0.3 | |

Auto-selected: Monotonic loop up to 5.0s per PRD §FR-0.3

---

## Audio Handler

| Option | Description | Selected |
|--------|-------------|----------|
| subprocess.Popen (fire and forget) | Non-blocking, ≤50ms dispatch latency achievable | ✓ |
| subprocess.run (blocking) | Blocks T-3 thread for duration of audio — violates NFR-P4 | |
| Background audio thread | Adds complexity, not needed with Popen | |

Auto-selected: subprocess.Popen — required for ≤50ms dispatch per PRD §FR-6.1, NFR-P4

---

## Event Logger

| Option | Description | Selected |
|--------|-------------|----------|
| Python RotatingFileHandler + JSON | Thread-safe, stdlib, 50MB rotation built-in | ✓ |
| Custom rotation logic | Reinvents wheel, more error-prone | |
| Write to SQLite | Overkill for this use case | |

Auto-selected: Python RotatingFileHandler — simplest correct approach matching PRD §FR-6.3

### Log format

| Option | Description | Selected |
|--------|-------------|----------|
| JSONL (newline-delimited, one obj/line) | Easy to stream/tail/parse | ✓ |
| Pretty JSON array | Requires full file parse | |
| CSV | Loses schema flexibility | |

Auto-selected: JSONL — one JSON object per line

---

## main.py Lifecycle

| Option | Description | Selected |
|--------|-------------|----------|
| threading.Event stop_flag + signal handler | Clean shutdown, standard Python pattern | ✓ |
| KeyboardInterrupt only | Misses SIGTERM (Docker/systemd), unclean | |
| multiprocessing | Overkill for GIL-limited ONNX inference | |

Auto-selected: threading.Event + SIGINT/SIGTERM signal handlers

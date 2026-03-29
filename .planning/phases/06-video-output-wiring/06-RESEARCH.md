# Phase 6: Video + Output + Full Wiring - Research

**Researched:** 2026-03-29
**Domain:** Python threading, OpenCV VideoCapture, subprocess.Popen, RotatingFileHandler, multi-thread pipeline wiring
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**D-01: Thread Architecture (PRD §3.3)**
- T-0 (VideoCapture): `threading.Thread` daemon, reads from OpenCV `VideoCapture`, puts `RawFrame` into `queue.Queue(maxsize=config.FRAME_QUEUE_DEPTH)`. Drop oldest on overflow (get() + put() pattern).
- T-1 (FacePerception): High-priority thread. Gets `RawFrame` from queue, runs `PerceptionStack.infer()`, puts partial `PerceptionBundle` (face + landmarks + gaze, phone=None/stale) into a result dict keyed by `frame_id`.
- T-2 (PhoneDetection): Normal-priority thread. Gets same `RawFrame` reference (shared from T-0 via a second queue or broadcast), runs `PhoneDetector.infer()`, puts `PhoneDetectionOutput` into a result dict keyed by `frame_id`.
- T-1/T-2 merge: T-1 sets a `threading.Event` when it completes. T-2 result is waited up to `config.PHONE_THREAD_TIMEOUT_MS / 1000.0` seconds. If T-2 times out, use last valid `PhoneDetectionOutput` with `phone_result_stale=True`. Stale count logged.
- T-3 (Pipeline): Normal-priority thread. Consumes merged `PerceptionBundle`, feeds it sequentially through `SignalProcessor` -> `TemporalEngine` -> `ScoringEngine` -> `AlertStateMachine`. Dispatches `AlertCommand` to `AudioHandler` and `EventLogger`.
- Frame distribution: T-0 puts `RawFrame` into two separate queues — one for T-1, one for T-2. Both queues have depth `FRAME_QUEUE_DEPTH`. This avoids any locking overhead and matches PRD "shared reference" intent.

**D-02: Webcam Source (PRD §FR-0.1–FR-0.4)**
- Class: `WebcamSource` in `layer0_video/webcam_source.py`
- Constructor: accepts `device_index: int = 0`
- Init timeout: poll `cap.isOpened()` in a monotonic loop up to 5.0 seconds; raise `SourceUnavailableError` if not ready
- Emits: `RawFrame(source_type='webcam', ...)` with `timestamp_ns=time.monotonic_ns()`, monotonically increasing `frame_id`
- `release()` method: calls `cap.release()`, logs closure
- Resolution: set `CAP_PROP_FRAME_WIDTH=config.CAPTURE_WIDTH`, `CAP_PROP_FRAME_HEIGHT=config.CAPTURE_HEIGHT`, `CAP_PROP_FPS=config.CAPTURE_FPS`
- Exception handling: all `cap.read()` failures caught, logged, returns None

**D-03: Audio Handler (PRD §FR-6.1)**
- Class: `AudioAlertHandler` in `layer6_output/audio_handler.py`
- Mac implementation: `subprocess.Popen(['afplay', sound_file])` — fire and forget, do NOT join/wait
- Dispatch latency must be ≤ 50ms from `AlertCommand` receipt to audio start (PRD NFR-P4)
- Sound file: `config.AUDIO_ALERT_SOUND` (default: `/System/Library/Sounds/Ping.aiff`); URGENT uses `config.AUDIO_ALERT_SOUND_URGENT` (default: `/System/Library/Sounds/Sosumi.aiff`)
- `play(command: AlertCommand)` is the public method
- Timestamp dispatch time for latency measurement; log if >50ms

**D-04: Event Logger (PRD §FR-6.3, §9)**
- Class: `EventLogger` in `layer6_output/event_logger.py`
- Uses Python `logging.handlers.RotatingFileHandler` with `maxBytes=config.LOG_MAX_BYTES` and `backupCount=config.LOG_BACKUP_COUNT`
- Log directory: `config.LOG_DIR` — create if not exists
- Log filename: `attentia_events.jsonl`
- Thread-safe: Python's `logging` module is thread-safe by default
- Public methods: `log_alert`, `log_state_transition`, `log_degraded`, `log_watchdog_timeout`, `log_calibration_complete`

**D-05: main.py Lifecycle**
- Instantiate all components, wire queues, start threads in order T-3 -> T-2 -> T-1 -> T-0
- `stop_event = threading.Event()` shared across all threads
- Signal handler: `signal.signal(SIGINT, ...)` and `signal.signal(SIGTERM, ...)` -> sets `stop_event`
- Optional `--display` cv2.imshow preview (off by default)
- `TemporalEngine.start()` / `.stop()` for watchdog + thermal threads

### Claude's Discretion
- Exact thread priority setting: Python doesn't expose OS thread priority cleanly on Mac; use default priority for all threads
- cv2.imshow display format: if enabled, draw face bbox + phone indicator overlay
- Logging verbosity at startup: log model paths, device index, config summary

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope.
</user_constraints>

---

## Summary

Phase 6 wires all previously-built layers into a single runnable process with real Mac webcam input. The four new files are: `layer0_video/webcam_source.py` (OpenCV camera source), `layer6_output/audio_handler.py` (afplay-backed audio), `layer6_output/event_logger.py` (rotating JSONL log), and `main.py` (thread orchestrator). All layer interfaces are already defined and fully implemented — this phase is primarily assembly work, not new algorithm development.

All critical dependencies are verified present on this machine: Python 3.10.14, OpenCV 4.10.0 with AVFoundation backend, onnxruntime 1.19.0, pytest 8.0.2, afplay at `/usr/bin/afplay`, and both default sound files (`Ping.aiff`, `Sosumi.aiff`). OpenCV `VideoCapture(0)` successfully opens the webcam and honors `cap.set()` for 1280x720@30fps. `subprocess.Popen(['afplay', ...])` launches with ~2ms latency — well within the 50ms NFR-P4 requirement. Python `RotatingFileHandler` produces valid JSONL output (one JSON object per line).

The most technically complex part of this phase is the T-1/T-2 merge logic. The design uses a `threading.Event` per frame and a shared result dict keyed by `frame_id`. This was verified to work correctly with a 5ms timeout. The LSTM hidden state must be carried forward from `bundle.lstm_hidden_state` to the next `PerceptionStack.infer(hidden_state=...)` call — this is a T-1-only concern because `PerceptionStack` manages both face and gaze models.

**Primary recommendation:** Build the four files in order — webcam source (simplest, self-contained), event logger (no threading complexity), audio handler (trivial), then main.py (hardest: thread lifecycle + merge logic). Write tests in parallel with each module.

---

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| opencv-python | 4.10.0 (installed) | VideoCapture, frame read | Already used in Phase 5; AVFoundation backend on Mac |
| onnxruntime | 1.19.0 (installed) | Model inference | Already used in all perception models |
| Python stdlib: `queue` | 3.10.14 | Thread-safe frame queues | No external dependency; Queue(maxsize=N) is the right primitive |
| Python stdlib: `threading` | 3.10.14 | Thread, Event, stop_event | No external dependency |
| Python stdlib: `subprocess` | 3.10.14 | Popen for afplay | Fire-and-forget audio launch |
| Python stdlib: `logging.handlers` | 3.10.14 | RotatingFileHandler for JSONL | Thread-safe; built-in rotation support |
| Python stdlib: `signal` | 3.10.14 | SIGINT/SIGTERM handling in main.py | Standard pattern for clean shutdown |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `json` (stdlib) | 3.10.14 | Serialize JSONL log entries | All EventLogger methods |
| `time` (stdlib) | 3.10.14 | `time.monotonic_ns()` for RawFrame timestamps | WebcamSource |
| `argparse` (stdlib) | 3.10.14 | CLI args in main.py (`--device`, `--no-display`) | main.py only |
| pytest | 8.0.2 (installed) | Test framework | Unit + integration tests |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `subprocess.Popen` for afplay | `playsound`, `pygame.mixer` | Popen is simpler, no external dep, fire-and-forget natively |
| `RotatingFileHandler` | `structlog`, custom file writer | stdlib handler is sufficient; no extra dep needed |
| `queue.Queue` per thread | `multiprocessing.Queue` | Threads share memory — Queue is correct; MP adds process overhead |
| Two separate queues T-0→T-1/T-2 | Broadcast queue or shared queue | Two queues avoids locking overhead; each consumer pops independently |

**Installation:** All required libraries are already installed. No `pip install` needed for Phase 6.

---

## Architecture Patterns

### Recommended Project Structure

```
layer0_video/
├── __init__.py     # exists
├── messages.py     # exists (RawFrame)
└── webcam_source.py  # NEW: WebcamSource class

layer6_output/
├── __init__.py     # exists (empty)
├── audio_handler.py  # NEW: AudioAlertHandler
└── event_logger.py   # NEW: EventLogger

main.py             # NEW: wires everything together

tests/
├── unit/
│   ├── test_webcam_source.py     # NEW
│   ├── test_audio_handler.py     # NEW
│   └── test_event_logger.py      # NEW
└── integration/
    └── test_pipeline_wiring.py   # NEW
```

### Pattern 1: Queue Drop-Oldest (T-0 overflow handling)

**What:** When a queue is full (consumer slower than producer), drop the oldest frame rather than blocking T-0.
**When to use:** T-0 puts into both `q_t1` and `q_t2`.

```python
# Source: verified on Python 3.10.14
def _put_dropping_oldest(q: queue.Queue, item) -> None:
    """Put item into queue; if full, drop oldest to make room."""
    try:
        q.put_nowait(item)
    except queue.Full:
        try:
            q.get_nowait()  # discard oldest
        except queue.Empty:
            pass
        try:
            q.put_nowait(item)
        except queue.Full:
            pass  # still full after discard — log and skip
```

### Pattern 2: T-1/T-2 Merge with threading.Event and Timeout

**What:** T-1 runs PerceptionStack.infer(), signals Event; T-3 waits for both T-1 result and T-2 result.
**When to use:** Frame-level synchronization with 5ms (PHONE_THREAD_TIMEOUT_MS) deadline.

The merge architecture uses a result dict per-frame keyed by `frame_id`. T-3 does NOT wait inline — T-1 produces the merged bundle and puts it into a T-3 queue.

```python
# Source: verified on Python 3.10.14 — threading.Event.wait() resolution
# Architecture: T-1 owns the merge, T-3 consumes finished bundles

# T-1 thread body (simplified):
def _t1_loop(self):
    while not stop_event.is_set():
        raw_frame = q_t1.get(timeout=0.1)  # blocks
        if raw_frame is None:
            break
        # signal T-2 to start (T-2 dequeues from q_t2 independently)
        bundle = perception_stack.infer(
            raw_frame.data, raw_frame.frame_id, lstm_hidden_state
        )
        # update LSTM state from bundle
        lstm_hidden_state = bundle.lstm_hidden_state

        # wait up to PHONE_THREAD_TIMEOUT_MS for T-2 phone result
        phone_result = _wait_for_t2_result(
            raw_frame.frame_id,
            timeout_s=config.PHONE_THREAD_TIMEOUT_MS / 1000.0
        )
        # merge phone result into bundle
        bundle.phone = phone_result.phone if phone_result else last_valid_phone
        bundle.phone_result_stale = (phone_result is None)
        q_t3.put(bundle)

# T-2 thread body (simplified):
def _t2_loop(self):
    while not stop_event.is_set():
        raw_frame = q_t2.get(timeout=0.1)
        phone_out = phone_detector.infer(raw_frame.data)
        # store result keyed by frame_id for T-1 to consume
        with t2_result_lock:
            t2_results[raw_frame.frame_id] = phone_out
```

**CRITICAL NOTE:** `PerceptionStack.infer()` already includes phone detection internally — it calls `self._phone.infer(frame)` in `_run()`. In the threaded architecture, T-1 runs `PerceptionStack` WITHOUT the phone detector (or with a dummy phone detector) and T-2 runs `PhoneDetector` independently. The `PhoneDetector` instance in `PerceptionStack` must be decoupled for this to work. See Anti-Patterns section.

### Pattern 3: LSTM Hidden State Threading in T-1

**What:** `lstm_hidden_state` is a per-session stateful variable that must persist across frames in T-1.
**When to use:** T-1 thread owns this state — it is NOT passed through the queue or shared with T-2/T-3.

```python
# T-1 local state — initialized once per session
_lstm_hidden_state: tuple | None = None

# Each frame:
bundle = perception_stack.infer(
    frame=raw_frame.data,
    frame_id=raw_frame.frame_id,
    hidden_state=_lstm_hidden_state,
)
_lstm_hidden_state = bundle.lstm_hidden_state  # carry forward
```

If `bundle.lstm_reset_occurred` is True, `bundle.lstm_hidden_state` is already None — no special handling needed, just carry it forward.

### Pattern 4: RotatingFileHandler for JSONL

**What:** Python logging.handlers.RotatingFileHandler writes JSONL without timestamps/level prefix.
**When to use:** EventLogger wraps this pattern.

```python
# Source: verified on Python 3.10.14 stdlib
import logging
import logging.handlers
import json, os

class EventLogger:
    def __init__(self) -> None:
        os.makedirs(config.LOG_DIR, exist_ok=True)
        log_path = os.path.join(config.LOG_DIR, 'attentia_events.jsonl')
        handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=config.LOG_MAX_BYTES,       # 52_428_800 (50MB)
            backupCount=config.LOG_BACKUP_COUNT, # 5
        )
        # Use a formatter that outputs ONLY the message (no level/timestamp prefix)
        handler.setFormatter(logging.Formatter('%(message)s'))
        self._logger = logging.getLogger('attentia.events')
        self._logger.addHandler(handler)
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False  # prevent double-logging to root

    def _write(self, entry: dict) -> None:
        self._logger.info(json.dumps(entry))
```

**Verified output format:** `'{"event_type": "ALERT", "alert_id": "abc"}\n'` — one JSON object per line, no prefix.

### Pattern 5: afplay Fire-and-Forget with Latency Measurement

**What:** Launch afplay without waiting, measure launch latency against 50ms NFR.
**When to use:** `AudioAlertHandler.play(command: AlertCommand)`.

```python
# Source: verified on Python 3.10.14, afplay 2.0
import subprocess, time, logging

def play(self, command: AlertCommand) -> None:
    sound_file = (
        config.AUDIO_ALERT_SOUND_URGENT
        if command.level == AlertLevel.URGENT
        else config.AUDIO_ALERT_SOUND
    )
    t0 = time.perf_counter()
    try:
        subprocess.Popen(['afplay', sound_file])
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if elapsed_ms > 50.0:
            _log.warning('Audio dispatch latency %.1fms exceeds 50ms NFR-P4', elapsed_ms)
    except Exception as exc:
        _log.error('AudioAlertHandler.play() failed: %s', exc)
```

**Verified latency:** ~1.88ms on this Mac — well under the 50ms limit.

### Pattern 6: Clean Thread Shutdown

**What:** All threads check `stop_event.is_set()` in their loop; T-0 stops first so queues drain.
**When to use:** main.py signal handler + join sequence.

```python
# main.py shutdown sequence
def _shutdown(signum, frame):
    stop_event.set()

# Thread join order: T-0 first, then T-1/T-2/T-3
# T-0 stops producing → queues drain → T-1/T-2/T-3 exit naturally
# Queue.get(timeout=0.1) allows checking stop_event regularly
```

### Pattern 7: Integration Test with Mock Source

**What:** Test pipeline wiring without real webcam by injecting synthetic frames.
**When to use:** `tests/integration/test_pipeline_wiring.py`.

```python
# Source: pattern verified in existing tests (test_perception_stack.py)
import numpy as np, queue, threading

def _make_synthetic_frame(frame_id: int) -> RawFrame:
    return RawFrame(
        timestamp_ns=frame_id * 33_333_333,  # 30fps spacing
        frame_id=frame_id,
        width=640, height=480, channels=3,
        data=np.zeros((480, 640, 3), dtype=np.uint8),
        source_type='webcam',
    )
```

### Anti-Patterns to Avoid

- **PerceptionStack.infer() for phone in T-1:** The current `PerceptionStack._run()` calls `self._phone.infer(frame)` on every frame. In the threaded design, T-1 must NOT run the phone detector — that would defeat T-2's purpose. Solution: either pass a `None`/no-op phone detector to the `PerceptionStack` used by T-1, OR have T-1 bypass phone detection and let T-2 supply the phone result before merging into the bundle. The `PerceptionStack` constructor takes a `phone_detector: PhoneDetector` argument — pass a stub that always returns `_safe_phone()`.

- **Joining afplay process:** `subprocess.Popen` returns immediately; calling `.wait()` or `.join()` blocks T-3 for the duration of audio playback (~0.5–2s). Never join the afplay process.

- **subprocess.run for afplay:** `subprocess.run()` blocks until the command completes. Use `subprocess.Popen()` only.

- **json.dumps with Enum values:** `AlertLevel` and `AlertType` are enums. `json.dumps({'level': command.level})` raises `TypeError`. Use `.value`: `command.level.value`, `command.alert_type.value`.

- **Logger propagation to root:** If `self._logger.propagate = True` (default), JSONL entries also appear in the root logger's handlers (console stderr). Always set `propagate = False` on the events logger.

- **queue.Queue.get() without timeout:** A blocking `q.get()` with no timeout prevents the thread from checking `stop_event.is_set()`. Use `q.get(timeout=0.1)` wrapped in a `try/except queue.Empty` loop.

- **Sharing a single queue for T-1 and T-2:** If T-1 and T-2 compete on the same queue, one thread will consume the frame the other needs. Always use two separate queues.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Rotating log files | Custom file size checker + rename logic | `logging.handlers.RotatingFileHandler` | Handles rotation atomically, thread-safe, handles concurrent writes |
| Thread-safe write to log | Manual lock around file.write() | Python `logging` module | Internal GIL + handler locks; already thread-safe |
| Audio playback timing | Sleep-based retry or audio library | `subprocess.Popen(['afplay', ...])` | Fire-and-forget; no blocking; verified ~2ms launch latency |
| Frame timestamping | `time.time_ns()` (wall clock) | `time.monotonic_ns()` | Monotonic never goes backward; wall clock can jump on NTP sync |
| Directory creation on startup | Recursive mkdir with checks | `os.makedirs(path, exist_ok=True)` | Atomic; handles race conditions; `exist_ok` avoids exception if already exists |

**Key insight:** The output layer (Layer 6) is almost entirely stdlib plumbing. Complexity lies in the thread architecture of main.py, not in audio or logging implementation.

---

## Layer Interface Reference

All interfaces are implemented and stable. This section documents the exact signatures T-3 must call.

### PerceptionStack (T-1 consumer)

```python
# layer1_perception/perception_stack.py
class PerceptionStack:
    def __init__(self,
        face_detector: FaceDetector,
        landmark_model: LandmarkModel,
        gaze_model: GazeModel,
        phone_detector: PhoneDetector,  # use stub in T-1 to disable phone
    ) -> None: ...

    def infer(
        self,
        frame: np.ndarray,          # BGR (H, W, 3) uint8
        frame_id: int,
        hidden_state: tuple | None = None,
    ) -> PerceptionBundle: ...
```

**PerceptionBundle output fields used by T-1/T-3:**
- `bundle.lstm_hidden_state` — carry forward to next `infer()` call in T-1
- `bundle.lstm_reset_occurred` — informational only; `lstm_hidden_state` is already None if reset
- `bundle.phone` — will be overwritten with T-2 result during merge
- `bundle.phone_result_stale` — set to True if T-2 timed out

### PhoneDetector (T-2 consumer)

```python
# layer1_perception/phone_detector.py
class PhoneDetector:
    def __init__(self, model_path: str | None = None) -> None: ...
    def infer(self, frame: np.ndarray) -> PhoneDetectionOutput: ...
    # frame: BGR (H, W, 3) uint8 — full frame (not cropped)
```

### SignalProcessor (T-3 step 1)

```python
# layer2_signals/signal_processor.py
class SignalProcessor:
    def __init__(self) -> None: ...

    # Optional startup: load persisted calibration
    def set_neutral_pose(self, yaw_offset: float, pitch_offset: float) -> None: ...
    def set_ear_baseline(self, baseline_ear: float, close_threshold: float) -> None: ...

    def process(
        self,
        bundle: PerceptionBundle,
        speed_mps: float = 0.0,
        speed_stale: bool = False,
    ) -> SignalFrame: ...
```

**NOTE:** `SignalProcessor.process()` takes `speed_mps` and `speed_stale` as keyword args. On Mac dev, speed source defaults to URBAN fallback — pass `speed_mps=0.0, speed_stale=True` unless SpeedSource provides a value.

### TemporalEngine (T-3 step 2)

```python
# layer3_temporal/temporal_engine.py
class TemporalEngine:
    def __init__(self) -> None: ...
    def start(self) -> None: ...   # MUST call before first process() — starts watchdog + thermal threads
    def stop(self) -> None: ...    # MUST call on shutdown
    def process(self, frame: SignalFrame) -> TemporalFeatures: ...
```

### ScoringEngine (T-3 step 3)

```python
# layer4_scoring/scoring_engine.py
class ScoringEngine:
    def __init__(self, weights: FeatureWeights = DEFAULT_WEIGHTS) -> None: ...
    def score(self, features: TemporalFeatures) -> DistractionScore: ...
    # Stateless — no start/stop needed
```

### AlertStateMachine (T-3 step 4)

```python
# layer5_alert/alert_state_machine.py
class AlertStateMachine:
    def __init__(self) -> None: ...
    def process(self, score: DistractionScore) -> Optional[AlertCommand]: ...
    @property
    def state(self) -> str: ...   # 'NOMINAL' | 'PRE_ALERT' | 'ALERTING' | 'COOLDOWN' | 'DEGRADED'
```

### AlertCommand fields (for EventLogger and AudioHandler)

```python
# layer5_alert/messages.py
@dataclass
class AlertCommand:
    alert_id: str           # UUID string
    timestamp_ns: int
    level: AlertLevel       # AlertLevel.HIGH or AlertLevel.URGENT — use .value for JSON
    alert_type: AlertType   # e.g. AlertType.PHONE_USE — use .value for JSON ('D-D')
    composite_score: float
    suppress_until_ns: int
```

---

## PRD §9 Event Log Schemas (Exact)

All six event types and their required JSON fields. The EventLogger must produce exactly these structures (no additional fields required, but missing fields are a bug).

### ALERT

```json
{
    "event_type": "ALERT",
    "timestamp_ns": 1740000000000000000,
    "alert_id": "uuid-v4-string",
    "alert_type": "D-A",
    "alert_level": "HIGH",
    "composite_score": 0.67,
    "active_classes": ["D-A", "D-B"],
    "speed_mps": 14.2,
    "speed_zone": "HIGHWAY",
    "speed_source": "OBD2",
    "gaze_continuous_secs": 2.1,
    "head_continuous_secs": 1.6,
    "perclos": 0.08,
    "phone_confidence": 0.0,
    "suppress_until_ns": 1740000008000000000
}
```

`log_alert(command: AlertCommand, features: TemporalFeatures)` sources:
- `alert_id`, `alert_type.value`, `level.value`, `composite_score`, `suppress_until_ns` → from `AlertCommand`
- `active_classes` → from `DistractionScore.active_classes` (pass through via TemporalFeatures or keep as param)
- `speed_mps`, `speed_zone` → from `TemporalFeatures`
- `speed_source` → hardcode `'NONE'` on Mac dev (no OBD-II)
- `gaze_continuous_secs`, `head_continuous_secs`, `perclos`, `phone_confidence` → from `TemporalFeatures` (use `phone_confidence_mean`)

**DESIGN NOTE:** `log_alert` needs `active_classes` which is on `DistractionScore`, not `TemporalFeatures`. Either pass `DistractionScore` as a third arg, or accept `active_classes: list[str]` as a separate param.

### STATE_TRANSITION

```json
{
    "event_type": "STATE_TRANSITION",
    "timestamp_ns": 1740000000000000000,
    "previous_state": "NOMINAL",
    "new_state": "ALERTING",
    "trigger": "ALT-01",
    "frame_id": 4521
}
```

`log_state_transition(prev: str, new: str, trigger: str, frame_id: int, ts_ns: int)` — all params explicit.

### DEGRADED

```json
{
    "event_type": "DEGRADED",
    "timestamp_ns": 1740000000000000000,
    "reason": "perception_invalid_60_frames",
    "duration_secs": 2.0
}
```

`log_degraded(reason: str, duration_secs: float, ts_ns: int)`.

### THERMAL_WARNING

```json
{
    "event_type": "THERMAL_WARNING",
    "timestamp_ns": 1740000000000000000,
    "temperature_c": 82.0,
    "action": "reduced_yolo_resolution_to_256",
    "inference_ms_mean": 95.3
}
```

This is emitted by `ThermalMonitor` (already implemented in Phase 3). EventLogger provides `log_thermal_warning` or this is called directly from the thermal monitor. Check if ThermalMonitor already logs internally — it likely does not write to EventLogger yet.

### WATCHDOG_TIMEOUT

```json
{
    "event_type": "WATCHDOG_TIMEOUT",
    "timestamp_ns": 1740000000000000000,
    "last_frame_id": 10531,
    "secs_since_last_frame": 2.14,
    "recovery_action": "thread_restart_attempted"
}
```

`log_watchdog_timeout(last_frame_id: int, secs_since: float, ts_ns: int)`.

### CALIBRATION_COMPLETE

```json
{
    "event_type": "CALIBRATION_COMPLETE",
    "timestamp_ns": 1740000000000000000,
    "baseline_ear": 0.31,
    "neutral_yaw_offset": -4.2,
    "neutral_pitch_offset": 3.1,
    "vehicle_vin": "1HGCM82633A004352",
    "frames_collected": 298
}
```

`log_calibration_complete(event: dict)` — receives the dict and writes it directly after adding `event_type` and `timestamp_ns`.

---

## Config Values for Phase 6

All required config constants are confirmed present in `config.py`:

| Constant | Value | Used By |
|----------|-------|---------|
| `CAPTURE_WIDTH` | 1280 | WebcamSource cap.set() |
| `CAPTURE_HEIGHT` | 720 | WebcamSource cap.set() |
| `CAPTURE_FPS` | 30 | WebcamSource cap.set() |
| `FRAME_QUEUE_DEPTH` | 2 | Both T-0→T-1 and T-0→T-2 queues |
| `PHONE_THREAD_TIMEOUT_MS` | 5 | T-1/T-2 merge wait timeout |
| `LOG_DIR` | `'logs/'` | EventLogger directory |
| `LOG_MAX_BYTES` | 52_428_800 (50MB) | RotatingFileHandler maxBytes |
| `LOG_BACKUP_COUNT` | 5 | RotatingFileHandler backupCount |
| `AUDIO_ALERT_SOUND` | `'/System/Library/Sounds/Ping.aiff'` | AudioAlertHandler HIGH alerts |
| `AUDIO_ALERT_SOUND_URGENT` | `'/System/Library/Sounds/Sosumi.aiff'` | AudioAlertHandler URGENT alerts |

Both audio files are confirmed to exist on this Mac.

**No new config constants needed for Phase 6** — all were pre-populated in config.py.

---

## Common Pitfalls

### Pitfall 1: PerceptionStack Runs Phone Detector Internally

**What goes wrong:** T-1 instantiates a full `PerceptionStack` (including `PhoneDetector`), runs it, and the phone result in `bundle.phone` is overwritten before the T-2 result is merged. The T-2 result is then discarded because the bundle already has a phone result.
**Why it happens:** `PerceptionStack._run()` calls `self._phone.infer(frame)` unconditionally.
**How to avoid:** Pass a no-op/stub `PhoneDetector` to the `PerceptionStack` used by T-1. The stub's `infer()` returns `_safe_phone()` (detected=False, max_confidence=0.0, bbox_norm=None) immediately. T-1 then replaces `bundle.phone` with the T-2 result during merge.
**Warning signs:** Phone detection results are always `detected=False` or always fresh (never stale), even when T-2 should be timing out.

### Pitfall 2: Enum Serialization Failure in EventLogger

**What goes wrong:** `json.dumps({'level': command.level})` raises `TypeError: Object of type AlertLevel is not JSON serializable`.
**Why it happens:** `AlertLevel` and `AlertType` are Python Enums, not raw strings/ints.
**How to avoid:** Always use `.value`: `command.level.value` (returns `'HIGH'` or `'URGENT'`), `command.alert_type.value` (returns `'D-A'`, `'D-D'`, etc.).
**Warning signs:** `TypeError` in EventLogger._write() during first live alert.

### Pitfall 3: T-3 Queue Starvation on T-1 Slowness

**What goes wrong:** T-1 inference is slow (~80-200ms per frame), causing `q_t3` to be empty most of the time and T-3 to spin-wait. Meanwhile T-0 drops frames because `q_t1` fills faster than T-1 consumes.
**Why it happens:** The ONNX inference on Mac can be 60-100ms per frame for the full face pipeline.
**How to avoid:** This is expected behavior — `FRAME_QUEUE_DEPTH=2` is intentionally shallow to prefer freshness over completeness. T-0 drops oldest to keep latency low. Accept frame drops. Log dropped frame count for diagnostics.
**Warning signs:** High drop rate (>50%) at 30fps input. This is not a bug — it means models are running at effective <15fps.

### Pitfall 4: cv2.imshow Not on Main Thread on Mac

**What goes wrong:** Calling `cv2.imshow()` from a non-main thread on macOS causes a crash or no-op.
**Why it happens:** macOS requires GUI operations on the main thread (Cocoa framework constraint).
**How to avoid:** If `--display` is enabled, run the imshow loop on the main thread. Pass frames to the main thread via a separate display queue. Alternatively, simply disable display by default (already the decision in D-05).
**Warning signs:** Crash with `NSInternalInconsistencyException` or silent no-op imshow.

### Pitfall 5: RotatingFileHandler Logger Propagation

**What goes wrong:** Every JSONL entry also appears on stderr (root logger), creating noise.
**Why it happens:** Python loggers propagate to parent by default. The root logger may have a StreamHandler.
**How to avoid:** Set `self._logger.propagate = False` on the events logger instance.
**Warning signs:** JSONL entries appearing in console during tests.

### Pitfall 6: T-2 Result Dict Memory Leak

**What goes wrong:** `t2_results` dict accumulates entries for every frame_id indefinitely, causing unbounded memory growth.
**Why it happens:** T-1 adds to dict for each frame; if cleanup isn't done, dict grows at 30fps.
**How to avoid:** After T-1 reads (or times out on) the T-2 result for a given `frame_id`, delete the dict entry. Use a `threading.Lock` when accessing the dict. Alternatively, use a bounded structure like `collections.deque` or just keep the single "last result" value.
**Warning signs:** Memory growth in NFR-R2 soak test.

### Pitfall 7: main.py thread start order

**What goes wrong:** T-0 starts before T-3 is ready to consume. First frames are dropped into a queue that nobody processes yet.
**Why it happens:** T-3 starts after T-0 if order is wrong.
**How to avoid:** Start threads in reverse order: T-3 first, then T-2, T-1, T-0 last (as specified in D-05). T-0 starting last ensures consumers are ready before the source begins emitting.
**Warning signs:** Initial frames always lost; T-3 queue is momentarily empty for first N frames.

### Pitfall 8: queue.Empty exception on stop

**What goes wrong:** When `stop_event` is set, threads calling `q.get(timeout=0.1)` may raise `queue.Empty` after timeout. Code that doesn't catch this in the loop body raises an unhandled exception.
**Why it happens:** `queue.Queue.get(timeout=N)` raises `queue.Empty` if no item arrives in time.
**How to avoid:** Wrap queue.get() in `try/except queue.Empty: continue` inside the `while not stop_event.is_set()` loop.

---

## Code Examples

### WebcamSource skeleton

```python
# Source: verified pattern — cv2.VideoCapture with AVFoundation on Mac
import cv2, time, logging
import config
from layer0_video.messages import RawFrame

class SourceUnavailableError(RuntimeError):
    pass

class WebcamSource:
    def __init__(self, device_index: int = 0) -> None:
        self._cap = cv2.VideoCapture(device_index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAPTURE_WIDTH)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAPTURE_HEIGHT)
        self._cap.set(cv2.CAP_PROP_FPS, config.CAPTURE_FPS)
        deadline = time.monotonic() + 5.0
        while not self._cap.isOpened():
            if time.monotonic() > deadline:
                raise SourceUnavailableError(f"Camera {device_index} not available after 5s")
            time.sleep(0.05)
        self._frame_id: int = 0

    def read(self) -> RawFrame | None:
        try:
            ret, frame = self._cap.read()
            if not ret or frame is None:
                return None
            self._frame_id += 1
            return RawFrame(
                timestamp_ns=time.monotonic_ns(),
                frame_id=self._frame_id,
                width=frame.shape[1],
                height=frame.shape[0],
                channels=3,
                data=frame,
                source_type='webcam',
            )
        except Exception as exc:
            logging.getLogger(__name__).error("cap.read() failed: %s", exc)
            return None

    def release(self) -> None:
        self._cap.release()
```

### EventLogger.log_alert signature

```python
def log_alert(
    self,
    command: AlertCommand,
    features: TemporalFeatures,
    active_classes: list[str],
) -> None:
    entry = {
        "event_type": "ALERT",
        "timestamp_ns": command.timestamp_ns,
        "alert_id": command.alert_id,
        "alert_type": command.alert_type.value,
        "alert_level": command.level.value,
        "composite_score": round(command.composite_score, 4),
        "active_classes": active_classes,
        "speed_mps": round(features.speed_modifier, 2),  # NOTE: speed_mps not on TemporalFeatures
        "speed_zone": features.speed_zone,
        "speed_source": "NONE",  # Mac dev: no OBD-II
        "gaze_continuous_secs": round(features.gaze_continuous_secs, 3),
        "head_continuous_secs": round(features.head_continuous_secs, 3),
        "perclos": round(features.perclos, 4),
        "phone_confidence": round(features.phone_confidence_mean, 4),
        "suppress_until_ns": command.suppress_until_ns,
    }
    self._write(entry)
```

**IMPORTANT:** `TemporalFeatures` does NOT have a `speed_mps` field — it has `speed_zone` and `speed_modifier`. The ALERT log entry requires `speed_mps`. Either pass it as a separate param or accept that Mac dev uses 0.0 as a placeholder. Verify against PRD §9.

---

## Integration Test Architecture

The integration test for threaded pipeline wiring should NOT require a real webcam. Use synthetic frames injected via a mock source.

### test_pipeline_wiring.py approach

```python
# Verified pattern from test_perception_stack.py mock approach
def _make_synthetic_raw_frame(frame_id: int) -> RawFrame:
    return RawFrame(
        timestamp_ns=frame_id * 33_333_333,
        frame_id=frame_id,
        width=640, height=480, channels=3,
        data=np.zeros((480, 640, 3), dtype=np.uint8),
        source_type='webcam',
    )

# Integration test: inject N frames, verify pipeline produces TemporalFeatures
# Use unittest.mock.patch to mock PerceptionStack.infer() returning a
# synthetic PerceptionBundle without real ONNX models

# Test goals:
# 1. T-0 → T-1/T-2 queues receive frames correctly
# 2. T-1 carries lstm_hidden_state correctly between frames
# 3. T-2 timeout triggers phone_result_stale=True on merged bundle
# 4. T-3 chain produces AlertCommand | None without exception
# 5. EventLogger writes JSONL on alert
# 6. stop_event causes clean thread shutdown within 2s
```

**Key constraint from testing.md:** Every module must be testable with mock inputs (no real webcam). `WebcamSource` tests must mock `cv2.VideoCapture`. `AudioAlertHandler` tests must mock `subprocess.Popen`.

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python | All | Yes | 3.10.14 | — |
| opencv-python (cv2) | WebcamSource, main.py | Yes | 4.10.0 | — |
| onnxruntime | PerceptionStack (existing) | Yes | 1.19.0 | — |
| pytest | Test suite | Yes | 8.0.2 | — |
| afplay | AudioAlertHandler | Yes | /usr/bin/afplay (2.0) | — |
| /System/Library/Sounds/Ping.aiff | AudioAlertHandler HIGH | Yes | — | — |
| /System/Library/Sounds/Sosumi.aiff | AudioAlertHandler URGENT | Yes | — | — |
| Mac webcam (device 0) | WebcamSource E2E test | Yes | 1920x1080 native | Synthetic frames for unit tests |
| OpenCV AVFoundation backend | VideoCapture Mac support | Yes | Verified | — |

**Missing dependencies with no fallback:** None.

**Note on webcam resolution:** The webcam natively returns 1920x1080 but honors `cap.set()` to 1280x720@30fps correctly. Verified on this machine.

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest 8.0.2 |
| Config file | none (uses standard pytest discovery) |
| Quick run command | `python3 -m pytest tests/unit/test_webcam_source.py tests/unit/test_audio_handler.py tests/unit/test_event_logger.py -x -q` |
| Full suite command | `python3 -m pytest tests/ -x -q` |

### Phase Requirements -> Test Map

| Behavior | Test Type | Automated Command | File Exists? |
|----------|-----------|-------------------|-------------|
| WebcamSource opens device, raises SourceUnavailableError on timeout | unit | `pytest tests/unit/test_webcam_source.py -x` | No — Wave 0 |
| WebcamSource emits RawFrame with correct fields | unit | `pytest tests/unit/test_webcam_source.py -x` | No — Wave 0 |
| WebcamSource returns None on cap.read() failure | unit | `pytest tests/unit/test_webcam_source.py -x` | No — Wave 0 |
| AudioAlertHandler.play() calls Popen with correct sound file | unit | `pytest tests/unit/test_audio_handler.py -x` | No — Wave 0 |
| AudioAlertHandler maps AlertLevel.URGENT to URGENT sound | unit | `pytest tests/unit/test_audio_handler.py -x` | No — Wave 0 |
| AudioAlertHandler does not block (no .wait()) | unit | `pytest tests/unit/test_audio_handler.py -x` | No — Wave 0 |
| EventLogger writes valid JSONL ALERT entry | unit | `pytest tests/unit/test_event_logger.py -x` | No — Wave 0 |
| EventLogger writes STATE_TRANSITION, DEGRADED, WATCHDOG, CALIBRATION entries | unit | `pytest tests/unit/test_event_logger.py -x` | No — Wave 0 |
| EventLogger file rotates at LOG_MAX_BYTES | unit | `pytest tests/unit/test_event_logger.py -x` | No — Wave 0 |
| T-0→T-1/T-2 queues receive frames | integration | `pytest tests/integration/test_pipeline_wiring.py -x` | No — Wave 0 |
| T-2 timeout produces phone_result_stale=True | integration | `pytest tests/integration/test_pipeline_wiring.py -x` | No — Wave 0 |
| Pipeline produces AlertCommand on threshold breach | integration | `pytest tests/integration/test_pipeline_wiring.py -x` | No — Wave 0 |
| Thread stop_event causes clean shutdown | integration | `pytest tests/integration/test_pipeline_wiring.py -x` | No — Wave 0 |

### Sampling Rate

- Per task commit: `python3 -m pytest tests/unit/test_webcam_source.py tests/unit/test_audio_handler.py tests/unit/test_event_logger.py -x -q`
- Per wave merge: `python3 -m pytest tests/ -x -q`
- Phase gate: Full suite (518 existing + all new tests) green before verify-work

### Wave 0 Gaps

- [ ] `tests/unit/test_webcam_source.py` — WebcamSource unit tests (mock cv2.VideoCapture)
- [ ] `tests/unit/test_audio_handler.py` — AudioAlertHandler unit tests (mock subprocess.Popen)
- [ ] `tests/unit/test_event_logger.py` — EventLogger unit tests (write to tempdir, validate JSONL)
- [ ] `tests/integration/test_pipeline_wiring.py` — threaded pipeline integration tests (mock models, synthetic frames)

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Sequential perception pipeline | T-1/T-2 parallel threads (PRD v2.0 CHANGE-07) | v2.0 | T-2 phone detection independent of face detection gate |
| Fixed 1/30s timer intervals | Frame-delta-seconds from timestamps (PRD v2.0 NFR-R4) | v2.0 | Duration timers accurate under thermal throttling |
| stdout-based alerts | afplay + RotatingFileHandler JSONL | Phase 6 | Mac-native audio; structured log for Phase 7 validation |

---

## Open Questions

1. **PerceptionStack phone detector stub: how to implement?**
   - What we know: `PerceptionStack.__init__` takes `phone_detector: PhoneDetector`
   - What's unclear: Should we create a `NoOpPhoneDetector` class, or monkey-patch after construction?
   - Recommendation: Create a lightweight `_NoOpPhoneDetector` inline in `main.py` (or in `perception_stack.py`) that implements `infer(frame) -> PhoneDetectionOutput` returning `_safe_phone()`. No separate file needed.

2. **log_alert active_classes source: DistractionScore or TemporalFeatures?**
   - What we know: `active_classes` is a field of `DistractionScore` (Layer 4 output), not `TemporalFeatures`
   - What's unclear: Does T-3 need to pass `DistractionScore` to `log_alert`, or should `active_classes` be extracted before calling?
   - Recommendation: Pass `score: DistractionScore` as a third argument to `log_alert`, alongside `command: AlertCommand` and `features: TemporalFeatures`. This matches the PRD §9 schema cleanly.

3. **speed_mps in ALERT log entry: not on TemporalFeatures**
   - What we know: PRD §9 ALERT schema requires `"speed_mps": 14.2`, but `TemporalFeatures` only has `speed_zone` and `speed_modifier`, not raw `speed_mps`
   - What's unclear: Where does T-3 get the raw speed value for logging?
   - Recommendation: Pass `speed_mps: float = 0.0` as an explicit param to `log_alert`. On Mac dev, this is always 0.0 (URBAN fallback). Alternatively, T-3 can track speed from `SignalProcessor.process()` call (which receives `speed_mps`).

4. **WatchdogManager and ThermalMonitor integration with EventLogger**
   - What we know: `WatchdogManager` and `ThermalMonitor` already exist and run in background threads inside `TemporalEngine`. They log via Python `logging` but do NOT currently call `EventLogger`.
   - What's unclear: Should `EventLogger` be injected into `TemporalEngine`, or should T-3 poll for watchdog/thermal events?
   - Recommendation: Inject `EventLogger` into `TemporalEngine` (optional arg) so watchdog and thermal monitor can call `event_logger.log_watchdog_timeout()` and `event_logger.log_thermal_warning()` directly from their background threads. Python logging is thread-safe so this is safe.

---

## Sources

### Primary (HIGH confidence)
- Direct code reading: `layer0_video/messages.py`, `layer1_perception/perception_stack.py`, `layer1_perception/messages.py`, `layer2_signals/signal_processor.py`, `layer3_temporal/temporal_engine.py`, `layer4_scoring/scoring_engine.py`, `layer5_alert/alert_state_machine.py`, `layer5_alert/messages.py`, `config.py`
- `PRD_v2.md` §3.3, §8.1–8.7, §9, §FR-0.1–FR-0.4, §FR-6.1, §FR-6.3, §NFR-P4 — read directly from file
- Python 3.10.14 stdlib: `queue`, `threading`, `subprocess`, `logging.handlers`, `signal`, `json` — verified via live execution on this machine

### Secondary (MEDIUM confidence)
- OpenCV 4.10.0 AVFoundation backend: verified via `cv2.videoio_registry.getBackendName(cv2.CAP_AVFOUNDATION)` — returns `'AVFOUNDATION'`
- `cap.set()` for 1280x720@30fps: verified live on this Mac webcam — returns correct values
- afplay launch latency: measured ~1.88ms via `subprocess.Popen` — well within 50ms NFR-P4
- RotatingFileHandler JSONL output: verified format via live Python execution

### Tertiary (LOW confidence)
- cv2.imshow main-thread constraint on macOS: based on Cocoa framework behavior — this was not live-tested but is a well-documented macOS constraint

---

## Metadata

**Confidence breakdown:**
- Layer interfaces: HIGH — read directly from implemented source files
- Standard stack: HIGH — all libraries verified installed and functional on this machine
- Architecture patterns: HIGH — derived from PRD + CONTEXT.md locked decisions, patterns verified via live Python execution
- PRD schemas: HIGH — read directly from PRD_v2.md §9
- Pitfalls: HIGH (PerceptionStack phone coupling, enum serialization) / MEDIUM (imshow main thread) — based on code analysis
- Config values: HIGH — read directly from config.py, all constants confirmed present

**Research date:** 2026-03-29
**Valid until:** 2026-04-28 (stable stdlib + locked PRD decisions; unlikely to change)

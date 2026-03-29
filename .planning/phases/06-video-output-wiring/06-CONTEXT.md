# Phase 6: Video + Output + Full Wiring — Context

**Gathered:** 2026-03-29
**Status:** Ready for planning
**Mode:** auto (--auto flag)

<domain>
## Phase Boundary

Deliver the complete runnable pipeline on Mac webcam. This phase wires all previously-built layers (1–5) together with:

1. `layer0_video/webcam_source.py` — OpenCV-based Mac webcam source, emits RawFrame
2. `layer6_output/audio_handler.py` — afplay-backed audio alert dispatcher (Mac dev)
3. `layer6_output/event_logger.py` — rotating JSON log, 50MB max, 5 files retained
4. `main.py` — entry point, instantiates all layers and runs T-0/T-1/T-2/T-3 thread architecture

Scope boundary: Mac dev only. No V4L2, no IMX219, no RKNN, no hardware. All hardware is Phase 8.

</domain>

<decisions>
## Implementation Decisions

### D-01: Thread Architecture (PRD §3.3)
- **T-0 (VideoCapture):** `threading.Thread` daemon, reads from OpenCV `VideoCapture`, puts `RawFrame` into `queue.Queue(maxsize=config.FRAME_QUEUE_DEPTH)`. Drop oldest on overflow (get() + put() pattern).
- **T-1 (FacePerception):** High-priority thread. Gets `RawFrame` from queue, runs `PerceptionStack.infer()`, puts partial `PerceptionBundle` (face + landmarks + gaze, phone=None/stale) into a result dict keyed by `frame_id`.
- **T-2 (PhoneDetection):** Normal-priority thread. Gets same `RawFrame` reference (shared from T-0 via a second queue or broadcast), runs `PhoneDetector.infer()`, puts `PhoneDetectionOutput` into a result dict keyed by `frame_id`.
- **T-1/T-2 merge:** T-1 sets a `threading.Event` when it completes. T-2 result is waited up to `config.PHONE_THREAD_TIMEOUT_MS / 1000.0` seconds. If T-2 times out, use last valid `PhoneDetectionOutput` with `phone_result_stale=True`. Stale count logged.
- **T-3 (Pipeline):** Normal-priority thread. Consumes merged `PerceptionBundle`, feeds it sequentially through `SignalProcessor` → `TemporalEngine` → `ScoringEngine` → `AlertStateMachine`. Dispatches `AlertCommand` to `AudioHandler` and `EventLogger`.
- **Frame distribution:** T-0 puts `RawFrame` into **two separate queues** — one for T-1, one for T-2. Both queues have depth `FRAME_QUEUE_DEPTH`. This avoids any locking overhead and matches PRD "shared reference" intent.

### D-02: Webcam Source (PRD §FR-0.1–FR-0.4)
- Class: `WebcamSource` in `layer0_video/webcam_source.py`
- Constructor: accepts `device_index: int = 0` (Mac dev; string path is Phase 8)
- Init timeout: poll `cap.isOpened()` in a monotonic loop up to 5.0 seconds; raise `SourceUnavailableError` if not ready
- Emits: `RawFrame(source_type='webcam', ...)` with `timestamp_ns=time.monotonic_ns()`, monotonically increasing `frame_id`
- `release()` method: calls `cap.release()`, logs closure
- Resolution: set `CAP_PROP_FRAME_WIDTH=config.CAPTURE_WIDTH`, `CAP_PROP_FRAME_HEIGHT=config.CAPTURE_HEIGHT`, `CAP_PROP_FPS=config.CAPTURE_FPS` via `cap.set()`
- Exception handling: all `cap.read()` failures caught, logged, returns None (T-0 skips None frames)

### D-03: Audio Handler (PRD §FR-6.1)
- Class: `AudioAlertHandler` in `layer6_output/audio_handler.py`
- Mac implementation: `subprocess.Popen(['afplay', sound_file])` — fire and forget, do NOT join/wait
- Dispatch latency must be ≤ 50ms from `AlertCommand` receipt to audio start (PRD NFR-P4)
- Sound file: use a bundled `.aiff` or `.wav` from the OS (`/System/Library/Sounds/Ping.aiff` as default); path configurable via `config.AUDIO_ALERT_SOUND` (add to config.py)
- Each `AlertLevel` (HIGH, URGENT) maps to a distinct sound file (configurable)
- No additional debouncing — the AlertStateMachine already enforces per-type cooldowns
- `play(command: AlertCommand)` is the public method
- Timestamp dispatch time for latency measurement; log if >50ms

### D-04: Event Logger (PRD §FR-6.3, §9)
- Class: `EventLogger` in `layer6_output/event_logger.py`
- Uses Python `logging.handlers.RotatingFileHandler` with `maxBytes=config.LOG_MAX_BYTES` and `backupCount=config.LOG_BACKUP_COUNT`
- Log directory: `config.LOG_DIR` — create if not exists
- Log filename: `attentia_events.jsonl` (newline-delimited JSON, one JSON object per line)
- Thread-safe: Python's `logging` module is thread-safe by default (GIL + internal locks)
- Public methods:
  - `log_alert(command: AlertCommand, features: TemporalFeatures)` → writes ALERT entry per §9
  - `log_state_transition(prev: str, new: str, trigger: str, frame_id: int, ts_ns: int)` → writes STATE_TRANSITION entry
  - `log_degraded(reason: str, duration_secs: float, ts_ns: int)` → writes DEGRADED entry
  - `log_watchdog_timeout(last_frame_id: int, secs_since: float, ts_ns: int)` → writes WATCHDOG_TIMEOUT entry
  - `log_calibration_complete(event: dict)` → writes CALIBRATION_COMPLETE entry
- All entries serialized with `json.dumps()` — no custom serializer needed

### D-05: main.py Lifecycle
- `main()` function: parse CLI args (optional `--device INT`, `--no-display`)
- Instantiate all components: `WebcamSource`, `PerceptionStack` (with models), `SignalProcessor`, `TemporalEngine`, `ScoringEngine`, `AlertStateMachine`, `AudioAlertHandler`, `EventLogger`
- `stop_event = threading.Event()` shared across all threads
- Signal handler: `signal.signal(SIGINT, ...)` and `signal.signal(SIGTERM, ...)` → sets `stop_event`
- Thread start order: T-3 → T-2 → T-1 → T-0 (sources start last so threads are ready)
- Thread stop order: stop_event set → T-0 stops first (source stops) → T-1, T-2, T-3 drain and stop → join all
- Display (optional): cv2.imshow preview if `--display` flag; off by default (headless)
- TemporalEngine.start() / .stop() called for watchdog + thermal threads

### Claude's Discretion
- Exact thread priority setting: Python doesn't expose OS thread priority cleanly on Mac; use default priority for all threads (SCHED_FIFO is Linux/hardware only, Phase 8)
- cv2.imshow display format: if enabled, draw face bbox + phone indicator overlay
- Logging verbosity at startup: log model paths, device index, config summary

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### PRD Sections
- `PRD_v2.md` §FR-0.1–FR-0.4 — VideoSource requirements (device input, RawFrame emission, SourceUnavailableError, release())
- `PRD_v2.md` §3.3 — Thread architecture: T-0/T-1/T-2/T-3, queue depth, 5ms T-2 timeout, phone_result_stale
- `PRD_v2.md` §FR-6.1 — AudioAlertHandler: afplay on Mac, ≤50ms dispatch latency
- `PRD_v2.md` §FR-6.3 — EventLogger: rotating JSON log, 50MB max, 5 files retained
- `PRD_v2.md` §9 — Event log format: all 5 JSON schemas (ALERT, STATE_TRANSITION, DEGRADED, THERMAL_WARNING, WATCHDOG_TIMEOUT, CALIBRATION_COMPLETE)
- `PRD_v2.md` §NFR-P4 — Audio dispatch latency ≤ 50ms P95

### Config
- `config.py` — All constants: PHONE_THREAD_TIMEOUT_MS, FRAME_QUEUE_DEPTH, LOG_DIR, LOG_MAX_BYTES, LOG_BACKUP_COUNT, CAPTURE_WIDTH, CAPTURE_HEIGHT, CAPTURE_FPS

### Existing Layer Interfaces
- `layer0_video/messages.py` — RawFrame dataclass (source of truth for T-0 output)
- `layer1_perception/perception_stack.py` — PerceptionStack.infer(frame, frame_id, hidden_state)
- `layer1_perception/messages.py` — PerceptionBundle, PhoneDetectionOutput
- `layer2_signals/signal_processor.py` — SignalProcessor.process(bundle) → SignalFrame
- `layer3_temporal/temporal_engine.py` — TemporalEngine.process(frame) → TemporalFeatures; .start()/.stop()
- `layer4_scoring/scoring_engine.py` — ScoringEngine.score(features) → DistractionScore
- `layer5_alert/alert_state_machine.py` — AlertStateMachine.update(score) → AlertCommand | None
- `layer5_alert/messages.py` — AlertCommand dataclass

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `layer0_video/messages.py:RawFrame` — fully defined, use as-is for T-0 output
- `layer1_perception/perception_stack.py:PerceptionStack` — thread-safe per instance (no shared state across calls); instantiate once for T-1
- `layer1_perception/phone_detector.py:PhoneDetector` — instantiate separately for T-2 (separate ONNX session)
- `config.py` — PHONE_THREAD_TIMEOUT_MS=5, FRAME_QUEUE_DEPTH=2, LOG_DIR='logs/', LOG_MAX_BYTES=52_428_800, LOG_BACKUP_COUNT=5

### Established Patterns
- All inter-layer communication via typed dataclasses only (no raw dicts)
- All exceptions caught at module boundaries — nothing propagates silently
- LSTM hidden state passes through PerceptionBundle.lstm_hidden_state between frames
- PerceptionStack.infer() is already exception-safe (returns safe bundle on error)

### Integration Points
- T-0 → T-1/T-2: Two `queue.Queue(maxsize=FRAME_QUEUE_DEPTH)` — one per thread
- T-1 → T-3: Partial PerceptionBundle (phone=_safe_phone(), phone_result_stale=True until T-2 result arrives)
- T-2 → T-3: PhoneDetectionOutput keyed by frame_id, merged into PerceptionBundle
- T-3 → AudioHandler + EventLogger: AlertCommand from AlertStateMachine

### Missing Config Values
- `config.AUDIO_ALERT_SOUND` — needs adding (default: '/System/Library/Sounds/Ping.aiff')
- `config.AUDIO_ALERT_SOUND_URGENT` — needs adding (default: '/System/Library/Sounds/Sosumi.aiff')

</code_context>

<specifics>
## Specific Requirements

- ONNX runtime only — no RKNN, no V4L2, no IMX219 (Phase 8)
- Mac webcam: `cv2.VideoCapture(device_index)` with integer index
- Audio: `afplay` via `subprocess.Popen` (not `subprocess.run` — must not block)
- Log format: newline-delimited JSON (.jsonl), one object per line, per PRD §9 schemas exactly
- LSTM hidden state must persist across frames in T-1: pass `bundle.lstm_hidden_state` forward
- Phone detector runs independently in T-2 every frame regardless of face detection result (PRD §FR1.4)

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 06-video-output-wiring*
*Context gathered: 2026-03-29*

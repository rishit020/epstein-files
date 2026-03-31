# main.py — Attentia Drive entry point
# PRD §3.3 — T-0/T-1/T-2/T-3 thread architecture
# Mac dev: ONNX runtime, OpenCV webcam, afplay audio, rotating JSONL log.
# No RKNN, no V4L2, no IMX219 (Phase 8).

from __future__ import annotations

import argparse
import logging
import queue
import signal
import sys
import threading
import time
from typing import Optional

import cv2
import numpy as np

import config
from layer0_video.webcam_source import WebcamSource, SourceUnavailableError
from layer0_video.messages import RawFrame
from layer1_perception.face_detector import FaceDetector
from layer1_perception.gaze_model import GazeModel
from layer1_perception.landmark_model import LandmarkModel
from layer1_perception.messages import PerceptionBundle, PhoneDetectionOutput
from layer1_perception.perception_stack import PerceptionStack
from layer1_perception.phone_detector import PhoneDetector
from layer2_signals.signal_processor import SignalProcessor
from layer3_temporal.temporal_engine import TemporalEngine
from layer4_scoring.scoring_engine import ScoringEngine
from layer5_alert.alert_state_machine import AlertStateMachine
from layer6_output.audio_handler import AudioAlertHandler
from layer6_output.event_logger import EventLogger

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)-8s %(name)s: %(message)s',
    datefmt='%H:%M:%S',
)
_log = logging.getLogger(__name__)


# ── NoOp phone detector stub for T-1 ─────────────────────────────────────────

class _NoOpPhoneDetector:
    """Stub phone detector passed to T-1's PerceptionStack.

    T-1 must NOT run real phone detection — that is T-2's job.
    Duck-typing: satisfies PerceptionStack's phone_detector argument without
    inheriting PhoneDetector (avoids loading the ONNX model twice).
    """
    def infer(self, frame: np.ndarray) -> PhoneDetectionOutput:
        return PhoneDetectionOutput(detected=False, max_confidence=0.0, bbox_norm=None)


# ── Queue helper ──────────────────────────────────────────────────────────────

def _put_dropping_oldest(q: queue.Queue, item: object) -> None:
    """Put item into queue; if full, drop oldest to make room.

    PRD §3.3 — T-0 must never block. Drop oldest frame to keep latency low.
    """
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
            pass  # still full — skip this frame


# ── T-0: VideoCapture ─────────────────────────────────────────────────────────

def _t0_video_capture(
    source: WebcamSource,
    q_t1: queue.Queue,
    q_t2: queue.Queue,
    stop_event: threading.Event,
    display_queue: Optional[queue.Queue],
) -> None:
    """T-0: Read frames from WebcamSource and distribute to T-1 and T-2 queues.

    PRD §3.3 — T-0 daemon. Puts same RawFrame reference into both queues.
    Drops oldest on overflow (shallow queue = low latency).
    """
    _log.info("T-0 started")
    while not stop_event.is_set():
        try:
            raw_frame = source.read()
        except Exception as exc:
            _log.error("T-0 source.read() exception: %s", exc)
            continue

        if raw_frame is None:
            time.sleep(0.005)  # brief back-off on read failure
            continue

        _put_dropping_oldest(q_t1, raw_frame)
        _put_dropping_oldest(q_t2, raw_frame)

        # Optional display queue (main thread polls this for cv2.imshow)
        if display_queue is not None:
            try:
                display_queue.put_nowait(raw_frame)
            except queue.Full:
                try:
                    display_queue.get_nowait()
                    display_queue.put_nowait(raw_frame)
                except (queue.Full, queue.Empty):
                    pass

    _log.info("T-0 stopped")


# ── T-2: PhoneDetection ───────────────────────────────────────────────────────

def _t2_phone_detection(
    phone_detector: PhoneDetector,
    q_t2: queue.Queue,
    t2_results: dict,
    t2_lock: threading.Lock,
    stop_event: threading.Event,
) -> None:
    """T-2: Run PhoneDetector independently on each frame.

    PRD §3.3 — T-2 normal-priority. Stores results keyed by frame_id for T-1 merge.
    T-1 is responsible for cleaning up t2_results after merge (pop on consume).
    """
    _log.info("T-2 started")
    while not stop_event.is_set():
        try:
            raw_frame: RawFrame = q_t2.get(timeout=0.1)
        except queue.Empty:
            continue

        try:
            phone_out = phone_detector.infer(raw_frame.data)
        except Exception as exc:
            _log.error("T-2 phone_detector.infer() error frame_id=%d: %s", raw_frame.frame_id, exc)
            phone_out = PhoneDetectionOutput(detected=False, max_confidence=0.0, bbox_norm=None)

        with t2_lock:
            t2_results[raw_frame.frame_id] = phone_out

    _log.info("T-2 stopped")


# ── T-1: FacePerception ───────────────────────────────────────────────────────

def _t1_face_perception(
    perception_stack: PerceptionStack,
    q_t1: queue.Queue,
    q_t3: queue.Queue,
    t2_results: dict,
    t2_lock: threading.Lock,
    stop_event: threading.Event,
) -> None:
    """T-1: Run PerceptionStack (face+landmarks+gaze) and merge T-2 phone result.

    PRD §3.3 — T-1 high-priority. Carries LSTM hidden state across frames.
    Waits up to PHONE_THREAD_TIMEOUT_MS for T-2 result before using stale value.
    Cleans up t2_results after each frame to prevent memory growth (Pitfall 6).
    """
    _log.info("T-1 started")
    lstm_hidden_state: tuple | None = None
    last_valid_phone = PhoneDetectionOutput(detected=False, max_confidence=0.0, bbox_norm=None)
    timeout_s = config.PHONE_THREAD_TIMEOUT_MS / 1000.0

    while not stop_event.is_set():
        try:
            raw_frame: RawFrame = q_t1.get(timeout=0.1)
        except queue.Empty:
            continue

        # Run face perception with NoOp phone detector (T-2 provides real phone result)
        try:
            bundle = perception_stack.infer(
                raw_frame.data,
                raw_frame.frame_id,
                hidden_state=lstm_hidden_state,
            )
        except Exception as exc:
            _log.error("T-1 perception_stack.infer() error frame_id=%d: %s", raw_frame.frame_id, exc)
            continue

        # Carry LSTM state forward (Pattern 3)
        lstm_hidden_state = bundle.lstm_hidden_state

        # Wait for T-2 phone result (up to PHONE_THREAD_TIMEOUT_MS)
        phone_result: Optional[PhoneDetectionOutput] = None
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            with t2_lock:
                if raw_frame.frame_id in t2_results:
                    phone_result = t2_results.pop(raw_frame.frame_id)  # clean up (Pitfall 6)
                    break
            time.sleep(0.0005)  # 0.5ms poll interval

        # Merge phone result or mark stale
        if phone_result is not None:
            bundle.phone = phone_result
            bundle.phone_result_stale = False
            last_valid_phone = phone_result
        else:
            bundle.phone = last_valid_phone
            bundle.phone_result_stale = True
            _log.debug("T-1 phone result stale for frame_id=%d", raw_frame.frame_id)

        _put_dropping_oldest(q_t3, bundle)

    _log.info("T-1 stopped")


# ── T-3: Pipeline ─────────────────────────────────────────────────────────────

def _t3_pipeline(
    signal_processor: SignalProcessor,
    temporal_engine: TemporalEngine,
    scoring_engine: ScoringEngine,
    alert_state_machine: AlertStateMachine,
    audio_handler: AudioAlertHandler,
    event_logger: EventLogger,
    q_t3: queue.Queue,
    stop_event: threading.Event,
) -> None:
    """T-3: Feed merged PerceptionBundle through the full processing chain.

    PRD §3.3 — T-3 normal-priority.
    Chain: SignalProcessor -> TemporalEngine -> ScoringEngine -> AlertStateMachine.
    Dispatches AlertCommand to AudioAlertHandler and EventLogger.
    Logs state transitions when AlertStateMachine.state changes.
    """
    _log.info("T-3 started")
    prev_state: str = alert_state_machine.state
    frame_count = 0

    while not stop_event.is_set():
        try:
            bundle: PerceptionBundle = q_t3.get(timeout=0.1)
        except queue.Empty:
            continue

        try:
            # Step 1: SignalProcessor — extract per-frame signals
            signal_frame = signal_processor.process(
                bundle,
                speed_mps=0.0,        # Mac dev: no OBD-II, URBAN fallback
                speed_stale=True,     # always stale on Mac dev
            )

            # Step 2: TemporalEngine — compute window-aggregated features
            temporal_features = temporal_engine.process(signal_frame)

            # Step 3: ScoringEngine — compute composite distraction score
            distraction_score = scoring_engine.score(temporal_features)

            # Step 4: AlertStateMachine — determine if an alert should fire
            alert_cmd = alert_state_machine.process(distraction_score)

            # Log state transition if state changed
            current_state = alert_state_machine.state
            if current_state != prev_state:
                event_logger.log_state_transition(
                    prev=prev_state,
                    new=current_state,
                    trigger=alert_cmd.alert_type.value if alert_cmd else 'internal',
                    frame_id=bundle.frame_id,
                    ts_ns=bundle.timestamp_ns,
                )
                prev_state = current_state

            # Dispatch alert if fired
            if alert_cmd is not None:
                audio_handler.play(alert_cmd)
                event_logger.log_alert(
                    command=alert_cmd,
                    features=temporal_features,
                    score=distraction_score,
                    speed_mps=0.0,
                )
                _log.info(
                    "ALERT fired: type=%s level=%s composite=%.3f frame_id=%d",
                    alert_cmd.alert_type.value,
                    alert_cmd.level.name,
                    alert_cmd.composite_score,
                    bundle.frame_id,
                )

        except Exception as exc:
            _log.error("T-3 pipeline error frame_id=%d: %s", bundle.frame_id, exc)

        frame_count += 1
        if frame_count % 300 == 0:  # log every ~10s at 30fps
            _log.debug("T-3 processed %d frames", frame_count)

    _log.info("T-3 stopped after %d frames", frame_count)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description='Attentia Drive — distraction detection')
    parser.add_argument('--device', type=int, default=0,
                        help='Webcam device index (default: 0)')
    parser.add_argument('--display', action='store_true', default=False,
                        help='Show live preview window (cv2.imshow on main thread)')
    args = parser.parse_args()

    _log.info("Starting Attentia Drive — device=%d display=%s", args.device, args.display)
    _log.info("Config: FRAME_QUEUE_DEPTH=%d PHONE_THREAD_TIMEOUT_MS=%d",
              config.FRAME_QUEUE_DEPTH, config.PHONE_THREAD_TIMEOUT_MS)

    # ── Instantiate all components ─────────────────────────────────────────
    try:
        source = WebcamSource(device_index=args.device)
    except SourceUnavailableError as exc:
        _log.error("Cannot open webcam: %s", exc)
        sys.exit(1)

    # T-1 gets a PerceptionStack with NoOp phone detector (Pitfall 1)
    perception_stack = PerceptionStack(
        face_detector=FaceDetector(),
        landmark_model=LandmarkModel(),
        gaze_model=GazeModel(),
        phone_detector=_NoOpPhoneDetector(),
    )

    # T-2 gets a real PhoneDetector with its own ONNX session
    phone_detector = PhoneDetector()

    signal_processor = SignalProcessor()
    temporal_engine = TemporalEngine()
    scoring_engine = ScoringEngine()
    alert_state_machine = AlertStateMachine()
    audio_handler = AudioAlertHandler()
    event_logger = EventLogger()

    _log.info("All components initialised")

    # ── Queues and sync primitives ────────────────────────────────────────
    q_t1 = queue.Queue(maxsize=config.FRAME_QUEUE_DEPTH)
    q_t2 = queue.Queue(maxsize=config.FRAME_QUEUE_DEPTH)
    q_t3 = queue.Queue(maxsize=config.FRAME_QUEUE_DEPTH)
    display_queue: Optional[queue.Queue] = queue.Queue(maxsize=2) if args.display else None

    t2_results: dict = {}
    t2_lock = threading.Lock()

    stop_event = threading.Event()

    # ── Signal handlers ───────────────────────────────────────────────────
    def _shutdown(signum, frame_obj):
        _log.info("Signal %s received — initiating shutdown", signum)
        stop_event.set()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # ── Start TemporalEngine background threads (watchdog + thermal) ──────
    temporal_engine.start()

    # ── Start worker threads — T-3 first, T-0 last (Pitfall 7) ──────────
    t3 = threading.Thread(
        target=_t3_pipeline,
        args=(signal_processor, temporal_engine, scoring_engine, alert_state_machine,
              audio_handler, event_logger, q_t3, stop_event),
        name='T-3-Pipeline', daemon=False,
    )
    t2 = threading.Thread(
        target=_t2_phone_detection,
        args=(phone_detector, q_t2, t2_results, t2_lock, stop_event),
        name='T-2-Phone', daemon=False,
    )
    t1 = threading.Thread(
        target=_t1_face_perception,
        args=(perception_stack, q_t1, q_t3, t2_results, t2_lock, stop_event),
        name='T-1-Face', daemon=False,
    )
    t0 = threading.Thread(
        target=_t0_video_capture,
        args=(source, q_t1, q_t2, stop_event, display_queue),
        name='T-0-Capture', daemon=True,  # daemon — exits if main thread exits
    )

    # Start in reverse order: consumers ready before source emits (D-05)
    t3.start()
    t2.start()
    t1.start()
    t0.start()
    _log.info("All threads started. Press Ctrl-C to stop.")

    # ── Main thread: optional display loop (macOS requires imshow on main thread — Pitfall 4)
    try:
        if args.display:
            while not stop_event.is_set():
                try:
                    raw_frame: RawFrame = display_queue.get(timeout=0.1)
                except queue.Empty:
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        stop_event.set()
                    continue
                cv2.imshow('Attentia Drive', raw_frame.data)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    stop_event.set()
            cv2.destroyAllWindows()
        else:
            # Headless: main thread blocks until stop_event
            while not stop_event.is_set():
                time.sleep(0.5)
    except KeyboardInterrupt:
        _log.info("KeyboardInterrupt received")
        stop_event.set()

    # ── Shutdown sequence ─────────────────────────────────────────────────
    _log.info("Shutting down...")
    stop_event.set()

    # T-0 stops first (source stops producing; queues drain)
    t0.join(timeout=2.0)
    # Then T-1, T-2, T-3 drain their queues and exit
    t1.join(timeout=5.0)
    t2.join(timeout=5.0)
    t3.join(timeout=5.0)

    if t1.is_alive() or t2.is_alive() or t3.is_alive():
        _log.warning("Some threads did not exit cleanly: T-1=%s T-2=%s T-3=%s",
                     t1.is_alive(), t2.is_alive(), t3.is_alive())

    # Stop TemporalEngine background threads (watchdog + thermal)
    temporal_engine.stop()

    source.release()
    _log.info("Attentia Drive stopped cleanly")


if __name__ == '__main__':
    main()

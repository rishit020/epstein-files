#!/usr/bin/env python3
"""
Diagnostic script to test live distraction detection with detailed output.
Shows real-time signal values, why signals are invalid, and alert state.

Run: python test_live_diagnostic.py --display --duration 60
"""

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
from layer1_perception.landmark_model import LandmarkModel
from layer1_perception.gaze_model import GazeModel
from layer1_perception.perception_stack import PerceptionStack
from layer1_perception.phone_detector import PhoneDetector
from layer2_signals.signal_processor import SignalProcessor
from layer3_temporal.temporal_engine import TemporalEngine
from layer4_scoring.scoring_engine import ScoringEngine
from layer5_alert.alert_state_machine import AlertStateMachine
from layer6_output.audio_handler import AudioAlertHandler
from layer6_output.event_logger import EventLogger

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(name)-20s %(levelname)-8s %(message)s',
    datefmt='%H:%M:%S',
)
_log = logging.getLogger(__name__)


class _NoOpPhoneDetector:
    def infer(self, frame: np.ndarray):
        from layer1_perception.messages import PhoneDetectionOutput
        return PhoneDetectionOutput(detected=False, max_confidence=0.0, bbox_norm=None)


def main() -> None:
    parser = argparse.ArgumentParser(description='Attentia diagnostic — live signal inspection')
    parser.add_argument('--device', type=int, default=0, help='Webcam device index')
    parser.add_argument('--display', action='store_true', help='Show live preview')
    parser.add_argument('--duration', type=int, default=30, help='Run duration in seconds')
    args = parser.parse_args()

    _log.info("Starting diagnostic — device=%d display=%s duration=%ds",
              args.device, args.display, args.duration)

    try:
        source = WebcamSource(device_index=args.device)
    except SourceUnavailableError as exc:
        _log.error("Cannot open webcam: %s", exc)
        sys.exit(1)

    perception_stack = PerceptionStack(
        face_detector=FaceDetector(),
        landmark_model=LandmarkModel(),
        gaze_model=GazeModel(),
        phone_detector=_NoOpPhoneDetector(),
    )
    phone_detector = PhoneDetector()
    signal_processor = SignalProcessor()
    temporal_engine = TemporalEngine()
    scoring_engine = ScoringEngine()
    alert_state_machine = AlertStateMachine()
    audio_handler = AudioAlertHandler()
    event_logger = EventLogger()

    stop_event = threading.Event()
    start_time = time.monotonic()

    def _shutdown(signum, frame_obj):
        _log.info("Shutdown signal received")
        stop_event.set()

    signal.signal(signal.SIGINT, _shutdown)
    temporal_engine.start()

    frame_count = 0
    invalid_frame_count = 0
    alert_fired_count = 0

    try:
        while not stop_event.is_set() and (time.monotonic() - start_time) < args.duration:
            try:
                raw_frame = source.read()
            except Exception as exc:
                _log.error("source.read() error: %s", exc)
                continue

            if raw_frame is None:
                time.sleep(0.01)
                continue

            frame_count += 1

            # ── Perception ─────────────────────────────────────────────
            try:
                bundle = perception_stack.infer(raw_frame.data, raw_frame.frame_id)
            except Exception as exc:
                _log.error("perception_stack.infer() error: %s", exc)
                continue

            # ── Signal Processing ──────────────────────────────────────
            try:
                signal_frame = signal_processor.process(bundle, speed_mps=5.0, speed_stale=False)
            except Exception as exc:
                _log.error("signal_processor.process() error: %s", exc)
                continue

            # ── Temporal ────────────────────────────────────────────────
            try:
                temporal_features = temporal_engine.process(signal_frame)
            except Exception as exc:
                _log.error("temporal_engine.process() error: %s", exc)
                continue

            # ── Scoring ─────────────────────────────────────────────────
            try:
                distraction_score = scoring_engine.score(temporal_features)
            except Exception as exc:
                _log.error("scoring_engine.score() error: %s", exc)
                continue

            # ── Alert State Machine ─────────────────────────────────────
            try:
                alert_cmd = alert_state_machine.process(distraction_score)
            except Exception as exc:
                _log.error("alert_state_machine.process() error: %s", exc)
                continue

            # ── Diagnostic output ───────────────────────────────────────
            if not signal_frame.signals_valid:
                invalid_frame_count += 1

            if alert_cmd is not None:
                alert_fired_count += 1
                audio_handler.play(alert_cmd)
                _log.warning(
                    "🚨 ALERT FIRED: type=%s level=%s composite=%.3f",
                    alert_cmd.alert_type.value,
                    alert_cmd.level.name,
                    alert_cmd.composite_score,
                )

            # Every 15 frames, print diagnostic snapshot
            if frame_count % 15 == 0:
                print(
                    f"\n[Frame {frame_count:4d}] "
                    f"Face={'Y' if bundle.face.present else 'N'} "
                    f"| Gaze={bundle.gaze.combined_yaw if bundle.gaze else 'N/A':+6.1f}° "
                    f"| Head={signal_frame.head_pose.yaw if signal_frame.head_pose else 'N/A':+6.1f}° "
                    f"| PERCLOS={temporal_features.perclos:.2f} "
                    f"| Composite={distraction_score.composite_score:.3f} "
                    f"| State={alert_state_machine.state:10s} "
                    f"| Phone={'Y' if bundle.phone.detected else 'N'}"
                )
                print(
                    f"  Signals_valid={signal_frame.signals_valid:5} "
                    f"| Invalid_consecutive={invalid_frame_count:3d}/60 "
                    f"| Alerts_fired={alert_fired_count}"
                )
                print(
                    f"  Gaze_off_road={temporal_features.gaze_off_road_fraction:.2f} "
                    f"| Gaze_continuous={temporal_features.gaze_continuous_secs:.2f}s "
                    f"| Head_continuous={temporal_features.head_continuous_secs:.2f}s"
                )

            if args.display:
                cv2.imshow('Diagnostic', raw_frame.data)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    stop_event.set()

    except KeyboardInterrupt:
        pass
    finally:
        temporal_engine.stop()
        source.release()
        if args.display:
            cv2.destroyAllWindows()

        elapsed = time.monotonic() - start_time
        _log.info(
            "\n\n=== DIAGNOSTIC SUMMARY ===\n"
            f"Duration: {elapsed:.1f}s\n"
            f"Frames processed: {frame_count}\n"
            f"Invalid signal frames: {invalid_frame_count} ({100*invalid_frame_count/max(1,frame_count):.1f}%)\n"
            f"Alerts fired: {alert_fired_count}\n"
            f"Final state: {alert_state_machine.state}"
        )


if __name__ == '__main__':
    main()

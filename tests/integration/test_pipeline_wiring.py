"""test_pipeline_wiring.py — TDD stub tests for full pipeline wiring (main.py).

PRD §3.3: T-0/T-1/T-2/T-3 thread architecture.
PRD §FR-0.1–FR-0.4: WebcamSource integration.
Decision D-01 (06-CONTEXT.md): Frame distribution, T-1/T-2 merge, lifecycle.
Decision D-05 (06-CONTEXT.md): main.py lifecycle (start, stop, signal handling).

All tests skip until main.py is implemented (Plan 05).
"""

from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import threading
import numpy as np
import pytest
from unittest.mock import MagicMock

_IMPL_MISSING = False
try:
    from main import build_pipeline, run_pipeline
except ImportError:
    _IMPL_MISSING = True

from layer0_video.messages import RawFrame
from layer1_perception.messages import (
    FaceDetection,
    GazeOutput,
    LandmarkOutput,
    PerceptionBundle,
    PhoneDetectionOutput,
)
from layer5_alert.alert_types import AlertLevel, AlertType
from layer5_alert.messages import AlertCommand

import config

pytestmark = pytest.mark.skipif(
    _IMPL_MISSING,
    reason="main.py not yet implemented — will be implemented in Plan 05",
)


# ═══════════════════════════════════════════════════════════════════════════════
# Helper factories
# ═══════════════════════════════════════════════════════════════════════════════

def _make_synthetic_frame(frame_id: int) -> RawFrame:
    """Returns a synthetic RawFrame at 30fps spacing (33.33ms per frame)."""
    return RawFrame(
        timestamp_ns=frame_id * 33_333_333,
        frame_id=frame_id,
        width=640,
        height=480,
        channels=3,
        data=np.zeros((480, 640, 3), dtype=np.uint8),
        source_type='webcam',
    )


def _make_mock_perception_stack() -> MagicMock:
    """Returns a MagicMock PerceptionStack whose .infer() returns a safe PerceptionBundle."""
    mock = MagicMock()
    safe_bundle = PerceptionBundle(
        timestamp_ns=0,
        frame_id=0,
        face=FaceDetection(present=False, confidence=0.0, bbox_norm=None, face_size_px=0),
        landmarks=None,
        gaze=None,
        phone=PhoneDetectionOutput(detected=False, max_confidence=0.0, bbox_norm=None),
        phone_result_stale=False,
        inference_ms=0.0,
        lstm_hidden_state=None,
        lstm_reset_occurred=False,
    )
    mock.infer.return_value = safe_bundle
    return mock


def _make_mock_phone_detector() -> MagicMock:
    """Returns a MagicMock PhoneDetector whose .infer() returns a safe PhoneDetectionOutput."""
    mock = MagicMock()
    mock.infer.return_value = PhoneDetectionOutput(
        detected=False,
        max_confidence=0.0,
        bbox_norm=None,
    )
    return mock


# ═══════════════════════════════════════════════════════════════════════════════
# TestPipelineWiring
# PRD §3.3: T-0/T-1/T-2/T-3 thread architecture
# D-01: Frame distribution into two queues, T-1/T-2 merge, stale handling
# D-05: Lifecycle — stop_event terminates all threads cleanly
# ═══════════════════════════════════════════════════════════════════════════════

class TestPipelineWiring:

    def test_t0_puts_frame_into_both_queues(self):
        """D-01: T-0 frame distribution must put RawFrame into both T-1 queue and T-2 queue."""
        pytest.skip("Stub — implementation in Plan 05")
        import queue
        q1 = queue.Queue(maxsize=config.FRAME_QUEUE_DEPTH)
        q2 = queue.Queue(maxsize=config.FRAME_QUEUE_DEPTH)
        frame = _make_synthetic_frame(0)

        # Simulate T-0 put logic (implementation will do this in the thread loop)
        q1.put(frame)
        q2.put(frame)

        assert q1.get_nowait() is frame
        assert q2.get_nowait() is frame

    def test_t1_produces_perception_bundle(self):
        """D-01: Feeding a RawFrame to PerceptionStack.infer() returns a PerceptionBundle."""
        pytest.skip("Stub — implementation in Plan 05")
        frame = _make_synthetic_frame(1)
        stack = _make_mock_perception_stack()
        result = stack.infer(frame.data, frame_id=frame.frame_id)
        assert isinstance(result, PerceptionBundle)

    def test_t2_produces_phone_detection(self):
        """D-01: Feeding a RawFrame to PhoneDetector.infer() returns a PhoneDetectionOutput."""
        pytest.skip("Stub — implementation in Plan 05")
        frame = _make_synthetic_frame(2)
        phone_det = _make_mock_phone_detector()
        result = phone_det.infer(frame.data)
        assert isinstance(result, PhoneDetectionOutput)

    def test_t1_t2_merge_uses_phone_result_when_available(self):
        """D-01: When T-2 result is available within timeout, phone_result_stale=False in merged bundle."""
        pytest.skip("Stub — implementation in Plan 05")
        # Simulate T-1 bundle (without phone result yet)
        partial_bundle = PerceptionBundle(
            timestamp_ns=0,
            frame_id=5,
            face=FaceDetection(present=False, confidence=0.0, bbox_norm=None, face_size_px=0),
            landmarks=None,
            gaze=None,
            phone=PhoneDetectionOutput(detected=False, max_confidence=0.0, bbox_norm=None),
            phone_result_stale=True,  # initially stale
            inference_ms=10.0,
            lstm_hidden_state=None,
            lstm_reset_occurred=False,
        )
        # T-2 result arrives in time — merged result should have stale=False
        phone_result = PhoneDetectionOutput(detected=False, max_confidence=0.1, bbox_norm=None)
        # After merge logic (implementation will do this):
        merged_stale = False  # T-2 result was available
        assert merged_stale is False

    def test_t1_t2_merge_marks_stale_on_timeout(self):
        """D-01: When T-2 times out (>PHONE_THREAD_TIMEOUT_MS), phone_result_stale=True in merged bundle."""
        pytest.skip("Stub — implementation in Plan 05")
        # Simulate scenario where T-2 does not respond within 5ms
        # The merged bundle should use last valid result with stale=True
        timeout_ms = config.PHONE_THREAD_TIMEOUT_MS
        assert timeout_ms == 5  # PRD §3.3 specifies 5ms T-2 timeout
        # After merge logic (implementation will set stale=True on timeout):
        merged_stale = True
        assert merged_stale is True

    def test_t3_runs_full_pipeline_chain(self):
        """D-01: T-3 feeds PerceptionBundle through SignalProcessor->TemporalEngine->ScoringEngine->AlertStateMachine."""
        pytest.skip("Stub — implementation in Plan 05")
        from layer2_signals.signal_processor import SignalProcessor
        from layer3_temporal.temporal_engine import TemporalEngine
        from layer4_scoring.scoring_engine import ScoringEngine
        from layer5_alert.alert_state_machine import AlertStateMachine
        from layer4_scoring.messages import DistractionScore

        # Real (not mocked) pipeline chain
        sig_proc = SignalProcessor()
        temporal = TemporalEngine()
        scoring = ScoringEngine()
        asm = AlertStateMachine()

        bundle = _make_mock_perception_stack().infer(
            np.zeros((480, 640, 3), dtype=np.uint8),
            frame_id=0,
        )
        sig_frame = sig_proc.process(bundle)
        features = temporal.process(sig_frame)
        score = scoring.score(features)
        alert_cmd = asm.update(score)

        # Result is either None (no alert) or an AlertCommand
        assert alert_cmd is None or isinstance(alert_cmd, AlertCommand)

    def test_stop_event_terminates_all_threads(self):
        """D-05: Setting stop_event causes all pipeline threads to exit within 2s."""
        pytest.skip("Stub — implementation in Plan 05")
        stop_event = threading.Event()
        threads = []

        def _dummy_thread():
            while not stop_event.is_set():
                stop_event.wait(timeout=0.05)

        for _ in range(4):  # T-0, T-1, T-2, T-3
            t = threading.Thread(target=_dummy_thread, daemon=True)
            t.start()
            threads.append(t)

        stop_event.set()
        for t in threads:
            t.join(timeout=2.0)
            assert not t.is_alive(), "Thread did not terminate within 2s after stop_event"

    def test_shutdown_joins_all_threads(self):
        """D-05: All threads exit cleanly when source produces N frames then stops."""
        pytest.skip("Stub — implementation in Plan 05")
        stop_event = threading.Event()
        results = []

        def _finite_source(n_frames: int) -> None:
            for i in range(n_frames):
                results.append(_make_synthetic_frame(i))
            stop_event.set()

        t = threading.Thread(target=_finite_source, args=(5,), daemon=True)
        t.start()
        t.join(timeout=2.0)
        assert not t.is_alive()
        assert len(results) == 5
        assert stop_event.is_set()

    def test_lstm_hidden_state_persists_across_frames(self):
        """FR-1.3 / D-01: T-1 must pass bundle.lstm_hidden_state from frame N to frame N+1 infer() call."""
        pytest.skip("Stub — implementation in Plan 05")
        stack = _make_mock_perception_stack()

        # Simulate hidden state returned from frame 0
        fake_hidden = (np.zeros((1, 64), dtype=np.float32), np.ones((1, 64), dtype=np.float32))
        first_bundle = PerceptionBundle(
            timestamp_ns=0,
            frame_id=0,
            face=FaceDetection(present=True, confidence=0.95, bbox_norm=(0.2, 0.2, 0.4, 0.4), face_size_px=160),
            landmarks=None,
            gaze=None,
            phone=PhoneDetectionOutput(detected=False, max_confidence=0.0, bbox_norm=None),
            phone_result_stale=False,
            inference_ms=10.0,
            lstm_hidden_state=fake_hidden,
            lstm_reset_occurred=False,
        )
        stack.infer.return_value = first_bundle

        frame0 = _make_synthetic_frame(0)
        result0 = stack.infer(frame0.data, frame_id=0, hidden_state=None)

        # T-1 loop must feed result0.lstm_hidden_state into the next call
        frame1 = _make_synthetic_frame(1)
        stack.infer(frame1.data, frame_id=1, hidden_state=result0.lstm_hidden_state)

        second_call_kwargs = stack.infer.call_args_list[1]
        # Verify hidden_state was passed from result0 to frame1's infer call
        assert second_call_kwargs[1].get('hidden_state') is fake_hidden or \
               (len(second_call_kwargs[0]) > 2 and second_call_kwargs[0][2] is fake_hidden)

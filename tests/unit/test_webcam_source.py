"""test_webcam_source.py — TDD stub tests for WebcamSource.

PRD §FR-0.1–FR-0.4: VideoSource requirements.
Decision D-02 (06-CONTEXT.md): WebcamSource behaviour spec.

All tests skip until layer0_video.webcam_source is implemented (Plan 02).
"""

from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pytest
from unittest.mock import MagicMock, patch, call

_IMPL_MISSING = False
try:
    from layer0_video.webcam_source import WebcamSource, SourceUnavailableError
    from layer0_video.messages import RawFrame
except ImportError:
    _IMPL_MISSING = True

import config

pytestmark = pytest.mark.skipif(
    _IMPL_MISSING,
    reason="WebcamSource not yet implemented — will be implemented in Plan 02",
)


# ═══════════════════════════════════════════════════════════════════════════════
# TestWebcamSource
# PRD §FR-0.1: Accepts device index (Mac: integer index 0)
# PRD §FR-0.2: Sets resolution via CAP_PROP_FRAME_WIDTH/HEIGHT/FPS
# PRD §FR-0.3: Raises SourceUnavailableError if device cannot be opened within timeout
# PRD §FR-0.4: release() closes the capture device
# ═══════════════════════════════════════════════════════════════════════════════

class TestWebcamSource:

    def test_init_opens_device_index_zero(self):
        """FR-0.1: Constructor must call VideoCapture(0) for default Mac device."""
        pytest.skip("Stub — implementation in Plan 02")
        with patch('cv2.VideoCapture') as mock_capture_cls:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_capture_cls.return_value = mock_cap

            _ws = WebcamSource(device_index=0)

            mock_capture_cls.assert_called_once_with(0)

    def test_init_sets_resolution(self):
        """FR-0.2: Constructor sets CAP_PROP_FRAME_WIDTH=1280, HEIGHT=720, FPS=30 via cap.set()."""
        pytest.skip("Stub — implementation in Plan 02")
        import cv2
        with patch('cv2.VideoCapture') as mock_capture_cls:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_capture_cls.return_value = mock_cap

            _ws = WebcamSource(device_index=0)

            set_calls = mock_cap.set.call_args_list
            props_set = {c[0][0]: c[0][1] for c in set_calls}
            assert props_set.get(cv2.CAP_PROP_FRAME_WIDTH) == config.CAPTURE_WIDTH
            assert props_set.get(cv2.CAP_PROP_FRAME_HEIGHT) == config.CAPTURE_HEIGHT
            assert props_set.get(cv2.CAP_PROP_FPS) == config.CAPTURE_FPS

    def test_init_raises_source_unavailable_on_timeout(self):
        """FR-0.3: SourceUnavailableError raised if isOpened() is always False within timeout."""
        pytest.skip("Stub — implementation in Plan 02")
        with patch('cv2.VideoCapture') as mock_capture_cls:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = False
            mock_capture_cls.return_value = mock_cap

            # Monkeypatch time.monotonic to simulate fast-forward past timeout
            import time
            _calls = [0]
            _start = time.monotonic()

            def _fast_monotonic():
                _calls[0] += 1
                return _start + (_calls[0] * 2.0)  # jumps 2s per call → exceeds 5s timeout fast

            with patch('time.monotonic', side_effect=_fast_monotonic):
                with pytest.raises(SourceUnavailableError):
                    WebcamSource(device_index=0)

    def test_read_returns_raw_frame(self):
        """FR-0.1: read() returns a RawFrame with correct metadata when cap.read() succeeds."""
        pytest.skip("Stub — implementation in Plan 02")
        dummy_frame = np.zeros((config.CAPTURE_HEIGHT, config.CAPTURE_WIDTH, 3), dtype=np.uint8)
        with patch('cv2.VideoCapture') as mock_capture_cls:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.read.return_value = (True, dummy_frame)
            mock_capture_cls.return_value = mock_cap

            ws = WebcamSource(device_index=0)
            result = ws.read()

            assert isinstance(result, RawFrame)
            assert result.source_type == 'webcam'
            assert result.frame_id == 0
            assert result.width == config.CAPTURE_WIDTH
            assert result.height == config.CAPTURE_HEIGHT
            assert result.channels == 3

    def test_read_monotonic_frame_id(self):
        """FR-0.1: Successive read() calls produce frame_id 0, 1, 2 (monotonically increasing)."""
        pytest.skip("Stub — implementation in Plan 02")
        dummy_frame = np.zeros((config.CAPTURE_HEIGHT, config.CAPTURE_WIDTH, 3), dtype=np.uint8)
        with patch('cv2.VideoCapture') as mock_capture_cls:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.read.return_value = (True, dummy_frame)
            mock_capture_cls.return_value = mock_cap

            ws = WebcamSource(device_index=0)
            ids = [ws.read().frame_id for _ in range(3)]
            assert ids == [0, 1, 2]

    def test_read_returns_none_on_failure(self):
        """FR-0.3 / D-02: Returns None (not raises) when cap.read() returns success=False."""
        pytest.skip("Stub — implementation in Plan 02")
        with patch('cv2.VideoCapture') as mock_capture_cls:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.read.return_value = (False, None)
            mock_capture_cls.return_value = mock_cap

            ws = WebcamSource(device_index=0)
            result = ws.read()
            assert result is None

    def test_release_calls_cap_release(self):
        """FR-0.4: release() must call cap.release() to close the device."""
        pytest.skip("Stub — implementation in Plan 02")
        with patch('cv2.VideoCapture') as mock_capture_cls:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_capture_cls.return_value = mock_cap

            ws = WebcamSource(device_index=0)
            ws.release()
            mock_cap.release.assert_called_once()

    def test_read_timestamp_uses_monotonic_ns(self):
        """D-02: timestamp_ns field is a positive integer (from time.monotonic_ns)."""
        pytest.skip("Stub — implementation in Plan 02")
        dummy_frame = np.zeros((config.CAPTURE_HEIGHT, config.CAPTURE_WIDTH, 3), dtype=np.uint8)
        with patch('cv2.VideoCapture') as mock_capture_cls:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.read.return_value = (True, dummy_frame)
            mock_capture_cls.return_value = mock_cap

            ws = WebcamSource(device_index=0)
            result = ws.read()
            assert isinstance(result.timestamp_ns, int)
            assert result.timestamp_ns > 0

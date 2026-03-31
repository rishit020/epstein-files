# layer0_video/webcam_source.py — Mac webcam video source
# PRD §FR-0.1–FR-0.4: VideoSource requirements
# Decision D-02 (06-CONTEXT.md): WebcamSource behaviour spec

from __future__ import annotations

import logging
import time

import cv2
import numpy as np

import config
from layer0_video.messages import RawFrame

_log = logging.getLogger(__name__)


class SourceUnavailableError(Exception):
    """Raised when video source cannot be opened within timeout."""
    pass


class WebcamSource:
    """OpenCV-backed Mac webcam source that emits RawFrame objects.

    PRD §FR-0.1: Accepts integer device index (Mac default: 0).
    PRD §FR-0.2: Sets resolution and FPS via cap.set().
    PRD §FR-0.3: Raises SourceUnavailableError if camera not available within 5s.
    PRD §FR-0.4: release() closes the capture device without resource leaks.
    """

    def __init__(self, device_index: int = 0) -> None:
        self._cap = cv2.VideoCapture(device_index)

        # FR-0.2: Configure resolution and frame rate from config — no magic numbers
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAPTURE_WIDTH)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAPTURE_HEIGHT)
        self._cap.set(cv2.CAP_PROP_FPS, config.CAPTURE_FPS)

        # FR-0.3: Poll until opened or timeout
        _OPEN_TIMEOUT_S = 5.0
        _POLL_INTERVAL_S = 0.1
        deadline = time.monotonic() + _OPEN_TIMEOUT_S
        while time.monotonic() < deadline:
            if self._cap.isOpened():
                break
            time.sleep(_POLL_INTERVAL_S)
        else:
            self._cap.release()
            raise SourceUnavailableError(
                f"Camera {device_index} not available after {_OPEN_TIMEOUT_S}s"
            )

        self._frame_id: int = 0
        _log.info(
            "WebcamSource opened: device=%d, resolution=%dx%d@%dfps",
            device_index,
            config.CAPTURE_WIDTH,
            config.CAPTURE_HEIGHT,
            config.CAPTURE_FPS,
        )

    def read(self) -> RawFrame | None:
        """Capture one frame and return a RawFrame, or None on failure.

        Never raises — all exceptions are caught and logged.
        """
        try:
            ret, frame = self._cap.read()
            if not ret or frame is None:
                _log.debug("cap.read() failed frame_id=%d", self._frame_id)
                return None

            ts = time.monotonic_ns()
            h, w = frame.shape[:2]
            raw = RawFrame(
                timestamp_ns=ts,
                frame_id=self._frame_id,
                width=w,
                height=h,
                channels=frame.shape[2],
                data=frame,
                source_type="webcam",
            )
            self._frame_id += 1
            return raw
        except Exception as exc:
            _log.error("WebcamSource.read() error: %s", exc)
            return None

    def release(self) -> None:
        """Close the capture device and release resources."""
        if self._cap is not None:
            self._cap.release()
            _log.info("WebcamSource released")

# layer0_video/messages.py — Layer 0 output message
# PRD §4.1 — RawFrame: Layer 0 → Layer 1

from dataclasses import dataclass
import numpy as np


@dataclass
class RawFrame:
    """Output of VideoSource — carries a single captured frame with metadata.

    PRD §4.1: Layer 0 → Layer 1 message.
    """
    timestamp_ns: int     # V4L2 buffer timestamp (tv_sec * 1e9 + tv_usec * 1e3).
                          # Falls back to Python monotonic if V4L2 timestamp unavailable.
    frame_id: int         # Monotonically increasing; resets on source restart.
    width: int            # Pixels — 1280 in production (IMX219 1280×720 mode).
    height: int           # Pixels — 720 in production.
    channels: int         # Always 3 (BGR24 — post NV12 conversion).
    data: np.ndarray      # Shape: (height, width, 3), dtype: uint8, BGR channel order.
    source_type: str      # 'imx219_v4l2' | 'webcam' | 'file'

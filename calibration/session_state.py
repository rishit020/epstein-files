# calibration/session_state.py — Session State JSON Persistence
# PRD §24 — Session State Persistence
#
# Handles reading and writing of calibration/session_state.json.
# Only the fields listed in PRD §24.1 are persisted.
#
# Schema version: "2.0" (PRD §23.3)

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import config

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = "2.0"


@dataclass
class SessionState:
    """In-memory representation of session_state.json (PRD §23.3, §24.1)."""

    schema_version: str
    calibrated_at: str           # ISO-8601 UTC timestamp
    vehicle_vin: str             # OBD-II VIN (empty string if unavailable)
    neutral_yaw_offset: float    # degrees
    neutral_pitch_offset: float  # degrees
    baseline_ear: float
    close_threshold: float
    calibration_complete: bool
    frames_collected: int
    pose_std_yaw: float
    pose_std_pitch: float


def load_session_state(path: str = config.NEUTRAL_POSE_FILE) -> SessionState | None:
    """Load and validate session_state.json.

    Returns:
        SessionState if the file exists and is valid, None otherwise.
        Logs a warning on any parse or validation error.
    """
    if not os.path.exists(path):
        logger.info("CALIBRATION_REQUIRED: no session_state.json at %s", path)
        return None

    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw: dict[str, Any] = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("SESSION_STATE_LOAD_ERROR: %s — %s", path, exc)
        return None

    try:
        state = SessionState(
            schema_version=str(raw["schema_version"]),
            calibrated_at=str(raw["calibrated_at"]),
            vehicle_vin=str(raw.get("vehicle_vin", "")),
            neutral_yaw_offset=float(raw["neutral_yaw_offset"]),
            neutral_pitch_offset=float(raw["neutral_pitch_offset"]),
            baseline_ear=float(raw["baseline_ear"]),
            close_threshold=float(raw["close_threshold"]),
            calibration_complete=bool(raw["calibration_complete"]),
            frames_collected=int(raw["frames_collected"]),
            pose_std_yaw=float(raw["pose_std_yaw"]),
            pose_std_pitch=float(raw["pose_std_pitch"]),
        )
    except (KeyError, ValueError, TypeError) as exc:
        logger.warning("SESSION_STATE_SCHEMA_ERROR: %s — %s", path, exc)
        return None

    if state.schema_version != _SCHEMA_VERSION:
        logger.warning(
            "SESSION_STATE_VERSION_MISMATCH: expected %s, got %s",
            _SCHEMA_VERSION,
            state.schema_version,
        )
        return None

    return state


def save_session_state(
    state: SessionState,
    path: str = config.NEUTRAL_POSE_FILE,
) -> bool:
    """Write session_state.json atomically.

    Writes to a temp file first, then renames to avoid partial writes.

    Returns:
        True on success, False on any IO error (logs the error).
    """
    payload = {
        "schema_version": state.schema_version,
        "calibrated_at": state.calibrated_at,
        "vehicle_vin": state.vehicle_vin,
        "neutral_yaw_offset": state.neutral_yaw_offset,
        "neutral_pitch_offset": state.neutral_pitch_offset,
        "baseline_ear": state.baseline_ear,
        "close_threshold": state.close_threshold,
        "calibration_complete": state.calibration_complete,
        "frames_collected": state.frames_collected,
        "pose_std_yaw": state.pose_std_yaw,
        "pose_std_pitch": state.pose_std_pitch,
    }

    tmp_path = path + ".tmp"
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=4)
        os.replace(tmp_path, path)
        logger.info("SESSION_STATE_SAVED: %s", path)
        return True
    except OSError as exc:
        logger.error("SESSION_STATE_SAVE_ERROR: %s — %s", path, exc)
        # Clean up temp file if it exists
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        return False


def make_session_state(
    vehicle_vin: str,
    neutral_yaw_offset: float,
    neutral_pitch_offset: float,
    baseline_ear: float,
    close_threshold: float,
    frames_collected: int,
    pose_std_yaw: float,
    pose_std_pitch: float,
) -> SessionState:
    """Construct a new SessionState with current timestamp."""
    return SessionState(
        schema_version=_SCHEMA_VERSION,
        calibrated_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        vehicle_vin=vehicle_vin,
        neutral_yaw_offset=neutral_yaw_offset,
        neutral_pitch_offset=neutral_pitch_offset,
        baseline_ear=baseline_ear,
        close_threshold=close_threshold,
        calibration_complete=True,
        frames_collected=frames_collected,
        pose_std_yaw=pose_std_yaw,
        pose_std_pitch=pose_std_pitch,
    )


def vin_matches(state: SessionState | None, current_vin: str) -> bool:
    """Return True if the persisted VIN matches the current vehicle VIN.

    An empty VIN on either side always returns False so calibration re-runs
    whenever VIN is unavailable (safe default per PRD §23.1).
    """
    if state is None:
        return False
    if not state.vehicle_vin or not current_vin:
        return False
    return state.vehicle_vin == current_vin

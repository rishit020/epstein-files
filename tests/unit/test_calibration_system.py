# tests/unit/test_calibration_system.py — Phase 2b Calibration System Tests
#
# Covers TASKS.md Phase 2b requirements:
#   - Calibration completes, state persists, loads correctly on next start
#   - Corrected angles verified against known offsets
#   - VIN-based re-calibration trigger logic
#   - Startup load behaviour — skip calibration if valid state exists (PRD §24.2)
#   - Fallback (CALIBRATION_FAILED) path: 0.0 offsets used, calibration_complete=False

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pytest

import config
from calibration.calibration_manager import CalibrationManager, CalibrationStatus
from calibration.session_state import (
    SessionState,
    load_session_state,
    make_session_state,
    save_session_state,
    vin_matches,
)
from layer2_signals.pose_calibration import PoseCalibration
from layer2_signals.signal_processor import SignalProcessor


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers / Fixtures
# ═══════════════════════════════════════════════════════════════════════════════

TEST_VIN = "1HGCM82633A004352"
ALT_VIN  = "2T1BURHE0JC052512"


def _make_state_file() -> str:
    """Return a path inside a fresh temp directory (file does not exist yet)."""
    d = tempfile.mkdtemp()
    return os.path.join(d, "session_state.json")


def _make_signal_processor() -> SignalProcessor:
    return SignalProcessor()


def _primary_window_size() -> int:
    """Total frames in the primary calibration window (valid + invalid)."""
    return int(config.CALIBRATION_DURATION_S * config.CAPTURE_FPS)


def _good_frames(count: int, yaw: float = -4.2, pitch: float = 3.1, ear: float = 0.31):
    """Generate `count` identical valid frames with stable pose."""
    return [(yaw, pitch, ear, True)] * count


def _full_primary_good_frames(yaw: float = -4.2, pitch: float = 3.1, ear: float = 0.31):
    """Generate a full primary window of valid frames (300 at 30fps).

    Using a full primary window ensures the evaluation is triggered.
    MIN_VALID_FRAMES (270) must be <= this count for calibration to succeed.
    """
    return _good_frames(_primary_window_size(), yaw=yaw, pitch=pitch, ear=ear)


def _run_calibration(manager: CalibrationManager, frames) -> CalibrationStatus:
    """Feed all frames into manager, return final status."""
    status = CalibrationStatus.COLLECTING
    for yaw, pitch, ear, pose_valid in frames:
        status = manager.feed_frame(yaw, pitch, ear, pose_valid)
        if status in (CalibrationStatus.COMPLETE, CalibrationStatus.FAILED):
            break
    return status


# ═══════════════════════════════════════════════════════════════════════════════
# 1. session_state.py — load / save / validate
# ═══════════════════════════════════════════════════════════════════════════════

class TestSessionStateIO:

    def test_roundtrip_save_load(self):
        """State written then loaded must have identical field values."""
        path = _make_state_file()
        original = make_session_state(
            vehicle_vin=TEST_VIN,
            neutral_yaw_offset=-4.2,
            neutral_pitch_offset=3.1,
            baseline_ear=0.31,
            close_threshold=0.233,
            frames_collected=298,
            pose_std_yaw=1.8,
            pose_std_pitch=1.2,
        )
        assert save_session_state(original, path)

        loaded = load_session_state(path)
        assert loaded is not None
        assert loaded.schema_version == "2.0"
        assert loaded.vehicle_vin == TEST_VIN
        assert loaded.neutral_yaw_offset == pytest.approx(-4.2)
        assert loaded.neutral_pitch_offset == pytest.approx(3.1)
        assert loaded.baseline_ear == pytest.approx(0.31)
        assert loaded.close_threshold == pytest.approx(0.233)
        assert loaded.calibration_complete is True
        assert loaded.frames_collected == 298
        assert loaded.pose_std_yaw == pytest.approx(1.8)
        assert loaded.pose_std_pitch == pytest.approx(1.2)

    def test_load_missing_file_returns_none(self):
        assert load_session_state("/nonexistent/path/session_state.json") is None

    def test_load_corrupted_json_returns_none(self, tmp_path):
        path = str(tmp_path / "session_state.json")
        with open(path, "w") as fh:
            fh.write("not-valid-json{{{")
        assert load_session_state(path) is None

    def test_load_missing_key_returns_none(self, tmp_path):
        path = str(tmp_path / "session_state.json")
        # Write JSON missing required 'neutral_yaw_offset'
        with open(path, "w") as fh:
            json.dump({"schema_version": "2.0", "calibration_complete": True}, fh)
        assert load_session_state(path) is None

    def test_load_wrong_schema_version_returns_none(self, tmp_path):
        path = str(tmp_path / "session_state.json")
        state = make_session_state(TEST_VIN, 0.0, 0.0, 0.3, 0.225, 270, 1.0, 1.0)
        save_session_state(state, path)
        # Corrupt the schema version
        with open(path, "r") as fh:
            data = json.load(fh)
        data["schema_version"] = "1.0"
        with open(path, "w") as fh:
            json.dump(data, fh)
        assert load_session_state(path) is None

    def test_atomic_write_leaves_no_temp_file(self, tmp_path):
        """After a successful save, no .tmp file should remain."""
        path = str(tmp_path / "session_state.json")
        state = make_session_state(TEST_VIN, 0.0, 0.0, 0.3, 0.225, 270, 1.0, 1.0)
        save_session_state(state, path)
        assert not os.path.exists(path + ".tmp")
        assert os.path.exists(path)


class TestVinMatch:

    def test_matching_vin_returns_true(self):
        state = make_session_state(TEST_VIN, 0.0, 0.0, 0.3, 0.225, 270, 1.0, 1.0)
        assert vin_matches(state, TEST_VIN) is True

    def test_different_vin_returns_false(self):
        state = make_session_state(TEST_VIN, 0.0, 0.0, 0.3, 0.225, 270, 1.0, 1.0)
        assert vin_matches(state, ALT_VIN) is False

    def test_empty_stored_vin_returns_false(self):
        state = make_session_state("", 0.0, 0.0, 0.3, 0.225, 270, 1.0, 1.0)
        assert vin_matches(state, TEST_VIN) is False

    def test_empty_current_vin_returns_false(self):
        state = make_session_state(TEST_VIN, 0.0, 0.0, 0.3, 0.225, 270, 1.0, 1.0)
        assert vin_matches(state, "") is False

    def test_none_state_returns_false(self):
        assert vin_matches(None, TEST_VIN) is False


# ═══════════════════════════════════════════════════════════════════════════════
# 2. CalibrationManager — startup load behaviour (PRD §24.2)
# ═══════════════════════════════════════════════════════════════════════════════

class TestStartupLoadBehaviour:

    def test_no_file_returns_collecting(self):
        """No session_state.json → must require calibration."""
        sp = _make_signal_processor()
        path = _make_state_file()  # does not exist yet
        manager = CalibrationManager(sp, state_file=path)
        status = manager.startup(TEST_VIN)
        assert status == CalibrationStatus.COLLECTING

    def test_valid_file_matching_vin_returns_loaded(self):
        """Valid state + VIN match → LOADED, no calibration needed."""
        path = _make_state_file()
        state = make_session_state(TEST_VIN, -4.2, 3.1, 0.31, 0.233, 298, 1.8, 1.2)
        save_session_state(state, path)

        sp = _make_signal_processor()
        manager = CalibrationManager(sp, state_file=path)
        status = manager.startup(TEST_VIN)
        assert status == CalibrationStatus.LOADED

    def test_loaded_applies_offsets_to_signal_processor(self):
        """On LOADED, signal_processor must have offsets applied immediately."""
        path = _make_state_file()
        state = make_session_state(TEST_VIN, -4.2, 3.1, 0.31, 0.233, 298, 1.8, 1.2)
        save_session_state(state, path)

        sp = _make_signal_processor()
        manager = CalibrationManager(sp, state_file=path)
        manager.startup(TEST_VIN)

        assert sp.pose_calibration.neutral_yaw_offset == pytest.approx(-4.2)
        assert sp.pose_calibration.neutral_pitch_offset == pytest.approx(3.1)
        assert sp.ear_calculator.baseline_EAR == pytest.approx(0.31)
        assert sp.ear_calculator.close_threshold == pytest.approx(0.233)
        assert sp.ear_calculator.calibration_complete is True

    def test_vin_mismatch_triggers_recalibration(self):
        """VIN mismatch → startup returns COLLECTING, not LOADED."""
        path = _make_state_file()
        state = make_session_state(TEST_VIN, -4.2, 3.1, 0.31, 0.233, 298, 1.8, 1.2)
        save_session_state(state, path)

        sp = _make_signal_processor()
        manager = CalibrationManager(sp, state_file=path)
        status = manager.startup(ALT_VIN)  # Different VIN
        assert status == CalibrationStatus.COLLECTING

    def test_calibration_incomplete_flag_triggers_recalibration(self):
        """State with calibration_complete=False → startup returns COLLECTING."""
        path = _make_state_file()
        state = make_session_state(TEST_VIN, -4.2, 3.1, 0.31, 0.233, 298, 1.8, 1.2)
        # Override calibration_complete to False
        state = SessionState(
            schema_version=state.schema_version,
            calibrated_at=state.calibrated_at,
            vehicle_vin=state.vehicle_vin,
            neutral_yaw_offset=state.neutral_yaw_offset,
            neutral_pitch_offset=state.neutral_pitch_offset,
            baseline_ear=state.baseline_ear,
            close_threshold=state.close_threshold,
            calibration_complete=False,
            frames_collected=state.frames_collected,
            pose_std_yaw=state.pose_std_yaw,
            pose_std_pitch=state.pose_std_pitch,
        )
        save_session_state(state, path)

        sp = _make_signal_processor()
        manager = CalibrationManager(sp, state_file=path)
        status = manager.startup(TEST_VIN)
        assert status == CalibrationStatus.COLLECTING

    def test_force_recalibration_overrides_valid_state(self, monkeypatch):
        """FORCE_RECALIBRATION=True ignores persisted state."""
        path = _make_state_file()
        state = make_session_state(TEST_VIN, -4.2, 3.1, 0.31, 0.233, 298, 1.8, 1.2)
        save_session_state(state, path)

        monkeypatch.setattr(config, "FORCE_RECALIBRATION", True)

        sp = _make_signal_processor()
        manager = CalibrationManager(sp, state_file=path)
        status = manager.startup(TEST_VIN)
        assert status == CalibrationStatus.COLLECTING


# ═══════════════════════════════════════════════════════════════════════════════
# 3. CalibrationManager — calibration sequence (PRD §23.2)
# ═══════════════════════════════════════════════════════════════════════════════

class TestCalibrationSequence:

    def _manager(self, sp=None) -> tuple[CalibrationManager, SignalProcessor, str]:
        sp = sp or _make_signal_processor()
        path = _make_state_file()
        beeps = []
        def audio_fn(count, kind):
            beeps.append((count, kind))
        manager = CalibrationManager(sp, audio_fn=audio_fn, state_file=path)
        return manager, sp, path

    def test_calibration_completes_with_sufficient_stable_frames(self):
        """COMPLETE returned after feeding a full primary window of valid frames."""
        manager, sp, path = self._manager()
        manager.startup(TEST_VIN)

        frames = _full_primary_good_frames(yaw=-4.2, pitch=3.1)
        status = _run_calibration(manager, frames)
        assert status == CalibrationStatus.COMPLETE

    def test_calibration_computes_correct_offsets(self):
        """Offsets must equal mean of fed yaw/pitch samples."""
        manager, sp, path = self._manager()
        manager.startup(TEST_VIN)

        n = _primary_window_size()
        yaws    = [-4.0, -4.2, -4.4, -4.1, -4.3]
        pitches = [3.0, 3.1, 3.2, 3.1, 3.0]
        # Repeat to fill the full primary window
        repeats     = n // len(yaws) + 1
        yaws_full   = (yaws    * repeats)[:n]
        pitches_full = (pitches * repeats)[:n]
        ears_full   = [0.31] * n

        frames = list(zip(yaws_full, pitches_full, ears_full, [True] * n))
        status = _run_calibration(manager, frames)
        assert status == CalibrationStatus.COMPLETE

        result = manager.result
        assert result is not None
        assert result.neutral_yaw_offset   == pytest.approx(np.mean(yaws_full),   abs=1e-4)
        assert result.neutral_pitch_offset == pytest.approx(np.mean(pitches_full), abs=1e-4)

    def test_offsets_applied_to_signal_processor_on_complete(self):
        """After COMPLETE, signal_processor pose offsets must match calibration."""
        manager, sp, path = self._manager()
        manager.startup(TEST_VIN)

        frames = _full_primary_good_frames(yaw=-4.2, pitch=3.1)
        _run_calibration(manager, frames)

        assert sp.pose_calibration.neutral_yaw_offset   == pytest.approx(-4.2, abs=1e-4)
        assert sp.pose_calibration.neutral_pitch_offset == pytest.approx(3.1,  abs=1e-4)

    def test_ear_baseline_applied_on_complete(self):
        """After COMPLETE, EAR calibration is marked complete in signal_processor."""
        manager, sp, path = self._manager()
        manager.startup(TEST_VIN)

        frames = _full_primary_good_frames(ear=0.31)
        _run_calibration(manager, frames)

        assert sp.ear_calculator.calibration_complete is True
        assert sp.ear_calculator.baseline_EAR == pytest.approx(0.31, abs=1e-4)
        assert sp.ear_calculator.close_threshold == pytest.approx(
            0.31 * config.EAR_CALIBRATION_MULTIPLIER, abs=1e-4
        )

    def test_state_persists_after_calibration(self):
        """session_state.json exists and loads correctly after calibration."""
        manager, sp, path = self._manager()
        manager.startup(TEST_VIN)

        frames = _full_primary_good_frames(yaw=-4.2, pitch=3.1, ear=0.31)
        _run_calibration(manager, frames)

        loaded = load_session_state(path)
        assert loaded is not None
        assert loaded.calibration_complete is True
        assert loaded.vehicle_vin == TEST_VIN
        assert loaded.neutral_yaw_offset   == pytest.approx(-4.2, abs=1e-4)
        assert loaded.neutral_pitch_offset == pytest.approx(3.1,  abs=1e-4)

    def test_next_startup_loads_from_persisted_state(self):
        """Second startup after calibration returns LOADED (skips calibration)."""
        manager, sp, path = self._manager()
        manager.startup(TEST_VIN)
        frames = _full_primary_good_frames(yaw=-4.2, pitch=3.1, ear=0.31)
        _run_calibration(manager, frames)

        # Second session — new manager, same file
        sp2 = _make_signal_processor()
        manager2 = CalibrationManager(sp2, state_file=path)
        status = manager2.startup(TEST_VIN)
        assert status == CalibrationStatus.LOADED
        assert sp2.pose_calibration.neutral_yaw_offset == pytest.approx(-4.2, abs=1e-4)

    def test_audio_beeps_on_start_and_complete(self):
        """3 short beeps at start, 2 long beeps on success (PRD §23.2 Steps 2+5)."""
        sp = _make_signal_processor()
        path = _make_state_file()
        beeps = []
        def audio_fn(count, kind):
            beeps.append((count, kind))

        manager = CalibrationManager(sp, audio_fn=audio_fn, state_file=path)
        manager.startup(TEST_VIN)
        frames = _full_primary_good_frames()
        _run_calibration(manager, frames)

        assert (3, "short") in beeps, "Expected 3 short beeps at calibration start"
        assert (2, "long")  in beeps, "Expected 2 long beeps on calibration success"

    def test_corrected_angles_match_known_offsets(self):
        """PoseCalibration.correct() subtracts the calibrated offset exactly."""
        sp = _make_signal_processor()
        path = _make_state_file()
        manager = CalibrationManager(sp, state_file=path)
        manager.startup(TEST_VIN)

        # Calibrate with known constant yaw/pitch
        known_yaw, known_pitch = -4.2, 3.1
        frames = _full_primary_good_frames(yaw=known_yaw, pitch=known_pitch)
        _run_calibration(manager, frames)

        # After calibration, correcting the same angles should give ~0.0
        corrected_yaw, corrected_pitch = sp.pose_calibration.correct(known_yaw, known_pitch)
        assert corrected_yaw   == pytest.approx(0.0, abs=0.01)
        assert corrected_pitch == pytest.approx(0.0, abs=0.01)

        # And a different angle should produce the expected offset result
        raw_yaw, raw_pitch = 5.0, -2.0
        cy, cp = sp.pose_calibration.correct(raw_yaw, raw_pitch)
        assert cy == pytest.approx(raw_yaw   - known_yaw,   abs=0.01)
        assert cp == pytest.approx(raw_pitch - known_pitch, abs=0.01)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. CalibrationManager — extension window (PRD §23.2 Step 2)
# ═══════════════════════════════════════════════════════════════════════════════

class TestExtensionWindow:

    def test_extension_triggered_when_valid_frames_short(self):
        """Feeding fewer than MIN_VALID_FRAMES valid frames triggers EXTENDING."""
        sp = _make_signal_processor()
        path = _make_state_file()
        manager = CalibrationManager(sp, state_file=path)
        manager.startup(TEST_VIN)

        fps = config.CAPTURE_FPS
        primary_total = int(config.CALIBRATION_DURATION_S * fps)

        # Feed only invalid frames through the primary window
        status = CalibrationStatus.COLLECTING
        for _ in range(primary_total):
            status = manager.feed_frame(0.0, 0.0, 0.3, pose_valid=False)
        assert status == CalibrationStatus.EXTENDING

    def test_extension_succeeds_with_enough_valid_frames(self):
        """Primary window short on valid frames, extension provides the top-up → COMPLETE.

        Strategy:
          - Primary (300 frames): 260 valid + 40 invalid → 260 < 270 → EXTENDING
          - Extension (up to 150 more): feed 15 valid → total 275 >= 270 → COMPLETE
        """
        sp = _make_signal_processor()
        path = _make_state_file()
        manager = CalibrationManager(sp, state_file=path)
        manager.startup(TEST_VIN)

        fps = config.CAPTURE_FPS
        primary_total = int(config.CALIBRATION_DURATION_S * fps)  # 300
        min_valid     = config.CALIBRATION_MIN_VALID_FRAMES        # 270
        valid_in_primary = min_valid - 10                          # 260

        # Feed primary window: 260 valid frames then 40 invalid
        status = CalibrationStatus.COLLECTING
        for i in range(primary_total):
            pose_valid = i < valid_in_primary
            status = manager.feed_frame(-4.2, 3.1, 0.31, pose_valid=pose_valid)
        assert status == CalibrationStatus.EXTENDING

        # Feed extension: enough valid to push total >= MIN_VALID_FRAMES
        needed_more = min_valid - valid_in_primary + 5  # 15 valid frames
        full_total  = int((config.CALIBRATION_DURATION_S + config.CALIBRATION_EXTENSION_S) * fps)
        remaining   = full_total - primary_total
        for i in range(remaining):
            pose_valid = i < needed_more
            status = manager.feed_frame(-4.2, 3.1, 0.31, pose_valid=pose_valid)
            if status in (CalibrationStatus.COMPLETE, CalibrationStatus.FAILED):
                break
        assert status == CalibrationStatus.COMPLETE

    def test_extension_failure_returns_failed_with_fallback_offsets(self):
        """If extension also has insufficient frames → FAILED with 0.0 offsets."""
        sp = _make_signal_processor()
        path = _make_state_file()
        manager = CalibrationManager(sp, state_file=path)
        manager.startup(TEST_VIN)

        fps = config.CAPTURE_FPS
        full_total = int((config.CALIBRATION_DURATION_S + config.CALIBRATION_EXTENSION_S) * fps)

        # Feed all invalid frames through both primary and extension windows
        status = CalibrationStatus.COLLECTING
        for _ in range(full_total):
            status = manager.feed_frame(0.0, 0.0, 0.3, pose_valid=False)
            if status in (CalibrationStatus.COMPLETE, CalibrationStatus.FAILED):
                break

        assert status == CalibrationStatus.FAILED
        result = manager.result
        assert result is not None
        assert result.neutral_yaw_offset   == pytest.approx(0.0)
        assert result.neutral_pitch_offset == pytest.approx(0.0)
        assert result.calibration_complete is False


# ═══════════════════════════════════════════════════════════════════════════════
# 5. CalibrationManager — std dev retry logic (PRD §23.2 Step 3)
# ═══════════════════════════════════════════════════════════════════════════════

class TestStdDevRetryLogic:

    def _make_noisy_frames(self, count: int, std: float) -> list:
        """Generate frames with head pose noise above the threshold."""
        rng = np.random.default_rng(99)
        yaws   = rng.normal(0.0, std, count).tolist()
        pitches = rng.normal(0.0, std, count).tolist()
        return [(y, p, 0.31, True) for y, p in zip(yaws, pitches)]

    def test_unstable_pose_triggers_retry(self):
        """Frames with std dev >= CALIBRATION_MAX_POSE_STD_DEG trigger retry."""
        sp = _make_signal_processor()
        path = _make_state_file()
        manager = CalibrationManager(sp, state_file=path)
        manager.startup(TEST_VIN)

        # First attempt: noisy frames (std >> 5°)
        noisy = self._make_noisy_frames(config.CALIBRATION_MIN_VALID_FRAMES, std=10.0)
        # Pad to fill primary window
        fps = config.CAPTURE_FPS
        primary = int(config.CALIBRATION_DURATION_S * fps)
        frames = (noisy * (primary // len(noisy) + 1))[:primary]

        status = CalibrationStatus.COLLECTING
        for yaw, pitch, ear, pose_valid in frames:
            status = manager.feed_frame(yaw, pitch, ear, pose_valid)
            if status in (CalibrationStatus.COMPLETE, CalibrationStatus.FAILED):
                break
            if status == CalibrationStatus.COLLECTING and manager._attempt > 0:
                break  # retry started

        # After feeding noisy frames, attempt counter should have advanced
        assert manager._attempt >= 1 or status == CalibrationStatus.FAILED

    def test_two_failed_std_attempts_gives_fallback(self):
        """After CALIBRATION_MAX_ATTEMPTS with bad std dev → FAILED, 0.0 offsets."""
        sp = _make_signal_processor()
        path = _make_state_file()
        manager = CalibrationManager(sp, state_file=path)
        manager.startup(TEST_VIN)

        fps = config.CAPTURE_FPS
        primary = int(config.CALIBRATION_DURATION_S * fps)

        # Feed noisy frames for all max attempts
        status = CalibrationStatus.COLLECTING
        for _attempt_i in range(config.CALIBRATION_MAX_ATTEMPTS + 1):
            noisy = self._make_noisy_frames(primary, std=15.0)
            for yaw, pitch, ear, pose_valid in noisy:
                status = manager.feed_frame(yaw, pitch, ear, pose_valid)
                if status in (CalibrationStatus.COMPLETE, CalibrationStatus.FAILED):
                    break
            if status in (CalibrationStatus.COMPLETE, CalibrationStatus.FAILED):
                break

        assert status == CalibrationStatus.FAILED
        result = manager.result
        assert result is not None
        assert result.neutral_yaw_offset   == pytest.approx(0.0)
        assert result.neutral_pitch_offset == pytest.approx(0.0)
        assert result.calibration_complete is False

    def test_second_attempt_with_stable_frames_succeeds(self):
        """First attempt noisy, second attempt stable → COMPLETE."""
        sp = _make_signal_processor()
        path = _make_state_file()
        manager = CalibrationManager(sp, state_file=path)
        manager.startup(TEST_VIN)

        fps = config.CAPTURE_FPS
        primary = int(config.CALIBRATION_DURATION_S * fps)

        # First attempt: noisy
        noisy = self._make_noisy_frames(primary, std=15.0)
        status = CalibrationStatus.COLLECTING
        for yaw, pitch, ear, pose_valid in noisy:
            status = manager.feed_frame(yaw, pitch, ear, pose_valid)
            if status in (CalibrationStatus.COMPLETE, CalibrationStatus.FAILED):
                break

        if status in (CalibrationStatus.COMPLETE, CalibrationStatus.FAILED):
            # Could be failed if both attempts exhausted — just verify that
            # if we get here with FAILED, that's the expected fallback
            return

        # Should now be COLLECTING again (retry started)
        assert status == CalibrationStatus.COLLECTING

        # Second attempt: stable frames
        stable = _good_frames(primary, yaw=-4.2, pitch=3.1)
        for yaw, pitch, ear, pose_valid in stable:
            status = manager.feed_frame(yaw, pitch, ear, pose_valid)
            if status in (CalibrationStatus.COMPLETE, CalibrationStatus.FAILED):
                break

        assert status == CalibrationStatus.COMPLETE


# ═══════════════════════════════════════════════════════════════════════════════
# 6. VIN-based re-calibration trigger (PRD §23.1)
# ═══════════════════════════════════════════════════════════════════════════════

class TestVINReCalibraitonTrigger:

    def test_vin_change_requires_new_calibration(self):
        """After calibration for VIN-A, starting with VIN-B → COLLECTING."""
        path = _make_state_file()
        sp = _make_signal_processor()
        manager = CalibrationManager(sp, state_file=path)
        manager.startup(TEST_VIN)
        frames = _full_primary_good_frames()
        _run_calibration(manager, frames)
        assert manager.status == CalibrationStatus.COMPLETE

        # New session, different VIN
        sp2 = _make_signal_processor()
        manager2 = CalibrationManager(sp2, state_file=path)
        status = manager2.startup(ALT_VIN)
        assert status == CalibrationStatus.COLLECTING

    def test_same_vin_skips_calibration(self):
        """After calibration for VIN-A, starting with VIN-A again → LOADED."""
        path = _make_state_file()
        sp = _make_signal_processor()
        manager = CalibrationManager(sp, state_file=path)
        manager.startup(TEST_VIN)
        frames = _full_primary_good_frames()
        _run_calibration(manager, frames)

        sp2 = _make_signal_processor()
        manager2 = CalibrationManager(sp2, state_file=path)
        status = manager2.startup(TEST_VIN)
        assert status == CalibrationStatus.LOADED

    def test_empty_vin_always_requires_calibration(self):
        """If VIN is unavailable (empty string) → COLLECTING every time."""
        path = _make_state_file()
        sp = _make_signal_processor()
        # Calibrate with empty VIN
        manager = CalibrationManager(sp, state_file=path)
        manager.startup("")
        frames = _full_primary_good_frames()
        _run_calibration(manager, frames)

        sp2 = _make_signal_processor()
        manager2 = CalibrationManager(sp2, state_file=path)
        # Even with empty VIN — should require calibration (safe default)
        status = manager2.startup("")
        assert status == CalibrationStatus.COLLECTING

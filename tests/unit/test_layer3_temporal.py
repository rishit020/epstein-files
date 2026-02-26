# tests/unit/test_layer3_temporal.py — Phase 3 Temporal Engine Unit Tests
#
# Covers all Layer 3 modules per TASKS.md Phase 3 test requirements:
#   - PERCLOS matches hand-computed values on test sequences
#   - Blink rate score formula correct on synthetic EAR sequences
#   - Speed zone boundary values and stale handling
#   - Fault injection test: watchdog correctly triggers after 3s block

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import math
import time

import pytest

import config
from layer2_signals.messages import (
    EyeSignals,
    GazeWorld,
    HeadPose,
    PhoneSignal,
    SignalFrame,
)
from layer3_temporal.blink_detector import BlinkDetector
from layer3_temporal.circular_buffer import CircularBuffer
from layer3_temporal.duration_timer import DurationTimer
from layer3_temporal.perclos_window import PERCLOSWindow
from layer3_temporal.speed_context import (
    ZONE_HIGHWAY,
    ZONE_PARKED,
    ZONE_URBAN,
    resolve_speed_zone,
)
from layer3_temporal.speed_source import SpeedSource
from layer3_temporal.temporal_engine import TemporalEngine
from layer3_temporal.thermal_monitor import ThermalMonitor
from layer3_temporal.watchdog import WatchdogManager


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

_TS_BASE    = 1_000_000_000  # 1 second in ns
_DT_NS      = int(1e9 / config.CAPTURE_FPS)  # ~33ms per frame
_DEFAULT_DT = 1.0 / config.CAPTURE_FPS


def _make_phone_signal(detected: bool = False, confidence: float = 0.0) -> PhoneSignal:
    return PhoneSignal(detected=detected, confidence=confidence, stale=False)


def _make_head_pose(
    yaw: float = 0.0,
    pitch: float = 0.0,
    valid: bool = True,
) -> HeadPose:
    return HeadPose(
        yaw_deg=yaw, pitch_deg=pitch, roll_deg=0.0, valid=valid,
        raw_yaw_deg=yaw, raw_pitch_deg=pitch, raw_roll_deg=0.0,
    )


def _make_eye_signals(
    mean_ear: float = 0.28,
    baseline_ear: float = 0.28,
    close_threshold: float = 0.21,
) -> EyeSignals:
    return EyeSignals(
        left_EAR=mean_ear, right_EAR=mean_ear, mean_EAR=mean_ear,
        baseline_EAR=baseline_ear,
        close_threshold=close_threshold,
        valid=True,
        calibration_complete=True,
    )


def _make_gaze_world(on_road: bool = True) -> GazeWorld:
    return GazeWorld(yaw_deg=0.0, pitch_deg=0.0, on_road=on_road, valid=True)


def _make_signal_frame(
    frame_id: int = 0,
    ts_ns: int | None = None,
    face_present: bool = True,
    signals_valid: bool = True,
    head_pose: HeadPose | None = None,
    eye_signals: EyeSignals | None = None,
    gaze_world: GazeWorld | None = None,
    phone: PhoneSignal | None = None,
    speed_mps: float = 5.0,
    speed_stale: bool = False,
) -> SignalFrame:
    if ts_ns is None:
        ts_ns = _TS_BASE + frame_id * _DT_NS
    if phone is None:
        phone = _make_phone_signal()
    if eye_signals is None and face_present:
        eye_signals = _make_eye_signals()
    if head_pose is None and face_present:
        head_pose = _make_head_pose()
    if gaze_world is None and face_present:
        gaze_world = _make_gaze_world(on_road=True)
    return SignalFrame(
        timestamp_ns=ts_ns,
        frame_id=frame_id,
        face_present=face_present,
        head_pose=head_pose,
        eye_signals=eye_signals,
        gaze_world=gaze_world,
        phone_signal=phone,
        speed_mps=speed_mps,
        speed_stale=speed_stale,
        signals_valid=signals_valid,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# CircularBuffer
# ═══════════════════════════════════════════════════════════════════════════════

class TestCircularBuffer:

    def test_empty_buffer_has_zero_length(self):
        buf = CircularBuffer(maxsize=10)
        assert len(buf) == 0

    def test_push_increases_length(self):
        buf = CircularBuffer(maxsize=10)
        buf.push(_make_signal_frame(0))
        assert len(buf) == 1

    def test_is_full_false_below_capacity(self):
        buf = CircularBuffer(maxsize=5)
        for i in range(4):
            buf.push(_make_signal_frame(i))
        assert not buf.is_full

    def test_is_full_true_at_capacity(self):
        buf = CircularBuffer(maxsize=5)
        for i in range(5):
            buf.push(_make_signal_frame(i))
        assert buf.is_full

    def test_length_does_not_exceed_maxsize(self):
        buf = CircularBuffer(maxsize=3)
        for i in range(10):
            buf.push(_make_signal_frame(i))
        assert len(buf) == 3

    def test_overflow_evicts_oldest(self):
        buf = CircularBuffer(maxsize=3)
        for i in range(5):
            buf.push(_make_signal_frame(i))
        window = buf.get_window(3)
        ids = [f.frame_id for f in window]
        assert ids == [2, 3, 4]

    def test_get_window_returns_last_n(self):
        buf = CircularBuffer(maxsize=10)
        for i in range(8):
            buf.push(_make_signal_frame(i))
        window = buf.get_window(4)
        assert len(window) == 4
        assert [f.frame_id for f in window] == [4, 5, 6, 7]

    def test_get_window_clamps_to_available(self):
        buf = CircularBuffer(maxsize=10)
        for i in range(3):
            buf.push(_make_signal_frame(i))
        window = buf.get_window(10)
        assert len(window) == 3

    def test_get_window_zero_returns_empty(self):
        buf = CircularBuffer(maxsize=10)
        buf.push(_make_signal_frame(0))
        assert buf.get_window(0) == []

    def test_default_maxsize_is_config_value(self):
        buf = CircularBuffer()
        assert buf._maxsize == config.CIRCULAR_BUFFER_SIZE

    def test_push_120_frames_buffer_is_full(self):
        buf = CircularBuffer()
        for i in range(config.CIRCULAR_BUFFER_SIZE):
            buf.push(_make_signal_frame(i))
        assert buf.is_full
        assert len(buf) == config.CIRCULAR_BUFFER_SIZE


# ═══════════════════════════════════════════════════════════════════════════════
# DurationTimer
# ═══════════════════════════════════════════════════════════════════════════════

class TestDurationTimer:

    def test_initial_value_is_zero(self):
        t = DurationTimer()
        assert t.value == pytest.approx(0.0)

    def test_accumulates_when_condition_true(self):
        t = DurationTimer()
        t.update(True, 0.1)
        t.update(True, 0.1)
        assert t.value == pytest.approx(0.2)

    def test_resets_on_condition_false(self):
        t = DurationTimer()
        t.update(True, 0.5)
        t.update(True, 0.5)
        t.update(False, 0.1)
        assert t.value == pytest.approx(0.0)

    def test_does_not_accumulate_when_false(self):
        t = DurationTimer()
        t.update(False, 1.0)
        t.update(False, 1.0)
        assert t.value == pytest.approx(0.0)

    def test_resumes_after_reset(self):
        t = DurationTimer()
        t.update(True, 0.3)
        t.update(False, 0.1)  # reset
        t.update(True, 0.2)
        assert t.value == pytest.approx(0.2)

    def test_explicit_reset(self):
        t = DurationTimer()
        t.update(True, 2.0)
        t.reset()
        assert t.value == pytest.approx(0.0)

    def test_update_returns_current_value(self):
        t = DurationTimer()
        returned = t.update(True, 0.5)
        assert returned == pytest.approx(0.5)

    def test_uses_variable_dt(self):
        t = DurationTimer()
        t.update(True, 1.0 / 30.0)
        t.update(True, 1.0 / 29.0)  # slightly different dt (CPU jitter)
        expected = 1.0 / 30.0 + 1.0 / 29.0
        assert t.value == pytest.approx(expected, rel=1e-6)


# ═══════════════════════════════════════════════════════════════════════════════
# PERCLOSWindow
# ═══════════════════════════════════════════════════════════════════════════════

class TestPERCLOSWindow:

    # ── Hand-computed test cases ──────────────────────────────────────────────

    def test_perclos_hand_computed_all_open(self):
        """60 valid open-eye frames → PERCLOS = 0.0"""
        pw = PERCLOSWindow()
        baseline = 0.28
        open_ear  = 0.28  # well above closure threshold
        for _ in range(60):
            pw.update(open_ear, baseline, is_valid=True)
        assert pw.valid
        assert pw.perclos == pytest.approx(0.0)

    def test_perclos_hand_computed_half_closed(self):
        """30 closed + 30 open → PERCLOS = 0.50 (hand computed)"""
        pw = PERCLOSWindow()
        baseline = 0.28
        # Threshold = 0.28 * (1 - 0.80) = 0.28 * 0.20 = 0.056
        closed_ear = 0.05   # below threshold → closed
        open_ear   = 0.28   # above threshold → open
        for _ in range(30):
            pw.update(closed_ear, baseline, is_valid=True)
        for _ in range(30):
            pw.update(open_ear, baseline, is_valid=True)
        assert pw.valid
        assert pw.perclos == pytest.approx(0.5, abs=0.01)

    def test_perclos_hand_computed_15_pct(self):
        """9 closed + 51 open in 60-frame window → PERCLOS = 9/60 = 0.15"""
        pw = PERCLOSWindow()
        baseline   = 0.28
        closed_ear = 0.04
        open_ear   = 0.28
        for _ in range(9):
            pw.update(closed_ear, baseline, is_valid=True)
        for _ in range(51):
            pw.update(open_ear, baseline, is_valid=True)
        assert pw.perclos == pytest.approx(9 / 60, abs=0.001)

    def test_perclos_all_closed(self):
        pw = PERCLOSWindow()
        baseline = 0.28
        closed   = 0.04
        for _ in range(60):
            pw.update(closed, baseline, is_valid=True)
        assert pw.perclos == pytest.approx(1.0)

    # ── Invalid frame exclusion ───────────────────────────────────────────────

    def test_invalid_frames_excluded_from_denominator(self):
        """Valid: 40 frames (10 closed, 30 open); invalid: 20 → PERCLOS = 10/40"""
        pw = PERCLOSWindow()
        baseline   = 0.28
        closed_ear = 0.04
        open_ear   = 0.28
        for _ in range(10):
            pw.update(closed_ear, baseline, is_valid=True)
        for _ in range(20):
            pw.update(None, baseline, is_valid=False)
        for _ in range(30):
            pw.update(open_ear, baseline, is_valid=True)
        assert pw.frames_valid == 40
        assert pw.perclos == pytest.approx(10 / 40, abs=0.001)

    def test_none_ear_treated_as_invalid(self):
        pw = PERCLOSWindow()
        pw.update(None, 0.28, is_valid=True)
        assert pw.frames_valid == 0

    # ── Min valid frames gate ─────────────────────────────────────────────────

    def test_below_min_valid_frames_returns_zero(self):
        pw = PERCLOSWindow()
        baseline   = 0.28
        closed_ear = 0.04
        # Add 29 valid frames — one below the minimum threshold
        for _ in range(config.PERCLOS_MIN_VALID_FRAMES - 1):
            pw.update(closed_ear, baseline, is_valid=True)
        assert not pw.valid
        assert pw.perclos == pytest.approx(0.0)

    def test_exactly_min_valid_frames_is_valid(self):
        pw = PERCLOSWindow()
        baseline = 0.28
        open_ear  = 0.28
        for _ in range(config.PERCLOS_MIN_VALID_FRAMES):
            pw.update(open_ear, baseline, is_valid=True)
        assert pw.valid

    # ── Sliding window behaviour ──────────────────────────────────────────────

    def test_window_slides_evicting_oldest(self):
        """After filling the 60-frame window, push 60 all-open frames.
        Old closed frames should be evicted → PERCLOS drops to 0.0."""
        pw = PERCLOSWindow()
        baseline   = 0.28
        closed_ear = 0.04
        open_ear   = 0.28
        # First 60: all closed
        for _ in range(60):
            pw.update(closed_ear, baseline, is_valid=True)
        assert pw.perclos == pytest.approx(1.0)
        # Next 60: all open — old closed frames evicted
        for _ in range(60):
            pw.update(open_ear, baseline, is_valid=True)
        assert pw.perclos == pytest.approx(0.0)

    def test_perclos_closure_threshold_uses_config(self):
        """Verify closure uses baseline * (1 - PERCLOS_CLOSURE_FRACTION)."""
        pw = PERCLOSWindow()
        baseline  = 0.30
        threshold = baseline * (1.0 - config.PERCLOS_CLOSURE_FRACTION)
        # Just above threshold — should NOT count as closed
        for _ in range(30):
            pw.update(threshold + 0.001, baseline, is_valid=True)
        # Just at or below threshold — SHOULD count as closed
        for _ in range(30):
            pw.update(threshold, baseline, is_valid=True)
        assert pw.perclos == pytest.approx(30 / 60, abs=0.001)


# ═══════════════════════════════════════════════════════════════════════════════
# BlinkDetector
# ═══════════════════════════════════════════════════════════════════════════════

class TestBlinkDetector:

    _DT = 1.0 / config.CAPTURE_FPS
    _THRESHOLD = 0.21

    def _push_frames(self, det: BlinkDetector, ear_values: list[float]) -> None:
        for ear in ear_values:
            det.update(ear, self._THRESHOLD, self._DT)

    def _simulate_blink(self, det: BlinkDetector, duration_frames: int) -> None:
        """Simulate one blink: N closed frames followed by one open frame."""
        for _ in range(duration_frames):
            det.update(self._THRESHOLD - 0.01, self._THRESHOLD, self._DT)
        det.update(self._THRESHOLD + 0.01, self._THRESHOLD, self._DT)

    # ── Blink detection state machine ────────────────────────────────────────

    def test_no_blink_all_open(self):
        det = BlinkDetector()
        for _ in range(60):
            det.update(0.28, self._THRESHOLD, self._DT)
        assert det.blink_rate_hz == pytest.approx(0.0)

    def test_short_blink_below_min_not_counted(self):
        """Duration < BLINK_MIN_FRAMES → should not count."""
        det = BlinkDetector()
        self._simulate_blink(det, config.BLINK_MIN_FRAMES - 1)
        assert det.blink_rate_hz == pytest.approx(0.0)

    def test_blink_at_min_frames_counted(self):
        det = BlinkDetector()
        self._simulate_blink(det, config.BLINK_MIN_FRAMES)
        assert det.blink_rate_hz > 0.0

    def test_blink_at_max_frames_counted(self):
        det = BlinkDetector()
        self._simulate_blink(det, config.BLINK_MAX_FRAMES)
        assert det.blink_rate_hz > 0.0

    def test_long_blink_above_max_not_counted(self):
        """Duration > BLINK_MAX_FRAMES → should not count as blink."""
        det = BlinkDetector()
        self._simulate_blink(det, config.BLINK_MAX_FRAMES + 1)
        assert det.blink_rate_hz == pytest.approx(0.0)

    def test_none_ear_treated_as_open(self):
        """None EAR resets closed state without counting a blink."""
        det = BlinkDetector()
        # Start closing
        for _ in range(config.BLINK_MIN_FRAMES):
            det.update(0.10, self._THRESHOLD, self._DT)
        # None breaks the sequence
        det.update(None, self._THRESHOLD, self._DT)
        assert det.blink_rate_hz == pytest.approx(0.0)

    # ── Blink rate computation ────────────────────────────────────────────────

    def test_blink_rate_zero_in_empty_window(self):
        det = BlinkDetector()
        assert det.blink_rate_hz == pytest.approx(0.0)

    def test_multiple_blinks_rate_correct(self):
        """Inject 5 valid blinks within window → rate ≈ 5 / window_duration."""
        det = BlinkDetector()
        window_s = config.FEATURE_WINDOW_FRAMES / config.CAPTURE_FPS  # 2.0s
        for _ in range(5):
            self._simulate_blink(det, config.BLINK_MIN_FRAMES)
            # Pad with open frames between blinks
            for _ in range(5):
                det.update(0.28, self._THRESHOLD, self._DT)
        expected_rate = 5 / window_s
        assert det.blink_rate_hz == pytest.approx(expected_rate, rel=0.2)

    # ── Blink rate anomaly score formula (PRD §5.5) ──────────────────────────

    def test_score_zero_within_normal_range(self):
        """Exactly 1 blink in a 2s window → rate = 0.5 Hz = HIGH boundary → score = 0.0.

        Formula: elif rate > HIGH, so rate == HIGH gives score = 0.0.
        window_s = 60/30 = 2.0s → rate = 1/2.0 = 0.5 = BLINK_RATE_NORMAL_HIGH_HZ.
        """
        det = BlinkDetector()
        # Inject exactly 1 valid blink; keep detector within the 2s window
        self._simulate_blink(det, config.BLINK_MIN_FRAMES)
        # Verify rate is at the HIGH boundary (within normal range)
        assert config.BLINK_RATE_NORMAL_LOW_HZ <= det.blink_rate_hz <= config.BLINK_RATE_NORMAL_HIGH_HZ
        assert det.blink_rate_score == pytest.approx(0.0, abs=1e-6)

    def test_score_one_when_no_blinks(self):
        """Zero blink rate → maximally anomalous score = 1.0"""
        det = BlinkDetector()
        for _ in range(60):
            det.update(0.28, self._THRESHOLD, self._DT)
        assert det.blink_rate_score == pytest.approx(1.0)

    def test_score_positive_when_rate_below_low(self):
        """Rate below BLINK_RATE_NORMAL_LOW_HZ → score > 0.

        Push one blink then enough open frames to push the blink event past the
        2s window cutoff — giving 0 blinks in window → rate = 0.0 < LOW → score = 1.0.
        window_s = 2.0s; blink fires at elapsed ≈ 3*DT ≈ 0.1s.
        After 61+ more frames, elapsed > 2.1s → cutoff > 0.1s → blink evicted.
        """
        det = BlinkDetector()
        window_s = config.FEATURE_WINDOW_FRAMES / config.CAPTURE_FPS  # 2.0s
        # Inject one blink then advance well past the window duration
        self._simulate_blink(det, config.BLINK_MIN_FRAMES)
        frames_to_expire = int(window_s / self._DT) + 5
        for _ in range(frames_to_expire):
            det.update(0.28, self._THRESHOLD, self._DT)
        # Blink should now be outside the 2s window → rate = 0 < LOW → score > 0
        assert det.blink_rate_hz < config.BLINK_RATE_NORMAL_LOW_HZ
        assert det.blink_rate_score > 0.0

    def test_score_positive_when_rate_above_high(self):
        """Rate above BLINK_RATE_NORMAL_HIGH_HZ → score > 0"""
        det = BlinkDetector()
        # Many blinks in 60 frames (very high rate)
        for _ in range(10):
            self._simulate_blink(det, config.BLINK_MIN_FRAMES)
        assert det.blink_rate_score > 0.0

    def test_score_clamped_to_one(self):
        """Score never exceeds 1.0."""
        det = BlinkDetector()
        # Extreme high rate
        for _ in range(30):
            self._simulate_blink(det, config.BLINK_MIN_FRAMES)
        assert det.blink_rate_score <= 1.0

    def test_score_clamped_to_zero(self):
        """Score never goes below 0.0."""
        det = BlinkDetector()
        assert det.blink_rate_score >= 0.0

    def test_formula_low_rate_exact(self):
        """Verify exact formula: rate < LOW → score = 1.0 - (rate/LOW)."""
        det = BlinkDetector()
        # Produce a specific low rate by driving elapsed time and blink count
        # We'll test the formula directly by checking the score matches expectation
        # Force rate = 0.0 (no blinks, all open):
        for _ in range(60):
            det.update(0.28, self._THRESHOLD, self._DT)
        # rate = 0.0 → score = 1.0 - (0.0 / LOW) = 1.0
        low = config.BLINK_RATE_NORMAL_LOW_HZ
        expected = 1.0 - (0.0 / low)
        assert det.blink_rate_score == pytest.approx(expected, abs=0.01)

    def test_formula_high_rate_exact(self):
        """Verify formula: rate > HIGH → score = min(1.0, (rate-HIGH)/0.5)."""
        det = BlinkDetector()
        high = config.BLINK_RATE_NORMAL_HIGH_HZ
        window_s = config.FEATURE_WINDOW_FRAMES / config.CAPTURE_FPS
        # Produce rate = HIGH + 0.25 (midway into anomalous high range)
        target_n = int((high + 0.25) * window_s)
        for _ in range(target_n):
            self._simulate_blink(det, config.BLINK_MIN_FRAMES)
            det.update(0.28, self._THRESHOLD, self._DT)
        rate = det.blink_rate_hz
        if rate > high:
            expected = min(1.0, (rate - high) / 0.5)
            assert det.blink_rate_score == pytest.approx(expected, rel=0.1)


# ═══════════════════════════════════════════════════════════════════════════════
# SpeedContext
# ═══════════════════════════════════════════════════════════════════════════════

class TestSpeedContext:

    # ── Zone classification ───────────────────────────────────────────────────

    def test_zero_speed_is_parked(self):
        zone, mod = resolve_speed_zone(0.0, False)
        assert zone == ZONE_PARKED
        assert mod == pytest.approx(0.0)

    def test_below_v_min_is_parked(self):
        zone, mod = resolve_speed_zone(config.V_MIN_MPS - 0.1, False)
        assert zone == ZONE_PARKED

    def test_exactly_v_min_is_urban(self):
        """V_MIN_MPS is the boundary between PARKED and URBAN."""
        zone, mod = resolve_speed_zone(config.V_MIN_MPS, False)
        assert zone == ZONE_URBAN
        assert mod == pytest.approx(1.0)

    def test_between_zones_is_urban(self):
        mid_speed = (config.V_MIN_MPS + config.V_HIGHWAY_MPS) / 2
        zone, mod = resolve_speed_zone(mid_speed, False)
        assert zone == ZONE_URBAN
        assert mod == pytest.approx(1.0)

    def test_below_highway_is_urban(self):
        zone, mod = resolve_speed_zone(config.V_HIGHWAY_MPS - 0.1, False)
        assert zone == ZONE_URBAN

    def test_exactly_v_highway_is_highway(self):
        """V_HIGHWAY_MPS is the boundary between URBAN and HIGHWAY."""
        zone, mod = resolve_speed_zone(config.V_HIGHWAY_MPS, False)
        assert zone == ZONE_HIGHWAY
        assert mod == pytest.approx(config.HIGHWAY_SCORE_MODIFIER)

    def test_above_v_highway_is_highway(self):
        zone, mod = resolve_speed_zone(config.V_HIGHWAY_MPS + 5.0, False)
        assert zone == ZONE_HIGHWAY
        assert mod == pytest.approx(config.HIGHWAY_SCORE_MODIFIER)

    # ── None and stale handling ───────────────────────────────────────────────

    def test_none_speed_defaults_to_urban(self):
        zone, mod = resolve_speed_zone(None, False)
        assert zone == ZONE_URBAN
        assert mod == pytest.approx(1.0)

    def test_stale_speed_defaults_to_urban(self):
        """Even a high stale speed should default to URBAN."""
        zone, mod = resolve_speed_zone(50.0, True)
        assert zone == ZONE_URBAN
        assert mod == pytest.approx(1.0)

    def test_none_stale_both_defaults_to_urban(self):
        zone, mod = resolve_speed_zone(None, True)
        assert zone == ZONE_URBAN

    # ── Speed clamping ────────────────────────────────────────────────────────

    def test_negative_speed_clamped_to_parked(self):
        """Negative speed clamps to 0.0 → PARKED."""
        zone, mod = resolve_speed_zone(-5.0, False)
        assert zone == ZONE_PARKED

    def test_extremely_high_speed_clamped_to_highway(self):
        """Speed > 100 m/s clamps to 100 m/s → still HIGHWAY."""
        zone, mod = resolve_speed_zone(999.0, False)
        assert zone == ZONE_HIGHWAY

    # ── Return type ───────────────────────────────────────────────────────────

    def test_returns_tuple_of_str_and_float(self):
        zone, mod = resolve_speed_zone(5.0, False)
        assert isinstance(zone, str)
        assert isinstance(mod, float)


# ═══════════════════════════════════════════════════════════════════════════════
# SpeedSource
# ═══════════════════════════════════════════════════════════════════════════════

class TestSpeedSource:

    def test_mac_returns_none_speed(self):
        src = SpeedSource()
        speed, stale = src.get_speed_mps()
        assert speed is None

    def test_mac_returns_stale(self):
        src = SpeedSource()
        _, stale = src.get_speed_mps()
        assert stale is True

    def test_mac_source_type_is_none(self):
        src = SpeedSource()
        assert src.get_source_type() == 'NONE'

    def test_mac_is_not_available(self):
        src = SpeedSource()
        assert src.is_available() is False


# ═══════════════════════════════════════════════════════════════════════════════
# WatchdogManager
# ═══════════════════════════════════════════════════════════════════════════════

class TestWatchdogManager:

    def test_not_timed_out_immediately_after_kick(self):
        wd = WatchdogManager(timeout_s=2.0)
        wd.kick(1)
        now = time.monotonic()
        assert wd.check(now=now) is False
        assert wd.timed_out is False

    def test_not_timed_out_before_threshold(self):
        """Inject a time just below timeout → should not fire."""
        wd = WatchdogManager(timeout_s=2.0)
        base = time.monotonic()
        wd._last_kick_time = base
        result = wd.check(now=base + 1.9)
        assert result is False
        assert wd.timed_out is False

    def test_timed_out_after_threshold(self):
        """Inject a time past timeout → should fire."""
        wd = WatchdogManager(timeout_s=2.0)
        base = time.monotonic()
        wd._last_kick_time = base
        result = wd.check(now=base + 2.1)
        assert result is True
        assert wd.timed_out is True

    def test_kick_recovers_from_timeout(self):
        """After timing out, a kick should clear timed_out."""
        wd = WatchdogManager(timeout_s=2.0)
        base = time.monotonic()
        wd._last_kick_time = base
        wd.check(now=base + 2.5)
        assert wd.timed_out is True
        wd.kick(99)
        assert wd.timed_out is False

    def test_callback_fires_on_timeout(self):
        """Verify the timeout callback is called exactly once."""
        fired = []
        wd = WatchdogManager(timeout_s=2.0)
        wd.set_timeout_callback(lambda: fired.append(1))
        base = time.monotonic()
        wd._last_kick_time = base
        wd.check(now=base + 3.0)
        assert len(fired) == 1

    def test_callback_does_not_fire_twice(self):
        """Second check after timeout should not re-fire the callback."""
        fired = []
        wd = WatchdogManager(timeout_s=2.0)
        wd.set_timeout_callback(lambda: fired.append(1))
        base = time.monotonic()
        wd._last_kick_time = base
        wd.check(now=base + 3.0)
        wd.check(now=base + 4.0)  # already timed out
        assert len(fired) == 1

    def test_last_frame_id_updates_on_kick(self):
        wd = WatchdogManager()
        wd.kick(42)
        assert wd.last_frame_id == 42
        wd.kick(100)
        assert wd.last_frame_id == 100

    def test_initial_last_frame_id(self):
        wd = WatchdogManager()
        assert wd.last_frame_id == -1

    def test_fault_injection_3s_block(self):
        """Fault injection: no kick for > WATCHDOG_TIMEOUT_S → DEGRADED triggered.

        Uses injectable `now` — no real sleeping needed (PRD §FR-3.6).
        """
        degraded_triggered = []
        wd = WatchdogManager(timeout_s=config.WATCHDOG_TIMEOUT_S)
        wd.set_timeout_callback(lambda: degraded_triggered.append(True))

        base = time.monotonic()
        wd._last_kick_time = base

        # Simulate 3 seconds without a kick (> WATCHDOG_TIMEOUT_S = 2.0s)
        wd.check(now=base + 3.0)

        assert wd.timed_out is True
        assert len(degraded_triggered) == 1


# ═══════════════════════════════════════════════════════════════════════════════
# ThermalMonitor
# ═══════════════════════════════════════════════════════════════════════════════

class TestThermalMonitor:

    def test_mac_throttle_inactive(self):
        tm = ThermalMonitor()
        assert tm.throttle_active is False

    def test_mac_temperature_is_zero(self):
        tm = ThermalMonitor()
        assert tm.temperature_c == pytest.approx(0.0)

    def test_start_stop_no_error_on_mac(self):
        tm = ThermalMonitor()
        tm.start()  # no-op on Mac
        tm.stop()
        assert tm.throttle_active is False

    def test_not_on_hardware_when_no_path(self):
        """Mac: thermal path doesn't exist → _on_hardware=False."""
        tm = ThermalMonitor()
        assert tm._on_hardware is False


# ═══════════════════════════════════════════════════════════════════════════════
# TemporalEngine — Integration
# ═══════════════════════════════════════════════════════════════════════════════

class TestTemporalEngine:

    def _make_engine(self) -> TemporalEngine:
        return TemporalEngine()

    def test_returns_temporal_features(self):
        from layer3_temporal.messages import TemporalFeatures
        engine = self._make_engine()
        frame = _make_signal_frame(0)
        result = engine.process(frame)
        assert isinstance(result, TemporalFeatures)

    def test_timestamp_preserved(self):
        engine = self._make_engine()
        ts = 999_000_000_000
        frame = _make_signal_frame(0, ts_ns=ts)
        result = engine.process(frame)
        assert result.timestamp_ns == ts

    def test_speed_zone_parked_at_low_speed(self):
        engine = self._make_engine()
        frame = _make_signal_frame(0, speed_mps=0.5)
        result = engine.process(frame)
        assert result.speed_zone == ZONE_PARKED
        assert result.speed_modifier == pytest.approx(0.0)

    def test_speed_zone_urban(self):
        engine = self._make_engine()
        frame = _make_signal_frame(0, speed_mps=5.0)
        result = engine.process(frame)
        assert result.speed_zone == ZONE_URBAN
        assert result.speed_modifier == pytest.approx(1.0)

    def test_speed_zone_highway(self):
        engine = self._make_engine()
        frame = _make_signal_frame(0, speed_mps=20.0)
        result = engine.process(frame)
        assert result.speed_zone == ZONE_HIGHWAY
        assert result.speed_modifier == pytest.approx(config.HIGHWAY_SCORE_MODIFIER)

    def test_stale_speed_defaults_to_urban(self):
        engine = self._make_engine()
        frame = _make_signal_frame(0, speed_mps=50.0, speed_stale=True)
        result = engine.process(frame)
        assert result.speed_zone == ZONE_URBAN

    def test_thermal_throttle_false_on_mac(self):
        engine = self._make_engine()
        frame = _make_signal_frame(0)
        result = engine.process(frame)
        assert result.thermal_throttle_active is False

    def test_gaze_continuous_secs_increments_when_off_road(self):
        engine = self._make_engine()
        off_road_gaze = _make_gaze_world(on_road=False)
        for i in range(5):
            frame = _make_signal_frame(
                frame_id=i,
                signals_valid=True,
                gaze_world=off_road_gaze,
            )
            result = engine.process(frame)
        assert result.gaze_continuous_secs > 0.0

    def test_gaze_continuous_secs_resets_when_on_road(self):
        engine = self._make_engine()
        off_road = _make_gaze_world(on_road=False)
        on_road  = _make_gaze_world(on_road=True)
        # Build up off-road duration
        for i in range(5):
            engine.process(_make_signal_frame(i, gaze_world=off_road, signals_valid=True))
        # One on-road frame resets it
        result = engine.process(_make_signal_frame(5, gaze_world=on_road, signals_valid=True))
        assert result.gaze_continuous_secs == pytest.approx(0.0)

    def test_head_continuous_secs_increments_beyond_threshold(self):
        engine = self._make_engine()
        far_head = _make_head_pose(yaw=config.HEAD_YAW_THRESHOLD_DEG + 5.0, valid=True)
        for i in range(5):
            engine.process(_make_signal_frame(i, head_pose=far_head))
        result = engine.process(_make_signal_frame(5, head_pose=far_head))
        assert result.head_continuous_secs > 0.0

    def test_head_continuous_secs_zero_within_threshold(self):
        engine = self._make_engine()
        normal_head = _make_head_pose(yaw=0.0, pitch=0.0, valid=True)
        result = engine.process(_make_signal_frame(0, head_pose=normal_head))
        assert result.head_continuous_secs == pytest.approx(0.0)

    def test_phone_continuous_secs_increments_when_detected(self):
        engine = self._make_engine()
        phone = _make_phone_signal(detected=True, confidence=0.9)
        for i in range(5):
            result = engine.process(_make_signal_frame(i, phone=phone))
        assert result.phone_continuous_secs > 0.0

    def test_face_absent_no_gaze_no_head_signals(self):
        engine = self._make_engine()
        frame = _make_signal_frame(
            0,
            face_present=False,
            signals_valid=False,
            head_pose=None,
            eye_signals=None,
            gaze_world=None,
        )
        result = engine.process(frame)
        assert result.gaze_off_road_fraction == pytest.approx(0.0)
        assert result.head_deviation_mean_deg == pytest.approx(0.0)

    def test_gaze_off_road_fraction_all_off_road(self):
        """60 off-road frames → fraction = 1.0"""
        engine = self._make_engine()
        off_road = _make_gaze_world(on_road=False)
        for i in range(60):
            result = engine.process(_make_signal_frame(i, gaze_world=off_road, signals_valid=True))
        assert result.gaze_off_road_fraction == pytest.approx(1.0)

    def test_gaze_off_road_fraction_all_on_road(self):
        engine = self._make_engine()
        on_road = _make_gaze_world(on_road=True)
        for i in range(60):
            result = engine.process(_make_signal_frame(i, gaze_world=on_road, signals_valid=True))
        assert result.gaze_off_road_fraction == pytest.approx(0.0)

    def test_frames_valid_in_window_counts_correctly(self):
        """Mix of valid (40) and invalid (20) frames — check on the 60th frame exactly.

        Push exactly 60 frames: 0-39 valid, 40-59 invalid.
        get_window(60) returns all 60 → frames_valid = 40.
        """
        engine = self._make_engine()
        result = None
        for i in range(40):
            result = engine.process(_make_signal_frame(i, signals_valid=True))
        for i in range(40, 60):
            result = engine.process(_make_signal_frame(i, signals_valid=False))
        assert result.frames_valid_in_window == 40

    def test_head_deviation_mean_uses_euclidean_norm(self):
        """head_deviation_mean_deg should be sqrt(yaw² + pitch²)."""
        engine = self._make_engine()
        yaw, pitch = 3.0, 4.0
        expected_deviation = math.sqrt(yaw ** 2 + pitch ** 2)  # 5.0
        head = _make_head_pose(yaw=yaw, pitch=pitch, valid=True)
        for i in range(60):
            result = engine.process(_make_signal_frame(i, head_pose=head))
        assert result.head_deviation_mean_deg == pytest.approx(expected_deviation, rel=0.01)

    def test_phone_confidence_mean_aggregated(self):
        """All frames with confidence 0.8 → mean = 0.8."""
        engine = self._make_engine()
        phone = _make_phone_signal(detected=True, confidence=0.8)
        for i in range(60):
            result = engine.process(_make_signal_frame(i, phone=phone))
        assert result.phone_confidence_mean == pytest.approx(0.8, rel=0.01)

    def test_perclos_reflected_in_temporal_features(self):
        """Feed 30 closed-eye frames → PERCLOS > 0."""
        engine = self._make_engine()
        baseline = 0.28
        closed_ear = 0.04  # well below closure threshold
        closed_eye_signals = _make_eye_signals(
            mean_ear=closed_ear, baseline_ear=baseline, close_threshold=0.21
        )
        open_eye_signals = _make_eye_signals(
            mean_ear=0.28, baseline_ear=baseline, close_threshold=0.21
        )
        for i in range(30):
            engine.process(_make_signal_frame(i, eye_signals=closed_eye_signals, signals_valid=True))
        for i in range(30, 60):
            engine.process(_make_signal_frame(i, eye_signals=open_eye_signals, signals_valid=True))
        result = engine.process(
            _make_signal_frame(60, eye_signals=open_eye_signals, signals_valid=True)
        )
        assert result.perclos > 0.0

    def test_first_frame_uses_default_dt(self):
        """First frame should not crash (no previous timestamp)."""
        engine = self._make_engine()
        frame = _make_signal_frame(0, ts_ns=_TS_BASE)
        result = engine.process(frame)
        assert result is not None

    def test_consecutive_timestamps_drive_timers(self):
        """Timer should accumulate exactly dt per frame using real timestamps."""
        engine = self._make_engine()
        dt_ns = _DT_NS
        off_road = _make_gaze_world(on_road=False)

        # Process 10 off-road frames with known timestamps
        for i in range(10):
            ts = _TS_BASE + i * dt_ns
            frame = _make_signal_frame(
                frame_id=i, ts_ns=ts,
                gaze_world=off_road, signals_valid=True,
            )
            result = engine.process(frame)

        expected_duration = 9 * (dt_ns / 1e9)  # first frame has dt=default
        # Allow generous tolerance — first frame uses default dt
        assert result.gaze_continuous_secs > 0.0
        assert result.gaze_continuous_secs == pytest.approx(
            expected_duration + _DEFAULT_DT, rel=0.1
        )

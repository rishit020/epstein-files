# layer3_temporal/temporal_engine.py — Layer 3 Temporal Engine Orchestrator
# PRD §3.1 Layer 3
#
# Consumes SignalFrame objects from Layer 2, maintains temporal state,
# and emits TemporalFeatures for Layer 4.
#
# Processing order per frame:
#   1. Compute frame dt from consecutive timestamps
#   2. Push SignalFrame into CircularBuffer
#   3. Update gaze duration timer
#   4. Update head pose duration timer
#   5. Update phone detection duration timer
#   6. Update face-absent duration timer
#   7. Update PERCLOS window
#   8. Update blink detector
#   9. Resolve speed zone
#  10. Kick watchdog
#  11. Poll thermal monitor
#  12. Compute window aggregates (gaze fraction, head deviation, phone mean)
#  13. Assemble and return TemporalFeatures

import math
from typing import Optional

import config
from layer2_signals.messages import SignalFrame
from layer3_temporal.blink_detector import BlinkDetector
from layer3_temporal.circular_buffer import CircularBuffer
from layer3_temporal.duration_timer import DurationTimer
from layer3_temporal.messages import TemporalFeatures
from layer3_temporal.perclos_window import PERCLOSWindow
from layer3_temporal.speed_context import resolve_speed_zone
from layer3_temporal.thermal_monitor import ThermalMonitor
from layer3_temporal.watchdog import WatchdogManager

_DEFAULT_DT: float = 1.0 / config.CAPTURE_FPS


class TemporalEngine:
    """Orchestrates all Layer 3 temporal processing.

    PRD §3.1 — Layer 3
    Inputs:  SignalFrame (from Layer 2)
    Outputs: TemporalFeatures (to Layer 4)
    """

    def __init__(self) -> None:
        self._buffer        = CircularBuffer()
        self._perclos       = PERCLOSWindow()
        self._blink         = BlinkDetector()

        self._gaze_timer       = DurationTimer()
        self._head_timer       = DurationTimer()
        self._phone_timer      = DurationTimer()
        self._face_absent_timer = DurationTimer()

        self._watchdog = WatchdogManager()
        self._thermal  = ThermalMonitor()

        self._prev_ts_ns: Optional[int] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start background threads (watchdog + thermal monitor)."""
        self._watchdog.start()
        self._thermal.start()

    def stop(self) -> None:
        """Stop background threads."""
        self._watchdog.stop()
        self._thermal.stop()

    def process(self, frame: SignalFrame) -> TemporalFeatures:
        """Process one SignalFrame and return aggregated TemporalFeatures.

        Args:
            frame: Layer 2 output for this frame.

        Returns:
            TemporalFeatures with all windowed metrics computed.
        """
        # ── 1. Compute frame delta ──────────────────────────────────────────
        if self._prev_ts_ns is None:
            dt = _DEFAULT_DT
        else:
            dt = max(0.0, (frame.timestamp_ns - self._prev_ts_ns) / 1e9)
            if dt == 0.0:
                dt = _DEFAULT_DT
        self._prev_ts_ns = frame.timestamp_ns

        # ── 2. Buffer ───────────────────────────────────────────────────────
        self._buffer.push(frame)

        # ── 3. Gaze duration timer ──────────────────────────────────────────
        gaze_condition = (
            frame.face_present
            and frame.signals_valid
            and frame.gaze_world is not None
            and not frame.gaze_world.on_road
        )
        self._gaze_timer.update(gaze_condition, dt)

        # ── 4. Head pose duration timer ─────────────────────────────────────
        head_condition = (
            frame.head_pose is not None
            and frame.head_pose.valid
            and (
                abs(frame.head_pose.yaw_deg) > config.HEAD_YAW_THRESHOLD_DEG
                or abs(frame.head_pose.pitch_deg) > config.HEAD_PITCH_THRESHOLD_DEG
            )
        )
        self._head_timer.update(head_condition, dt)

        # ── 5. Phone duration timer ─────────────────────────────────────────
        phone_condition = frame.phone_signal.detected
        self._phone_timer.update(phone_condition, dt)

        # ── 6. Face-absent duration timer ───────────────────────────────────
        self._face_absent_timer.update(not frame.face_present, dt)

        # ── 7. PERCLOS window ───────────────────────────────────────────────
        if frame.eye_signals is not None:
            self._perclos.update(
                mean_ear=frame.eye_signals.mean_EAR,
                baseline_ear=frame.eye_signals.baseline_EAR,
                is_valid=frame.signals_valid,
            )
        else:
            self._perclos.update(mean_ear=None, baseline_ear=0.0, is_valid=False)

        # ── 8. Blink detector ───────────────────────────────────────────────
        if frame.eye_signals is not None:
            self._blink.update(
                mean_ear=frame.eye_signals.mean_EAR,
                close_threshold=frame.eye_signals.close_threshold,
                dt_seconds=dt,
            )
        else:
            self._blink.update(mean_ear=None, close_threshold=config.EAR_DEFAULT_CLOSE_THRESHOLD, dt_seconds=dt)

        # ── 9. Speed zone ───────────────────────────────────────────────────
        speed_zone, speed_modifier = resolve_speed_zone(frame.speed_mps, frame.speed_stale)

        # ── 10. Kick watchdog ───────────────────────────────────────────────
        self._watchdog.kick(frame.frame_id)

        # ── 11. Thermal state ───────────────────────────────────────────────
        thermal_throttle = self._thermal.throttle_active

        # ── 12. Window aggregates ───────────────────────────────────────────
        window = self._buffer.get_window(config.FEATURE_WINDOW_FRAMES)
        (
            gaze_off_road_fraction,
            head_deviation_mean_deg,
            phone_confidence_mean,
            frames_valid_in_window,
        ) = self._compute_window_aggregates(window)

        # ── 13. Assemble ────────────────────────────────────────────────────
        return TemporalFeatures(
            timestamp_ns=frame.timestamp_ns,
            gaze_off_road_fraction=gaze_off_road_fraction,
            gaze_continuous_secs=self._gaze_timer.value,
            head_deviation_mean_deg=head_deviation_mean_deg,
            head_continuous_secs=self._head_timer.value,
            perclos=self._perclos.perclos,
            blink_rate_score=self._blink.blink_rate_score,
            phone_confidence_mean=phone_confidence_mean,
            phone_continuous_secs=self._phone_timer.value,
            speed_zone=speed_zone,
            speed_modifier=speed_modifier,
            frames_valid_in_window=frames_valid_in_window,
            face_absent_continuous_secs=self._face_absent_timer.value,
            thermal_throttle_active=thermal_throttle,
        )

    # ── Accessors for testing ─────────────────────────────────────────────────

    @property
    def watchdog(self) -> WatchdogManager:
        return self._watchdog

    @property
    def thermal_monitor(self) -> ThermalMonitor:
        return self._thermal

    # ── Private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _compute_window_aggregates(
        window: list[SignalFrame],
    ) -> tuple[float, float, float, int]:
        """Compute feature aggregates over a window of SignalFrames.

        Returns:
            (gaze_off_road_fraction, head_deviation_mean_deg,
             phone_confidence_mean, frames_valid_in_window)
        """
        frames_valid = 0
        gaze_valid_count = 0
        gaze_off_road_count = 0
        head_deviations: list[float] = []
        phone_confidences: list[float] = []

        for f in window:
            if f.signals_valid:
                frames_valid += 1

            # Gaze off-road fraction — valid gaze frames only
            if (
                f.face_present
                and f.signals_valid
                and f.gaze_world is not None
            ):
                gaze_valid_count += 1
                if not f.gaze_world.on_road:
                    gaze_off_road_count += 1

            # Head deviation — valid pose frames only
            if f.head_pose is not None and f.head_pose.valid:
                deviation = math.sqrt(
                    f.head_pose.yaw_deg ** 2 + f.head_pose.pitch_deg ** 2
                )
                head_deviations.append(deviation)

            # Phone confidence — all frames in window
            phone_confidences.append(f.phone_signal.confidence)

        gaze_off_road_fraction = (
            gaze_off_road_count / gaze_valid_count if gaze_valid_count > 0 else 0.0
        )
        head_deviation_mean_deg = (
            sum(head_deviations) / len(head_deviations) if head_deviations else 0.0
        )
        phone_confidence_mean = (
            sum(phone_confidences) / len(phone_confidences) if phone_confidences else 0.0
        )

        return (
            gaze_off_road_fraction,
            head_deviation_mean_deg,
            phone_confidence_mean,
            frames_valid,
        )

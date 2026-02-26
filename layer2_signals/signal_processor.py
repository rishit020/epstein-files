# layer2_signals/signal_processor.py — Signal Processor Orchestrator
# PRD §3.1 — Layer 2: Signal Processor
#
# Accepts a PerceptionBundle from Layer 1 and returns a SignalFrame for Layer 3.
#
# Processing order per frame:
#   1. Head pose: PnP solve → raw Euler angles
#   2. Kalman filter: head yaw, pitch, roll (reset on face-lost)
#   3. Neutral pose correction: corrected yaw, pitch (from PoseCalibration)
#   4. EAR computation: left/right/mean EAR from landmarks
#   5. EAR calibration update (if driving)
#   6. Gaze world transform: camera-space gaze + corrected head pose
#   7. Kalman filter: gaze world yaw, pitch
#   8. On-road check
#   9. Phone signal extraction
#  10. Assemble SignalFrame
#
# All angle values placed in SignalFrame are Kalman-filtered and
# neutral-pose-corrected (PRD CHANGE-06, CHANGE-08).
# Raw values are stored in HeadPose debug fields.

import config
from layer1_perception.messages import PerceptionBundle
from layer2_signals.ear_calculator import EARCalculator
from layer2_signals.gaze_transformer import is_on_road, transform_gaze
from layer2_signals.head_pose_solver import HeadPoseSolver
from layer2_signals.kalman_filter import KalmanFilter1D
from layer2_signals.messages import (
    EyeSignals,
    GazeWorld,
    HeadPose,
    SignalFrame,
)
from layer2_signals.phone_signal import extract_phone_signal
from layer2_signals.pose_calibration import PoseCalibration


class SignalProcessor:
    """Orchestrates all Layer 2 signal computation.

    PRD §3.1 — Layer 2
    Inputs:  PerceptionBundle (from Layer 1)
    Outputs: SignalFrame (to Layer 3)
    """

    def __init__(self) -> None:
        # Sub-components
        self._head_solver     = HeadPoseSolver()
        self._ear_calc        = EARCalculator()
        self._pose_cal        = PoseCalibration()

        # Kalman filters — one per angle axis (PRD §5.6)
        self._kf_head_yaw   = KalmanFilter1D()
        self._kf_head_pitch = KalmanFilter1D()
        self._kf_head_roll  = KalmanFilter1D()
        self._kf_gaze_yaw   = KalmanFilter1D()
        self._kf_gaze_pitch = KalmanFilter1D()

        # Track consecutive face-absent frames for Kalman reset (PRD §5.6)
        self._face_absent_frames: int = 0

        # Frame size (updated each process call from bundle or defaults)
        self._frame_width:  int = config.CAPTURE_WIDTH
        self._frame_height: int = config.CAPTURE_HEIGHT

    # ── Configuration API (called from main.py on startup) ──────────────────

    def set_neutral_pose(self, yaw_offset: float, pitch_offset: float) -> None:
        """Load persisted neutral pose offsets (PRD §24.2)."""
        self._pose_cal.set_offsets(yaw_offset, pitch_offset)

    def set_ear_baseline(self, baseline_ear: float, close_threshold: float) -> None:
        """Load persisted EAR baseline (PRD §24.2)."""
        self._ear_calc.load_baseline(baseline_ear, close_threshold)

    def reset_calibration(self) -> None:
        """Force re-calibration (e.g., VIN change)."""
        self._ear_calc.reset_calibration()
        self._pose_cal.reset()

    @property
    def ear_calculator(self) -> EARCalculator:
        """Expose EARCalculator for calibration_manager access."""
        return self._ear_calc

    @property
    def pose_calibration(self) -> PoseCalibration:
        """Expose PoseCalibration for calibration_manager access."""
        return self._pose_cal

    # ── Main processing entry point ─────────────────────────────────────────

    def process(self, bundle: PerceptionBundle, speed_mps: float = 0.0, speed_stale: bool = False) -> SignalFrame:
        """Process one PerceptionBundle and return a SignalFrame.

        Args:
            bundle:      Layer 1 output for this frame.
            speed_mps:   Current vehicle speed (from SpeedSource, Layer 3 context).
                         Passed in here to feed EAR calibration driving-gate logic.
            speed_stale: True if speed reading is stale.

        Returns:
            SignalFrame with all signals computed and flags set.
        """
        face_present = bundle.face.present

        # ── Kalman filter reset logic (PRD §5.6) ────────────────────────────
        if not face_present:
            self._face_absent_frames += 1
            if self._face_absent_frames > config.LSTM_RESET_ABSENT_FRAMES:
                self._reset_all_kalman()
        else:
            self._face_absent_frames = 0

        # ── 1–3: Head pose ───────────────────────────────────────────────────
        head_pose_out: HeadPose | None = None
        raw_yaw = raw_pitch = raw_roll = 0.0
        filtered_yaw = filtered_pitch = filtered_roll = 0.0
        corrected_yaw = corrected_pitch = 0.0
        pose_valid = False

        if face_present and bundle.landmarks is not None and bundle.landmarks.pose_valid:
            raw_yaw, raw_pitch, raw_roll, reproj_err, pose_valid = self._head_solver.solve(
                bundle.landmarks.landmarks,
                self._frame_width,
                self._frame_height,
            )

            if pose_valid:
                filtered_yaw   = self._kf_head_yaw.update(raw_yaw)
                filtered_pitch = self._kf_head_pitch.update(raw_pitch)
                filtered_roll  = self._kf_head_roll.update(raw_roll)
                corrected_yaw, corrected_pitch = self._pose_cal.correct(filtered_yaw, filtered_pitch)
            else:
                # Pose invalid — reset Kalman filters (PRD §5.6)
                self._reset_pose_kalman()

            head_pose_out = HeadPose(
                yaw_deg=corrected_yaw,
                pitch_deg=corrected_pitch,
                roll_deg=filtered_roll,
                valid=pose_valid,
                raw_yaw_deg=raw_yaw,
                raw_pitch_deg=raw_pitch,
                raw_roll_deg=raw_roll,
            )

        # ── 4–5: EAR computation + calibration update ────────────────────────
        eye_signals_out: EyeSignals | None = None

        if face_present and bundle.landmarks is not None:
            left_ear, right_ear, mean_ear = self._ear_calc.compute(bundle.landmarks.landmarks)
            is_driving = speed_mps > config.V_MIN_MPS
            self._ear_calc.update_calibration(mean_ear, is_driving)

            eye_signals_out = EyeSignals(
                left_EAR=left_ear,
                right_EAR=right_ear,
                mean_EAR=mean_ear,
                baseline_EAR=self._ear_calc.baseline_EAR,
                close_threshold=self._ear_calc.close_threshold,
                valid=True,
                calibration_complete=self._ear_calc.calibration_complete,
            )

        # ── 6–8: Gaze world transform + Kalman + on-road check ───────────────
        gaze_world_out: GazeWorld | None = None

        gaze_valid = (
            face_present
            and bundle.gaze is not None
            and bundle.gaze.valid
            and pose_valid                          # PRD §5.3: both required
        )

        if gaze_valid:
            raw_gw_yaw, raw_gw_pitch = transform_gaze(
                gaze_camera_yaw=bundle.gaze.combined_yaw,
                gaze_camera_pitch=bundle.gaze.combined_pitch,
                head_yaw_corrected=corrected_yaw,
                head_pitch_corrected=corrected_pitch,
            )
            # Apply Kalman to gaze world angles (PRD §5.3)
            filtered_gw_yaw   = self._kf_gaze_yaw.update(raw_gw_yaw)
            filtered_gw_pitch = self._kf_gaze_pitch.update(raw_gw_pitch)

            on_road = is_on_road(filtered_gw_yaw, filtered_gw_pitch)

            gaze_world_out = GazeWorld(
                yaw_deg=filtered_gw_yaw,
                pitch_deg=filtered_gw_pitch,
                on_road=on_road,
                valid=True,
            )
        else:
            # Gaze unavailable — reset gaze Kalman filters
            self._reset_gaze_kalman()

        # ── 9: Phone signal ──────────────────────────────────────────────────
        phone_signal_out = extract_phone_signal(bundle.phone, bundle.phone_result_stale)

        # ── 10: Signals validity flag ────────────────────────────────────────
        # signals_valid = False if face present but ALL of head+gaze are invalid.
        # (Phone is always considered a valid independent signal.)
        if face_present:
            signals_valid = pose_valid or gaze_valid
        else:
            signals_valid = False

        return SignalFrame(
            timestamp_ns=bundle.timestamp_ns,
            frame_id=bundle.frame_id,
            face_present=face_present,
            head_pose=head_pose_out,
            eye_signals=eye_signals_out,
            gaze_world=gaze_world_out,
            phone_signal=phone_signal_out,
            speed_mps=speed_mps,
            speed_stale=speed_stale,
            signals_valid=signals_valid,
        )

    # ── Private ──────────────────────────────────────────────────────────────

    def _reset_pose_kalman(self) -> None:
        self._kf_head_yaw.reset()
        self._kf_head_pitch.reset()
        self._kf_head_roll.reset()

    def _reset_gaze_kalman(self) -> None:
        self._kf_gaze_yaw.reset()
        self._kf_gaze_pitch.reset()

    def _reset_all_kalman(self) -> None:
        self._reset_pose_kalman()
        self._reset_gaze_kalman()

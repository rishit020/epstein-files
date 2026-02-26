# ATTENTIA DRIVE — DISTRACTION DETECTION ENGINE

## Product Requirements Document v2.0.0

| Field | Value |
|---|---|
| **Document Version** | 2.0.0 — MVP (Revised from v1.0.0) |
| **Status** | IMPLEMENTATION READY — RK3568 + IMX219 Target |
| **Scope** | Software Only — No Hardware |
| **Target Platform** | Python 3.10+ / Linux (RK3568 production; macOS acceptable for development) |
| **Date** | February 2026 |
| **Confidentiality** | Proprietary — Attentia Inc. |
| **Revised By** | Hardware Compatibility Review — RTR-1.0.0 |

> **GOLDEN SOURCE OF TRUTH** This document is the sole authoritative specification for the Attentia Drive Distraction Detection Engine (DDE), Version 2.0 MVP. It supersedes v1.0.0 entirely. No requirement in this document is aspirational or subject to interpretation without a formal document revision. During development, no feature, threshold, assumption, or architectural decision may be changed without updating this document first.

---

## ⚠️ v2.0.0 Revision Summary

v1.0.0 was written for a generic Python/Linux target and did not account for Attentia's production hardware (RK3568 SoC + IMX219 camera). The following categories of changes were made in this revision. **No code should be written against v1.0.0.**

| Change ID | Severity | Summary |
|---|---|---|
| CHANGE01 | CRITICAL | Replace L2CS-Net (ResNet-50) with MobileNetV3+LSTM — incompatible with RK3568 NPU budget |
| CHANGE02 | CRITICAL | RKNN model conversion moved from post-MVP to Phase 5 entry requirement |
| CHANGE03 | CRITICAL | LSTM stateful interface — hidden state must persist across frames, not be discarded |
| CHANGE04 | CRITICAL | IMX219 ISP pipeline fully specified — NV12 output, V4L2 configuration, NV12→BGR conversion |
| CHANGE05 | HIGH | Speed signal source specified — OBD-II / CAN / GPS priority stack + SpeedSource module |
| CHANGE06 | HIGH | Kalman filter added to head pose and gaze outputs — raw PnP jitter breaks duration timers |
| CHANGE07 | HIGH | Thread architecture defined — parallel phone + face perception pipelines |
| CHANGE08 | HIGH | Per-vehicle mounting calibration protocol — 10s startup calibration, persisted to disk |
| CHANGE09 | HIGH | Thermal throttling detection and DEGRADED fallback for RK3568 in dashboard enclosures |
| CHANGE10 | MEDIUM | V4L2 auto-exposure with face ROI weighting for driving light conditions |
| CHANGE11 | MEDIUM | Blink rate anomaly score formula formally defined (was referenced but undefined in v1.0.0) |
| CHANGE12 | MEDIUM | EAR and pose calibration persisted across power cycles (critical for ignition-powered device) |
| CHANGE13 | MEDIUM | Performance benchmarks revised — RK3568 targets added alongside M1 dev targets |
| CHANGE14 | MEDIUM | Watchdog manager added — detects hung inference threads silently |

---

## Table of Contents

1. Document Purpose and Scope
2. Formal Distraction Definition
3. System Architecture
4. API Contracts — All Inter-Layer Messages
5. Signal Processor — Mathematical Definitions
6. Distraction Scoring Engine
7. Alert State Machine
8. Functional Requirements
9. Event Log Format
10. Non-Functional Requirements
11. Model Specifications
12. Dataset Requirements and Labeling Schema
13. Evaluation and Validation Framework
14. Failure Modes and System Responses
15. Known Limitations
16. Out of Scope — MVP
17. Engineering Risks
18. Future Roadmap — Post-MVP Only
19. Configuration Reference — config.py
20. Recommended Implementation Order
21. RKNN Model Conversion Pipeline
22. Speed Signal Acquisition Module
23. Per-Vehicle Mounting Calibration Protocol
24. Session State Persistence

---

## 1. Document Purpose and Scope

This PRD covers software only: the perception stack, signal processing pipeline, temporal engine, distraction scoring model, alert state machine, dataset requirements, and evaluation framework.

Hardware selection, PCB design, enclosure, thermal management, and physical vehicle integration are explicitly out of scope.

### 1.1 Explicit Assumptions

v2.0.0 Change: Assumptions A-01 through A-07 have been revised to reflect IMX219 + RK3568 hardware reality. A-08 and A-09 are new.

| ID | Assumption | Impact if Violated |
|---|---|---|
| A01 [REVISED] | IMX219 delivers NV12 frames via V4L2 at 30 FPS (1280×720 mode). VideoSource converts NV12 to BGR24 before emitting RawFrame. Frame timestamps are sourced from V4L2 buffer metadata. V4L2 device node is /dev/video0 unless overridden in config. | ISP misconfiguration delivers raw Bayer data; BlazeFace input will be garbage. Re-verify ISP pipeline on each new board BSP. |
| A02 | Face-to-camera distance is 40–80 cm under typical dashboard mounting | Landmark model may fail outside this range; recalibration required |
| A03 | Vehicle speed is available via OBD-II, CAN, or GPS as specified in §22. 0.0 m/s is a valid value. If no source is available, URBAN zone is used as fallback per FM-05. | Speed-adaptive thresholding cannot function; static URBAN thresholds are used |
| A04 | System runs on Python 3.10+ on Linux (RK3568 Debian BSP); macOS acceptable for development only | Library compatibility must be re-verified for other environments |
| A05 | Ambient lighting is sufficient for RGB face detection (>5 lux minimum). Auto-exposure is enabled via V4L2 per FR-0.9. | BlazeFace detection rate degrades below 5 lux; IR pathway declared out of scope for MVP |
| A06 | Driver is the only person in the front seat during detection operation | Multi-face disambiguation is out of scope for MVP |
| A07 | All pretrained model weights are available under permissive open-source licenses. RKNN-converted weights are derived works — legal review required before distribution. | Legal review required before any commercial distribution |
| A08 [NEW] | RKNN Toolkit 2 version is pinned to the board BSP version. Mismatched versions can produce silent inference errors. | Model outputs incorrect. Re-convert all models if BSP is updated. |
| A09 [NEW] | RK3568 thermal design allows sustained 30fps inference without exceeding 90°C junction temperature. Thermal monitor per §14 FM-08 provides fallback. | Thermal throttling degrades timing-based detection logic. |

---

## 2. Formal Distraction Definition

**Critical Boundary:** The system detects physical proxies of distraction only. Cognitive distraction (mind-wandering, highway hypnosis) is not camera-detectable and is explicitly out of scope. No claim is made that detection of these physical signals guarantees detection of all dangerous cognitive states.

Distraction is defined as one or more of four measurable physical conditions, each sustained beyond a defined duration threshold while vehicle speed exceeds the minimum alerting speed.

### 2.1 Distraction Taxonomy

| Class | Name | Definition | Threshold | Trigger Duration |
|---|---|---|---|---|
| D-A | Visual Inattention | Estimated gaze direction falls outside the forward road zone (Yaw: ±15°, Pitch: -10° to +5° from level). All angles are corrected by neutral_pose_offset from calibration (§23) before comparison. | Gaze outside road zone continuously | ≥ 2.0 seconds |
| D-B | Head Pose Inattention | Head yaw > ±30° OR head pitch > ±20° from neutral, even when gaze is unavailable. Angles are neutral-pose-corrected per §23 and Kalman-filtered per §5.6 before comparison. | Head exceeds angular threshold continuously | ≥ 1.5 seconds |
| D-C | Drowsiness (PERCLOS) | Fraction of frames where eyes are ≥ 80% closed, measured over a 60-frame sliding window (2.0 seconds at 30 FPS) | PERCLOS > 0.15 in current window | Immediate on threshold breach (window-based) |
| D-D | Phone Use | Mobile phone detected with confidence ≥ 0.70 anywhere in frame. Independent alert path — does not require composite score threshold | Phone confidence ≥ 0.70 continuously | ≥ 1.0 second |

### 2.2 Forward Road Zone Definition

```python
ROAD_ZONE = {
    yaw_min_deg: -15.0,  # Negative = left of forward
    yaw_max_deg: +15.0,  # Positive = right of forward
    pitch_min_deg: -10.0,  # Negative = downward (road surface)
    pitch_max_deg: +5.0,  # Positive = upward
}

# Gaze is ON_ROAD if:
# yaw_min_deg <= gaze_world_yaw <= yaw_max_deg
# AND pitch_min_deg <= gaze_world_pitch <= pitch_max_deg
```

All ROAD_ZONE values are configurable via config.py. Gaze world angles are always neutral-pose-corrected before this comparison.

### 2.3 Speed Context Zones

| Zone | Speed Range | Alert Modifier |
|---|---|---|
| PARKED | 0.0 – 1.4 m/s (0 – 5 km/h) | 0.0 — All distraction alerts suppressed. Phone alert remains active. |
| URBAN | 1.4 – 13.9 m/s (5 – 50 km/h) | 1.0 — Baseline thresholds apply as specified. |
| HIGHWAY | > 13.9 m/s (> 50 km/h) | 1.4 — Composite score multiplied by 1.4 before threshold comparison. Effectively tightens alert sensitivity by 40%. |

Speed is sourced per the priority stack in §22. If no speed source is available, URBAN zone is used as the default (speed_modifier = 1.0).

---

## 3. System Architecture

### 3.1 Architecture Overview

The system is a six-layer unidirectional pipeline. Data flows from Layer 0 to Layer 5 in one direction only. No layer calls into the internals of any other layer. All inter-layer communication is through typed message objects with validated schemas. A failure in any layer causes that layer to emit a defined error signal; downstream layers handle error signals explicitly.

v2.0.0 Change (CHANGE-07): The pipeline is no longer purely sequential. Layers 0–1 run across three threads (T-0 VideoCapture, T-1 FacePerception, T-2 PhoneDetection) that merge before Layer 2. See §3.3 for the thread architecture.

v2.0.0 Change (CHANGE-03): The gaze/feature model in Layer 1 is stateful (LSTM hidden state). The PerceptionBundle carries lstm_hidden_state back to Layer 1 on the next frame call.

```
┌──────────────────────────────────────────────────────────────────────
│           ATTENTIA DRIVE — DETECTION ENGINE v2.0                    │
│                    DATA FLOW (UNIDIRECTIONAL)                        │
├──────────────────────────────────────────────────────────────────────
│                                                                      │
│  [IMX219 MIPI CSI-2] ── [V4L2 / RKISP1 ISP] ── [NV12 → BGR24]    │
│                          │ RawFrame{ts, id, bgr_data}               │
│                          ▼                                           │
│ ┌──────────────────────────────────────────────────────────────────┐ │
│ │         LAYER 1: PARALLEL PERCEPTION (CHANGE-07: T-1 ǁ T-2)    │ │
│ │                                                                  │ │
│ │  T-1: FacePerception             T-2: PhoneDetection            │ │
│ │  ├── BlazeFace Detector          └── YOLOv8-nano (every frame,  │ │
│ │  ├── PFLD Landmark (98pt)            independent of face gate)  │ │
│ │  └── MobileNetV3+LSTM Gaze ───────────────── stateful h_t, c_t │ │
│ │      (CHANGE-01, CHANGE-03)                                      │ │
│ │                                                                  │ │
│ │  [Synchronize on frame_id — 5ms phone timeout — merge PerceptionBundle] │ │
│ └──────────────────────────────────────────────────────────────────┘ │
│                          │ PerceptionBundle{..., lstm_hidden_state} │
│                          ▼                                           │
│              [LAYER 2: Signal Processor]                             │
│              ├── PnP Head Pose Solver                                │
│              ├── Kalman Filter (yaw, pitch, roll) (CHANGE-06)       │
│              ├── EAR / PERCLOS Calculator                            │
│              ├── Gaze World-Space Transformer                        │
│              ├── Kalman Filter (gaze yaw, pitch) (CHANGE-06)        │
│              ├── Neutral Pose Offset Correction (CHANGE-08)         │
│              └── Phone Signal Extractor                              │
│                          │ SignalFrame{...}                          │
│                          ▼                                           │
│              [LAYER 3: Temporal Engine]                              │
│              ├── CircularBuffer (120 frames)                         │
│              ├── Continuous Duration Timers                          │
│              ├── PERCLOS Sliding Window                              │
│              ├── Speed Context Resolver ── SpeedSource (§22)        │
│              ├── Blink Detector                                      │
│              ├── WatchdogManager (CHANGE-14)                        │
│              └── ThermalMonitor (CHANGE-09)                         │
│                          │ TemporalFeatures{F1..F5}                 │
│                          ▼                                           │
│              [LAYER 4: Scoring Engine]                               │
│              ├── Weighted Composite Scorer                           │
│              ├── Speed Modifier Application                          │
│              └── Threshold Evaluator                                 │
│                          │ DistractionScore{D, components}          │
│                          ▼                                           │
│              [LAYER 5: Alert State Machine]                          │
│              ├── State: NOMINAL / PRE_ALERT /                        │
│              │         ALERTING / COOLDOWN / DEGRADED               │
│              ├── Priority Arbitration                                │
│              └── Cooldown Manager                                    │
│                          │ AlertCommand{level, type}                 │
│                          ▼                                           │
│              [LAYER 6: Output Manager]                               │
│              ├── AudioAlertHandler                                   │
│              └── EventLogger                                         │
│                                                                      │
└──────────────────────────────────────────────────────────────────────
```

### 3.2 Module Directory Structure

```
attentia_drive/
├── main.py                          # Entry point. Wires all layers and threads.
├── config.py                        # Runtime configuration (thresholds, paths, hardware)
├── layer0_video/
│   ├── video_source.py              # Abstract VideoSource base class
│   ├── webcam_source.py             # Development webcam implementation
│   └── imx219_v4l2_source.py       # [NEW v2.0] RK3568 + IMX219 production source
│                                    # Handles: V4L2 open, ISP init via media-ctl,
│                                    # NV12 capture, NV12→BGR24 conversion,
│                                    # V4L2 timestamp extraction
├── layer1_perception/
│   ├── perception_stack.py          # Orchestrates models. Manages LSTM hidden state.
│   ├── face_detector.py             # BlazeFace wrapper (RKNN runtime)
│   ├── landmark_model.py            # PFLD 98-point wrapper (RKNN runtime)
│   ├── gaze_model.py                # MobileNetV3+LSTM wrapper — STATEFUL (RKNN)
│   └── phone_detector.py            # YOLOv8-nano wrapper (RKNN runtime)
├── layer2_signals/
│   ├── signal_processor.py          # Orchestrates signal computation
│   ├── head_pose_solver.py          # PnP solver, Euler angles
│   ├── kalman_filter.py             # [NEW v2.0] 1D Kalman filter (yaw, pitch, roll, gaze)
│   ├── ear_calculator.py            # Eye Aspect Ratio
│   ├── gaze_transformer.py          # Camera-space to world-space
│   ├── pose_calibration.py          # [NEW v2.0] Neutral pose offset correction
│   └── phone_signal.py              # Phone signal extractor
├── layer3_temporal/
│   ├── temporal_engine.py           # Orchestrates all temporal logic
│   ├── circular_buffer.py           # Fixed-length frame history
│   ├── duration_timer.py            # Continuous condition timers
│   ├── perclos_window.py            # Sliding window PERCLOS
│   ├── blink_detector.py            # Blink rate anomaly scorer
│   ├── speed_context.py             # Speed zone resolver
│   ├── speed_source.py              # [NEW v2.0] OBD-II / CAN / GPS speed acquisition
│   ├── watchdog.py                  # [NEW v2.0] WatchdogManager — detects hung threads
│   └── thermal_monitor.py           # [NEW v2.0] RK3568 thermal throttle detection
├── layer4_scoring/
│   ├── scoring_engine.py            # Composite scorer + threshold eval
│   └── feature_weights.py           # Weight definitions
├── layer5_alert/
│   ├── alert_state_machine.py       # NOMINAL/PRE_ALERT/ALERTING/COOLDOWN/DEGRADED FSM
│   └── alert_types.py               # Alert enum definitions
├── layer6_output/
│   ├── audio_handler.py             # System beep interface
│   └── event_logger.py              # Structured JSON event log
├── calibration/
│   ├── calibration_manager.py       # [NEW v2.0] Startup calibration orchestrator
│   └── session_state.json           # [NEW v2.0] Persisted calibration (auto-generated)
├── models/
│   ├── blazeface.rknn               # [CHANGED] RKNN-converted (was .onnx in v1.0.0)
│   ├── pfld_98pt.rknn               # [CHANGED] RKNN-converted
│   ├── gaze_mobilenetv3_lstm.rknn   # [CHANGED] MobileNetV3+LSTM replaces L2CS-Net
│   ├── yolov8n_phone.rknn           # [CHANGED] RKNN-converted (was .pt in v1.0.0)
│   └── CONVERSION_LOG.json          # [NEW v2.0] RKNN conversion audit trail
└── tests/
    ├── unit/                        # Per-module unit tests
    ├── integration/                 # End-to-end pipeline tests
    ├── hardware/                    # [NEW v2.0] RK3568-specific hardware tests
    │   ├── test_imx219_pipeline.py  # V4L2 + ISP + NV12→BGR validation
    │   ├── test_rknn_accuracy.py    # Per-model RKNN accuracy delta tests
    │   └── test_thermal_soak.py     # 2-hour thermal soak on production hardware
    └── fixtures/                    # Synthetic signal sequences
```

### 3.3 Thread Architecture

[NEW v2.0 — CHANGE-07] Phone detection and face perception run in parallel. This is required to meet NFR-P2 on RK3568 NPU.

| Thread Name | Models Run | Input | Output | OS Priority |
|---|---|---|---|---|
| T-0 VideoCapture | None — V4L2 read loop | V4L2 device /dev/video0 | RawFrame queue (depth 2, drop oldest on overflow) | High — SCHED_FIFO if available |
| T-1 FacePerception | BlazeFace → PFLD → MobileNetV3+LSTM | RawFrame from queue | Partial PerceptionBundle (face + landmarks + gaze) | High |
| T-2 PhoneDetection | YOLOv8-nano | Same RawFrame as T-1 (shared reference) | PhoneDetectionOutput | Normal |
| T-3 Pipeline | Layers 2–6 | Merged PerceptionBundle (T-1 + T-2 sync) | AlertCommand | Normal |

Synchronization: T-1 and T-2 consume the same RawFrame simultaneously. Their outputs are merged on frame_id as the join key. If T-2 has not completed within 5ms of T-1 completing, the PerceptionBundle is emitted using the last valid PhoneDetectionOutput with `phone_result_stale=True`. Stale count is logged.

---

## 4. API Contracts — All Inter-Layer Messages

### 4.1 Layer 0 → Layer 1: RawFrame

```python
@dataclass
class RawFrame:
    timestamp_ns: int  # V4L2 buffer timestamp (tv_sec * 1e9 + tv_usec * 1e3)
                       # Falls back to Python monotonic if V4L2 timestamp unavailable
    frame_id: int      # Monotonically increasing, resets on source restart
    width: int         # Pixels — 1280 in production (IMX219 1280×720 mode)
    height: int        # Pixels — 720 in production
    channels: int      # Always 3 (BGR24 — post NV12 conversion)
    data: np.ndarray   # Shape: (height, width, 3), dtype: uint8, BGR channel order
    source_type: str   # 'imx219_v4l2' | 'webcam' | 'file'
```

### 4.2 Layer 1 → Layer 2: PerceptionBundle

v2.0.0 Change (CHANGE-03): Added `lstm_hidden_state` and `lstm_reset_occurred` fields. Added `phone_result_stale` field (CHANGE-07).

```python
@dataclass
class FaceDetection:
    present: bool
    confidence: float       # [0.0, 1.0]
    bbox_norm: tuple        # (x, y, w, h) normalized to [0,1]
    face_size_px: int       # Width of bounding box in pixels

@dataclass
class LandmarkOutput:
    landmarks: np.ndarray   # Shape: (98, 2), normalized [0,1]
    confidence: float       # [0.0, 1.0]
    pose_valid: bool        # False if face rotation > reliable limit

@dataclass
class GazeOutput:
    left_eye_yaw: float     # Degrees (camera space, pre-transform)
    left_eye_pitch: float
    right_eye_yaw: float
    right_eye_pitch: float
    combined_yaw: float     # Weighted mean
    combined_pitch: float
    confidence: float       # [0.0, 1.0]
    valid: bool

@dataclass
class PhoneDetectionOutput:
    detected: bool
    max_confidence: float   # [0.0, 1.0], 0.0 if not detected
    bbox_norm: tuple        # (x, y, w, h) normalized, None if not detected

@dataclass
class PerceptionBundle:
    timestamp_ns: int
    frame_id: int
    face: FaceDetection
    landmarks: LandmarkOutput | None    # None if face not present
    gaze: GazeOutput | None             # None if face not present
    phone: PhoneDetectionOutput
    phone_result_stale: bool            # [NEW v2.0] True if T-2 timed out; last result used
    inference_ms: float                 # Total perception inference time
    # LSTM state — passed back to Layer 1 on next frame call (CHANGE-03)
    lstm_hidden_state: Any | None       # Opaque (h_t, c_t) tuple. None on first frame
                                        # or after face-absent reset (> 10 frame gap).
    lstm_reset_occurred: bool           # True if hidden state was reset this frame
```

### 4.3 Layer 2 → Layer 3: SignalFrame

v2.0.0 Change (CHANGE-06, CHANGE-08): All angle values in HeadPose and GazeWorld are now Kalman-filtered and neutral-pose-corrected before being placed in SignalFrame. Raw (unfiltered, uncorrected) values are available in separate debug fields.

```python
@dataclass
class HeadPose:
    yaw_deg: float      # Kalman-filtered + neutral-pose-corrected. Positive = right.
    pitch_deg: float    # Kalman-filtered + neutral-pose-corrected. Positive = up.
    roll_deg: float     # Kalman-filtered. Range: [-45, 45]
    valid: bool         # False if reprojection error > 8.0 px
    # Debug fields — raw values for logging/tuning (not used in detection logic)
    raw_yaw_deg: float
    raw_pitch_deg: float
    raw_roll_deg: float

@dataclass
class EyeSignals:
    left_EAR: float             # Eye Aspect Ratio [0.0, ~0.4]
    right_EAR: float
    mean_EAR: float
    baseline_EAR: float         # Per-session calibrated open-eye baseline
    close_threshold: float      # baseline_EAR * 0.75
    valid: bool
    calibration_complete: bool  # True after 30s of valid driving data collected

@dataclass
class GazeWorld:
    yaw_deg: float      # World-space gaze yaw — Kalman-filtered + neutral-pose-corrected
    pitch_deg: float    # World-space gaze pitch — same
    on_road: bool       # True if within ROAD_ZONE (compared against corrected angles)
    valid: bool

@dataclass
class PhoneSignal:
    detected: bool
    confidence: float   # [0.0, 1.0]
    stale: bool         # True if using T-2 timeout fallback result

@dataclass
class SignalFrame:
    timestamp_ns: int
    frame_id: int
    face_present: bool
    head_pose: HeadPose | None
    eye_signals: EyeSignals | None
    gaze_world: GazeWorld | None
    phone_signal: PhoneSignal
    speed_mps: float        # From SpeedSource (§22). 0.0 if unavailable.
    speed_stale: bool       # True if speed reading is older than SPEED_STALE_THRESHOLD_S
    signals_valid: bool     # False if any critical signal is unavailable
```

### 4.4 Layer 3 → Layer 4: TemporalFeatures

```python
@dataclass
class TemporalFeatures:
    timestamp_ns: int
    # F1: Fraction of window where gaze was OFF_ROAD
    gaze_off_road_fraction: float   # [0.0, 1.0]
    gaze_continuous_secs: float     # Duration of current continuous off-road event
    # F2: Mean angular head deviation in current window
    head_deviation_mean_deg: float  # Euclidean norm of yaw+pitch (corrected angles)
    head_continuous_secs: float     # Duration of current continuous head pose breach
    # F3: PERCLOS value in current 60-frame window
    perclos: float                  # [0.0, 1.0]
    # F4: Blink rate anomaly score (formula defined in §5.5)
    blink_rate_score: float         # [0.0, 1.0], 1.0 = maximally anomalous
    # F5: Mean phone confidence in current window
    phone_confidence_mean: float    # [0.0, 1.0]
    phone_continuous_secs: float    # Duration of current continuous phone detection
    # Context
    speed_zone: str                 # 'PARKED' | 'URBAN' | 'HIGHWAY'
    speed_modifier: float           # 0.0 | 1.0 | 1.4
    frames_valid_in_window: int     # Count of valid frames in buffer
    # Thermal state (CHANGE-09)
    thermal_throttle_active: bool   # True if ThermalMonitor has declared throttle state
```

### 4.5 Layer 4 → Layer 5: DistractionScore

```python
@dataclass
class DistractionScore:
    timestamp_ns: int
    # Composite score AFTER speed modifier applied
    composite_score: float          # [0.0, 1.0+]
    # Component scores (pre-modifier)
    component_gaze: float           # W1 * F1
    component_head: float           # W2 * F2_norm
    component_perclos: float        # W3 * F3
    component_blink: float          # W4 * F4
    # Threshold breach flags (evaluated independently)
    gaze_threshold_breached: bool   # continuous_secs >= T_GAZE
    head_threshold_breached: bool   # continuous_secs >= T_HEAD
    perclos_threshold_breached: bool  # perclos >= PERCLOS_THRESHOLD
    phone_threshold_breached: bool  # phone_continuous_secs >= T_PHONE
    # Which distraction classes are active
    active_classes: list[str]       # e.g. ['D-A', 'D-C']
```

### 4.6 Layer 5 → Layer 6: AlertCommand

```python
class AlertLevel(Enum):
    LOW = 1     # Advisory — no current use in MVP
    HIGH = 2    # Standard distraction alert
    URGENT = 3  # Phone use alert (highest priority)

class AlertType(Enum):
    VISUAL_INATTENTION = 'D-A'
    HEAD_INATTENTION = 'D-B'
    DROWSINESS = 'D-C'
    PHONE_USE = 'D-D'
    FACE_ABSENT = 'FACE'

@dataclass
class AlertCommand:
    alert_id: str           # UUID
    timestamp_ns: int
    level: AlertLevel
    alert_type: AlertType
    composite_score: float  # Score that triggered alert
    suppress_until_ns: int  # Do not repeat until this time
```

---

## 5. Signal Processor — Mathematical Definitions

### 5.1 Head Pose via PnP Solve

Head pose is recovered from 2D landmarks and a 3D mean face model using the Perspective-n-Point (PnP) algorithm. The 3D mean face model uses 6 reference points: nose tip, chin, left eye corner, right eye corner, left mouth corner, right mouth corner.

```python
# Inputs
L = {l_1 .. l_6}  # 2D landmark pixel coordinates (from PFLD output)
M = {m_1 .. m_6}  # 3D mean face model coordinates (millimeters)
K = camera_matrix  # Intrinsic matrix, estimated as:
                   # [[focal_len, 0, cx],
                   #  [0, focal_len, cy],
                   #  [0, 0, 1]]
                   # focal_len = frame_width (pinhole approximation)
                   # cx = frame_width / 2, cy = frame_height / 2

# Solve
R, t = cv2.solvePnP(M, L, K, dist_coeffs=np.zeros(4))
    # Method: SOLVEPNP_ITERATIVE

# Convert rotation vector to matrix
R_mat, _ = cv2.Rodrigues(R)

# Extract raw Euler angles
raw_yaw_deg   = atan2(R_mat[1,0], R_mat[0,0]) * (180/pi)
raw_pitch_deg = atan2(-R_mat[2,0], sqrt(R_mat[2,1]**2 + R_mat[2,2]**2)) * (180/pi)
raw_roll_deg  = atan2(R_mat[2,1], R_mat[2,2]) * (180/pi)

# Validity check
reprojection_error = mean(||l_i - project(R*m_i + t, K)||) for all i
pose_valid = reprojection_error < 8.0  # pixels

# Apply Kalman filter (§5.6) → filtered_yaw, filtered_pitch, filtered_roll
# Apply neutral pose offset correction (§5.7) → corrected_yaw, corrected_pitch
# HeadPose output uses corrected values; raw values stored in debug fields
```

### 5.2 Eye Aspect Ratio (EAR)

EAR is computed using the Soukupová & Čech (2016) formulation. Landmark indices reference the PFLD 98-point model output.

```python
# For each eye:
# p1, p2 = horizontal endpoints (inner/outer canthus)
# p3, p4 = upper lid points (two points)
# p5, p6 = lower lid points (two points)
EAR = (||p3 - p6|| + ||p4 - p5||) / (2 * ||p1 - p2||)

# Expected ranges:
# EAR ≈ 0.00 → eye fully closed
# EAR ≈ 0.28-0.35 → typical open eye (population mean)

# Per-session calibration (executed during first 30s at speed > V_MIN):
baseline_EAR = mean(mean_EAR samples collected during calibration window)
close_threshold = baseline_EAR * 0.75

# If calibration window not yet complete, use population default:
DEFAULT_CLOSE_THRESHOLD = 0.21

# [v2.0.0 — CHANGE-12] After calibration completes, persist to session_state.json.
# On next startup, load persisted values immediately (skip 30s wait).
# See §24 for persistence spec.
```

### 5.3 Gaze World-Space Transform

```python
# Raw gaze is in camera coordinate space (from MobileNetV3+LSTM output)
# Head pose contribution is combined to get world-space gaze
# Weighting factors (head contributes ~70% of total angular shift
# for large deviations — empirically validated in gaze research)
ALPHA = 0.7  # Head-to-gaze coupling, yaw
BETA  = 0.7  # Head-to-gaze coupling, pitch

# Note: head_yaw and head_pitch here are already Kalman-filtered and
# neutral-pose-corrected (§5.6, §5.7)
gaze_world_yaw   = gaze_camera_yaw   + (head_yaw   * ALPHA)
gaze_world_pitch = gaze_camera_pitch + (head_pitch  * BETA)

# Apply Kalman filter to gaze_world_yaw and gaze_world_pitch (§5.6)
# Validity: gaze_world is invalid if gaze OR head_pose is invalid
gaze_world_valid = gaze_output.valid AND head_pose.valid
```

### 5.4 PERCLOS Sliding Window

```python
# Window: last 60 frames (2.0s at 30 FPS)
# P80 definition: fraction of frames where eyes >= 80% closed
PERCLOS_WINDOW_FRAMES = 60
CLOSURE_FRACTION = 0.80  # Eyes considered 'closed' at 80% closure

# Per frame:
eyes_80pct_closed = mean_EAR <= (baseline_EAR * (1.0 - CLOSURE_FRACTION))

# PERCLOS over window:
PERCLOS = count(eyes_80pct_closed == True in window) / len(valid_frames_in_window)

# Validity: minimum 30 valid frames required for PERCLOS to be computed
perclos_valid = valid_frames_in_window >= 30
```

### 5.5 Blink Rate Anomaly Score

[NEW v2.0 — CHANGE-11] This formula was referenced in §6.1 (F4) but undefined in v1.0.0. It is now formally specified.

```python
# Normal human blink rate: 15–20 blinks/minute = 0.25–0.33 blinks/second
# Drowsiness signatures:
# - Reduced blink rate: < 8 blinks/min (0.13 Hz) — vigilance decrement
# - Elevated blink rate: > 30 blinks/min (0.50 Hz) — microsleep recovery / fatigue
BLINK_RATE_NORMAL_LOW_HZ  = 0.13  # 8 blinks/min lower bound
BLINK_RATE_NORMAL_HIGH_HZ = 0.50  # 30 blinks/min upper bound

# blink_rate_hz = blink events in current window / window_duration_seconds
# Blink detection: EAR transition from above close_threshold to below it,
# sustained for BLINK_MIN_FRAMES to BLINK_MAX_FRAMES (67–333ms at 30fps)
if blink_rate_hz < BLINK_RATE_NORMAL_LOW_HZ:
    # Abnormally low blink rate — drowsiness indicator
    score = 1.0 - (blink_rate_hz / BLINK_RATE_NORMAL_LOW_HZ)
elif blink_rate_hz > BLINK_RATE_NORMAL_HIGH_HZ:
    # Abnormally high blink rate — fatigue/recovery indicator
    score = min(1.0, (blink_rate_hz - BLINK_RATE_NORMAL_HIGH_HZ) / 0.5)
else:
    # Within normal range
    score = 0.0

blink_rate_score = clamp(score, 0.0, 1.0)  # F4 — already [0,1] by construction above
```

### 5.6 Kalman Filter for Pose and Gaze Smoothing

[NEW v2.0 — CHANGE-06] Raw PnP output has frame-to-frame jitter of ±3–8°. Without filtering, duration timers in Layer 3 reset repeatedly on jitter spikes before the threshold is crossed, suppressing valid alerts. A 1D constant-velocity Kalman filter is applied independently to each angle output.

State vector: `[angle, angular_velocity]` (2D)

```python
# Applied independently to: head_yaw, head_pitch, head_roll, gaze_world_yaw, gaze_world_pitch

# Kalman parameters (tuned for human head movement dynamics):
KALMAN_PROCESS_NOISE_Q    = 0.01   # Low = trust model; increase for fast head movements
KALMAN_MEASUREMENT_NOISE_R = 4.0  # ±2° PnP measurement noise (variance)
KALMAN_INITIAL_COVARIANCE  = 1.0

# State transition (constant velocity model, dt = 1/FPS):
# x_k = F * x_{k-1} + process_noise
# F = [[1, dt],
#      [0, 1]]

# Measurement: z_k = H * x_k + measurement_noise
# H = [1, 0]  (observe angle only, not velocity)

# Filter reset conditions:
# - head_pose.valid transitions to False (face lost)
# - face absent for > 10 consecutive frames
# - Source restart (frame_id resets)
```

Validation target: Kalman filter SHALL reduce frame-to-frame standard deviation by ≥ 60% on a synthetic noisy angle sequence (injected noise σ = 5°, clean signal σ ≤ 2°).

### 5.7 Neutral Pose Offset Correction

[NEW v2.0 — CHANGE-08] Camera mounting angle varies between vehicles. Without correction, the same driver looking straight ahead at the road produces different yaw/pitch readings across mounting configurations.

```python
# Load from calibration/session_state.json on startup.
# If file not found or VIN mismatch: run calibration sequence (§23).
neutral_yaw_offset   = session_state['neutral_yaw_offset']    # degrees
neutral_pitch_offset = session_state['neutral_pitch_offset']  # degrees

# Applied every frame, after Kalman filtering, before threshold comparison:
corrected_yaw   = filtered_yaw   - neutral_yaw_offset
corrected_pitch = filtered_pitch - neutral_pitch_offset

# All downstream consumers (GazeWorld, HeadPose in SignalFrame) receive corrected values.
# Raw values are preserved in HeadPose.raw_yaw_deg, HeadPose.raw_pitch_deg for debugging.
```

---

## 6. Distraction Scoring Engine

### 6.1 Composite Score Formula

```python
# Feature weights — initialized from NHTSA 100-Car Study literature
# (Klauer et al. 2006) and PERCLOS validation studies (Wierwille 1994)
# These values are PROVISIONAL and must be validated against DMD dataset
W1 = 0.45  # Gaze off-road fraction (strongest predictor per NHTSA data)
W2 = 0.30  # Head deviation (secondary signal)
W3 = 0.20  # PERCLOS (drowsiness, orthogonal signal)
W4 = 0.05  # Blink rate anomaly (weak signal, supports PERCLOS)

# Assert weights sum to 1.0
assert abs(W1 + W2 + W3 + W4 - 1.0) < 1e-6

# Feature normalization
F1 = gaze_off_road_fraction                        # Already [0.0, 1.0]
F2_norm = clamp(head_deviation_mean / 30.0, 0.0, 1.0)
F3 = perclos                                       # Already [0.0, 1.0]
F4 = blink_rate_anomaly_score                      # [0.0, 1.0] per §5.5

# Raw composite score
D_raw = W1*F1 + W2*F2_norm + W3*F3 + W4*F4

# Speed-modulated composite score
D = D_raw * speed_modifier

# Composite alert threshold
COMPOSITE_ALERT_THRESHOLD = 0.55
composite_alert = D >= COMPOSITE_ALERT_THRESHOLD
```

### 6.2 Alert Threshold Table

Alerts fire when ANY individual threshold is breached OR the composite score exceeds its threshold. Conditions are evaluated independently and in parallel.

| Alert ID | Condition | Threshold Value | Priority | Cooldown |
|---|---|---|---|---|
| ALT-01 | Gaze continuous off-road (corrected angles) | ≥ 2.0 seconds | HIGH | 8.0 seconds |
| ALT-02 | Head pose continuous breach (corrected angles) | \|yaw\| > 30° OR \|pitch\| > 20° for ≥ 1.5s | HIGH | 8.0 seconds |
| ALT-03 | PERCLOS drowsiness threshold | PERCLOS ≥ 0.15 in 60-frame window | HIGH | 12.0 seconds |
| ALT-04 | Phone use detection | Confidence ≥ 0.70 for ≥ 1.0 second | URGENT | 5.0 seconds |
| ALT-05 | Composite score threshold | D ≥ 0.55 (speed-modulated) | HIGH | 8.0 seconds |
| ALT-06 | Face absent while moving | No face detected for ≥ 5.0s at speed > V_MIN | HIGH | 10.0 seconds |

---

## 7. Alert State Machine

### 7.1 State Definitions

| State | Description | Entry Condition |
|---|---|---|
| NOMINAL | No distraction condition active. System monitoring normally. | Initial state, OR all thresholds cleared AND cooldown expired |
| PRE_ALERT | A threshold has been breached. Duration timer is running. No alert fired yet. | Any threshold condition becomes true |
| ALERTING | Alert has fired. Audio output triggered. Suppression active. | Duration threshold crossed; AlertCommand emitted |
| COOLDOWN | Alert recently fired. Repeat suppression in effect. Monitoring continues. | Immediately after ALERTING; duration = per-alert cooldown value |
| DEGRADED | One or more perception modules failed or produced invalid output for > 2.0s, OR thermal throttle has entered critical state (FM-08). | perception_stack signals invalid for > 60 consecutive frames, OR ThermalMonitor signals CRITICAL |

### 7.2 State Transition Diagram

```
                [SYSTEM START]
                      │
                      ▼
          ┌─────────────────────┐
          │       NOMINAL       │ ─────────────────────────┐
          └─────────────────────┘                          │
                      │                                    │
          [Threshold  │            [Perception invalid     │
          condition   │            > 60 frames, OR         │
          becomes     │            thermal CRITICAL]        │
          true]       │                                    │
                      │                    ▼               │
                      │    ┌───────────────────────┐       │
                      │    │       DEGRADED         │       │
                      │    │  No alerts fire.       │       │
                      │    │  Log degraded event.   │       │
                      │    └───────────────────────┘       │
                      │                │                   │
                      │    [Perception │ recovers           │
                      │    30 consec  │ valid frames        │
                      │    AND thermal│ normal]             │
                      │               └────────────────────┘
                      ▼
          ┌─────────────────────┐
          │      PRE_ALERT      │ [Condition clears before
          │  Timer running.     │──duration threshold] ──────────────
          │  No alert yet.      │
          └─────────────────────┘
                      │
          [Duration   │
          threshold   │
          crossed]    │
                      ▼
          ┌─────────────────────┐
          │      ALERTING       │ Emit AlertCommand.
          │  Audio fires.       │ suppressed_until = now + cooldown
          └─────────────────────┘
                      │
          [Immediate] │
                      ▼
          ┌─────────────────────┐
          │      COOLDOWN       │ No repeat alert of same type.
          │                     │ Other alert types CAN fire.
          │  suppress_until_ns  │ Phone alert (URGENT) overrides
          │  tracked per type   │ ALL suppression.
          └─────────────────────┘
                      │
          [suppress_  │
          until_ns    │
          elapsed]    │
                      └───────────────────────────────────────────── NOMINAL
```

### 7.3 Priority Arbitration Rules

| Rule | Condition | Behavior |
|---|---|---|
| P-01 | Phone alert (ALT-04) fires while any other alert is in COOLDOWN | Phone alert overrides cooldown suppression and fires immediately |
| P-02 | Multiple non-phone conditions breach simultaneously | Alert fires once with active_classes listing all breached conditions |
| P-03 | Face absent (ALT-06) fires while distraction alert in COOLDOWN | Face absent alert fires immediately — independent suppression counter |
| P-04 | System is in DEGRADED state | No alert fires. DEGRADED state is logged. No false alert from invalid signals. |
| P-05 | Speed zone is PARKED | Only ALT-04 (phone) can fire. All other alerts suppressed regardless of score. |
| P-06 [NEW v2.0] | Thermal DEGRADED — system entered DEGRADED due to thermal throttle | No alert fires. THERMAL_DEGRADED event logged. Audio: 3 short low-tone beeps to signal driver of device issue. |

---

## 8. Functional Requirements

### 8.1 Layer 0 — VideoSource

v2.0.0 Change (CHANGE-04): Requirements FR-0.1 through FR-0.4 unchanged. FR-0.5 through FR-0.8 are new and specify the IMX219/V4L2 pipeline. FR-0.9 and FR-0.10 are new for lighting adaptation (CHANGE-10).

| ID | Requirement | Verification |
|---|---|---|
| FR0.1 | VideoSource SHALL accept a V4L2 device path string (e.g. `/dev/video0`) or a video file path (string) as its sole constructor argument. Webcam device index (integer) is supported for development only. | Unit test |
| FR0.2 | VideoSource SHALL emit RawFrame objects at the source's native frame rate with monotonically increasing frame_id values | Unit test |
| FR0.3 | VideoSource SHALL raise `SourceUnavailableError` if the source cannot be opened within 5.0 seconds | Unit test with mock |
| FR0.4 | VideoSource SHALL provide a `release()` method that closes the source cleanly without resource leaks | Memory/resource test |
| FR0.5 [NEW] | `imx219_v4l2_source.py` SHALL configure the V4L2 device with `VIDIOC_S_FMT` to request NV12 output format at 1280×720 resolution and 30 FPS before capture begins. If the ISP does not support 1280×720, fall back to 1920×1080 with center-crop to 1280×720. | V4L2 format query test on RK3568 |
| FR0.6 [NEW] | VideoSource SHALL convert each NV12 frame to BGR24 using `cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_NV12)` before emitting RawFrame. No raw Bayer data or unconverted YUV data shall ever be emitted. | Color channel validation test: verify 3-channel uint8 output |
| FR0.7 [NEW] | VideoSource SHALL read frame timestamp from V4L2 buffer metadata (`tv_sec`, `tv_usec` fields of struct timeval). Python `time.monotonic_ns()` is used as fallback only if V4L2 timestamp is unavailable. Timestamp source is logged at startup. | Timestamp monotonicity test |
| FR0.8 [NEW] | `imx219_v4l2_source.py` SHALL initialize the RKISP1 ISP pipeline via `media-ctl` command before opening the V4L2 device. Auto-exposure (AE) and auto-white-balance (AWB) SHALL be enabled in the ISP configuration. Initialization failure raises `ISPInitError`. | ISP pipeline state verification on RK3568 |
| FR0.9 [NEW] | VideoSource SHALL enable V4L2 auto-exposure (`V4L2_CID_EXPOSURE_AUTO = V4L2_EXPOSURE_AUTO`) during initialization. If a valid face bounding box is available from the previous frame, the AE region shall be weighted toward the face ROI using `V4L2_CID_AUTO_EXPOSURE_BIAS` or equivalent ISP control. | AE convergence test across lux range (100 lux, 1000 lux, 50,000 lux) |
| FR0.10 [NEW] | VideoSource SHALL monitor mean pixel intensity of the face ROI (when available). If mean intensity falls below `LOW_LIGHT_THRESHOLD` (default: 30 out of 255) or exceeds `OVEREXPOSE_THRESHOLD` (default: 240) for 10 consecutive frames, a `LIGHTING_WARNING` event is logged. DEGRADED entry threshold is raised to 90 invalid frames (instead of 60) to allow AE convergence time. | Synthetic brightness injection test |

### 8.2 Layer 1 — Perception Stack

v2.0.0 Change (CHANGE-01, CHANGE-02, CHANGE-03): L2CS-Net replaced with MobileNetV3+LSTM. All model wrappers use RKNN runtime (not PyTorch or ONNX runtime). Gaze model is stateful.

| ID | Requirement | Verification |
|---|---|---|
| FR1.1 | The face detector SHALL use BlazeFace (RKNN-converted, `blazeface.rknn`) and output FaceDetection with bounding box and confidence per frame | Unit test on DMD frames |
| FR1.2 | The landmark model SHALL be PFLD 98-point (RKNN-converted, `pfld_98pt.rknn`) and SHALL NOT execute if face confidence < 0.60 | Unit test: confidence gate |
| FR1.3 [REVISED] | The gaze model SHALL be MobileNetV3+LSTM (RKNN-converted, `gaze_mobilenetv3_lstm.rknn`) operating on the face crop extracted from PFLD landmarks. It SHALL maintain LSTM hidden state (h_t, c_t) across frames within a session. It SHALL NOT execute if landmark confidence < 0.65. The hidden state SHALL reset when face_present has been False for > 10 consecutive frames. | Unit test: stateful inference, hidden state persistence, reset trigger |
| FR1.4 | The phone detector SHALL be YOLOv8-nano (RKNN-converted, `yolov8n_phone.rknn`) running on the full frame at 320×320 resolution. It SHALL run every frame regardless of face detection result. It runs in thread T-2 (§3.3). | Unit test: independence, T-2 thread |
| FR1.5 | Each model module SHALL expose a standard interface: face/landmark/phone: `infer(frame: np.ndarray) -> OutputType`; gaze model: `infer(face_crop: np.ndarray, hidden_state: tuple | None) -> (GazeOutput, new_hidden_state)`. No module SHALL access another module directly. | Interface compliance test |
| FR1.6 | The Perception Stack SHALL catch all model runtime exceptions, log them, and return a PerceptionBundle with `face.present=False` and `lstm_reset_occurred=True` rather than propagating exceptions upward | Exception injection test |
| FR1.7 [NEW] | All four model files SHALL be `.rknn` format and loaded using `rknn_toolkit2_lite.RKNNLite`. No PyTorch, TensorFlow, or ONNX runtime SHALL be used in the production pipeline. ONNX runtime is permitted in `tests/` only. | Runtime check at model load |
| FR1.8 [NEW] | Model inference in T-1 and T-2 threads SHALL use separate RKNN runtime instances (one per model file). No runtime instance is shared across threads. | Concurrency isolation test |

### 8.3 Layer 2 — Signal Processor

| ID | Requirement | Verification |
|---|---|---|
| FR2.1 | Head pose SHALL be computed via OpenCV `solvePnP` (SOLVEPNP_ITERATIVE) using 6 reference landmarks against the defined 3D mean face model. Output SHALL be Euler angles in degrees before filtering. | Compare vs ground truth on 300W subset |
| FR2.2 | EAR SHALL be computed per the Soukupová (2016) formula. Baseline calibration SHALL execute during the first valid 30 seconds of operation at speed > V_MIN. Before calibration completes, the default threshold (0.21) SHALL be used. [REVISED] If session_state.json exists and `calibration_complete` is True, the persisted `baseline_EAR` and `close_threshold` are loaded immediately on startup — skipping the 30s wait. | Unit test: calibration sequence; cold start vs warm start |
| FR2.3 | Gaze world transform SHALL apply ALPHA=0.7, BETA=0.7 coupling factors. The ROAD_ZONE boundary check SHALL use neutral-pose-corrected gaze angles and produce a boolean `on_road` flag per frame. | Unit test with synthetic angles |
| FR2.4 | Signal Processor SHALL set `signals_valid=False` and propagate downstream when `head_pose.valid=False` AND `gaze_world.valid=False` simultaneously | Unit test: dual-invalid case |
| FR2.5 [NEW] | Head pose yaw, pitch, and roll SHALL each be filtered by an independent 1D Kalman filter (§5.6) before being placed in HeadPose. Raw (unfiltered) values SHALL be preserved in HeadPose debug fields. Kalman state SHALL reset when `head_pose.valid` transitions to False. | Unit test: verify filter reduces frame-to-frame std dev by ≥ 60% on synthetic noisy angle sequence |
| FR2.6 [NEW] | Gaze world yaw and pitch SHALL be Kalman-filtered (same parameterization as FR-2.5) after the world-space transform in §5.3. Filter state resets on gaze validity change. | Unit test: gaze jitter reduction |
| FR2.7 [NEW] | Neutral pose offsets (neutral_yaw_offset, neutral_pitch_offset) SHALL be applied to filtered head pose angles and corrected gaze world angles per §5.7. Offsets are loaded from session_state.json. If not available, offsets default to 0.0 and a `CALIBRATION_MISSING` warning is logged. | Unit test: offset application |

### 8.4 Layer 3 — Temporal Engine

| ID | Requirement | Verification |
|---|---|---|
| FR3.1 | Circular buffer SHALL maintain exactly 120 SignalFrame entries. On overflow, the oldest entry is evicted. Buffer SHALL be fully initialized before TemporalFeatures are computed. | Unit test: overflow behavior |
| FR3.2 | The continuous gaze-off-road timer SHALL increment by frame_delta_seconds each frame where `gaze_world.on_road=False` AND `face_present=True` AND `signals_valid=True`. It SHALL reset to 0.0 on the first frame where gaze is ON_ROAD. | Unit test: timer sequences |
| FR3.3 | PERCLOS SHALL be computed over the last 60 valid SignalFrames. Frames where `signals_valid=False` SHALL be excluded from the denominator. PERCLOS SHALL be marked invalid if fewer than 30 valid frames exist in the window. | Unit test: partial validity |
| FR3.4 | Speed zone classification SHALL use the thresholds defined in §2.3. Speed input SHALL be clamped to [0.0, 100.0] m/s before classification. A speed value of None or `speed_stale=True` SHALL default to URBAN zone and log a warning. | Unit test: boundary values, stale handling |
| FR3.5 | Blink detection SHALL identify EAR transitions from above `close_threshold` to below it, lasting 2–10 frames (67–333ms at 30 FPS). Events outside this window are not counted as blinks. Blink rate is computed as events per second over the current window. Blink_rate_score computed per §5.5. | Unit test with synthetic EAR sequences |
| FR3.6 [NEW] | WatchdogManager SHALL be instantiated in Layer 3 and kicked every frame via `watchdog.kick(frame_id)`. If no kick is received within `WATCHDOG_TIMEOUT_S` (2.0s), WatchdogManager SHALL log a `WATCHDOG_TIMEOUT` event and signal Layer 5 to enter DEGRADED state. WatchdogManager SHALL then attempt to join and restart T-1 and T-2 threads. | Fault injection test: block T-1 for 3s, verify DEGRADED entry and restart |
| FR3.7 [NEW] | ThermalMonitor SHALL poll `/sys/class/thermal/thermal_zone0/temp` every `THERMAL_CHECK_INTERVAL_S` (5.0s) on a low-priority background thread. At `THERMAL_WARN_TEMP_C` (80°C): log warning, reduce YOLOv8-nano input resolution from 320 to 256. At `THERMAL_CRITICAL_TEMP_C` (90°C): signal Layer 5 to enter DEGRADED via thermal path. | Hardware soak test on RK3568 in enclosure at 40°C ambient |

### 8.5 Layer 4 — Scoring Engine

| ID | Requirement | Verification |
|---|---|---|
| FR4.1 | Composite score D SHALL be computed per the formula in §6.1. All weights SHALL be loaded from config.py at startup and validated to sum to 1.0 ± 1e-6. | Unit test: weight validation |
| FR4.2 | Each of the 6 alert conditions (ALT-01 through ALT-06) SHALL be evaluated independently. A DistractionScore object SHALL be emitted every frame regardless of whether thresholds are breached. | Unit test: per-alert evaluation |
| FR4.3 | If `speed_modifier=0.0` (PARKED zone), `composite_score` SHALL be set to 0.0 and all non-phone threshold breach flags SHALL be set to False before DistractionScore is emitted. | Unit test: parked suppression |

### 8.6 Layer 5 — Alert State Machine

| ID | Requirement | Verification |
|---|---|---|
| FR5.1 | The state machine SHALL initialize in NOMINAL state. Alert FSM state SHALL be persisted in memory only — no FSM state is written to disk. (Calibration state is separate and IS persisted — see §24.) | Unit test: initialization |
| FR5.2 | Cooldown suppression SHALL be tracked per AlertType independently. An URGENT (phone) alert SHALL bypass all suppression for all other alert types. | Unit test: cooldown isolation |
| FR5.3 | Every state transition SHALL produce a log entry via the EventLogger. The log entry SHALL include: previous state, new state, triggering condition, and timestamp_ns. | Integration test: log validation |
| FR5.4 | The DEGRADED state SHALL suppress all alert emissions. The state machine SHALL transition back to NOMINAL when 30 consecutive valid PerceptionBundles are received AND ThermalMonitor is not in CRITICAL state. | Integration test: degraded recovery |

### 8.7 Layer 6 — Output Manager

| ID | Requirement | Verification |
|---|---|---|
| FR6.1 | AudioAlertHandler SHALL emit a system beep via the platform audio API (Linux: `aplay`; macOS: `afplay` for development). Alert dispatch latency from AlertCommand receipt to audio start SHALL be ≤ 50ms. | Latency measurement test |
| FR6.2 | AudioAlertHandler SHALL be swappable via a defined AudioBackend interface without changes to any other layer. The interface exposes a single method: `play_alert(level: AlertLevel) -> None`. | Interface compliance test |
| FR6.3 | EventLogger SHALL write structured JSON log entries to a rotating log file. Maximum file size: 50 MB. Maximum files retained: 5. Log format is defined in §9. | Log format validation test |

---

## 9. Event Log Format

Every alert event and every state transition is logged in the following JSON format. No image or video data is logged.

```json
// Alert Event Log Entry
{
    "event_type": "ALERT",
    "timestamp_ns": 1740000000000000000,
    "alert_id": "uuid-v4-string",
    "alert_type": "D-A",
    "alert_level": "HIGH",
    "composite_score": 0.67,
    "active_classes": ["D-A", "D-B"],
    "speed_mps": 14.2,
    "speed_zone": "HIGHWAY",
    "speed_source": "OBD2",
    "gaze_continuous_secs": 2.1,
    "head_continuous_secs": 1.6,
    "perclos": 0.08,
    "phone_confidence": 0.0,
    "suppress_until_ns": 1740000008000000000
}

// State Transition Log Entry
{
    "event_type": "STATE_TRANSITION",
    "timestamp_ns": 1740000000000000000,
    "previous_state": "NOMINAL",
    "new_state": "ALERTING",
    "trigger": "ALT-01",
    "frame_id": 4521
}

// Degraded Entry
{
    "event_type": "DEGRADED",
    "timestamp_ns": 1740000000000000000,
    "reason": "perception_invalid_60_frames",
    "duration_secs": 2.0
}

// [NEW v2.0] Thermal Warning Entry
{
    "event_type": "THERMAL_WARNING",
    "timestamp_ns": 1740000000000000000,
    "temperature_c": 82.0,
    "action": "reduced_yolo_resolution_to_256",
    "inference_ms_mean": 95.3
}

// [NEW v2.0] Watchdog Timeout Entry
{
    "event_type": "WATCHDOG_TIMEOUT",
    "timestamp_ns": 1740000000000000000,
    "last_frame_id": 10531,
    "secs_since_last_frame": 2.14,
    "recovery_action": "thread_restart_attempted"
}

// [NEW v2.0] Calibration Entry
{
    "event_type": "CALIBRATION_COMPLETE",
    "timestamp_ns": 1740000000000000000,
    "baseline_ear": 0.31,
    "neutral_yaw_offset": -4.2,
    "neutral_pitch_offset": 3.1,
    "vehicle_vin": "1HGCM82633A004352",
    "frames_collected": 298
}
```

---

## 10. Non-Functional Requirements

### 10.1 Performance

v2.0.0 Change (CHANGE-13): RK3568 production targets added. M1 Mac targets are development-only. Validation in §13 Suite 4 must use RK3568 hardware, not M1. All NFR-P requirements are bindings on the production target.

| ID | Requirement | RK3568 Production Target | M1 Mac Dev Target | Measurement Method |
|---|---|---|---|---|
| NFR-P1 | End-to-end pipeline latency: frame capture to alert dispatch | ≤ 200ms P95 | ≤ 150ms P95 | Pipeline timer, 1000 frame sample on RK3568 |
| NFR-P2 | Perception stack total inference time (all 4 models, RKNN, parallel T-1+T-2) | ≤ 80ms P95 (requires RKNN + threading) | ≤ 50ms P95 | Inference timer, 1000 frame sample |
| NFR-P2b | Perception inference during thermal throttle state | ≤ 120ms P95 acceptable — degrade gracefully, not hard fail | N/A | Same as NFR-P2 with RK3568 at ≥ 85°C |
| NFR-P3 | Temporal engine and scoring computation time per frame | ≤ 8ms P99 | ≤ 3ms P99 | Function timer |
| NFR-P4 | Audio dispatch latency from AlertCommand to audio start | ≤ 50ms P95 | ≤ 50ms P95 | Timestamp comparison |
| NFR-P5 | False positive rate: spurious alerts during non-distracted driving | < 1.0 alert per hour | < 1.0 alert per hour | Evaluation protocol §13 — platform independent |
| NFR-P6 | True positive rate (recall) on DMD labeled distraction events | ≥ 0.80 | ≥ 0.80 | Evaluation protocol §13 |

### 10.2 Reliability

| ID | Requirement | Verification |
|---|---|---|
| NFR-R1 | System SHALL recover from a single module crash (exception) within 2.0 seconds without requiring process restart. WatchdogManager (FR-3.6) provides active detection of hung threads in addition to exception handling. Recovery enters DEGRADED state, then NOMINAL when perception recovers. | Fault injection test + watchdog timeout test |
| NFR-R2 | System SHALL maintain NOMINAL operation for a minimum 4-hour continuous session without memory growth exceeding 50 MB above baseline | 4-hour soak test with profiler on RK3568 |
| NFR-R3 | Circular buffer write operations SHALL be atomic. No intermediate buffer state shall be observable by downstream layers. | Concurrency unit test |
| NFR-R4 [NEW] | System SHALL maintain correct operation during RK3568 thermal management events. Frame timing jitter due to CPU frequency scaling SHALL not cause duration timers to fire spuriously. All timers use frame_delta_seconds (computed from timestamps) rather than assuming a fixed 1/30s interval. | Thermal soak + timer accuracy test |

### 10.3 Modularity and Testability

| ID | Requirement | Verification |
|---|---|---|
| NFR-M1 | Every module SHALL be independently instantiable and testable with mock inputs, without requiring any other layer to be present | Module isolation tests |
| NFR-M2 | All threshold values SHALL be defined in config.py and referenced by name. No magic numbers shall appear in pipeline logic code. | Code review / grep audit |
| NFR-M3 | Each model file (`.rknn`) SHALL be replaceable by updating config.py model path alone, with no code changes required to any other file | Model swap test |
| NFR-M4 | Test coverage SHALL reach ≥ 85% line coverage across all pipeline logic code (layers 2–5). Models and layer 0/1 wrappers are excluded from coverage targets. | pytest-cov report |

---

## 11. Model Specifications

### 11.1 Model Registry

v2.0.0 Change (CHANGE-01, CHANGE-02): L2CS-Net replaced with MobileNetV3+LSTM. All models in `.rknn` format. RKNN conversion pipeline is §21.

| Role | Model | Architecture | Input | Output | Size (pre-RKNN) | Format |
|---|---|---|---|---|---|---|
| Face Detection | BlazeFace (MediaPipe) | MobileNet backbone | 128×128 RGB | BBox + confidence | ~400 KB | .rknn |
| Landmarks | PFLD 98-point | MobileNet + auxiliary branches | 112×112 RGB face crop | 98 × (x, y) normalized | ~4 MB | .rknn |
| Gaze Estimation | MobileNetV3+LSTM (Attentia internal) | MobileNetV3-Small + LSTM (8-frame temporal window) | 112×112 RGB face crop | Yaw, pitch (degrees) + temporal confidence | ~3–5 MB | .rknn |
| Phone Detection | YOLOv8-nano | CSP-DarkNet nano | 320×320 RGB full frame (256×256 if thermal throttle active) | BBox + class + confidence | ~3.2 MB | .rknn |

### 11.2 Model Confidence Gates

```
BlazeFace.confidence >= 0.60
 └─ PFLD executes (face crop passed to landmark model)
     PFLD.confidence >= 0.65
      └─ MobileNetV3+LSTM executes (face crop extracted from landmarks)
          LSTM hidden state (h_t, c_t) persists across frames
          Hidden state resets if face absent > 10 consecutive frames

YOLOv8-nano: executes EVERY frame, independent of face detection gate
 Runs in parallel thread T-2 (see §3.3)

If any gate fails:
 - Set downstream outputs to None
 - Set signals_valid=False for signals dependent on that model
 - Log gate failure event with frame_id
 - If gaze model gate fails: reset lstm_hidden_state, set lstm_reset_occurred=True
```

### 11.3 Training and Fine-Tuning Requirements

| Model | Training Source | Fine-Tuning Requirement | RKNN Validation Requirement |
|---|---|---|---|
| BlazeFace | MediaPipe pretrained — use as-is | No fine-tuning required for MVP | RKNN recall delta ≤ 3% relative vs FP32 on DMD |
| PFLD 98-point | Pretrained on 300W + WFLW datasets | No fine-tuning required; validate MAE < 5° on 300W test split | RKNN NME delta ≤ 0.5% absolute |
| MobileNetV3+LSTM | Attentia internal training; validate mean angular error < 6° on MPIIFaceGaze test split | No additional fine-tuning required for MVP — internal weights are the source | RKNN MAE delta ≤ 0.5° absolute; verify LSTM state correctness on RK3568 |
| YOLOv8-nano | Pretrained on COCO; fine-tune on AUC Distracted Driver Dataset (phone class only) | Fine-tuning required; target mAP50 ≥ 0.85 on held-out AUC test split | RKNN mAP50 delta ≤ 0.03 absolute (≥ 0.82 after conversion) |

---

## 12. Dataset Requirements and Labeling Schema

### 12.1 Required Public Datasets

| Dataset | Used For | Access | Size |
|---|---|---|---|
| DMD (Vicomtech 2020) | End-to-end pipeline validation, PERCLOS ground truth, distraction event labels | Free — Box account request at dmd.vicomtech.org | 41 hours, 37 drivers, RGB+IR+depth |
| AUC Distracted Driver | YOLOv8-nano phone detector fine-tuning (10 distraction classes including phone) | Free — Kaggle download | 22,424 images, 10 classes |
| 300W / WFLW | PFLD landmark model validation (head pose MAE benchmark) | Free — academic download | 300W: 3,148 train / 689 test images |
| MPIIFaceGaze | MobileNetV3+LSTM gaze estimation validation | Free — ETH Zurich academic request | 213,659 images, 15 participants |

### 12.2 DMD Labeling Schema for Validation

| Distraction Class | DMD Label(s) | Mapping Notes |
|---|---|---|
| D-A: Visual Inattention | looking_road / not_looking_road | not_looking_road for ≥ 2.0s continuous = positive D-A event |
| D-B: Head Pose | Head pose annotations (yaw/pitch) | Derived from gaze ground truth; head pose > angular thresholds = positive D-B. Neutral-pose correction should be applied if DMD driver calibration data is available. |
| D-C: Drowsiness | eyes_closed / drowsy labels | PERCLOS computed from frame-level eye state labels |
| D-D: Phone Use | phone_call_right / phone_call_left / texting_right / texting_left | Any phone interaction label = positive D-D event |

---

## 13. Evaluation and Validation Framework

### 13.1 Primary Evaluation Metrics

| Metric | Formula | MVP Target | Measurement Scope |
|---|---|---|---|
| Recall (TPR) | TP / (TP + FN) | ≥ 0.80 per distraction class | DMD held-out test sequences |
| Precision | TP / (TP + FP) | ≥ 0.75 per distraction class | DMD held-out test sequences |
| F1 Score | 2 × (P × R) / (P + R) | ≥ 0.77 composite | DMD held-out test sequences |
| False Positive Rate | FP alerts / total driving hours | < 1.0 per hour | Non-distracted DMD sequences only |
| Alert Latency | Time from event onset to alert dispatch | ≤ 200ms at P95 | Timestamped pipeline on synthetic test sequences on RK3568 |
| Phone mAP50 | COCO mAP at IoU 0.50 | ≥ 0.82 (RKNN) / ≥ 0.85 (FP32) | AUC Distracted Driver held-out test split |

### 13.2 Validation Protocol — Six Test Suites

v2.0.0 Change (CHANGE-13): Suite 4 (Performance) now runs on RK3568, not M1. Suite 6 (Hardware) is new.

**Suite 1: Unit Tests (Per Module)** Each module is tested in isolation with synthetic inputs. No real camera or model inference is required. Mock objects replace all dependencies. Target: 100% of defined behaviors covered by at least one unit test. Execution time target: < 60 seconds for full unit suite.

**Suite 2: Model Benchmark Tests** Each RKNN-converted model is evaluated against its respective held-out test dataset. Tests confirm that converted weights meet the accuracy targets defined in §11.3. Pass/fail is evaluated against RKNN targets (not FP32 baselines). RKNN conversion deltas are recorded in `models/CONVERSION_LOG.json`.

**Suite 3: Pipeline Integration Tests** The full pipeline (Layers 0–6) is run on pre-recorded DMD video sequences. The EventLogger output is compared against DMD ground truth labels. True positives, false positives, and false negatives are counted per distraction class.

**Suite 4: Performance Tests (RK3568 Required)** The pipeline is timed over 1,000 consecutive frames on RK3568 production hardware with RKNN models loaded. P50, P95, and P99 latency values are recorded for each layer and for the full end-to-end pipeline. Memory usage is measured at 30-minute and 4-hour marks. M1 Mac timing is recorded separately for developer reference but does not gate release.

**Suite 5: Failure Mode Tests** Fault injection tests verify that each failure mode defined in §14 (FM-01 through FM-08) produces the defined system behavior. Tests include: model timeout simulation, invalid frame injection, speed input None, face absent for extended duration, perception confidence below gate thresholds, watchdog timeout (block T-1 for 3s), and thermal critical signal injection.

**Suite 6: Hardware Integration Tests (New — RK3568 Only)** Hardware-specific tests in `tests/hardware/`. Covers:

- V4L2 + ISP pipeline: verify IMX219 delivers correct BGR output at 30fps under varied lighting
- RKNN runtime: per-model accuracy validation on RK3568 chip vs simulator
- Thermal soak: 2-hour session in enclosed chassis at 40°C ambient — verify graceful thermal DEGRADED behavior
- Speed source acquisition: OBD-II integration test on target vehicle

### 13.3 Release Gate

The MVP is considered validated and ready for live camera testing when ALL of the following conditions are met simultaneously:

1. All unit tests pass.
2. All model RKNN benchmarks meet targets in §11.3 (not FP32 targets — RKNN deltas must be within spec).
3. Pipeline recall ≥ 0.80 on DMD test sequences.
4. False positive rate < 1.0/hr on DMD non-distracted sequences.
5. All 8 failure modes (FM-01 through FM-08) produce defined behavior in §14.
6. No memory leak detected in 4-hour soak test on RK3568.
7. [NEW] NFR-P1 (≤ 200ms P95) and NFR-P2 (≤ 80ms P95) validated on RK3568 with RKNN models, not on M1.
8. [NEW] Thermal soak test passes (system enters DEGRADED gracefully at ≥ 90°C, recovers when temperature drops below THERMAL_WARN_TEMP_C).

---

## 14. Failure Modes and System Responses

| ID | Failure Mode | Detection | System Response |
|---|---|---|---|
| FM01 | Camera produces no frames for > 100ms | VideoSource frame timeout | `SourceUnavailableError` raised; system halts pipeline; logs error |
| FM02 | Face absent for > 5 seconds while moving | `face_present=False` for > 150 consecutive frames | ALT-06 fires. System continues monitoring. |
| FM03 | Landmark confidence consistently < 0.65 (e.g., sunglasses, extreme angle) | PFLD confidence gate fails consistently | Head pose falls back to PFLD-only. Gaze model disabled. LSTM hidden state reset. Only head pose and phone signals active. |
| FM04 | Model raises runtime exception (OOM, corrupt .rknn weights, RKNN driver error) | Exception caught in `PerceptionStack.infer()` | PerceptionBundle emitted with `face.present=False`. Error logged. System enters DEGRADED after 60 frames. LSTM state cleared. |
| FM05 | Speed input is None, stale (> SPEED_STALE_THRESHOLD_S), or all speed sources unavailable | Type check + staleness check in speed_source.py | Default to URBAN zone (speed_modifier=1.0). Log warning with source attempted. |
| FM06 | EAR calibration not completed (< 30 seconds at speed) | `calibration_complete` flag = False AND no persisted session_state.json | Use default `close_threshold` = 0.21 until calibration completes or persisted data loads. |
| FM07 | Event log disk full | IOException on log write | Log rotation discards oldest file. If disk remains full, logging silently drops events. Alert firing continues unaffected. |
| FM08 [NEW] | RK3568 CPU/NPU thermal throttling reduces effective frame rate below 25fps | ThermalMonitor: `/sys/class/thermal/thermal_zone0/temp` + rolling inference_ms mean | Warn at 80°C: log + reduce YOLOv8 input to 256×256. Critical at 90°C: enter DEGRADED, emit `THERMAL_DEGRADED` event, play 3 short low-tone beeps. If temp drops below THERMAL_WARN_TEMP_C, exit thermal DEGRADED and return to NOMINAL. |

---

## 15. Known Limitations

These limitations are known, accepted, and documented. They are not defects — they are the defined boundary of V2.0 MVP.

| ID | Limitation | Impact | Mitigation |
|---|---|---|---|
| L01 | Sunglasses degrade EAR reliability | PERCLOS path disabled; drowsiness detection unavailable | System falls back to head pose only. Declared in user documentation. |
| L02 | No night / IR detection in MVP | Detection degrades in low-light conditions (< 5 lux). V4L2 AE handles moderate low-light but cannot compensate below sensor sensitivity floor. | MVP is daylight-optimized. IR pathway is roadmap item V2-01. |
| L03 | Cognitive distraction not detectable | Highway hypnosis and mind-wandering produce no camera-visible signals | Explicitly out of scope. No claim is made for this category. |
| L04 | Single driver assumption | If passenger is in front seat, multi-face disambiguation may cause incorrect driver face selection | V2-02: Multi-face tracking with driver-seat position heuristic. |
| L05 | EAR calibration requires 30 seconds of valid driving (cold start) | First 30 seconds uses population-default EAR threshold on first-ever use. Subsequent sessions load persisted calibration immediately (CHANGE-12). | Persisted calibration eliminates this for return users after first session. |
| L06 | Alert efficacy not yet measured | No empirical data exists on driver re-engagement rate post-alert | Pilot study required post-deployment. Cannot be addressed in software alone. |
| L07 | Scoring weights are literature-initialized, not empirically tuned | Composite score may be suboptimal for specific driving scenarios | Weights are configurable in config.py. Tuning study planned for V1.1. |
| L08 [NEW] | LSTM hidden state is not meaningful at session start | First 8 frames (1 LSTM lookback window) have no temporal context. Early detection in the first 267ms is frame-by-frame equivalent. | Impact is negligible. Session start is low-risk (vehicle just started moving). |
| L09 [NEW] | Neutral pose calibration assumes driver looks straight ahead during calibration | If driver is not looking at road during the 10-second calibration window, offsets will be incorrect for that session | System validates calibration quality (std dev < 5°). Failed calibration falls back to 0 offset with warning. |

---

## 16. Out of Scope — MVP

| Out-of-Scope Item | Reason |
|---|---|
| Hardware selection and integration | Separate engineering domain. Handled after software validation is complete. |
| IR camera or multi-spectral sensing | Requires hardware changes. Roadmap item V2-01. |
| Cloud connectivity or data transmission | Violates privacy-first design principle. Any connectivity requires explicit design review. |
| Driver identity recognition | Explicitly prohibited by product privacy principle. |
| Video or image storage of any kind | Privacy violation. Only structured event logs are written. |
| Parent/fleet dashboard or mobile app | Roadmap item. Out of software engine scope. |
| ONNX/TensorRT (desktop) optimization | ONNX runtime is desktop-only and not deployed on RK3568. RKNN conversion IS in scope (§21). |
| Multi-language or i18n support | Not applicable to MVP system (audio alert is a beep, not speech). |
| Cognitive or emotional state detection | Not camera-detectable with current architecture. |
| Crash prediction or risk probability output | Not claimed and not implemented. System detects distraction states only. |

---

## 17. Engineering Risks

| ID | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R01 | False positive rate exceeds 1.0/hr in real driving — alert fatigue causes users to disable device | HIGH | CRITICAL | Duration thresholds set conservatively. Cooldown suppression implemented. Tune on DMD non-distracted sequences first. Kalman filter reduces false triggers from jitter. |
| R02 | MobileNetV3+LSTM gaze accuracy below L2CS-Net baseline on diverse demographics — lower accuracy for certain eye shapes | MED | HIGH | Validate on DMD dataset demographics. System falls back to head pose if gaze confidence < 0.65. Internal LSTM training data diversity should be audited. |
| R03 | Pipeline latency exceeds 200ms on RK3568 NPU — system falls behind real-time | MED | HIGH | Parallel T-1/T-2 thread architecture (CHANGE-07) mitigates. Profile each RKNN model independently. Reduce YOLOv8 resolution to 256×256 if needed. |
| R04 | YOLOv8-nano phone detection accuracy below target after RKNN conversion | LOW | MED | Use FP16 RKNN quantization first. Verify mAP50 per §21.3. Reduce from INT8 to FP16 if mAP drops > 0.03. |
| R05 | DMD dataset access delayed — validation cannot proceed | LOW | MED | Begin unit testing and model benchmarking immediately using 300W and AUC datasets while DMD request is processed. |
| R06 [NEW] | RKNN Toolkit quantization (INT8) degrades model accuracy beyond acceptable delta | MED | HIGH | Convert with FP16 first. Run accuracy benchmarks per §21.3 against RKNN weights, not original FP32. Fall back to FP16 or mixed precision if INT8 exceeds delta thresholds. |
| R07 [NEW] | RK3568 NPU driver incompatibility between RKNN Toolkit 2 version and board BSP | LOW | CRITICAL | Pin RKNN Toolkit 2 version in requirements.txt to BSP-matched release. Record in `models/CONVERSION_LOG.json`. Test on board before Phase 5. |
| R08 [NEW] | LSTM hidden state accumulates error over long sessions — gaze accuracy degrades after 30+ minutes | LOW | MED | Monitor gaze confidence score distribution over session duration in pilot data. If degradation detected, add periodic hidden state refresh (reset every N minutes if face absent momentarily). |

---

## 18. Future Roadmap — Post-MVP Only

All items in this section are explicitly outside V2.0 MVP scope. No implementation work on these items shall begin until the V2.0 release gate in §13.3 has been passed.

### V1.1 — Threshold Tuning and Robustness

| ID | Item | Description |
|---|---|---|
| V1.1-01 | Empirical weight tuning | Run systematic grid search over W1–W4 weights against DMD labeled sequences. Replace literature-initialized weights with empirically validated values. |
| V1.1-02 | Sunglasses fallback improvement | Train or fine-tune landmark model on sunglasses-occluded face dataset. Improve EAR reliability under partial occlusion. |
| V1.1-03 | Blink rate anomaly refinement | Collect baseline blink rate data from pilot deployments. Refine F4 scoring model with real-world distribution. |
| V1.1-04 | Alert efficacy pilot study | Deploy to N ≥ 20 pilot users with opt-in anonymous logging. Measure re-engagement latency post-alert. |
| V1.1-05 | LSTM training data expansion | Expand internal gaze training dataset with diverse demographics and driving scenarios. |

### V2 — Extended Capabilities

| ID | Item | Description |
|---|---|---|
| V2-01 | IR / Night detection | Add IR camera input path to VideoSource. Retrain face detector on IR dataset. Enable low-light operation. |
| V2-02 | Multi-face disambiguation | Track multiple faces per frame. Use geometric heuristics (driver seat position) to select correct face. |
| V2-03 | Eating / reaching detection | Add body pose estimation model (MediaPipe Pose). New distraction class D-E. |
| V2-04 | RKNN NPU further optimization | Profile and optimize RKNN models using NPU-specific quantization techniques. Target ≤ 30ms full pipeline. |
| V2-05 | Optional opt-in telemetry | Structured (non-video) event log transmission for fleet analytics. Requires explicit user consent UI and encrypted channel. |
| V2-06 | Parent/fleet dashboard | Read structured event logs, display driving session summaries and alert frequency trends. |

---

## 19. Configuration Reference — config.py

All tunable parameters are centralized in config.py. No threshold or path value appears in any other file.

```python
# ─── MODEL PATHS (RKNN FORMAT — v2.0.0) ─────────────────────────────────────
BLAZEFACE_MODEL_PATH = 'models/blazeface.rknn'
PFLD_MODEL_PATH      = 'models/pfld_98pt.rknn'
GAZE_MODEL_PATH      = 'models/gaze_mobilenetv3_lstm.rknn'  # Replaces L2CS-Net
YOLO_MODEL_PATH      = 'models/yolov8n_phone.rknn'

# ─── CAMERA / ISP (IMX219 + RK3568 — v2.0.0) ────────────────────────────────
V4L2_DEVICE       = '/dev/video0'
CAPTURE_WIDTH     = 1280
CAPTURE_HEIGHT    = 720
CAPTURE_FPS       = 30
V4L2_PIXEL_FORMAT = 'NV12'  # ISP output format; VideoSource converts to BGR24
LOW_LIGHT_THRESHOLD  = 30   # Mean pixel intensity (0-255)
OVEREXPOSE_THRESHOLD = 240  # Mean pixel intensity (0-255)

# ─── RKNN RUNTIME ────────────────────────────────────────────────────────
RKNN_TARGET_PLATFORM  = 'rk3568'
RKNN_TOOLKIT_VERSION  = '2.0.0b0'  # Must match BSP. Pin this.

# ─── CONFIDENCE GATES ─────────────────────────────────────────────────────
FACE_CONFIDENCE_GATE     = 0.60
LANDMARK_CONFIDENCE_GATE = 0.65

# ─── ROAD ZONE ───────────────────────────────────────────────────────────
ROAD_ZONE_YAW_MIN   = -15.0  # degrees
ROAD_ZONE_YAW_MAX   = +15.0
ROAD_ZONE_PITCH_MIN = -10.0
ROAD_ZONE_PITCH_MAX = +5.0

# ─── SPEED ZONES ──────────────────────────────────────────────────────────
V_MIN_MPS             = 1.4   # 5 km/h
V_HIGHWAY_MPS         = 13.9  # 50 km/h
HIGHWAY_SCORE_MODIFIER = 1.4
SPEED_STALE_THRESHOLD_S = 2.0  # Declare speed stale if no update in 2s

# ─── SPEED SIGNAL SOURCE (v2.0.0) ────────────────────────────────────────────
SPEED_SOURCE_PRIORITY = ['OBD2', 'CAN', 'GPS', 'NONE']  # Try in order
OBD2_PORT      = '/dev/ttyUSB0'
OBD2_BAUDRATE  = 38400
OBD2_POLL_HZ   = 10
CAN_INTERFACE  = 'can0'
CAN_PGN_SPEED  = 0xFEF1  # J1939 PGN 65265 Wheel Speed
GPS_PORT       = '/dev/ttyS3'

# ─── DURATION THRESHOLDS ──────────────────────────────────────────────────
T_GAZE_SECONDS         = 2.0
T_HEAD_SECONDS         = 1.5
T_PHONE_SECONDS        = 1.0
T_FACE_ABSENT_SECONDS  = 5.0

# ─── HEAD POSE THRESHOLDS ──────────────────────────────────────────────────
HEAD_YAW_THRESHOLD_DEG   = 30.0
HEAD_PITCH_THRESHOLD_DEG = 20.0
PNP_REPROJECTION_ERR_MAX = 8.0  # pixels

# ─── EAR / PERCLOS ─────────────────────────────────────────────────────────
EAR_DEFAULT_CLOSE_THRESHOLD  = 0.21
EAR_CALIBRATION_MULTIPLIER   = 0.75
EAR_CALIBRATION_DURATION_S   = 30.0
PERCLOS_WINDOW_FRAMES        = 60
PERCLOS_CLOSURE_FRACTION     = 0.80
PERCLOS_ALERT_THRESHOLD      = 0.15
PERCLOS_MIN_VALID_FRAMES     = 30

# ─── KALMAN FILTER (v2.0.0) ──────────────────────────────────────────────────
KALMAN_PROCESS_NOISE_Q     = 0.01
KALMAN_MEASUREMENT_NOISE_R = 4.0
KALMAN_INITIAL_COVARIANCE  = 1.0

# ─── GAZE TRANSFORM ──────────────────────────────────────────────────────
GAZE_HEAD_COUPLING_ALPHA = 0.7  # Yaw coupling
GAZE_HEAD_COUPLING_BETA  = 0.7  # Pitch coupling

# ─── LSTM GAZE MODEL (v2.0.0) ────────────────────────────────────────────────
GAZE_INPUT_RESOLUTION   = 112  # Down from 224 (L2CS-Net) — MobileNetV3 compatible
GAZE_TEMPORAL_FRAMES    = 8    # LSTM lookback window
LSTM_RESET_ABSENT_FRAMES = 10  # Reset hidden state after face absent N frames

# ─── PHONE DETECTION ──────────────────────────────────────────────────────
PHONE_CONFIDENCE_THRESHOLD      = 0.70
YOLO_INPUT_RESOLUTION           = 320  # Reduced to 256 if thermal throttle active
YOLO_INPUT_RESOLUTION_THROTTLE  = 256

# ─── SCORING WEIGHTS ──────────────────────────────────────────────────────
WEIGHT_GAZE    = 0.45
WEIGHT_HEAD    = 0.30
WEIGHT_PERCLOS = 0.20
WEIGHT_BLINK   = 0.05
COMPOSITE_ALERT_THRESHOLD = 0.55

# ─── BLINK DETECTION ──────────────────────────────────────────────────────
BLINK_MIN_FRAMES          = 2     # 67ms at 30fps
BLINK_MAX_FRAMES          = 10    # 333ms at 30fps
BLINK_RATE_NORMAL_LOW_HZ  = 0.13  # 8 blinks/min
BLINK_RATE_NORMAL_HIGH_HZ = 0.50  # 30 blinks/min

# ─── TEMPORAL BUFFER ──────────────────────────────────────────────────────
CIRCULAR_BUFFER_SIZE  = 120  # 4.0s at 30fps
FEATURE_WINDOW_FRAMES = 60   # 2.0s at 30fps

# ─── ALERT COOLDOWNS (seconds) ───────────────────────────────────────────────
COOLDOWN_VISUAL      = 8.0
COOLDOWN_HEAD        = 8.0
COOLDOWN_DROWSINESS  = 12.0
COOLDOWN_PHONE       = 5.0
COOLDOWN_FACE_ABSENT = 10.0
COOLDOWN_COMPOSITE   = 8.0

# ─── DEGRADED STATE ──────────────────────────────────────────────────────
DEGRADED_TRIGGER_FRAMES   = 60  # 2.0s of invalid frames
DEGRADED_RECOVERY_FRAMES  = 30  # 1.0s of valid frames to recover
DEGRADED_TRIGGER_LIGHTING = 90  # Extended trigger window during AE convergence

# ─── THREAD / SYNC (v2.0.0) ──────────────────────────────────────────────────
PHONE_THREAD_TIMEOUT_MS = 5  # Wait for T-2 before using stale phone result
FRAME_QUEUE_DEPTH       = 2  # T-0 → T-1/T-2 queue depth (drop oldest on overflow)

# ─── WATCHDOG (v2.0.0) ──────────────────────────────────────────────────────
WATCHDOG_TIMEOUT_S   = 2.0
WATCHDOG_HEARTBEAT_S = 0.5

# ─── THERMAL MONITOR (v2.0.0) ────────────────────────────────────────────────
THERMAL_WARN_TEMP_C     = 80
THERMAL_CRITICAL_TEMP_C = 90
THERMAL_MONITOR_PATH    = '/sys/class/thermal/thermal_zone0/temp'
THERMAL_CHECK_INTERVAL_S = 5.0

# ─── CALIBRATION (v2.0.0) ────────────────────────────────────────────────────
CALIBRATION_DURATION_S        = 10.0
CALIBRATION_MIN_VALID_FRAMES  = 270  # 90% of 300 frames
CALIBRATION_MAX_POSE_STD_DEG  = 5.0  # Reject calibration if std dev >= 5°
NEUTRAL_POSE_FILE             = 'calibration/session_state.json'
CALIBRATION_REQUIRED_ON_VIN_CHANGE = True

# ─── LOGGING ────────────────────────────────────────────────────────────
LOG_DIR         = 'logs/'
LOG_MAX_BYTES   = 52_428_800  # 50 MB
LOG_BACKUP_COUNT = 5
```

---

## 20. Recommended Implementation Order

v2.0.0 Change (CHANGE-02, CHANGE-04, CHANGE-08, CHANGE-12): Implementation order revised significantly from v1.0.0. RKNN conversion is now Phase 5 entry requirement. Phase 1b (hardware interface) is new. Phase 2b (calibration system) is new. Validation (Phase 7) runs on RK3568.

| Phase | Target | Key Tasks | Exit Criterion |
|---|---|---|---|
| 1 | Foundation | Create directory structure. Implement config.py with all parameters. Implement dataclasses for all inter-layer messages including new v2.0 fields. Write unit tests for all dataclass schemas. | All dataclass schema tests pass. |
| 1b | Hardware Interface | Validate V4L2 + RKISP1 pipeline on RK3568. Verify IMX219 delivers NV12 at 30fps. Verify NV12→BGR24 conversion produces correct output. Test OBD-II or CAN speed acquisition on target vehicle. Verify RKNN Toolkit version matches BSP. | imx219_v4l2_source.py delivers correct BGR frames at ≥ 28fps. SpeedSource returns valid speed. |
| 2 | Signal Processor | Implement Layer 2 entirely. PnP solver, Kalman filter (§5.6), neutral pose correction (§5.7), EAR calculator, gaze transformer, phone signal extractor. All use mock landmark inputs. | All Layer 2 unit tests pass. Kalman filter reduces frame-to-frame std dev by ≥ 60% on synthetic sequence. |
| 2b | Calibration System | Implement calibration_manager.py, session_state.json persistence, VIN-based re-calibration trigger. Integrate with Layer 2 pose correction. | Calibration completes in ≤ 15s. State persists across reboot. Corrected angles verified against known offsets. |
| 3 | Temporal Engine | Implement Layer 3: CircularBuffer, duration timers, PERCLOS window, blink detector (with §5.5 formula), SpeedSource integration, WatchdogManager (FR-3.6), ThermalMonitor (FR-3.7). | All Layer 3 unit tests pass. PERCLOS matches hand-computed values. Watchdog correctly triggers DEGRADED. Thermal monitor reads correct temperature. |
| 4 | Scoring + State Machine | Implement Layers 4 and 5. Scoring engine with weight validation. Alert state machine with all 5 states, 6 alert types, thermal DEGRADED path (P-06), and 8 failure modes (FM-01–FM-08). Cooldown logic. | All Layer 4/5 unit tests pass. All 8 failure modes produce defined behavior. |
| 5 | RKNN Conversion + Model Integration | RKNN conversion runs first. Convert all four models per §21. Validate each against §21.3 acceptance criteria on RK3568 before writing any wrapper code. Then implement Layer 1 wrappers using rknn_toolkit2_lite. Implement LSTM stateful interface. Implement parallel T-1/T-2 thread architecture. | All four RKNN models pass §21.3 acceptance on RK3568. Per-model accuracy benchmarks pass. Thread sync verified. |
| 6 | Video + Output | Implement imx219_v4l2_source.py (may already be done in Phase 1b — integrate). Implement Layer 6 AudioHandler and EventLogger. Wire all layers and threads in main.py. | Full pipeline runs on pre-recorded video file without errors on RK3568. End-to-end latency measured. |
| 7 | Validation | Run all 6 test suites (§13.2) on RK3568 hardware. Measure recall, precision, FPR, and latency against RK3568 targets. Run thermal soak test. Run 4-hour memory soak. | All 8 release gate criteria in §13.3 met. |

---

## 21. RKNN Model Conversion Pipeline

[NEW in v2.0.0 — CHANGE-02] This section is a required addition. No implementation of Layer 1 (Phase 5) shall begin until RKNN conversion is validated for all four perception models.

### 21.1 Why RKNN is Mandatory (Not Optional)

The RK3568 SoC contains a 1.0 TOPS NPU accessible only via RKNN Toolkit 2. Without NPU acceleration:

| Model | CPU-only (Cortex-A55) | NPU via RKNN | 80ms Budget |
|---|---|---|---|
| BlazeFace | ~25ms | ~4ms | |
| PFLD 98-point | ~40ms | ~6ms | |
| MobileNetV3+LSTM | ~35ms | ~8ms | |
| YOLOv8-nano | ~90ms | ~15ms | |
| **TOTAL** | **~190ms** | **~33ms** | **80ms** |

Running without RKNN on CPU alone costs ~190ms on perception alone — before any of Layers 2–6 run. The total pipeline would be ~400–600ms, 2–3x above the 200ms NFR-P1 requirement. Real-time is not achievable without NPU acceleration.

### 21.2 Conversion Toolchain

| Tool | Version | Purpose |
|---|---|---|
| RKNN Toolkit 2 | Pin to BSP version | Convert ONNX/PyTorch → .rknn on host machine |
| rknn_toolkit2 Python pkg | Same as above | Simulate and test RKNN inference on host before board deployment |
| rknn-toolkit2-lite | Same as above | On-device inference API — runs on RK3568 in production |
| RKNN Model Zoo | Latest | Reference conversion scripts for BlazeFace, PFLD, YOLO |

### 21.3 Conversion Acceptance Criteria

| Model | Metric | FP32 Baseline | Max RKNN Delta | Pass Condition |
|---|---|---|---|---|
| BlazeFace | Recall @ 0.60 conf on DMD | Establish on DMD | ≤ 3% relative | RKNN recall ≥ FP32 − 3% |
| PFLD 98-point | Mean landmark NME on 300W | < 4% NME | ≤ 0.5% NME absolute | RKNN NME < 4.5% |
| MobileNetV3+LSTM | Mean angular error on MPIIFaceGaze | < 6° MAE | ≤ 0.5° absolute | RKNN MAE < 6.5° |
| YOLOv8-nano (phone) | mAP50 on AUC held-out | ≥ 0.85 | ≤ 0.03 mAP absolute | RKNN mAP ≥ 0.82 |

### 21.4 Conversion Procedure (Per Model)

1. Export model to ONNX. Validate ONNX output matches PyTorch within 1e-4 tolerance using `onnxruntime`.
2. Run `rknn.load_onnx()` with `target_platform='rk3568'`. Use INT8 quantization as default. If accuracy delta exceeds §21.3 thresholds, fall back to FP16.
3. Run accuracy benchmark on held-out test set using rknn_toolkit2 simulator on host machine. Record metric delta vs FP32.
4. Transfer `.rknn` file to RK3568 board. Run on-device accuracy check using rknn-toolkit2-lite with same test set (200-sample subset). Confirm on-device metric matches simulator within 1%.
5. Record conversion metadata in `models/CONVERSION_LOG.json`: source model hash, RKNN Toolkit version, quantization type, FP32 accuracy, RKNN accuracy, on-device latency.

### 21.5 CONVERSION_LOG.json Format

```json
{
    "blazeface": {
        "source_model_sha256": "abc123...",
        "rknn_toolkit_version": "2.0.0b0",
        "quantization": "int8",
        "fp32_recall": 0.953,
        "rknn_recall": 0.941,
        "delta_pct": -1.26,
        "on_device_latency_ms": 3.8,
        "converted_at": "2026-02-20T14:32:00Z",
        "pass": true
    },
    "pfld_98pt": { "...": "..." },
    "gaze_mobilenetv3_lstm": { "...": "..." },
    "yolov8n_phone": { "...": "..." }
}
```

---

## 22. Speed Signal Acquisition Module

[NEW in v2.0.0 — CHANGE-05] Speed-adaptive thresholding is a core safety feature. This section specifies how speed is acquired, what sources are supported, and how the system behaves when speed is unavailable.

### 22.1 SpeedSource Module Interface

`layer3_temporal/speed_source.py`

```python
class SpeedSource:
    def get_speed_mps(self) -> tuple[float | None, bool]:
        """
        Returns (speed_mps, is_stale).
        is_stale=True if last update > SPEED_STALE_THRESHOLD_S ago.
        Returns (None, True) if no source is available — triggers FM-05 URBAN fallback.
        """

    def get_source_type(self) -> str:
        """Returns active source: 'OBD2' | 'CAN' | 'GPS' | 'NONE'"""

    def is_available(self) -> bool:
        """True if any speed source is active and non-stale"""
```

### 22.2 Speed Source Priority Stack

| Priority | Source | Interface | Update Rate | Latency | Accuracy |
|---|---|---|---|---|---|
| 1 (Primary) | OBD-II PID 0x0D | ELM327 UART on OBD2_PORT at OBD2_BAUDRATE | ~10 Hz | ~100–200ms | ±2 km/h |
| 2 (Secondary) | CAN Bus J1939 PGN 65265 | SocketCAN on CAN_INTERFACE | ~10–20 Hz | ~50ms | ±1 km/h |
| 3 (Fallback) | GPS NMEA $GPRMC field | UART on GPS_PORT | ~1 Hz | ~1000ms | ±3 km/h |
| 4 (Emergency) | None — URBAN default | N/A | N/A | N/A | N/A |

SpeedSource auto-discovers available sources at startup using `SPEED_SOURCE_PRIORITY`. It falls back to the next priority level if the primary source fails or becomes stale. Source failover is logged at WARNING level.

### 22.3 Speed Signal Behavior

```python
# In TemporalEngine.process(signal_frame):
speed_mps, is_stale = speed_source.get_speed_mps()
if speed_mps is None or is_stale:
    # FM-05: Default to URBAN
    effective_speed = None  # FM-05 handler sets speed_modifier = 1.0
    log_warning("SPEED_UNAVAILABLE", source=speed_source.get_source_type())
else:
    effective_speed = clamp(speed_mps, 0.0, 100.0)

signal_frame.speed_mps   = effective_speed or 0.0
signal_frame.speed_stale = is_stale
```

---

## 23. Per-Vehicle Mounting Calibration Protocol

[NEW in v2.0.0 — CHANGE-08] Head pose thresholds compare against angles relative to the road-forward direction, not the camera axis. Mounting angle varies between vehicles. Calibration corrects for this.

### 23.1 When Calibration Runs

- First-ever startup (no `calibration/session_state.json` exists)
- `session_state.json` exists but VIN does not match current vehicle OBD-II VIN
- User manually triggers re-calibration via config flag `FORCE_RECALIBRATION=True`

### 23.2 Calibration Sequence

| Step | Action | Duration | Pass Condition | Failure Behavior |
|---|---|---|---|---|
| 1 | Check session_state.json. If VIN matches, load and skip to monitoring. | < 1s | File exists and VIN matches | Proceed to Step 2 |
| 2 | 3 short audio beeps (calibration mode signal). Driver looks straight ahead at road. System collects head pose and EAR samples. | 10 seconds | ≥ 270 valid frames in 300-frame window | Extend window by 5s, retry once. If still failing: use 0.0 offsets and log `CALIBRATION_FAILED`. |
| 3 | Validate calibration quality: std dev of collected yaw and pitch samples must be < `CALIBRATION_MAX_POSE_STD_DEG` (5°). If driver was not stable, discard. | < 100ms | Pose std dev < 5° | Retry Step 2. After 2 failed attempts: use 0.0 offsets, log warning. |
| 4 | Compute offsets: `neutral_yaw_offset = mean(yaw_samples)`, `neutral_pitch_offset = mean(pitch_samples)`, `baseline_EAR = mean(EAR_samples)`. Write to `session_state.json`. | < 200ms | File write confirmed | Use in-memory values. Log disk error. |
| 5 | 2 long audio beeps (calibration success). Transition to NOMINAL monitoring state. | — | — | — |

### 23.3 session_state.json Schema

```json
{
    "schema_version": "2.0",
    "calibrated_at": "2026-02-20T14:30:00Z",
    "vehicle_vin": "1HGCM82633A004352",
    "neutral_yaw_offset": -4.2,
    "neutral_pitch_offset": 3.1,
    "baseline_ear": 0.31,
    "close_threshold": 0.233,
    "calibration_complete": true,
    "frames_collected": 298,
    "pose_std_yaw": 1.8,
    "pose_std_pitch": 1.2
}
```

---

## 24. Session State Persistence

[NEW in v2.0.0 — CHANGE-12] An embedded device that powers on with the ignition restarts on every drive. Without persistence, EAR calibration (30s warm-up) and neutral pose offsets are lost every session.

### 24.1 What Gets Persisted

| State Field | Persisted? | Notes |
|---|---|---|
| baseline_EAR | YES | Per-driver baseline. Recomputed if calibration reruns. |
| close_threshold | YES | Derived from baseline_EAR. |
| neutral_yaw_offset | YES | Per-vehicle mounting offset. |
| neutral_pitch_offset | YES | Per-vehicle mounting offset. |
| calibration_vehicle_vin | YES | OBD-II VIN if available. Used to detect vehicle change. |
| calibration_complete | YES | True once calibration has completed at least once for this vehicle. |
| Alert FSM state (NOMINAL/PRE_ALERT/etc.) | NO | Always reset to NOMINAL on startup. |
| LSTM hidden state (h_t, c_t) | NO | Session-specific. Always reset on startup. |
| Circular buffer contents | NO | Session-specific. Buffer starts empty. |
| Active cooldown timers | NO | Session-specific. All cooldowns reset on startup. |

### 24.2 Startup Load Behavior

```python
# In main.py initialization:
state = load_session_state(NEUTRAL_POSE_FILE)
if state and state['calibration_complete']:
    signal_processor.set_ear_baseline(state['baseline_ear'], state['close_threshold'])
    signal_processor.set_neutral_pose(state['neutral_yaw_offset'], state['neutral_pitch_offset'])
    log_info("CALIBRATION_LOADED", source=NEUTRAL_POSE_FILE)
    # Skip calibration phase — go directly to NOMINAL
else:
    log_info("CALIBRATION_REQUIRED")
    calibration_manager.run()  # §23.2
```

---

*END OF DOCUMENT — Attentia Drive DDE PRD v2.0.0*

*This document supersedes v1.0.0 entirely. Revision history: v1.0.0 (Feb 2026, Rishit Sharma) → v2.0.0 (Feb 2026, hardware compatibility revision RTR-1.0.0).*

*This document is frozen. No changes shall be made during development without a formal revision and version increment.*

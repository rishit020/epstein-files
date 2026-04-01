# config.py — Attentia Drive Configuration
# ALL tunable parameters are defined here. No threshold or path value appears in any other file.
# PRD §19 — Configuration Reference

# ─── MODEL PATHS (ONNX FORMAT — Mac development) ─────────────────────────────
# Production uses .rknn format; ONNX is Mac/dev only (per CLAUDE.md)
BLAZEFACE_MODEL_PATH = 'models/blazeface.onnx'
# PRD specifies pfld_98pt.onnx (98-point). Using 68-point model as accepted deviation. Indices remapped to iBUG 68-point convention.
# Corrected benchmark (2026-03-26): PFLD achieves ~4.3% NME on 300W common, ~5.97% on IBUG challenging.
# Previous 24.74% result was caused by evaluating on augmented images with wrong GT — see PFLD_HANDOFF.md.
PFLD_MODEL_PATH      = 'models/pfld.onnx'
GAZE_MODEL_PATH      = 'models/gaze_mobilenetv3_lstm.onnx'
YOLO_MODEL_PATH      = 'models/yolov8n_phone.onnx'

# ─── CAMERA / ISP (IMX219 + RK3568 — production targets) ─────────────────────
V4L2_DEVICE       = '/dev/video0'
CAPTURE_WIDTH     = 1280
CAPTURE_HEIGHT    = 720
CAPTURE_FPS       = 30
V4L2_PIXEL_FORMAT = 'NV12'   # ISP output format; VideoSource converts to BGR24
LOW_LIGHT_THRESHOLD  = 30    # Mean pixel intensity (0-255)
OVEREXPOSE_THRESHOLD = 240   # Mean pixel intensity (0-255)

# ─── RKNN RUNTIME ─────────────────────────────────────────────────────────────
RKNN_TARGET_PLATFORM = 'rk3568'
RKNN_TOOLKIT_VERSION = '2.0.0b0'  # Must match BSP. Pin this.

# ─── CONFIDENCE GATES ─────────────────────────────────────────────────────────
FACE_CONFIDENCE_GATE     = 0.35  # Lowered from 0.60 — Mac webcam BlazeFace scores real face 0.43-0.63; false positives outscore it at 0.60+ gate
LANDMARK_CONFIDENCE_GATE = 0.65

# ─── ROAD ZONE ────────────────────────────────────────────────────────────────
ROAD_ZONE_YAW_MIN   = -15.0  # degrees; negative = left of forward
ROAD_ZONE_YAW_MAX   = +15.0
ROAD_ZONE_PITCH_MIN = -10.0
ROAD_ZONE_PITCH_MAX = +5.0

# ─── SPEED ZONES ──────────────────────────────────────────────────────────────
V_MIN_MPS              = 1.4   # 5 km/h — below this: PARKED zone
V_HIGHWAY_MPS          = 13.9  # 50 km/h — above this: HIGHWAY zone
HIGHWAY_SCORE_MODIFIER = 1.4
SPEED_STALE_THRESHOLD_S = 2.0  # Declare speed stale if no update in 2s

# ─── SPEED SIGNAL SOURCE ──────────────────────────────────────────────────────
SPEED_SOURCE_PRIORITY = ['OBD2', 'CAN', 'GPS', 'NONE']  # Try in order
OBD2_PORT     = '/dev/ttyUSB0'
OBD2_BAUDRATE = 38400
OBD2_POLL_HZ  = 10
CAN_INTERFACE = 'can0'
CAN_PGN_SPEED = 0xFEF1   # J1939 PGN 65265 Wheel Speed
GPS_PORT      = '/dev/ttyS3'

# ─── DURATION THRESHOLDS ──────────────────────────────────────────────────────
T_GAZE_SECONDS        = 2.0
T_HEAD_SECONDS        = 1.5
T_PHONE_SECONDS       = 1.0
T_FACE_ABSENT_SECONDS = 5.0

# ─── HEAD POSE THRESHOLDS ─────────────────────────────────────────────────────
HEAD_YAW_THRESHOLD_DEG   = 30.0
HEAD_PITCH_THRESHOLD_DEG = 20.0
PNP_REPROJECTION_ERR_MAX = 75.0    # pixels (Mac dev: uncalibrated webcam; production=8.0 with calibrated camera)

# ─── EAR / PERCLOS ────────────────────────────────────────────────────────────
EAR_DEFAULT_CLOSE_THRESHOLD = 0.21
EAR_CALIBRATION_MULTIPLIER  = 0.75
EAR_CALIBRATION_DURATION_S  = 30.0
PERCLOS_WINDOW_FRAMES       = 60
PERCLOS_CLOSURE_FRACTION    = 0.80
PERCLOS_ALERT_THRESHOLD     = 0.15
PERCLOS_MIN_VALID_FRAMES    = 30

# ─── KALMAN FILTER ────────────────────────────────────────────────────────────
KALMAN_PROCESS_NOISE_Q     = 0.01
KALMAN_MEASUREMENT_NOISE_R = 4.0
KALMAN_INITIAL_COVARIANCE  = 1.0

# ─── GAZE TRANSFORM ───────────────────────────────────────────────────────────
GAZE_HEAD_COUPLING_ALPHA = 0.7  # Yaw coupling
GAZE_HEAD_COUPLING_BETA  = 0.7  # Pitch coupling

# ─── LSTM GAZE MODEL ──────────────────────────────────────────────────────────
GAZE_INPUT_RESOLUTION    = 112  # MobileNetV3 input size
GAZE_TEMPORAL_FRAMES     = 8    # LSTM lookback window
LSTM_RESET_ABSENT_FRAMES = 10   # Reset hidden state after face absent N frames

# ─── PHONE DETECTION ──────────────────────────────────────────────────────────
PHONE_CONFIDENCE_THRESHOLD     = 0.70
YOLO_INPUT_RESOLUTION          = 320   # Normal operating resolution
YOLO_INPUT_RESOLUTION_THROTTLE = 256   # Reduced during thermal throttle

# ─── SCORING WEIGHTS ──────────────────────────────────────────────────────────
WEIGHT_GAZE    = 0.45
WEIGHT_HEAD    = 0.30
WEIGHT_PERCLOS = 0.20
WEIGHT_BLINK   = 0.05
COMPOSITE_ALERT_THRESHOLD = 0.55

# ─── BLINK DETECTION ──────────────────────────────────────────────────────────
BLINK_MIN_FRAMES          = 2     # 67ms at 30fps
BLINK_MAX_FRAMES          = 10    # 333ms at 30fps
BLINK_RATE_NORMAL_LOW_HZ  = 0.13  # 8 blinks/min
BLINK_RATE_NORMAL_HIGH_HZ = 0.50  # 30 blinks/min

# ─── TEMPORAL BUFFER ──────────────────────────────────────────────────────────
CIRCULAR_BUFFER_SIZE  = 120  # 4.0s at 30fps
FEATURE_WINDOW_FRAMES = 60   # 2.0s at 30fps

# ─── ALERT COOLDOWNS (seconds) ────────────────────────────────────────────────
COOLDOWN_VISUAL      = 8.0
COOLDOWN_HEAD        = 8.0
COOLDOWN_DROWSINESS  = 12.0
COOLDOWN_PHONE       = 5.0
COOLDOWN_FACE_ABSENT = 10.0
COOLDOWN_COMPOSITE   = 8.0

# ─── DEGRADED STATE ───────────────────────────────────────────────────────────
DEGRADED_TRIGGER_FRAMES   = 60   # 2.0s of invalid frames
DEGRADED_RECOVERY_FRAMES  = 30   # 1.0s of valid frames to recover
DEGRADED_TRIGGER_LIGHTING = 90   # Extended trigger window during AE convergence

# ─── THREAD / SYNC ────────────────────────────────────────────────────────────
PHONE_THREAD_TIMEOUT_MS = 100  # Wait for T-2 before using stale phone result (YOLO ~45ms on Mac)
FRAME_QUEUE_DEPTH       = 2   # T-0 → T-1/T-2 queue depth (drop oldest on overflow)

# ─── WATCHDOG ─────────────────────────────────────────────────────────────────
WATCHDOG_TIMEOUT_S   = 2.0
WATCHDOG_HEARTBEAT_S = 0.5

# ─── THERMAL MONITOR ──────────────────────────────────────────────────────────
THERMAL_WARN_TEMP_C      = 80
THERMAL_CRITICAL_TEMP_C  = 90
THERMAL_MONITOR_PATH     = '/sys/class/thermal/thermal_zone0/temp'
THERMAL_CHECK_INTERVAL_S = 5.0

# ─── CALIBRATION ──────────────────────────────────────────────────────────────
CALIBRATION_DURATION_S           = 10.0
CALIBRATION_EXTENSION_S          = 5.0   # One-time window extension on low frame count (PRD §23.2 Step 2)
CALIBRATION_MIN_VALID_FRAMES     = 270   # 90% of 300 frames
CALIBRATION_MAX_POSE_STD_DEG     = 5.0   # Reject calibration if std dev >= 5°
CALIBRATION_MAX_ATTEMPTS         = 2     # Max std-dev validation attempts before fallback (PRD §23.2 Step 3)
NEUTRAL_POSE_FILE                = 'calibration/session_state.json'
CALIBRATION_REQUIRED_ON_VIN_CHANGE = True
FORCE_RECALIBRATION              = False # Set True to bypass persisted state (PRD §23.1)

# ─── AUDIO ALERT SOUNDS ──────────────────────────────────────────────────────
AUDIO_ALERT_SOUND        = '/System/Library/Sounds/Ping.aiff'       # HIGH alerts
AUDIO_ALERT_SOUND_URGENT = '/System/Library/Sounds/Sosumi.aiff'     # URGENT alerts

# ─── LOGGING ──────────────────────────────────────────────────────────────────
LOG_DIR          = 'logs/'
LOG_MAX_BYTES    = 52_428_800  # 50 MB
LOG_BACKUP_COUNT = 5

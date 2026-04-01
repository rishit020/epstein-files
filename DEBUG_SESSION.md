# Attentia Drive — Focused Diagnostic & Repair Session

## CONTEXT

Attentia Drive is a 6-layer driver monitoring system running on Mac with ONNX models.

**Current Status:** System partially works — alerts fire under some conditions but distraction detection is unreliable. **NOT a complete system failure.** The models are confirmed working. The problem is elsewhere.

**What's Already Known:**
- BlazeFace face detection ✅ working (0.95+ confidence consistently)
- PFLD landmarks ✅ working (1.0 confidence, 68-point output valid)
- Gaze model ✅ working (outputting yaw/pitch angles)
- YOLO phone detection ✅ working (mAP50 = 0.977)
- **Camera calibration** ❌ BROKEN — reprojection error ~726px vs 8px threshold

---

## THE ROOT CAUSE (ALREADY IDENTIFIED)

**File:** `layer2_signals/head_pose_solver.py` lines 79-86

The PnP head pose solver uses a **heuristic camera matrix** that is inaccurate for the Mac webcam:

```python
focal_len = float(frame_width)  # Using 1280 as focal length — this is a GUESS
cx = frame_width / 2.0
cy = frame_height / 2.0
camera_matrix = np.array([
    [focal_len, 0.0, cx],
    [0.0, focal_len, cy],
    [0.0, 0.0,  1.0],
], dtype=np.float64)
```

**Impact:**
- PnP solver produces huge reprojection errors (avg 726 pixels)
- Line 112: `valid = reprojection_error < config.PNP_REPROJECTION_ERR_MAX` → always False (8px threshold)
- `pose_valid = False` → head pose rejected
- `gaze_world = None` (requires valid pose per PRD §5.3)
- All signals marked invalid
- After 60 frames: FM-04 triggers DEGRADED state (watchdog detects hung pipeline)
- P-04: "DEGRADED state suppresses all alerts"
- Result: **No distraction alerts ever fire**

**Temporary Workaround Applied:** `config.py` line 62
```python
PNP_REPROJECTION_ERR_MAX = 100.0   # TEMPORARY: Mac webcam has no calibration
```
This allows the system to run, but pose estimates are still poor quality.

---

## YOUR MISSION

### PHASE 1: Verify the Camera Calibration Problem (5 min)

Run the diagnostic test:
```bash
python test_reprojection.py
```

Expected output: Print reprojection error for 20 frames.

**If you see:**
- avg error > 50px → camera matrix is severely wrong (confirms root cause)
- avg error > 100px → PnP solver is completely broken
- avg error < 8px → camera matrix is accurate (something else is wrong)

Print the actual numbers and confirm they match what we found.

---

### PHASE 2: Determine the Proper Fix (10 min)

You have THREE options:

**Option A: Temporary Test Fix (already applied)**
- Leave `PNP_REPROJECTION_ERR_MAX = 100.0`
- System runs, detection works but pose quality is poor
- Use for testing/verification only
- ⚠️ Not production-ready

**Option B: Proper Camera Calibration (recommended)**
- Use OpenCV checkerboard calibration to compute actual focal length and principal point
- Update `head_pose_solver.py` to use calibrated camera matrix
- Restore `PNP_REPROJECTION_ERR_MAX = 8.0`
- Requires: printed checkerboard pattern, ~10 minutes of calibration
- Result: accurate pose estimation

**Option C: Investigate Further (if reprojection error is NOT the issue)**
- If `test_reprojection.py` shows error < 8px but system still doesn't detect distraction
- Then the problem is elsewhere (move to Phase 3)

**Recommended:** Start with Option A verification, then decide whether to do Option B.

---

### PHASE 3: Test Detection Pipeline (10 min)

Once camera calibration is confirmed or worked around, run a live test:

```bash
python main.py --display --debug
```

For each test, watch the debug overlay and check the logs:

**Test 1: No Face**
- Point camera at wall or cover with hand
- Expected: `face=False`, no alerts
- If you see `face=True`: BlazeFace has a bug (unlikely, already verified working)

**Test 2: Neutral Gaze (baseline)**
- Look directly at camera, neutral position
- Expected: `score=0.050` (attentive), `state=NOMINAL`, no alerts
- Duration: 10 seconds

**Test 3: Visual Inattention (D-A)**
- Look away to the LEFT for 3+ seconds (gaze > 15° yaw)
- Expected:
  - After ~2 seconds: `score` rises above 0.55
  - After ~2.5 seconds: `state=PRE_ALERT` → `ALERTING`
  - Audio plays (Ping.aiff)
  - Log shows: `ALERT fired: type=D-A level=HIGH`
- If no alert: check if gaze is actually leaving road zone (debug overlay shows `Gaze=...`)

**Test 4: Phone Detection (D-D)**
- Hold phone clearly in frame for 2+ seconds
- Expected:
  - `phone=YES` appears in debug overlay
  - After ~1 second: `state=ALERTING`
  - Audio plays (Sosumi.aiff — URGENT)
  - Log shows: `ALERT fired: type=D-D level=URGENT`
- If no alert: phone detection may have wrong confidence threshold

**Test 5: Drowsiness (D-C)**
- Close eyes for 3+ seconds (simulate drowsiness)
- Expected:
  - `PERCLOS` rises above 0.15
  - Alert fires: `type=D-C level=HIGH`
- If no alert: EAR calibration may be wrong or landmarks are bad

---

### PHASE 4: Fix Issues Found (surgical only)

If tests reveal specific failures:

**If Test 3 fails (gaze not triggering):**
- Print: gaze yaw/pitch values (in debug overlay)
- Check: Is gaze actually leaving the road zone (±15° yaw, -10°–+5° pitch)?
- If gaze values are garbage (±170°): camera matrix is still wrong
- If gaze values are reasonable but alert doesn't fire: check thresholds in config.py

**If Test 4 fails (phone not triggering):**
- Check: `PHONE_CONFIDENCE_THRESHOLD` in config.py (should be 0.70)
- Run phone detector on a frame with phone visible
- Print raw detection confidence scores
- If confidence is always < 0.70 even with phone present: YOLO model needs threshold adjustment

**If Test 5 fails (drowsiness not triggering):**
- Check: EAR calibration state (should complete after 30s of driving)
- Print: current EAR value and close_threshold
- If EAR is always 0 or always 1: landmark indices wrong in `ear_calculator.py`

---

## REFERENCE MATERIALS

### PRD Sections (read if stuck)
- **PRD §5.1** — Head Pose via PnP (camera matrix, reprojection error formula)
- **PRD §5.2** — EAR & PERCLOS (eye aspect ratio thresholds)
- **PRD §5.3** — Gaze World Transform (requires valid head pose)
- **PRD §6** — Distraction Scoring (composite score formula, thresholds)
- **PRD §7** — Alert State Machine (DEGRADED state FM-04, P-04 rule)
- **PRD §11** — Model Specifications (input shapes, preprocessing)
- **PRD §19** — Configuration Reference (all config parameters)

### Key Files
- `config.py` — All thresholds and model paths
- `main.py` — T-0/T-1/T-2/T-3 thread architecture
- `layer2_signals/signal_processor.py` — Signal orchestration (where gaze_world is computed)
- `layer5_alert/alert_state_machine.py` — FSM states and transitions

### Diagnostic Scripts (already in repo)
- `test_reprojection.py` — Validates camera matrix accuracy
- `test_signal_debug.py` — Prints signal frame contents for debugging
- `test_live_diagnostic.py` — Real-time signal inspection with webcam

---

## KNOWN DEVIATIONS (documented)

**Gaze Model:**
- Current: Single-frame MobileNetV3 (no LSTM state I/O)
- PRD requires: MobileNetV3+LSTM with stateful hidden state
- Status: Accepted for Mac dev phase, must be replaced before Phase 7
- Impact: Gaze may be noisier (lacks temporal continuity), but system still functions

**PFLD Landmarks:**
- Current: 68-point iBUG format
- PRD specifies: 98-point format
- Status: Accepted deviation (benchmarks comparable)
- Impact: None on current system (landmark indices are correct for 68pt)

---

## SUCCESS CRITERIA

System is working when:
- ✅ Face detected only when real face in frame
- ✅ No face when camera covered
- ✅ Composite score rises above 0.55 when distracted (gaze > ±15°, eyes closed, phone present)
- ✅ Alert fires and audio plays (Ping.aiff for HIGH, Sosumi.aiff for URGENT)
- ✅ No spurious alerts when looking normally at camera
- ✅ JSONL event log records state transitions and alerts

---

## DO NOT CHANGE

- Layer architecture (6 layers, thread design, message passing)
- API contracts between layers (input/output message schemas)
- Model weights or inference pipelines (only config parameters)
- Core algorithms (Kalman filter, scoring formula, state machine logic)
- Existing test suite (only add tests if fixing bugs)

---

## CHECKLIST

- [ ] Run test_reprojection.py, print results, confirm camera matrix is the issue
- [ ] Decide between Option A (temporary) or Option B (calibrate) for fix
- [ ] Apply fix
- [ ] Run main.py --display --debug
- [ ] Complete all 5 test scenarios
- [ ] Document any NEW issues found (not camera calibration)
- [ ] Fix NEW issues (surgical changes only)
- [ ] Verify success criteria met
- [ ] Report: what was broken, what was fixed, confirmation of working detection


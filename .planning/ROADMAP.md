# Attentia Drive v2 — Roadmap

## Milestone 1: Mac Development Build

### Phase 1: Foundation ✅
Goal: Project skeleton, all config values, all data structures defined and tested.
Status: Complete (79 tests pass)

### Phase 2: Signal Processor ✅
Goal: All maths and signal logic working with mock/synthetic inputs.
Status: Complete (48 tests pass)

### Phase 2b: Calibration System ✅
Goal: Session state saves and loads correctly across restarts.
Status: Complete (34 tests pass)

### Phase 3: Temporal Engine ✅
Goal: All timing, buffering, and scoring inputs working correctly.
Status: Complete (96 tests pass)

### Phase 4: Scoring Engine + Alert State Machine ✅
Goal: Full scoring and alerting logic working end-to-end with synthetic inputs.
Status: Complete (102 tests pass)

### Phase 5: Model Integration (Mac / ONNX) ✅
Goal: Real models running on Mac with ONNX runtime.
Status: Complete (518/518 tests pass)
Canonical refs: PRD_v2.md §FR-1.1–FR-1.6, layer1_perception/

### Phase 6: Video + Output + Full Wiring
Goal: Full pipeline running live on Mac webcam.
Status: In progress
**Plans:** 4/5 plans executed

Plans:
- [x] 06-01-PLAN.md — Wave 0: test stub files for all Phase 6 modules
- [x] 06-02-PLAN.md — Wave 1: layer0_video/webcam_source.py (WebcamSource)
- [x] 06-03-PLAN.md — Wave 1: layer6_output/audio_handler.py (AudioAlertHandler)
- [x] 06-04-PLAN.md — Wave 1: layer6_output/event_logger.py (EventLogger, 6 log methods)
- [ ] 06-05-PLAN.md — Wave 2: main.py (T-0/T-1/T-2/T-3 thread architecture) + config.py verify + integration tests

Canonical refs: PRD_v2.md §FR-0.1–FR-0.4, §FR-6.1, §9, §3.3, config.py

### Phase 7: Validation (Mac)
Goal: Confirm system meets accuracy and reliability targets on Mac.
Status: Pending

### Phase 8: Hardware Handoff (RK3568 + IMX219)
Goal: Production hardware deployment.
Status: Pending (hardware team)

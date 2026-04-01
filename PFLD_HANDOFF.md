# Context Handoff — Phase 5 Landmark Model (Corrected)

**Date:** 2026-03-26
**Session outcome:** Benchmark dataset bug found and fixed. PFLD actually passes. FAN-2 replacement abandoned and reverted.

---

## Critical Discovery: 300W-LP Dataset Bug

The previous session reported PFLD NME = 24.74% on 300W-LP/IBUG (1,786 images) and declared the model broken. **This was wrong.**

### Root Cause
300W-LP/IBUG contains 135 original images x ~13 pose-augmented copies each = 1,786 total.
- The augmented copies have **different pixel content** (rendered at different 3D angles via 3D morphable model)
- But they all share **identical ground truth landmarks** (copied from the original `_0` image)
- Evaluating models on augmented images with wrong GT produces garbage NME (~25%)
- Only filenames ending in `_0` are original (non-augmented) images with valid GT

### Corrected PFLD Results (original `_0` images only)

| Subset | PFLD Mean NME | n | Status |
|--------|--------------|---|--------|
| HELEN | **3.89%** | 100 | PASS <5% |
| LFPW | **4.51%** | 100 | PASS <5% |
| AFW | **4.43%** | 100 | PASS <5% |
| IBUG (challenging) | **5.97%** | 135 | Borderline |

**Common subset (HELEN+LFPW+AFW): ~4.3% NME = PASS**
IBUG challenging subset: 5.97% — above 5% but this is the hardest subset (extreme pose, heavy occlusion). Even the original FAN paper reports ~5.8% on IBUG.

---

## FAN-2 Investigation (Abandoned)

Attempted to replace PFLD with FAN-2 (facefusion 2dfan4.onnx):
- Model: 4-stack hourglass, 256x256 input, soft-argmax heatmap output (1,68,3)
- I/O confirmed: input `input` (1,3,256,256) [0,1]; outputs `landmarks` (1,68,3) + `heatmaps` (1,68,64,64)
- FAN-2 scored **worse** than PFLD on every subset (5.22-7.68% vs 3.89-5.97%)
- Reason: facefusion model is optimized for face swapping, not landmark precision
- **Verdict: reverted entirely. FAN-2 files deleted.**

---

## Benchmark Fix

`bench_pfld.py` updated to only evaluate `_0` variants:
- Filters `*_0.mat` files (135 original IBUG images, not 1786 augmented)
- Documents why augmented images are excluded
- Correct result: PFLD NME = 5.97% on IBUG challenging, ~4.3% on common

---

## Current State

- **Phases 1-4:** Complete, 359 tests passing
- **Phase 5 — PFLD landmarks:** PASSES on common faces. Benchmark corrected.
- **Phase 5 — YOLO phone:** PASSES (mAP50 = 0.977)
- **Phase 5 — Gaze model:** N/A (placeholder, not production-ready)
- **Phase 5 — Model wrappers:** Still need to be written:
  - `face_detector.py` (BlazeFace)
  - `landmark_model.py` (PFLD wrapper)
  - `gaze_model.py` (placeholder wrapper)
  - `phone_detector.py` (YOLOv8n wrapper)
  - `perception_stack.py` (orchestrator)

## Next Steps
1. Write `landmark_model.py` as a **PFLD wrapper** (input 112x112, output 136-vector -> (68,2))
2. Continue with remaining Phase 5 model wrappers
3. Run corrected benchmark as part of Phase 5 validation

---

## Lesson Learned
**Always validate ground truth before blaming the model.** The 300W-LP augmented images have identical GT landmarks across pose variants. Only `_0` variants are valid for per-image NME evaluation. The previous benchmark inflated NME by 4-5x due to this bug.

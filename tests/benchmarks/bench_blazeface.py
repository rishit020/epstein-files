"""
bench_blazeface.py — BlazeFace recall benchmark on 300W common + challenging subsets.

Metric : Detection recall @ conf >= FACE_CONFIDENCE_GATE (0.60).
         Each image contains exactly one frontal face — a detection is a hit if
         face.present=True with confidence >= 0.60.
Dataset: data/300w/300W_LP/ — original _0 images only from HELEN, LFPW, AFW, IBUG.
         _Flip/ subdirectories are excluded (mirrored augmentation).
Target : Recall >= 0.90 (Mac/ONNX dev baseline — RKNN production target deferred to Phase 8).

Note: DMD dataset is unavailable for Mac dev phase. 300W provides a representative
      proxy of diverse face poses, lighting, and partial occlusion.

Run from project root:
    python tests/benchmarks/bench_blazeface.py
"""

import os
import sys
import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
os.chdir(_ROOT)

import numpy as np
import cv2
import onnxruntime as ort

import config

# ─── Constants ────────────────────────────────────────────────────────────────

SUBSETS     = ['HELEN', 'LFPW', 'AFW', 'IBUG']
DATA_ROOT   = _ROOT / 'data' / '300w' / '300W_LP'
RESULTS_MD  = Path(__file__).parent / 'BENCHMARK_RESULTS.md'

RECALL_TARGET = 0.90
INPUT_SIZE    = 128

# Use very low conf_threshold — model's built-in filter is unreliable for this export.
# We apply FACE_CONFIDENCE_GATE manually after getting all candidate detections.
CONF_THRESH   = np.array([0.001], dtype=np.float32)
IOU_THRESH    = np.array([0.30],  dtype=np.float32)
MAX_DET       = np.array([100],   dtype=np.int64)


# ─── Inference helpers ────────────────────────────────────────────────────────

def preprocess(img_bgr: np.ndarray) -> np.ndarray:
    resized = cv2.resize(img_bgr, (INPUT_SIZE, INPUT_SIZE))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return rgb.transpose(2, 0, 1)[np.newaxis]


def detect(sess: ort.InferenceSession, img_bgr: np.ndarray) -> bool:
    """Return True if a face is detected above the confidence gate."""
    tensor = preprocess(img_bgr)
    out = sess.run(None, {
        'image':          tensor,
        'conf_threshold': CONF_THRESH,
        'max_detections': MAX_DET,
        'iou_threshold':  IOU_THRESH,
    })[0]

    # Model emits (1, N, 16) normally; squeezes N=1 → (1, 16)
    if out.ndim == 2:
        out = out[np.newaxis]   # (1, 16) → (1, 1, 16)

    n = out.shape[1]
    if n == 0:
        return False

    confs = out[0, :, 4]
    return bool((confs >= config.FACE_CONFIDENCE_GATE).any())


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_benchmark():
    print(f"\nBlazeFace Recall Benchmark")
    print(f"Model: {config.BLAZEFACE_MODEL_PATH}")
    print(f"Conf gate: {config.FACE_CONFIDENCE_GATE}")
    print(f"Target recall: >= {RECALL_TARGET:.0%}\n")

    sess = ort.InferenceSession(str(_ROOT / config.BLAZEFACE_MODEL_PATH))

    results_by_subset = {}
    total_hits = total_images = 0

    for subset in SUBSETS:
        subset_dir = DATA_ROOT / subset
        if not subset_dir.exists():
            print(f"  [{subset}] directory not found, skipping")
            continue

        images = sorted(subset_dir.glob('*_0.jpg'))
        hits = skipped = 0

        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                skipped += 1
                continue
            if detect(sess, img):
                hits += 1

        n = len(images) - skipped
        recall = hits / n if n > 0 else 0.0
        results_by_subset[subset] = {'n': n, 'hits': hits, 'recall': recall}
        total_hits += hits
        total_images += n

        status = 'PASS' if recall >= RECALL_TARGET else 'FAIL'
        print(f"  [{subset}] n={n:4d}  hits={hits:4d}  recall={recall:.3f}  {status}")

    overall = total_hits / total_images if total_images > 0 else 0.0
    overall_status = 'PASS' if overall >= RECALL_TARGET else 'FAIL'
    print(f"\n  Overall  n={total_images:4d}  hits={total_hits:4d}  recall={overall:.3f}  {overall_status}")
    print(f"\n{'='*55}")

    _update_results_md(results_by_subset, overall, total_images, total_hits)
    return overall >= RECALL_TARGET


def _update_results_md(results_by_subset, overall, total_images, total_hits):
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    rows = ""
    for subset, r in results_by_subset.items():
        status = 'PASS' if r['recall'] >= RECALL_TARGET else 'FAIL'
        rows += f"| {subset} | {r['n']} | {r['hits']} | {r['recall']:.3f} | {status} |\n"

    overall_status = 'PASS' if overall >= RECALL_TARGET else 'FAIL'
    overall_emoji = '✅' if overall >= RECALL_TARGET else '❌'

    new_section = f"""---
## BlazeFace — Face Detector

**Date:** {now}
**Model:** `{config.BLAZEFACE_MODEL_PATH}`
**Dataset:** `data/300w/300W_LP/` — original `_0` images only (HELEN, LFPW, AFW, IBUG)
**Metric:** Detection recall @ conf >= {config.FACE_CONFIDENCE_GATE}
**Target:** Recall >= {RECALL_TARGET:.0%} (Mac/ONNX dev proxy — DMD unavailable)

| Subset | Images | Hits | Recall | Result |
|---|---|---|---|---|
{rows}| **Overall** | **{total_images}** | **{total_hits}** | **{overall:.3f}** | **{overall_emoji} {overall_status}** |

### Notes
- DMD dataset unavailable for Mac dev phase. 300W is used as a diverse proxy
  (varied pose, lighting, partial occlusion across HELEN/LFPW/AFW/IBUG splits).
- Production BlazeFace recall target (vs DMD) and RKNN delta validation deferred to Phase 8.
- Bug fix applied: BlazeFace ONNX model emits squeezed (1,16) shape for single detections;
  `face_detector._parse_detections` now normalises this to (1,1,16) before processing.

"""

    existing = RESULTS_MD.read_text() if RESULTS_MD.exists() else ""

    # Replace existing BlazeFace section or append
    import re
    pattern = r'---\n## BlazeFace.*?(?=\n---\n## |\Z)'
    if re.search(pattern, existing, re.DOTALL):
        updated = re.sub(pattern, new_section.rstrip('\n'), existing, flags=re.DOTALL)
    else:
        updated = existing + "\n" + new_section

    RESULTS_MD.write_text(updated)
    print(f"Results written to {RESULTS_MD}")


if __name__ == '__main__':
    passed = run_benchmark()
    sys.exit(0 if passed else 1)

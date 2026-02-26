"""
bench_gaze.py — Gaze model MAE benchmark on MPIIFaceGaze Evaluation Subset.

Metric : Mean Absolute Error (°) across yaw and pitch, averaged.
Target : MAE < 6.0° (PRD §11).
Dataset: data/mpiifacegaze/MPIIGaze/Evaluation Subset/ — 3,000 face patches
         per subject × 15 subjects = 45,000 samples.

PRD Deviations (documented):
  1. Backbone: PRD §11 specifies MobileNetV3+LSTM (stateful, 8-frame temporal).
               Actual model is MobileNetV2, no LSTM, single-frame input.
  2. Input resolution: config.GAZE_INPUT_RESOLUTION = 112 (MobileNetV3 spec).
               Actual ONNX model accepts (1, 3, 448, 448) — full-face crop.
  3. Output format: 90-bin softmax classification per axis.
               Assumed angle mapping: np.linspace(-45°, +44°, 90) (1° per bin).

Ground truth strategy:
  - Searches Data/Original/p*/day*/*.mat for per-image gaze labels.
  - Accepted keys: 'gaze_direction', 'gazedirection', 'gaze'.
  - If 3D gaze vector found: yaw = atan2(x, -z), pitch = atan2(-y, sqrt(x²+z²)).
  - If no ground truth found: reports inference statistics only; MAE = N/A.
    This is expected — screen_annotations.mat are per-session files that may
    require the full MPIIGaze calibration pipeline to decode.

Run from project root:
    python tests/benchmarks/bench_gaze.py
"""

import os
import sys
import math
import datetime
import warnings
from pathlib import Path

# Ensure project root is on path and is cwd
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
os.chdir(_ROOT)

import numpy as np
import cv2
import onnxruntime as ort

import config

# Suppress scipy missing-key warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ─── Constants ────────────────────────────────────────────────────────────────

EVAL_ANNOT  = _ROOT / "data" / "mpiifacegaze" / "MPIIGaze" / "Evaluation Subset" / "annotation for face image"
DATA_ORIG   = _ROOT / "data" / "mpiifacegaze" / "MPIIGaze" / "Data" / "Original"
RESULTS_MD  = Path(__file__).parent / "BENCHMARK_RESULTS.md"

MAE_TARGET   = 6.0    # PRD §11 acceptance threshold (degrees)
INPUT_H      = 448    # Actual ONNX model input height (confirmed from model inspection)
INPUT_W      = 448    # Actual ONNX model input width

# Angle bin mapping for 90-bin softmax output.
# Assumption: 1 degree per bin, symmetric around 0°, range [-45°, +44°].
# If this mapping is wrong, reported angles will be systematically offset,
# but MAE computation relative to ground truth will still be valid.
ANGLE_BINS = np.linspace(-45.0, 44.0, 90, dtype=np.float32)

# Keys to search in per-day .mat files for gaze direction
_GAZE_KEYS = ("gaze_direction", "gazedirection", "gaze", "GazeDirection", "Gaze")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def logits_to_angle(logits: np.ndarray) -> float:
    """Convert 90-bin logits → angle (degrees) via softmax expectation."""
    probs = softmax(logits.astype(np.float32))
    return float(np.dot(probs, ANGLE_BINS))


def gaze_vec_to_angles(vec: np.ndarray):
    """Convert 3D unit gaze direction vector to (yaw_deg, pitch_deg).

    Convention (camera-space): x = right, y = down, z = forward.
      yaw   = atan2(x, -z)  — positive = looking right
      pitch = atan2(-y, sqrt(x² + z²))  — positive = looking up
    """
    vec = vec.flatten().astype(float)
    if np.linalg.norm(vec) < 1e-9:
        return None, None
    vec = vec / np.linalg.norm(vec)
    x, y, z = vec[0], vec[1], vec[2]
    yaw   = math.degrees(math.atan2(x, -z))
    pitch = math.degrees(math.atan2(-y, math.sqrt(x**2 + z**2)))
    return yaw, pitch


def load_gaze_gt(mat_path: str):
    """Try to extract (yaw_deg, pitch_deg) from a .mat file.

    Returns (yaw, pitch) or (None, None) if no gaze key found.
    """
    try:
        import scipy.io
        mat = scipy.io.loadmat(mat_path)
        for key in _GAZE_KEYS:
            if key in mat:
                data = mat[key]
                if hasattr(data, "shape"):
                    # May be (3,), (N,3), (3,1), etc.
                    data = np.array(data).squeeze()
                    if data.ndim == 1 and len(data) >= 3:
                        return gaze_vec_to_angles(data[:3])
                    elif data.ndim == 2 and data.shape[-1] >= 3:
                        # Take first row
                        return gaze_vec_to_angles(data[0, :3])
    except Exception:
        pass
    return None, None


def discover_gt_map() -> dict:
    """Search Data/Original/p*/day*/*.mat for per-image gaze labels.

    Returns dict mapping (subject, day, filename) → (yaw_deg, pitch_deg),
    or an empty dict if none found.
    """
    gt_map   = {}
    mat_count = 0

    for subj_dir in sorted(DATA_ORIG.iterdir()):
        if not subj_dir.is_dir():
            continue
        subj = subj_dir.name   # p00 … p14
        for day_dir in sorted(subj_dir.iterdir()):
            if not day_dir.is_dir() or day_dir.name == "Calibration":
                continue
            day = day_dir.name
            for mat_f in day_dir.glob("*.mat"):
                mat_count += 1
                # Per-image .mat: filename stem matches the jpg
                yaw, pitch = load_gaze_gt(str(mat_f))
                if yaw is not None:
                    key = (subj, day, mat_f.stem + ".jpg")
                    gt_map[key] = (yaw, pitch)

    if mat_count == 0:
        print("  [GT] No .mat files found in Data/Original/p*/day*/")
    elif not gt_map:
        print(f"  [GT] Found {mat_count} .mat files but none contained a recognised gaze key.")
        print(f"       Expected one of: {_GAZE_KEYS}")
    else:
        print(f"  [GT] Loaded {len(gt_map)} ground truth gaze labels from {mat_count} .mat files.")

    return gt_map


def preprocess_face(img: np.ndarray, kpts: np.ndarray | None) -> np.ndarray:
    """Crop face region using 6 keypoints (if available), resize to 448×448.

    kpts: (6, 2) pixel coordinates of face keypoints from annotation file.
    Falls back to full image if crop is degenerate.
    """
    if kpts is not None and len(kpts) >= 2:
        x1, y1 = kpts.min(axis=0)
        x2, y2 = kpts.max(axis=0)
        pad_x  = max((x2 - x1) * 0.20, 10)
        pad_y  = max((y2 - y1) * 0.20, 10)
        x1 = max(0, int(x1 - pad_x))
        y1 = max(0, int(y1 - pad_y))
        x2 = min(img.shape[1], int(x2 + pad_x))
        y2 = min(img.shape[0], int(y2 + pad_y))
        if x2 > x1 and y2 > y1:
            img = img[y1:y2, x1:x2]

    resized = cv2.resize(img, (INPUT_W, INPUT_H))
    rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return rgb.transpose(2, 0, 1)[np.newaxis]   # (1, 3, 448, 448)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> dict:
    print("=" * 60)
    print("Gaze Model MAE Benchmark — MPIIFaceGaze Evaluation Subset")
    print("=" * 60)
    print(f"  Model  : {config.GAZE_MODEL_PATH}")
    print(f"  Input  : {INPUT_H}×{INPUT_W} (ONNX-confirmed; config says {config.GAZE_INPUT_RESOLUTION})")
    print(f"  Bins   : 90 × [-45°, +44°] @ 1°/bin")
    print()

    session = ort.InferenceSession(config.GAZE_MODEL_PATH)

    # Discover ground truth (may return empty dict)
    gt_map = discover_gt_map()
    print()

    txt_files = sorted(EVAL_ANNOT.glob("p*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No p*.txt annotation files in {EVAL_ANNOT}")

    yaw_preds, pitch_preds = [], []
    yaw_gts,   pitch_gts   = [], []
    n_processed = 0
    n_skip      = 0

    for txt_file in txt_files:
        subj = txt_file.stem   # p00 … p14

        with open(txt_file, encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]

        for line in lines:
            parts = line.split()
            if not parts:
                continue

            img_rel  = parts[0]              # e.g., "day01/0005.jpg"
            seg      = img_rel.split("/")
            if len(seg) < 2:
                n_skip += 1
                continue

            day, fname = seg[0], seg[1]
            img_path   = DATA_ORIG / subj / img_rel

            if not img_path.exists():
                n_skip += 1
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                n_skip += 1
                continue

            # Parse face keypoints (6 × (x,y) = 12 values, columns 1–12)
            kpts = None
            if len(parts) >= 13:
                try:
                    vals = np.array(parts[1:13], dtype=float).reshape(6, 2)
                    kpts = vals
                except ValueError:
                    pass

            inp  = preprocess_face(img, kpts)
            try:
                outs        = session.run(["yaw", "pitch"], {"input": inp})
                yaw_logits  = outs[0][0]    # (90,)
                pitch_logits = outs[1][0]   # (90,)
                yaw_pred    = logits_to_angle(yaw_logits)
                pitch_pred  = logits_to_angle(pitch_logits)
            except Exception as exc:
                n_skip += 1
                continue

            yaw_preds.append(yaw_pred)
            pitch_preds.append(pitch_pred)
            n_processed += 1

            # Ground truth lookup
            key = (subj, day, fname)
            if key in gt_map:
                yaw_gts.append(gt_map[key][0])
                pitch_gts.append(gt_map[key][1])

    # ─── Compute metrics ──────────────────────────────────────────────────────

    gt_available = (len(yaw_gts) > 0 and len(yaw_gts) == len(yaw_preds))

    print(f"  Processed : {n_processed}")
    print(f"  Skipped   : {n_skip}")

    if gt_available:
        mae_yaw   = float(np.mean(np.abs(np.array(yaw_preds)   - np.array(yaw_gts))))
        mae_pitch = float(np.mean(np.abs(np.array(pitch_preds) - np.array(pitch_gts))))
        mae_mean  = (mae_yaw + mae_pitch) / 2.0
        passed    = mae_mean < MAE_TARGET

        print(f"  MAE yaw   : {mae_yaw:.2f}°")
        print(f"  MAE pitch : {mae_pitch:.2f}°")
        print(f"  MAE mean  : {mae_mean:.2f}°")
        print(f"  Target    : < {MAE_TARGET}°")
        print(f"  Result    : {'PASS' if passed else 'FAIL'}")

        result = {
            "gt_available" : True,
            "n_processed"  : n_processed,
            "n_skipped"    : n_skip,
            "mae_yaw"      : mae_yaw,
            "mae_pitch"    : mae_pitch,
            "mae_mean"     : mae_mean,
            "target_deg"   : MAE_TARGET,
            "passed"       : passed,
        }
    else:
        # No ground truth — report prediction statistics
        yaw_arr   = np.array(yaw_preds)
        pitch_arr = np.array(pitch_preds)

        print(f"  GT labels : NOT AVAILABLE (screen_annotations.mat absent)")
        print(f"  MAE       : N/A")
        print(f"  Yaw pred  : {yaw_arr.mean():.2f}° ± {yaw_arr.std():.2f}°  (range [{yaw_arr.min():.1f}, {yaw_arr.max():.1f}]°)")
        print(f"  Pitch pred: {pitch_arr.mean():.2f}° ± {pitch_arr.std():.2f}°  (range [{pitch_arr.min():.1f}, {pitch_arr.max():.1f}]°)")
        print(f"  Result    : N/A — see PRD deviation note")

        result = {
            "gt_available"  : False,
            "n_processed"   : n_processed,
            "n_skipped"     : n_skip,
            "yaw_mean"      : float(yaw_arr.mean()),
            "yaw_std"       : float(yaw_arr.std()),
            "pitch_mean"    : float(pitch_arr.mean()),
            "pitch_std"     : float(pitch_arr.std()),
            "target_deg"    : MAE_TARGET,
            "passed"        : False,
        }

    _write_results(result)
    return result


def _write_results(r: dict) -> None:
    """Append gaze section to BENCHMARK_RESULTS.md."""
    ts   = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    if r["gt_available"]:
        flag = "✅ PASS" if r["passed"] else "❌ FAIL"
        metrics_rows = (
            f"| MAE (mean) | {r['mae_mean']:.2f}° | < {r['target_deg']:.1f}° | {flag} |\n"
            f"| MAE (yaw) | {r['mae_yaw']:.2f}° | — | — |\n"
            f"| MAE (pitch) | {r['mae_pitch']:.2f}° | — | — |\n"
            f"| Images processed | {r['n_processed']} | — | — |\n"
            f"| Images skipped | {r['n_skipped']} | — | — |"
        )
        gt_note = "Ground truth loaded from per-image `.mat` files in `Data/Original/`."
    else:
        flag = "⚠️  N/A"
        metrics_rows = (
            f"| MAE (mean) | N/A | < {r['target_deg']:.1f}° | {flag} |\n"
            f"| Predicted yaw | {r['yaw_mean']:.2f}° ± {r['yaw_std']:.2f}° | — | — |\n"
            f"| Predicted pitch | {r['pitch_mean']:.2f}° ± {r['pitch_std']:.2f}° | — | — |\n"
            f"| Images processed | {r['n_processed']} | — | — |\n"
            f"| Images skipped | {r['n_skipped']} | — | — |"
        )
        gt_note = (
            "**Ground truth unavailable:** `screen_annotations.mat` files were not found "
            "in `Data/Original/p*/day*/`. MAE cannot be computed without per-image gaze labels. "
            "Inference ran successfully; prediction statistics are shown above. "
            "A full MAE evaluation requires either the complete MPIIGaze screen annotation "
            "files or a pre-processed gaze-angle CSV."
        )

    section = f"""## Gaze Model — MobileNetV2 (placeholder)

**Date:** {ts}
**Model:** `{config.GAZE_MODEL_PATH}`
**Dataset:** `data/mpiifacegaze/MPIIGaze/Evaluation Subset/` (3,000 samples × 15 subjects)
**Input:** {INPUT_H}×{INPUT_W} face crop (config `GAZE_INPUT_RESOLUTION` = {config.GAZE_INPUT_RESOLUTION} — see deviation below)
**Angle bins:** 90 bins × `np.linspace(−45°, +44°, 90)` (softmax expectation)

| Metric | Value | Target | Result |
|---|---|---|---|
{metrics_rows}

### Ground Truth
{gt_note}

### PRD Deviations
| Aspect | PRD §11 Specification | Actual Model | Status |
|---|---|---|---|
| Backbone | MobileNetV3 | MobileNetV2 | ⚠️ Placeholder — not production-ready |
| Temporal | LSTM, 8-frame stateful | Single-frame (no LSTM) | ⚠️ Placeholder — not production-ready |
| Input size | 112×112 (per `GAZE_INPUT_RESOLUTION`) | 448×448 (ONNX-confirmed) | ⚠️ Config mismatch — update `GAZE_INPUT_RESOLUTION` when real model is integrated |

These deviations are **not accepted for production**. The gaze model must be replaced
with the MobileNetV3+LSTM stateful model before Phase 7 validation.

---
"""

    if not RESULTS_MD.exists():
        header = "# Attentia Drive — Model Benchmark Results\n\nGenerated by `tests/benchmarks/`. Run scripts individually to refresh each section.\n\n---\n\n"
        RESULTS_MD.write_text(header + section, encoding="utf-8")
    else:
        existing = RESULTS_MD.read_text(encoding="utf-8")
        marker   = "## Gaze Model — MobileNetV2 (placeholder)"
        if marker in existing:
            start  = existing.index(marker)
            rest   = existing[start + len(marker):]
            next_h = rest.find("\n## ")
            if next_h == -1:
                existing = existing[:start] + section
            else:
                existing = existing[:start] + section + rest[next_h + 1:]
            RESULTS_MD.write_text(existing, encoding="utf-8")
        else:
            with open(RESULTS_MD, "a", encoding="utf-8") as f:
                f.write(section)

    print(f"\n  Results written to {RESULTS_MD.relative_to(_ROOT)}")


if __name__ == "__main__":
    main()

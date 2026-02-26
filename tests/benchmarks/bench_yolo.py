"""
bench_yolo.py — YOLOv8n phone detector mAP50 benchmark on Roboflow test split.

Metric : mAP50 on data/phone/test/ (1,719 images, YOLO label format).
Target : mAP50 ≥ 0.85 (PRD §11).
Method : Pure ONNX + numpy mAP50 computation (no ultralytics dependency).
         Images resized to 640×640 (training resolution) for fair evaluation.
         Confidence swept from 0.001→1.0 to produce the full PR curve.

Run from project root:
    python tests/benchmarks/bench_yolo.py
"""

import os
import sys
import datetime
from pathlib import Path

# Ensure project root is on path and is cwd
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
os.chdir(_ROOT)

import numpy as np
import cv2
import onnxruntime as ort

import config

# ─── Constants ────────────────────────────────────────────────────────────────

TEST_IMG_DIR = _ROOT / "data" / "phone" / "test" / "images"
TEST_LBL_DIR = _ROOT / "data" / "phone" / "test" / "labels"
RESULTS_MD   = Path(__file__).parent / "BENCHMARK_RESULTS.md"

MAP50_TARGET = 0.85    # PRD §11 acceptance threshold
EVAL_IMGSZ   = 640     # Match training resolution
CONF_THRESH  = 0.001   # Low threshold — sweep full PR curve
NMS_IOU      = 0.6     # NMS IoU threshold
IOU_50       = 0.50    # mAP50 evaluation IoU


# ─── ONNX Inference ───────────────────────────────────────────────────────────

def preprocess(img_bgr: np.ndarray) -> np.ndarray:
    """BGR image → (1, 3, 640, 640) float32 [0, 1] RGB."""
    resized = cv2.resize(img_bgr, (EVAL_IMGSZ, EVAL_IMGSZ))
    rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return rgb.transpose(2, 0, 1)[np.newaxis]


def run_yolo(session: ort.InferenceSession, inp: np.ndarray) -> np.ndarray:
    """Return (N, 5) detections: [cx, cy, w, h, conf] in 640-pixel space."""
    raw = session.run(None, {"images": inp})[0]   # (1, 5, 8400)
    preds = raw[0].T                               # (8400, 5): [cx,cy,w,h,cls_conf]
    # Filter by confidence
    mask  = preds[:, 4] > CONF_THRESH
    return preds[mask]                             # (N, 5)


# ─── Box utilities ────────────────────────────────────────────────────────────

def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """(N,4) cx,cy,w,h → x1,y1,x2,y2."""
    out = np.empty_like(boxes)
    out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return out


def iou_matrix(a_xyxy: np.ndarray, b_xyxy: np.ndarray) -> np.ndarray:
    """Return (M, N) IoU matrix for M pred boxes vs N gt boxes."""
    x1 = np.maximum(a_xyxy[:, None, 0], b_xyxy[None, :, 0])
    y1 = np.maximum(a_xyxy[:, None, 1], b_xyxy[None, :, 1])
    x2 = np.minimum(a_xyxy[:, None, 2], b_xyxy[None, :, 2])
    y2 = np.minimum(a_xyxy[:, None, 3], b_xyxy[None, :, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    a_area = (a_xyxy[:, 2] - a_xyxy[:, 0]) * (a_xyxy[:, 3] - a_xyxy[:, 1])
    b_area = (b_xyxy[:, 2] - b_xyxy[:, 0]) * (b_xyxy[:, 3] - b_xyxy[:, 1])
    union  = a_area[:, None] + b_area[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def nms(boxes_xyxy: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    """Greedy NMS. Returns indices of kept boxes."""
    order  = scores.argsort()[::-1]
    kept   = []
    while order.size:
        i = order[0]
        kept.append(i)
        if order.size == 1:
            break
        iou = iou_matrix(boxes_xyxy[[i]], boxes_xyxy[order[1:]])[0]
        order = order[1:][iou <= iou_thr]
    return np.array(kept, dtype=np.int64)


def load_gt_labels(label_path: Path, img_w: int, img_h: int) -> np.ndarray:
    """Load YOLO format labels → (N, 4) x1y1x2y2 in pixel coords."""
    if not label_path.exists():
        return np.empty((0, 4), dtype=np.float32)
    lines = label_path.read_text().strip().splitlines()
    boxes = []
    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue
        cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        x1 = (cx - w / 2) * img_w
        y1 = (cy - h / 2) * img_h
        x2 = (cx + w / 2) * img_w
        y2 = (cy + h / 2) * img_h
        boxes.append([x1, y1, x2, y2])
    return np.array(boxes, dtype=np.float32) if boxes else np.empty((0, 4), dtype=np.float32)


# ─── mAP Computation ─────────────────────────────────────────────────────────

def compute_map50(all_dets: list, all_gt: list, iou_thr: float = IOU_50) -> float:
    """Compute mAP50 from per-image detection lists.

    all_dets: list of (N, 5) arrays [x1,y1,x2,y2,conf] per image
    all_gt  : list of (M, 4) arrays [x1,y1,x2,y2] per image
    Returns: AP at iou_thr (scalar)
    """
    n_gt = sum(len(gt) for gt in all_gt)
    if n_gt == 0:
        return 0.0

    # Flatten all detections, tracking which image each came from
    records = []   # (conf, is_tp)
    gt_matched = [np.zeros(len(gt), dtype=bool) for gt in all_gt]

    for img_idx, (dets, gt) in enumerate(zip(all_dets, all_gt)):
        if len(dets) == 0:
            continue
        # Sort descending by confidence (already done in main loop, but ensure)
        order = dets[:, 4].argsort()[::-1]
        dets  = dets[order]

        if len(gt) == 0:
            for d in dets:
                records.append((d[4], 0))
            continue

        iou = iou_matrix(dets[:, :4], gt)   # (N_det, N_gt)
        for di in range(len(dets)):
            conf = dets[di, 4]
            # Find best matching GT
            best_iou  = 0.0
            best_gi   = -1
            for gi in range(len(gt)):
                if not gt_matched[img_idx][gi] and iou[di, gi] > best_iou:
                    best_iou = iou[di, gi]
                    best_gi  = gi
            if best_iou >= iou_thr and best_gi >= 0:
                gt_matched[img_idx][best_gi] = True
                records.append((conf, 1))   # TP
            else:
                records.append((conf, 0))   # FP

    if not records:
        return 0.0

    records.sort(key=lambda x: -x[0])
    tp_cum = np.cumsum([r[1] for r in records])
    fp_cum = np.cumsum([1 - r[1] for r in records])
    prec   = tp_cum / (tp_cum + fp_cum + 1e-9)
    rec    = tp_cum / (n_gt + 1e-9)

    # Interpolated AP (101-point)
    ap = 0.0
    for thr in np.linspace(0, 1, 101):
        p_at_r = prec[rec >= thr]
        ap += (p_at_r.max() if len(p_at_r) else 0.0) / 101.0

    return float(ap)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> dict:
    print("=" * 60)
    print("YOLOv8n Phone Detector mAP50 Benchmark")
    print("=" * 60)
    print(f"  Model  : {config.YOLO_MODEL_PATH}")
    print(f"  Data   : {TEST_IMG_DIR}")
    print(f"  imgsz  : {EVAL_IMGSZ}px")
    print()

    session    = ort.InferenceSession(config.YOLO_MODEL_PATH)
    img_files  = sorted(TEST_IMG_DIR.glob("*.jpg")) + sorted(TEST_IMG_DIR.glob("*.png"))

    if not img_files:
        raise FileNotFoundError(f"No images in {TEST_IMG_DIR}")

    all_dets   = []
    all_gt     = []
    n_skip     = 0

    for img_path in img_files:
        lbl_path = TEST_LBL_DIR / (img_path.stem + ".txt")

        img = cv2.imread(str(img_path))
        if img is None:
            n_skip += 1
            continue

        img_h, img_w = img.shape[:2]
        gt_boxes = load_gt_labels(lbl_path, img_w, img_h)   # (M, 4) in pixel coords

        inp  = preprocess(img)                        # (1, 3, 640, 640)
        try:
            raw_dets = run_yolo(session, inp)         # (N, 5): cx,cy,w,h,conf in 640-space
        except Exception as e:
            n_skip += 1
            continue

        if len(raw_dets) == 0:
            all_dets.append(np.empty((0, 5), dtype=np.float32))
            all_gt.append(gt_boxes)
            continue

        # Convert from 640-space to original pixel space
        scale_x = img_w / EVAL_IMGSZ
        scale_y = img_h / EVAL_IMGSZ
        pred_xyxy = xywh_to_xyxy(raw_dets[:, :4])
        pred_xyxy[:, [0, 2]] *= scale_x
        pred_xyxy[:, [1, 3]] *= scale_y

        # NMS
        kept = nms(pred_xyxy, raw_dets[:, 4], NMS_IOU)
        pred_xyxy = pred_xyxy[kept]
        confs     = raw_dets[kept, 4]

        dets_final = np.concatenate([pred_xyxy, confs[:, None]], axis=1)  # (K,5)
        all_dets.append(dets_final)
        all_gt.append(gt_boxes)

    print(f"  Images processed : {len(img_files) - n_skip}")
    print(f"  Images skipped   : {n_skip}")

    map50  = compute_map50(all_dets, all_gt, iou_thr=IOU_50)
    passed = map50 >= MAP50_TARGET

    # Also compute precision/recall at conf=0.25 for reference
    ref_conf = 0.25
    tp_total, fp_total, gt_total = 0, 0, sum(len(g) for g in all_gt)
    for dets, gt in zip(all_dets, all_gt):
        hi_dets = dets[dets[:, 4] >= ref_conf] if len(dets) else dets
        if len(hi_dets) == 0:
            continue
        if len(gt) == 0:
            fp_total += len(hi_dets)
            continue
        iou = iou_matrix(hi_dets[:, :4], gt)
        matched_gt = set()
        for di in range(len(hi_dets)):
            best_gi = iou[di].argmax()
            if iou[di, best_gi] >= IOU_50 and best_gi not in matched_gt:
                tp_total += 1
                matched_gt.add(best_gi)
            else:
                fp_total += 1

    prec_ref = tp_total / (tp_total + fp_total + 1e-9)
    rec_ref  = tp_total / (gt_total + 1e-9)

    print(f"  mAP50            : {map50:.4f}")
    print(f"  Precision@{ref_conf}   : {prec_ref:.4f}")
    print(f"  Recall@{ref_conf}      : {rec_ref:.4f}")
    print(f"  Target           : mAP50 ≥ {MAP50_TARGET}")
    print(f"  Result           : {'PASS' if passed else 'FAIL'}")

    result = {
        "n_images"    : len(img_files) - n_skip,
        "n_skipped"   : n_skip,
        "map50"       : map50,
        "precision"   : prec_ref,
        "recall"      : rec_ref,
        "ref_conf"    : ref_conf,
        "target_map50": MAP50_TARGET,
        "passed"      : passed,
    }

    _write_results(result)
    return result


def _write_results(r: dict) -> None:
    """Append YOLO section to BENCHMARK_RESULTS.md."""
    ts   = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    flag = "✅ PASS" if r["passed"] else "❌ FAIL"

    section = f"""## YOLOv8n — Phone Detector

**Date:** {ts}
**Model:** `{config.YOLO_MODEL_PATH}`
**Dataset:** `data/phone/test/` (Roboflow — {r['n_images']} images, class: cell-phones)
**Evaluation imgsz:** {EVAL_IMGSZ}px  |  **NMS IoU:** {NMS_IOU}  |  **mAP IoU:** {IOU_50}

| Metric | Value | Target | Result |
|---|---|---|---|
| mAP50 | {r['map50']:.4f} | ≥ {r['target_map50']:.2f} | {flag} |
| Precision (conf≥{r['ref_conf']}) | {r['precision']:.4f} | — | — |
| Recall (conf≥{r['ref_conf']}) | {r['recall']:.4f} | — | — |
| Images evaluated | {r['n_images']} | — | — |
| Images skipped | {r['n_skipped']} | — | — |

### Notes
- mAP50 computed using 101-point interpolated PR curve with IoU threshold 0.50.
- Model was fine-tuned from YOLOv8n COCO pretrained weights on the Roboflow
  pedestrian cell-phone detection dataset. No PRD deviations.
- Evaluation imgsz=640 (training resolution). Runtime uses `YOLO_INPUT_RESOLUTION={config.YOLO_INPUT_RESOLUTION}`.

---
"""

    if not RESULTS_MD.exists():
        header = "# Attentia Drive — Model Benchmark Results\n\nGenerated by `tests/benchmarks/`. Run scripts individually to refresh each section.\n\n---\n\n"
        RESULTS_MD.write_text(header + section, encoding="utf-8")
    else:
        existing = RESULTS_MD.read_text(encoding="utf-8")
        marker   = "## YOLOv8n — Phone Detector"
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

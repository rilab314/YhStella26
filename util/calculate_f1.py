import csv
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Tuple
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

import numpy as np
import torch
from tqdm import tqdm

from configs.config import CfgNode
from model.instance_generator import GeneratePolylineInstances
from util.compare_iou import evaluate_one_image, prf_from_counts


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_label_maps(labels: List[dict]) -> Tuple[Dict[int, str], Dict[int, int]]:
    id2name: Dict[int, str] = {}
    cat2id: Dict[int, int] = {}
    for l in labels:
        try:
            lid = int(l.get("id"))
        except Exception:
            continue
        name = l.get("name", f"class_{lid}")
        id2name[lid] = name
        try:
            cat = int(l.get("category_id"))
            cat2id[cat] = lid
        except Exception:
            pass
    return id2name, cat2id


def _clip_pts(pts: np.ndarray, img_size: List[int]) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 2:
        return np.zeros((0, 2), dtype=np.float32)
    pts[:, 0] = np.clip(pts[:, 0], 0, img_size[0] - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, img_size[1] - 1)
    return pts


def _to_pixel_pts(pts: np.ndarray, img_size: List[int]) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 2:
        return np.zeros((0, 2), dtype=np.float32)
    if float(np.max(pts)) <= 1.5 and float(np.min(pts)) >= -0.5:
        pts[:, 0] *= float(img_size[0])
        pts[:, 1] *= float(img_size[1])
    return _clip_pts(pts, img_size)


def parse_gt_instances(gt_data: Any, img_size: List[int], cat2id: Dict[int, int]) -> Dict[int, List[np.ndarray]]:
    out: Dict[int, List[np.ndarray]] = defaultdict(list)
    if not isinstance(gt_data, list):
        return out

    for obj in gt_data:
        if not isinstance(obj, dict):
            continue

        if "label" in obj and "points" in obj:
            try:
                label_id = int(obj["label"])
            except Exception:
                continue
            pts = _to_pixel_pts(np.asarray(obj["points"], dtype=np.float32), img_size)
            if pts.shape[0] < 2:
                continue
            out[label_id].append(pts)
            continue

        if obj.get("class") != "RoadObject" or obj.get("geometry_type") != "LINE_STRING":
            continue

        pts = obj.get("image_points", None)
        cat = obj.get("category_id", None)
        if pts is None or cat is None:
            continue

        try:
            cat_i = int(cat)
        except Exception:
            continue

        label_id = int(cat2id.get(cat_i, 0))
        pts = _to_pixel_pts(np.asarray(pts, dtype=np.float32), img_size)
        if pts.shape[0] < 2:
            continue

        out[label_id].append(pts)

    return out


def parse_pred_instances(pred_data: Any, img_size: List[int]) -> Dict[int, List[np.ndarray]]:
    out: Dict[int, List[np.ndarray]] = defaultdict(list)

    if isinstance(pred_data, list) and len(pred_data) == 1 and isinstance(pred_data[0], list):
        pred_data = pred_data[0]
    if not isinstance(pred_data, list):
        return out

    for obj in pred_data:
        if not isinstance(obj, dict):
            continue
        if "label" not in obj or "points" not in obj:
            continue

        try:
            label_id = int(obj["label"])
        except Exception:
            continue

        pts = _to_pixel_pts(np.asarray(obj["points"], dtype=np.float32), img_size)
        if pts.shape[0] < 2:
            continue

        out[label_id].append(pts)

    return out


def infer_pred_format(pred_dir: str) -> str:
    has_json = any(f.lower().endswith(".json") for f in os.listdir(pred_dir))
    has_pt = any(f.lower().endswith(".pt") for f in os.listdir(pred_dir))
    if has_json and not has_pt:
        return "json"
    if has_pt and not has_json:
        return "pt"
    if has_json and has_pt:
        return "json"
    return "unknown"


def load_pred_from_pt(
    pt_path: str,
    img_size: List[int],
    labels: List[dict],
    conf_threshold: float,
) -> Dict[int, List[np.ndarray]]:
    output = torch.load(pt_path, map_location="cpu")
    if isinstance(output, list):
        outputs = output
    elif isinstance(output, dict):
        outputs = [output]
    else:
        raise ValueError(f"Unsupported .pt format: {type(output)}")

    generator = GeneratePolylineInstances(class_ids=labels, conf_threshold=conf_threshold)
    pred_list = generator(outputs)
    if not pred_list:
        return defaultdict(list)
    return parse_pred_instances(pred_list[0], img_size)


def build_file_pairs(gt_dir: str, pred_dir: str, pred_ext: str) -> List[Tuple[str, str, str]]:
    pred_files = [f for f in os.listdir(pred_dir) if f.lower().endswith(pred_ext)]
    stems = sorted([os.path.splitext(f)[0] for f in pred_files])
    pairs = []
    for stem in stems:
        gt_path = os.path.join(gt_dir, stem + ".json")
        pred_path = os.path.join(pred_dir, stem + pred_ext)
        pairs.append((stem, gt_path, pred_path))
    return pairs


def evaluate_dataset(
    gt_dir: str,
    pred_dir: str,
    pred_format: str,
    labels: List[dict],
    cat2id: Dict[int, int],
    img_size: List[int],
    thickness: int,
    iou_th: float,
    bbox_pad: int,
    conf_threshold: float,
    limit: int | None = None,
    print_per_image: bool = False,
    show_progress: bool = True,
) -> Tuple[Dict[str, int], Dict[int, Dict[str, int]]]:
    pred_ext = ".json" if pred_format == "json" else ".pt"
    pairs = build_file_pairs(gt_dir, pred_dir, pred_ext)
    if limit:
        pairs = pairs[:limit]

    class_total = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    total = {"tp": 0, "fp": 0, "fn": 0}

    it = pairs
    if show_progress:
        it = tqdm(pairs, desc=f"threshold={conf_threshold:.2f}", ncols=100)

    for stem, gt_path, pred_path in it:
        if not os.path.exists(gt_path):
            continue

        gt_data = load_json(gt_path)
        gt_instances = parse_gt_instances(gt_data, img_size=img_size, cat2id=cat2id)

        if os.path.exists(pred_path):
            if pred_format == "json":
                pred_data = load_json(pred_path)
                pred_instances = parse_pred_instances(pred_data, img_size=img_size)
            else:
                pred_instances = load_pred_from_pt(
                    pred_path,
                    img_size=img_size,
                    labels=labels,
                    conf_threshold=conf_threshold,
                )
        else:
            pred_instances = defaultdict(list)

        micro, per_class = evaluate_one_image(
            gt_instances=gt_instances,
            pred_instances=pred_instances,
            img_size=img_size,
            thickness=thickness,
            iou_th=iou_th,
            bbox_pad=bbox_pad,
        )

        for lid, stats in per_class.items():
            class_total[lid]["tp"] += stats["tp"]
            class_total[lid]["fp"] += stats["fp"]
            class_total[lid]["fn"] += stats["fn"]

        total["tp"] += micro["tp"]
        total["fp"] += micro["fp"]
        total["fn"] += micro["fn"]

        if print_per_image:
            p, r, f1 = prf_from_counts(micro["tp"], micro["fp"], micro["fn"])
            print(f"[{stem}] TP={micro['tp']} FP={micro['fp']} FN={micro['fn']} | P={p:.4f} R={r:.4f} F1={f1:.4f}")

    return total, class_total


def build_thresholds(start: float = 0.5, step: float = 0.05, end: float = 0.95) -> List[float]:
    thresholds = []
    t = start
    while t <= end + 1e-9:
        thresholds.append(round(t, 4))
        t += step
    return thresholds


def sweep_thresholds(
    thresholds: List[float],
    gt_dir: str,
    pred_dir: str,
    pred_format: str,
    labels: List[dict],
                cat2id: Dict[int, int],
                img_size: List[int],
    thickness: int,
    iou_th: float,
    bbox_pad: int,
    limit: int | None = None,
) -> Tuple[float, Dict[str, int], Dict[int, Dict[str, int]], List[Tuple[float, float]]]:
    best_th = thresholds[0]
    best_total: Dict[str, int] = {"tp": 0, "fp": 0, "fn": 0}
    best_class: Dict[int, Dict[str, int]] = {}
    best_f1 = -1.0
    series: List[Tuple[float, float]] = []

    for th in thresholds:
        total, class_total = evaluate_dataset(
            gt_dir=gt_dir,
            pred_dir=pred_dir,
            pred_format=pred_format,
            labels=labels,
            cat2id=cat2id,
            img_size=img_size,
            thickness=thickness,
            iou_th=iou_th,
            bbox_pad=bbox_pad,
            conf_threshold=th,
            limit=limit,
            print_per_image=False,
        )
        p, r, f1 = prf_from_counts(total["tp"], total["fp"], total["fn"])
        series.append((th, f1))
        if f1 > best_f1:
            best_f1 = f1
            best_th = th
            best_total = total
            best_class = class_total

    return best_th, best_total, best_class, series


def main() -> None:
    cfg_name = "stella_cfg"
    gt_dir = "/home/gorilla/kyh_workspace/project/dataset/satellite_lane/validation/json"
    pred_dir = "/home/gorilla/kyh_workspace/project/results/tblog_260127_2211/checkpoints/epoch=19_pt"
    pred_format = infer_pred_format(pred_dir)
    if pred_format == "unknown":
        raise ValueError(f"Could not infer prediction format from: {pred_dir}")
    if pred_format != "pt":
        raise ValueError(
            "Threshold sweep for F1 requires .pt predictions so that "
            "GeneratePolylineInstances(conf_threshold=...) can filter detections."
        )
    output_dir = pred_dir.replace('_pt', '_f1')

    thickness = 5
    iou_th = 0.3
    bbox_pad = max(3, thickness * 2)
    limit = None

    cfg = CfgNode.from_file(cfg_name)
    labels = cfg.dataset.labels
    id2name, cat2id = build_label_maps(labels)
    img_h = int(getattr(cfg.dataset, "image_height", 768))
    img_w = int(getattr(cfg.dataset, "image_width", img_h))
    img_size = [img_w, img_h]

    thresholds = build_thresholds(0.5, 0.05, 0.95)

    os.makedirs(output_dir, exist_ok=True)

    for conf_threshold in thresholds:
        print(f"[THRESHOLD] conf_threshold={conf_threshold:.2f}")
        total, class_total = evaluate_dataset(
            gt_dir=gt_dir,
            pred_dir=pred_dir,
            pred_format=pred_format,
            labels=labels,
            cat2id=cat2id,
            img_size=img_size,
            thickness=thickness,
            iou_th=iou_th,
            bbox_pad=bbox_pad,
            conf_threshold=conf_threshold,
            limit=limit,
            print_per_image=False,
            show_progress=True,
        )

        tp, fp, fn = total["tp"], total["fp"], total["fn"]
        p, r, f1 = prf_from_counts(tp, fp, fn)

        th_str = f"{conf_threshold*100}".rstrip("0").rstrip(".")
        csv_name = f"thres_0.{th_str}.csv"
        csv_path = os.path.join(output_dir, csv_name)

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["scope", "id", "name", "tp", "fp", "fn", "precision", "recall", "f1"])
            writer.writerow(["micro", "-", "-", tp, fp, fn, f"{p:.6f}", f"{r:.6f}", f"{f1:.6f}"])

            for lid in sorted(class_total.keys()):
                ctp = class_total[lid]["tp"]
                cfp = class_total[lid]["fp"]
                cfn = class_total[lid]["fn"]
                cp, cr, cf1 = prf_from_counts(ctp, cfp, cfn)
                class_name = id2name.get(lid, f"class_{lid}")
                writer.writerow(["class", lid, class_name, ctp, cfp, cfn, f"{cp:.6f}", f"{cr:.6f}", f"{cf1:.6f}"])

        print(f"[SAVED] {csv_path}")


if __name__ == "__main__":
    main()

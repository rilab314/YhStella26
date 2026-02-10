import os
import json
import csv
from typing import Any, Dict, List, Tuple
from collections import defaultdict

import cv2
import numpy as np
from tqdm import tqdm


LABELS = [
    dict(category_id="000", id=0, priority=11, name="ignore", color=(0, 0, 0)),
    dict(category_id="501", id=1, priority=10, name="center_line", color=(77, 77, 255)),
    dict(category_id="502", id=2, priority=6, name="u_turn_zone_line", color=(77, 178, 255)),
    dict(category_id="503", id=3, priority=7, name="lane_line", color=(77, 255, 77)),
    dict(category_id="504", id=4, priority=3, name="bus_only_lane", color=(255, 153, 77)),
    dict(category_id="505", id=5, priority=8, name="edge_line", color=(255, 77, 77)),
    dict(category_id="506", id=6, priority=4, name="path_change_restriction_line", color=(178, 77, 255)),
    dict(category_id="515", id=7, priority=5, name="no_parking_stopping_line", color=(77, 255, 178)),
    dict(category_id="525", id=8, priority=9, name="guiding_line", color=(255, 178, 77)),
    dict(category_id="530", id=9, priority=0, name="stop_line", color=(77, 102, 255)),
    dict(category_id="531", id=10, priority=1, name="safety_zone", color=(255, 77, 128)),
    dict(category_id="535", id=11, priority=2, name="bicycle_lane", color=(128, 255, 77)),
]


def _build_color_maps(labels: List[dict]) -> Tuple[Dict[int, Tuple[int, int, int]], Dict[int, int]]:
    id2color: Dict[int, Tuple[int, int, int]] = {}
    cat2id: Dict[int, int] = {}
    for l in labels:
        lid = int(l["id"])
        id2color[lid] = tuple(map(int, l["color"]))  # BGR
        cat2id[int(l["category_id"])] = lid
    return id2color, cat2id


ID2COLOR, CAT2ID = _build_color_maps(LABELS)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_image_bgr(path: str, img_size: List[int]) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    if img.shape[0] != img_size[0] or img.shape[1] != img_size[1]:
        img = cv2.resize(img, (img_size[0], img_size[1]), interpolation=cv2.INTER_LINEAR)
    return img


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
    # If coordinates look normalized ([0, 1]), convert to pixels.
    if float(np.max(pts)) <= 1.5 and float(np.min(pts)) >= -0.5:
        if isinstance(img_size, (list, tuple, np.ndarray)):
            sx, sy = float(img_size[0]), float(img_size[1])
        else:
            sx = sy = float(img_size)
        pts[:, 0] *= sx
        pts[:, 1] *= sy
    return _clip_pts(pts, img_size)


def parse_gt_instances(gt_data: Any, img_size: int) -> Dict[int, List[np.ndarray]]:
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

        label_id = int(CAT2ID.get(cat_i, 0))
        pts = _to_pixel_pts(np.asarray(pts, dtype=np.float32), img_size)
        if pts.shape[0] < 2:
            continue

        out[label_id].append(pts)

    return out


def parse_pred_instances(pred_data: Any, img_size: int) -> Dict[int, List[np.ndarray]]:
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


def _draw_polyline(img_bgr: np.ndarray, pts_xy: np.ndarray, color_bgr: Tuple[int, int, int], thickness: int) -> None:
    pts_xy = np.asarray(pts_xy, dtype=np.float32)
    if pts_xy.shape[0] < 2:
        return
    pts_i32 = np.round(pts_xy).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img_bgr, [pts_i32], isClosed=False, color=color_bgr, thickness=thickness, lineType=cv2.LINE_8)


def overlay_instances(img_bgr: np.ndarray, instances: Dict[int, List[np.ndarray]], thickness: int) -> np.ndarray:
    out = img_bgr.copy()
    for lid, polylines in instances.items():
        color = ID2COLOR.get(int(lid), (255, 255, 255))
        for pts in polylines:
            _draw_polyline(out, pts, color, thickness)
    return out


def visualize_side_by_side(
    img_bgr: np.ndarray,
    gt_instances: Dict[int, List[np.ndarray]],
    pred_instances: Dict[int, List[np.ndarray]],
    thickness: int = 3,
    window_name: str = "GT(left) vs PRED(right)",
) -> np.ndarray:
    left = overlay_instances(img_bgr, gt_instances, thickness)
    right = overlay_instances(img_bgr, pred_instances, thickness)
    cv2.putText(left, "GT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(right, "PRED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    panel = cv2.hconcat([left, right])
    cv2.imshow(window_name, panel)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return panel


# =========================
# FAST IoU via bbox pre-filter + ROI masks
# =========================
def polyline_bbox_xyxy(polyline_xy: np.ndarray, img_size: List[int], pad: int = 0) -> Tuple[int, int, int, int]:
    pts = np.asarray(polyline_xy, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] == 0:
        return (0, 0, -1, -1)

    x1 = int(np.floor(np.min(pts[:, 0]))) - pad
    y1 = int(np.floor(np.min(pts[:, 1]))) - pad
    x2 = int(np.ceil(np.max(pts[:, 0]))) + pad
    y2 = int(np.ceil(np.max(pts[:, 1]))) + pad

    x1 = max(0, min(x1, img_size[0] - 1))
    y1 = max(0, min(y1, img_size[1] - 1))
    x2 = max(0, min(x2, img_size[0] - 1))
    y2 = max(0, min(y2, img_size[1] - 1))

    if x2 < x1 or y2 < y1:
        return (0, 0, -1, -1)
    return (x1, y1, x2, y2)


def bbox_intersects(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    if ax2 < ax1 or ay2 < ay1 or bx2 < bx1 or by2 < by1:
        return False
    return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)


def _draw_polyline_on_roi(
    roi_mask: np.ndarray,
    pts_xy: np.ndarray,
    roi_x1: int,
    roi_y1: int,
    thickness: int,
) -> None:
    pts = np.asarray(pts_xy, dtype=np.float32)
    if pts.shape[0] < 2:
        return
    pts2 = pts.copy()
    pts2[:, 0] -= float(roi_x1)
    pts2[:, 1] -= float(roi_y1)
    pts2 = np.round(pts2).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(roi_mask, [pts2], isClosed=False, color=1, thickness=thickness, lineType=cv2.LINE_8)


def iou_polylines_roi(
    pred_pts: np.ndarray,
    gt_pts: np.ndarray,
    img_size: List[int],
    thickness: int,
    pad: int,
) -> float:
    pb = polyline_bbox_xyxy(pred_pts, img_size, pad=pad)
    gb = polyline_bbox_xyxy(gt_pts, img_size, pad=pad)
    if not bbox_intersects(pb, gb):
        return 0.0

    x1 = max(pb[0], gb[0])
    y1 = max(pb[1], gb[1])
    x2 = min(pb[2], gb[2])
    y2 = min(pb[3], gb[3])
    if x2 < x1 or y2 < y1:
        return 0.0

    w = int(x2 - x1 + 1)
    h = int(y2 - y1 + 1)

    pred_mask = np.zeros((h, w), dtype=np.uint8)
    gt_mask = np.zeros((h, w), dtype=np.uint8)

    _draw_polyline_on_roi(pred_mask, pred_pts, roi_x1=x1, roi_y1=y1, thickness=thickness)
    _draw_polyline_on_roi(gt_mask, gt_pts, roi_x1=x1, roi_y1=y1, thickness=thickness)

    inter = cv2.countNonZero(cv2.bitwise_and(pred_mask, gt_mask))
    union = cv2.countNonZero(cv2.bitwise_or(pred_mask, gt_mask))
    if union == 0:
        return 0.0
    return float(inter) / float(union)


def compute_iou_matrix_one_class_fast(
    gt_polylines: List[np.ndarray],
    pred_polylines: List[np.ndarray],
    img_size: List[int],
    thickness: int,
    bbox_pad: int,
) -> np.ndarray:
    """
    Fast version:
    - bbox pre-filter implicitly (non-overlap -> 0)
    - ROI-based masks instead of full 768x768 masks
    Returns iou_mat shape (P, G)
    """
    G = len(gt_polylines)
    P = len(pred_polylines)
    iou_mat = np.zeros((P, G), dtype=np.float32)
    if P == 0 or G == 0:
        return iou_mat

    # precompute bboxes once
    gt_bboxes = [polyline_bbox_xyxy(g, img_size, pad=bbox_pad) for g in gt_polylines]
    pred_bboxes = [polyline_bbox_xyxy(p, img_size, pad=bbox_pad) for p in pred_polylines]

    for p in range(P):
        pb = pred_bboxes[p]
        for g in range(G):
            if not bbox_intersects(pb, gt_bboxes[g]):
                continue
            iou_mat[p, g] = iou_polylines_roi(
                pred_pts=pred_polylines[p],
                gt_pts=gt_polylines[g],
                img_size=img_size,
                thickness=thickness,
                pad=bbox_pad,
            )
    return iou_mat


# =========================
# Instance Matching (IoU >= th)
# =========================
def hopcroft_karp_pairs(adj: List[List[int]], n_left: int, n_right: int) -> List[Tuple[int, int]]:
    INF = 10**9
    pair_u = [-1] * n_left
    pair_v = [-1] * n_right
    dist = [0] * n_left

    from collections import deque

    def bfs() -> bool:
        q = deque()
        for u in range(n_left):
            if pair_u[u] == -1:
                dist[u] = 0
                q.append(u)
            else:
                dist[u] = INF

        found_free = False
        while q:
            u = q.popleft()
            for v in adj[u]:
                u2 = pair_v[v]
                if u2 != -1 and dist[u2] == INF:
                    dist[u2] = dist[u] + 1
                    q.append(u2)
                if u2 == -1:
                    found_free = True
        return found_free

    def dfs(u: int) -> bool:
        for v in adj[u]:
            u2 = pair_v[v]
            if u2 == -1 or (dist[u2] == dist[u] + 1 and dfs(u2)):
                pair_u[u] = v
                pair_v[v] = u
                return True
        dist[u] = INF
        return False

    while bfs():
        for u in range(n_left):
            if pair_u[u] == -1:
                dfs(u)

    return [(u, pair_u[u]) for u in range(n_left) if pair_u[u] != -1]


def match_instances_one_class(iou_mat: np.ndarray, iou_th: float) -> List[Tuple[int, int, float]]:
    P, G = iou_mat.shape
    if P == 0 or G == 0:
        return []

    adj: List[List[int]] = []
    for p in range(P):
        g_list = np.where(iou_mat[p] >= float(iou_th))[0].tolist()
        adj.append(g_list)

    pairs = hopcroft_karp_pairs(adj, n_left=P, n_right=G)
    out = [(p, g, float(iou_mat[p, g])) for (p, g) in pairs]
    out.sort(key=lambda x: x[2], reverse=True)
    return out


def prf_from_counts(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return float(p), float(r), float(f1)


def evaluate_one_image(
    gt_instances: Dict[int, List[np.ndarray]],
    pred_instances: Dict[int, List[np.ndarray]],
    img_size: List[int],
    thickness: int,
    iou_th: float,
    bbox_pad: int=10,
) -> Tuple[Dict[str, int], Dict[int, Dict[str, int]]]:
    total = {"tp": 0, "fp": 0, "fn": 0}
    per_class: Dict[int, Dict[str, int]] = {}
    all_classes = set(gt_instances.keys()) | set(pred_instances.keys())

    for lid in sorted(all_classes):
        gt_list = gt_instances.get(lid, [])
        pred_list = pred_instances.get(lid, [])

        iou_mat = compute_iou_matrix_one_class_fast(
            gt_polylines=gt_list,
            pred_polylines=pred_list,
            img_size=img_size,
            thickness=thickness,
            bbox_pad=bbox_pad,
        )
        matches = match_instances_one_class(iou_mat, iou_th=iou_th)
        tp = len(matches)
        fp = len(pred_list) - tp
        fn = len(gt_list) - tp

        total["tp"] += tp
        total["fp"] += fp
        total["fn"] += fn
        per_class[int(lid)] = {"tp": tp, "fp": fp, "fn": fn}

    return total, per_class


def compute_iou_metrics(
    gt_data: Any,
    pred_data: Any,
    img_size: List[int],
    thickness: int,
    iou_th: float,
    bbox_pad: int = 10,
) -> Tuple[Dict[str, int], Dict[int, Dict[str, int]]]:
    """
    GT, PRED를 통해 image 단위로 micro 및 per-class 개수를 반환합니다.
    """
    gt_instances = parse_gt_instances(gt_data, img_size=img_size)
    pred_instances = parse_pred_instances(pred_data, img_size=img_size)
    micro, per_class = evaluate_one_image(
        gt_instances=gt_instances,
        pred_instances=pred_instances,
        img_size=img_size,
        thickness=thickness,
        iou_th=iou_th,
        bbox_pad=bbox_pad,
    )
    return micro, per_class


def save_metrics_csv(path: str, total: Dict[str, int], class_total: Dict[int, Dict[str, int]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label_id", "class_name", "tp", "fp", "fn", "precision", "recall", "f1"])

        p, r, f1 = prf_from_counts(total["tp"], total["fp"], total["fn"])
        writer.writerow(["total", "micro", total["tp"], total["fp"], total["fn"], f"{p:.4f}", f"{r:.4f}", f"{f1:.4f}"])

        for lid in sorted(class_total.keys()):
            tp = class_total[lid]["tp"]
            fp = class_total[lid]["fp"]
            fn = class_total[lid]["fn"]
            p, r, f1 = prf_from_counts(tp, fp, fn)
            class_name = next((l["name"] for l in LABELS if l["id"] == lid), f"class_{lid}")
            writer.writerow([lid, class_name, tp, fp, fn, f"{p:.4f}", f"{r:.4f}", f"{f1:.4f}"])


def build_file_triplets(img_dir: str, gt_dir: str, pred_dir: str, img_ext: str = ".png") -> List[Tuple[str, str, str, str]]:
    pred_files = [f for f in os.listdir(pred_dir) if f.lower().endswith(".json")]
    stems = sorted([os.path.splitext(f)[0] for f in pred_files])
    return [
        (stem, os.path.join(img_dir, stem + img_ext), os.path.join(gt_dir, stem + ".json"), os.path.join(pred_dir, stem + ".json"))
        for stem in stems
    ]


def main(pred_dir=None):
    img_dir = "/home/gorilla/kyh_workspace/project/dataset/satellite_lane/validation/image"
    gt_dir = "/home/gorilla/kyh_workspace/project/dataset/satellite_lane/validation/json"
    # gt_dir = "/home/gorilla/kyh_workspace/project/SatelliteDet/dataset/seedmap_cfg/label"
    pred_dir = "/home/gorilla/kyh_workspace/project/results/tblog_260127_2211/checkpoints/last_instance" if (pred_dir == None) else pred_dir
    save_path = pred_dir.replace('_instance', '_iou')
    os.makedirs(save_path, exist_ok=True)

    for thickness, iou_th in ((3, 0.2), (5, 0.2), (3, 0.3), (5, 0.3), (3, 0.4), (5, 0.4)):

        class_total = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

        img_size = [768, 768]
        thickness = thickness
        iou_th = iou_th

        # bbox_pad는 thickness보다 조금 크게 주는 게 안전함
        bbox_pad = max(3, thickness * 2)  # 예: 6

        triplets = build_file_triplets(img_dir, gt_dir, pred_dir, img_ext=".png")

        total = {"tp": 0, "fp": 0, "fn": 0}
        # triplets = triplets[:200]  # 디버깅 시 제한

        for stem, img_path, gt_path, pred_path in tqdm(triplets):
            if not os.path.exists(img_path) or not os.path.exists(gt_path):
                continue

            gt_data = load_json(gt_path)
            pred_data = load_json(pred_path) if os.path.exists(pred_path) else []

            gt_instances = parse_gt_instances(gt_data, img_size=img_size)
            pred_instances = parse_pred_instances(pred_data, img_size=img_size)

            # 시각화 필요 없으면 주석
            # img_bgr = load_image_bgr(img_path, img_size=img_size)
            # visualize_side_by_side(img_bgr, gt_instances, pred_instances, thickness=thickness, window_name=stem)

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

            p, r, f1 = prf_from_counts(micro["tp"], micro["fp"], micro["fn"])

            total["tp"] += micro["tp"]
            total["fp"] += micro["fp"]
            total["fn"] += micro["fn"]

        tp, fp, fn = total["tp"], total["fp"], total["fn"]
        p, r, f1 = prf_from_counts(tp, fp, fn)
        # print("==============================")
        # print(f"[TOTAL] IoU>={iou_th:.2f}, thickness={thickness}, bbox_pad={bbox_pad}")
        # print(f"MICRO TP={tp} FP={fp} FN={fn} | P={p:.4f} R={r:.4f} F1={f1:.4f}")
        for lid in sorted(class_total.keys()):
            tp = class_total[lid]["tp"]
            fp = class_total[lid]["fp"]
            fn = class_total[lid]["fn"]

            p, r, f1 = prf_from_counts(tp, fp, fn)

            class_name = next(
                (l["name"] for l in LABELS if l["id"] == lid),
                f"class_{lid}"
            )

            # print(
            #     f"[{lid:2d}] {class_name:30s} | "
            #     f"TP={tp:6d} FP={fp:6d} FN={fn:6d} | "
            #     f"P={p:.4f} R={r:.4f} F1={f1:.4f}"
            # )

        csv_name = os.path.join(save_path, f"iou(thk:{thickness}, th:{iou_th}).csv")
        save_metrics_csv(csv_name, total=total, class_total=class_total)
        print(f"Saved CSV: {csv_name}")


if __name__ == "__main__":
    pred_dir = "/home/gorilla/kyh_workspace/project/results/tblog_260127_2211/checkpoints/last_instance"
    main(pred_dir=pred_dir)
    pred_dir = "/home/gorilla/kyh_workspace/project/results/tblog_260127_2211/checkpoints/epoch=19_instance"
    main(pred_dir=pred_dir)

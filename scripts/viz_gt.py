"""GT 인코딩·데이터셋 육안 확인 (impl_plan M1).

사용: python scripts/viz_gt.py --config configs.exp_synthetic --split train --count 4
"""

import argparse
import importlib
import time
from pathlib import Path

import numpy as np

from stella.builder import build_instance
from stella.data.types import GridDatasetBase
from stella.train import viz


def main() -> None:
    args = parse_args()
    cfg = importlib.import_module(args.config).get_config()
    dataset = build_instance(cfg.data, cfg, base=GridDatasetBase, split=args.split)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    elapsed = draw_samples(dataset, cfg, out_dir, args.count)
    print(f"{args.count} samples -> {out_dir}  (평균 {elapsed * 1000:.1f} ms/sample)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs.exp_synthetic")
    parser.add_argument("--split", default="train")
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--out", default="viz_gt_out")
    return parser.parse_args()


def draw_samples(dataset: GridDatasetBase, cfg, out_dir: Path, count: int) -> float:
    stride = cfg.data.grid_stride
    total = 0.0
    for idx in range(min(count, len(dataset))):
        start = time.perf_counter()
        sample = dataset[idx]
        total += time.perf_counter() - start
        write_sample(sample, out_dir, idx, stride)
    return total / max(count, 1)


def write_sample(sample: dict, out_dir: Path, idx: int, stride: int) -> None:
    import cv2

    image = sample["image"].numpy()
    positive = sample["class_map"].numpy() > 0
    coord = sample["coord_map"].numpy()
    conn = sample["conn_cells"].numpy()
    pages = {
        "heat": viz.draw_heatmap(image, positive.astype(np.float32)),
        "class": viz.draw_class_map(image, sample["class_map"].numpy(), positive, stride),
        "slot": viz.draw_slots(image, coord, *_gt_slots(conn, coord, stride), positive, stride),
        "inst": viz.draw_instances(image, sample["instances"]),
        "end": viz.draw_heatmap(image, sample["end_map"].numpy()),
    }
    for name, page in pages.items():
        cv2.imwrite(str(out_dir / f"{idx:03d}_{name}.png"), page[..., ::-1])


def _gt_slots(conn: np.ndarray, coord: np.ndarray, stride: int) -> tuple[np.ndarray, np.ndarray]:
    """GT 이웃 셀 좌표에서 방향·존재를 유도해 예측과 같은 형태로 만든다 (6.2절 유도식)."""
    side, degree = conn.shape[0], conn.shape[2]
    rows, cols = np.meshgrid(np.arange(side), np.arange(side), indexing="ij")
    origin = np.stack([cols + 0.5, rows + 0.5], axis=-1)[:, :, None, :]
    target_ij = np.clip(conn, 0, side - 1)
    target = (
        np.stack([target_ij[..., 1], target_ij[..., 0]], axis=-1)
        + coord[target_ij[..., 0], target_ij[..., 1]]
    )
    delta = target - origin
    norm = np.linalg.norm(delta, axis=-1, keepdims=True)
    direction = delta / np.maximum(norm, 1e-6)
    exist = (conn[..., 0] >= 0).astype(np.float32)
    return direction.reshape(side, side, degree, 2), exist


if __name__ == "__main__":
    main()

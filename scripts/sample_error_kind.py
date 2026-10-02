"""원인 하나(예: 긴 가짜 선)의 표본을 무작위로 뽑아 그린다 — 사람이 눈으로 원인을 확인한다 (E5).

E2 는 "정답이 없는 곳에 길게 그린 선"을 기하 조건으로만 센다. 그중 **정답 라벨이 빠진 곳**
(사진에는 차선이 있는데 라벨이 없음)이 얼마나 섞였는지는 사진을 봐야 안다.

    python scripts/sample_error_kind.py --method stella --kind spurious_long --count 30

한 장 = 원본 | 정답 | 예측(문제의 선만 굵은 빨강, 나머지는 얇은 회색).
→ experiment/data/e5/{방법}_{원인}/NN_{타일}_{선 번호}.png
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

from stella.builder import build_instance
from stella.config_io import load_config
from stella.data.types import GridDatasetBase
from stella.eval.error_kinds import ErrorKinds
from stella.eval.instance_io import load_instances
from stella.eval.methods import target_tiles
from stella.train import viz

PAPER_ROOT = Path("/data/ongoing/2026_stella/paper")
OUT_ROOT = Path(__file__).resolve().parents[1] / "experiment" / "data" / "e5"
SEED = 0
HIGHLIGHT = (230, 40, 40)  # 문제의 선 (RGB)
CONTEXT = (170, 170, 170)  # 나머지 예측
GAP = 12


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, [])
    found = collect(cfg, args)
    print(f"[{args.method}] {args.kind} {len(found)}개 중 {min(args.count, len(found))}개를 그린다")
    sample = random.Random(SEED).sample(found, min(args.count, len(found)))
    render_all(cfg, args, sample)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="stella")
    parser.add_argument("--kind", default="spurious_long")
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--split", default="val")
    parser.add_argument("--config", default="configs.base")
    return parser.parse_args()


def collect(cfg, args) -> list[tuple[str, int]]:
    """(타일, 예측 선 번호) — 그 원인으로 판정된 선 전부."""
    kinds = ErrorKinds(cfg)
    found = []
    for tile, target in target_tiles(cfg, args.split):
        labels = kinds.label_predictions(read_prediction(args, tile), target)
        found += [(tile, index) for index, label in enumerate(labels) if label == args.kind]
    return found


def read_prediction(args, tile: str) -> list[dict]:
    path = PAPER_ROOT / "pred" / args.method / args.split / f"{tile}.npz"
    return load_instances(path) if path.exists() else []


def render_all(cfg, args, sample: list[tuple[str, int]]) -> None:
    dataset = build_instance(cfg.data, cfg, base=GridDatasetBase, split=args.split)
    position = {stem: index for index, stem in enumerate(dataset.stems)}
    out_dir = OUT_ROOT / f"{args.method}_{args.kind}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for order, (tile, index) in enumerate(sorted(sample), start=1):
        item = dataset[position[tile]]
        sheet = render(item["image"].numpy(), item["instances"], read_prediction(args, tile), index)
        cv2.imwrite(str(out_dir / f"{order:02d}_{tile}_{index}.png"), sheet[:, :, ::-1])
    print(f"[{args.method}] 저장 {out_dir}")


def render(image: np.ndarray, truth: list[dict], prediction: list[dict], index: int) -> np.ndarray:
    plain = viz.to_uint8_image(image)
    others = plain.copy()
    for number, item in enumerate(prediction):
        if number != index:
            _polyline(others, item["points"], CONTEXT, 1)
    _polyline(others, prediction[index]["points"], HIGHLIGHT, 3)
    gap = np.full((plain.shape[0], GAP, 3), 255, np.uint8)
    return np.concatenate([plain, gap, viz.draw_instances(image, truth, 2), gap, others], axis=1)


def _polyline(canvas: np.ndarray, points, color: tuple, thickness: int) -> None:
    pixels = np.round(np.asarray(points, np.float32)).astype(np.int32)
    cv2.polylines(canvas, [pixels], False, color, thickness, cv2.LINE_AA)


if __name__ == "__main__":
    main()

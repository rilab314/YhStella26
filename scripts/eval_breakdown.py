"""종류별 분해 · 중복 선 비율을 5단계 산출물 옆에 남긴다 (6단계 · GPU 없음).

`scripts/eval_methods.py` 가 떨궈 둔 **예측 폴리라인 파일**만 다시 읽는다 — 디코딩을 다시 하지
않으므로 방법마다 몇 분이면 끝난다. 정답은 정답 주입 캐시(`pred_cache/gt_*`)에 함께 저장된
것을 쓴다(모든 캐시의 정답은 같은 라벨에서 나왔다).

    python scripts/eval_breakdown.py --method stella --split val

남기는 것:

    paper/score/{방법}_{split}/breakdown.json    종류별 표 · 중복 선 비율

**`eval_methods.py` 를 다시 돌리면 그 폴더가 통째로 지워진다** — 이 스크립트도 뒤이어 다시 돌린다.
"""

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import torch

from stella.config_io import load_config
from stella.eval.breakdown import ClassBreakdown, RedundancyCounter
from stella.eval.instance_io import load_instances

DATA_ROOT = Path("/data/ongoing/2026_stella")
# `scripts/eval_methods.py` 와 같은 값이다 — 경로를 바꾸면 함께 고친다.
PAPER_ROOT = DATA_ROOT / "paper"
CACHE_ROOT = DATA_ROOT / "pred_cache"
GT_CACHE = {"val": "gt_val_v12"}


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    tiles = list_tiles(args)
    classes, redundancy = measure(tiles, args)
    save(args, classes, redundancy, len(tiles))
    print(f"[{args.method}] 타일 {len(tiles)}장 · {time.perf_counter() - started:.0f} s")
    print(f"  중복 선 비율 {redundancy.summary()['redundant_ratio']:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--config", default="configs.base")
    parser.add_argument("--gt-cache", default="", help="정답을 읽을 캐시 폴더 이름")
    parser.add_argument("--workers", type=int, default=12)
    return parser.parse_args()


def list_tiles(args: argparse.Namespace) -> list[tuple[Path, Path]]:
    """(예측 파일, 정답 파일) 짝. 정답이 없는 타일이 있으면 멈춘다 — 조용히 빼면 값이 틀린다."""
    pred_dir = PAPER_ROOT / "pred" / args.method / args.split
    gt_dir = CACHE_ROOT / (args.gt_cache or GT_CACHE[args.split])
    pairs = [(path, gt_dir / path.name) for path in sorted(pred_dir.glob("*.npz"))]
    missing = [pred.name for pred, gt in pairs if not gt.exists()]
    if not pairs or missing:
        raise SystemExit(f"예측 {len(pairs)}장 · 정답 없음 {len(missing)}장 ({pred_dir})")
    return pairs


def measure(tiles: list, args: argparse.Namespace) -> tuple[ClassBreakdown, RedundancyCounter]:
    """타일을 일꾼 수만큼 나눠 누적하고 마지막에 더한다."""
    chunks = [(tiles[index :: args.workers], args.config) for index in range(args.workers)]
    with Pool(args.workers) as pool:
        parts = pool.map(measure_chunk, chunks)
    classes, redundancy = parts[0]
    for other_classes, other_redundancy in parts[1:]:
        classes.merge(other_classes)
        redundancy.merge(other_redundancy)
    return classes, redundancy


def measure_chunk(job: tuple) -> tuple[ClassBreakdown, RedundancyCounter]:
    tiles, config = job
    torch.set_num_threads(1)
    cfg = load_config(config, [])
    classes = ClassBreakdown(cfg)
    redundancy = RedundancyCounter.from_eval_cfg(cfg.eval)
    for pred_path, gt_path in tiles:
        prediction = load_instances(pred_path)
        classes.update(prediction, load_instances(gt_path))
        redundancy.update(prediction)
    return classes, redundancy


def save(args, classes: ClassBreakdown, redundancy: RedundancyCounter, count: int) -> None:
    payload = {
        "method": args.method,
        "split": args.split,
        "tiles": count,
        "redundancy": redundancy.summary(),
        "per_class": classes.summary(),
    }
    path = PAPER_ROOT / "score" / f"{args.method}_{args.split}" / "breakdown.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[{args.method}] 저장 {path}")


if __name__ == "__main__":
    main()

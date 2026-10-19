"""분할 비교 대상을 우리 자로 잰다 — 3-A (GPU 불필요).

칠해진 예측 PNG 를 폴리라인으로 바꿔(`SegmentVectorizer`) 우리 평가 코드(`InstanceCCQ`)로
채점한다. **우리 모델과 같은 데이터·같은 split·같은 지표**라야 비교가 성립한다.

사용:
    python scripts/eval_baseline_seg.py --pred <예측폴더> --split val --count 0
"""

import argparse
import time
from pathlib import Path

from torch.utils.data import DataLoader

from stella.baseline.segment_vectorize import CLASS_RGB, SegmentVectorizer, read_prediction
from stella.builder import build_instance
from stella.config_io import apply_override, load_config
from stella.data.types import GridDatasetBase, collate_fn
from stella.eval.ccq import InstanceCCQ

REPORTED = ("f1", "f1_macro", "precision", "recall", "coverage", "correctness", "frag")


def main() -> None:
    args = parse_args()
    cfg = build_cfg(args)
    loader = build_loader(cfg, args.split)
    vectorizer = build_vectorizer(cfg, args)
    metric = InstanceCCQ.from_cfg(cfg.eval, cfg)
    start = time.perf_counter()
    scored = run(loader, vectorizer, metric, Path(args.pred), args.count)
    report(metric, scored, time.perf_counter() - start, cfg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs.base")
    parser.add_argument("--pred", required=True, help="분할 예측 PNG 폴더")
    parser.add_argument("--split", default="val")
    parser.add_argument("--count", type=int, default=0, help="0이면 전체")
    parser.add_argument("--epsilon", type=float, default=2.0, help="꼭짓점 줄이기 허용 오차(px)")
    parser.add_argument("--min-pixels", type=int, default=20, help="이보다 작은 덩어리는 버린다")
    parser.add_argument("--override", nargs="*", default=[])
    return parser.parse_args()


def build_cfg(args: argparse.Namespace):
    cfg = load_config(args.config, [])
    for item in args.override:
        path, _, raw = item.partition("=")
        apply_override(cfg, path.split("."), raw)
    return cfg


def build_loader(cfg, split: str) -> DataLoader:
    dataset = build_instance(cfg.data, cfg, base=GridDatasetBase, split=split)
    print(f"[baseline] {split} {len(dataset)} samples")
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn)


def build_vectorizer(cfg, args: argparse.Namespace) -> SegmentVectorizer:
    """채점에서 빼는 종류는 애초에 뽑지도 않는다 — 자와 벡터화를 같은 목록으로 맞춘다."""
    excluded = set(int(label) for label in cfg.eval.exclude_classes)
    classes = tuple(label for label in sorted(CLASS_RGB) if label not in excluded)
    print(f"[baseline] 세는 종류 {len(classes)}개: {classes}")
    return SegmentVectorizer(classes=classes, epsilon=args.epsilon, min_pixels=args.min_pixels)


def run(loader, vectorizer, metric, pred_dir: Path, count: int) -> int:
    scored, missing = 0, 0
    for batch in loader:
        if count and scored >= count:
            break
        stem = str(batch["meta"][0].get("filename", "")).replace("/", "_")
        path = pred_dir / f"{stem}.png"
        if not path.exists():
            missing += 1
            continue
        metric.update(vectorizer(read_prediction(path)), batch["instances"][0])
        scored += 1
    if missing:
        print(f"[baseline] 예측이 없는 사진 {missing}장은 건너뛰었다")
    return scored


def report(metric, scored: int, elapsed: float, cfg) -> None:
    computed = {key: float(value) for key, value in metric.compute().items()}
    print(f"\n[baseline] 사진 {scored}장 · {elapsed:.1f} s · 버퍼 {cfg.eval.buffer_rho} px")
    for key in REPORTED:
        if key in computed:
            print(f"  {key:14} {computed[key]:.4f}")
    print("  — 종류별 f1 (논문 Table III 의 유도선 열이 여기서 나온다)")
    for key in sorted(key for key in computed if key.startswith("f1/")):
        print(f"  {key:34} {computed[key]:.4f}")


if __name__ == "__main__":
    main()

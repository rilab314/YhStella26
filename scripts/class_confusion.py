"""클래스 헤드 병목 분석 — "배경이라 부름" vs "종류 혼동" (E07 · 개선 루프 진단 트랙).

학습된 모델의 셀 단위 클래스 정확도가 낮다. 그런데 디코더는 `argmax != 0`(배경이
아님)인 셀만 정점으로 쓰므로, GT 양성 셀을 **배경이라 부르는 것**(정점이 아예 안 생김,
치명적)과 **다른 차선 종류로 부르는 것**(정점은 생기고 사슬 클래스만 틀림, 덜 치명적)은
디코더에 미치는 피해가 다르다. 이 스크립트는 그 둘을 갈라서 잰다.

`dump_predictions.py`가 만든 예측 캐시(npz)와 데이터셋의 GT `class_map`을 대조해
① 배경 호출/종류 혼동/정답 3분류 비율, ② 전경 클래스끼리의 혼동 행렬,
③ 클래스별 지원 수(빈도)·정확도, ④ 배경 셀의 전경 오판율과 분포를 낸다.

사용:
    python scripts/class_confusion.py --cache <캐시> --split val --count 200 --workers 4
"""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from stella.builder import build_instance
from stella.config_io import load_config
from stella.data.types import CLASS_NAMES, GridDatasetBase
from stella.decode.cache import load_prediction
from stella.decode.sweep import list_files, read_meta, shape_of

FOREGROUND_NAMES = CLASS_NAMES[1:]  # 11종, 인덱스 0..10 = 라벨 1..11
NUM_CLASSES = len(CLASS_NAMES)  # 12 (0=background + 11종)
TOP_CONFUSION_PAIRS = 5
_WORKER: dict[str, object] = {}


def main() -> None:
    args = parse_args()
    cache = Path(args.cache)
    meta = read_meta(cache)
    files = list_files(cache, args.count)
    print(f"[class_confusion] {len(files)} samples from {cache} (split={args.split})")
    stats = accumulate(args.config, args.split, files, shape_of(meta), args.workers)
    report(stats)
    if args.out:
        _write_json(args.out, stats.summary())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    default_cache = (
        "/media/humpback/435806fd-079f-4ba1-ad80-109c8f6e2ec0/Ongoing/2026_stella"
        "/pred_cache/reff_ep11_val200"
    )
    parser.add_argument("--cache", default=default_cache)
    parser.add_argument("--config", default="configs.base")
    parser.add_argument("--split", default="val")
    parser.add_argument("--count", type=int, default=0, help="0이면 캐시 전체")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", default="", help="집계 결과 JSON 저장 경로")
    return parser.parse_args()


def accumulate(config: str, split: str, files: list[Path], shape: dict, workers: int):
    """캐시 파일마다 GT와 예측을 대조해 `ConfusionStats`로 모은다."""
    if workers <= 1:
        _init_worker(config, split, shape)
        records = [_scan_one(path) for path in files]
    else:
        args_tuple = (config, split, shape)
        with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=args_tuple) as pool:
            records = list(pool.map(_scan_one, files, chunksize=4))
    stats = ConfusionStats()
    for record in records:
        stats.merge(record)
    return stats


def _init_worker(config: str, split: str, shape: dict) -> None:
    cfg = load_config(config, [])
    _WORKER["dataset"] = build_instance(cfg.data, cfg, base=GridDatasetBase, split=split)
    _WORKER["shape"] = shape


def _scan_one(path: Path) -> "ConfusionStats":
    """워커 안에서 한 타일을 채점한다. 이미지는 읽지 않는다(라벨 인코딩만 필요)."""
    dataset = _WORKER["dataset"]
    class_map = _load_gt_class_map(dataset, path.stem)
    output, _ = load_prediction(path, _WORKER["shape"])
    predicted = output.class_logit.argmax(dim=-1).numpy()
    stats = ConfusionStats()
    stats.add(class_map, output.node_mask.numpy(), predicted)
    return stats


def _load_gt_class_map(dataset, stem: str) -> np.ndarray:
    """`__getitem__`은 이미지까지 읽으므로 우회한다 — `stat_labels.py`와 같은 패턴."""
    instances = dataset._load_instances(stem)
    target = dataset.encoder.encode(instances)
    return target["class_map"]


class ConfusionStats:
    """셀 단위 GT-예측 대조 누적기. 배경 호출/종류 혼동/클래스별 지표를 한 번에 쌓는다."""

    def __init__(self) -> None:
        self.total = 0  # GT 양성 & 선택된 셀 수 (both)
        self.called_bg = 0
        self.wrong_fg = 0
        self.correct = 0
        self.confusion = np.zeros((len(FOREGROUND_NAMES), len(FOREGROUND_NAMES)), dtype=np.int64)
        self.support = np.zeros(len(FOREGROUND_NAMES), dtype=np.int64)
        self.hits = np.zeros(len(FOREGROUND_NAMES), dtype=np.int64)
        self.bg_total = 0  # GT 배경 & 선택된 셀 수
        self.bg_fp = 0
        self.bg_fp_dist: Counter = Counter()

    def add(self, class_map: np.ndarray, node_mask: np.ndarray, predicted: np.ndarray) -> None:
        positive = class_map > 0
        both = positive & node_mask
        self._add_triage(class_map, predicted, both)
        self._add_confusion(class_map, predicted, both)
        self._add_per_class(class_map, predicted, both)
        self._add_background(class_map, node_mask, predicted)

    def _add_triage(self, class_map: np.ndarray, predicted: np.ndarray, both: np.ndarray) -> None:
        gt, pred = class_map[both], predicted[both]
        self.total += int(gt.size)
        self.called_bg += int((pred == 0).sum())
        self.wrong_fg += int(((pred > 0) & (pred != gt)).sum())
        self.correct += int((pred == gt).sum())

    def _add_confusion(
        self, class_map: np.ndarray, predicted: np.ndarray, both: np.ndarray
    ) -> None:
        """전경끼리만(GT 양성·선택·예측도 전경) — 배경 호출은 항목 1에서 이미 다룬다."""
        mask = both & (predicted > 0)
        gt, pred = class_map[mask] - 1, predicted[mask] - 1
        np.add.at(self.confusion, (gt, pred), 1)

    def _add_per_class(
        self, class_map: np.ndarray, predicted: np.ndarray, both: np.ndarray
    ) -> None:
        gt = class_map[both]
        hit_labels = gt[predicted[both] == gt]
        self.support += np.bincount(gt, minlength=NUM_CLASSES)[1:]
        self.hits += np.bincount(hit_labels, minlength=NUM_CLASSES)[1:]

    def _add_background(self, class_map: np.ndarray, node_mask: np.ndarray, pred) -> None:
        bg_selected = (class_map == 0) & node_mask
        self.bg_total += int(bg_selected.sum())
        false_positive = pred[bg_selected]
        false_positive = false_positive[false_positive > 0]
        self.bg_fp += int(false_positive.size)
        self.bg_fp_dist.update(false_positive.tolist())

    def merge(self, other: "ConfusionStats") -> None:
        self.total += other.total
        self.called_bg += other.called_bg
        self.wrong_fg += other.wrong_fg
        self.correct += other.correct
        self.confusion += other.confusion
        self.support += other.support
        self.hits += other.hits
        self.bg_total += other.bg_total
        self.bg_fp += other.bg_fp
        self.bg_fp_dist.update(other.bg_fp_dist)

    def summary(self) -> dict:
        return {
            "total_both": self.total,
            "called_bg_ratio": _ratio(self.called_bg, self.total),
            "wrong_fg_ratio": _ratio(self.wrong_fg, self.total),
            "correct_ratio": _ratio(self.correct, self.total),
            "class_names": list(FOREGROUND_NAMES),
            "confusion": self.confusion.tolist(),
            "support": self.support.tolist(),
            "hits": self.hits.tolist(),
            "class_acc": [_ratio(h, s) for h, s in zip(self.hits, self.support)],
            "bg_total": self.bg_total,
            "bg_fp_ratio": _ratio(self.bg_fp, self.bg_total),
            "bg_fp_dist": {FOREGROUND_NAMES[k - 1]: v for k, v in self.bg_fp_dist.items()},
        }


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator) / denominator if denominator > 0 else 0.0


def report(stats: ConfusionStats) -> None:
    summary = stats.summary()
    _report_triage(summary)
    _report_confusion_pairs(stats)
    _report_class_table(summary)
    _report_background(summary)


def _report_triage(summary: dict) -> None:
    print("\n[1] GT 양성 & 선택된 셀의 3분류 (합 = 1.0)")
    print(f"  배경이라고 부름   : {summary['called_bg_ratio']:.4f}")
    print(f"  전경인데 종류 틀림 : {summary['wrong_fg_ratio']:.4f}")
    print(f"  정확히 맞힘       : {summary['correct_ratio']:.4f}")
    print(f"  (분모 = {summary['total_both']} 셀)")


def _report_confusion_pairs(stats: ConfusionStats) -> None:
    print(f"\n[2] 전경 혼동 상위 {TOP_CONFUSION_PAIRS}쌍 (GT -> 예측, 대각선 제외)")
    off_diag = stats.confusion.copy()
    np.fill_diagonal(off_diag, 0)
    flat_order = np.argsort(off_diag, axis=None)[::-1]
    for rank in flat_order[:TOP_CONFUSION_PAIRS]:
        gt_idx, pred_idx = np.unravel_index(rank, off_diag.shape)
        count = off_diag[gt_idx, pred_idx]
        if count <= 0:
            continue
        share = count / max(stats.support[gt_idx], 1)
        print(
            f"  {FOREGROUND_NAMES[gt_idx]:<32} -> {FOREGROUND_NAMES[pred_idx]:<32} "
            f"{count:>6d}건  (해당 GT 클래스의 {share:.1%})"
        )


def _report_class_table(summary: dict) -> None:
    print("\n[3] 클래스별 지원 수·정확도 (지원 수 오름차순 — 희소한 클래스가 위)")
    rows = sorted(
        zip(summary["class_names"], summary["support"], summary["class_acc"]), key=lambda r: r[1]
    )
    print(f"  {'클래스':<32}{'지원수':>10}{'정확도':>10}")
    for name, support, acc in rows:
        print(f"  {name:<32}{support:>10d}{acc:>10.4f}")


def _report_background(summary: dict) -> None:
    print("\n[4] 배경 셀(GT=배경 & 선택됨)의 오판")
    print(
        f"  전경이라고 오판한 비율: {summary['bg_fp_ratio']:.4f}  (분모 = {summary['bg_total']} 셀)"
    )
    top = Counter(summary["bg_fp_dist"]).most_common(5)
    for name, count in top:
        print(f"    -> {name:<32}{count:>6d}건")


def _write_json(out_path: str, payload: dict) -> None:
    import json

    Path(out_path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[class_confusion] 결과 저장 -> {out_path}")


if __name__ == "__main__":
    main()

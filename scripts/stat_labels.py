"""SEED-MAP 라벨 통계 재집계 (impl_plan 6.7.5절, M9).

전체 데이터로 ① 6.7.1 표에 없는 category_id, ② `n_max`가 충분한지,
③ 차수 > D 빈도, ④ 인코딩 시간을 다시 잰다.

사용: python scripts/stat_labels.py --split train --limit 500 --workers 8
"""

import argparse
import json
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from configs.base import get_config
from stella.builder import build_instance
from stella.data.types import GridDatasetBase

_WORKER: dict[str, object] = {}


def main() -> None:
    args = parse_args()
    cfg = get_config()
    cfg.data.limit = args.limit
    cfg.data.cache_gt = "none"
    cfg.data.augment = False
    dataset = build_instance(cfg.data, cfg, base=GridDatasetBase, split=args.split)
    print(f"[stat] split={args.split} tiles={len(dataset)} workers={args.workers}")
    records = collect(cfg, args.split, len(dataset), args.workers)
    report(records, cfg)
    if args.out:
        Path(args.out).write_text(json.dumps(_summary(records), indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=0, help="0이면 split 전체")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", default="")
    return parser.parse_args()


def collect(cfg, split: str, count: int, workers: int) -> list[dict]:
    if workers <= 1:
        _init_worker(cfg, split)
        return [_scan_one(index) for index in range(count)]
    with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(cfg, split)) as pool:
        return list(pool.map(_scan_one, range(count), chunksize=8))


def _init_worker(cfg, split: str) -> None:
    _WORKER["dataset"] = build_instance(cfg.data, cfg, base=GridDatasetBase, split=split)


def _scan_one(index: int) -> dict:
    """워커 안에서 한 타일을 훑는다. 누적 카운터는 샘플마다 비워 이중 집계를 막는다."""
    dataset = _WORKER["dataset"]
    dataset.unknown_categories.clear()
    dataset.encoder.stats.clear()
    stem = dataset.stems[index]
    start = time.perf_counter()
    instances = dataset._load_instances(stem)
    label_ms = (time.perf_counter() - start) * 1000
    start = time.perf_counter()
    target = dataset.encoder.encode(instances)
    encode_ms = (time.perf_counter() - start) * 1000
    return _sample_record(dataset, instances, target, label_ms, encode_ms)


def _sample_record(dataset, instances, target, label_ms, encode_ms) -> dict:
    class_map = target["class_map"]
    degree = (target["conn_cells"][..., 0] >= 0).sum(axis=-1)[class_map > 0]
    graph = {int(k.split("_")[1]): v for k, v in dataset.encoder.stats.items() if "degree_" in k}
    return {
        "graph_degree": Counter(graph),
        "ends": int(target["end_map"].sum()),
        "nodes": int((class_map > 0).sum()),
        "instances": len(instances),
        "points": [int(len(item["points"])) for item in instances],
        "classes": Counter(int(item["class"]) for item in instances),
        "degree": Counter(degree.tolist()),
        "unknown": Counter(dataset.unknown_categories),
        "truncated": int(dataset.encoder.stats.get("truncated_cells", 0)),
        "label_ms": label_ms,
        "encode_ms": encode_ms,
    }


def report(records: list[dict], cfg) -> None:
    nodes = np.array([r["nodes"] for r in records])
    counts = np.array([r["instances"] for r in records])
    points = np.array([p for r in records for p in r["points"]])
    print(
        f"\n[노드 수/장]  평균 {nodes.mean():.0f}  p90 {_p(nodes, 90):.0f}  "
        f"p99 {_p(nodes, 99):.0f}  최대 {nodes.max()}   (n_max = {cfg.model.n_max})"
    )
    print(
        f"[인스턴스/장] 평균 {counts.mean():.1f}  중앙 {np.median(counts):.0f}  최대 {counts.max()}"
    )
    print(
        f"[점/폴리라인] 중앙 {np.median(points):.0f}  p99 {_p(points, 99):.0f}  최대 {points.max()}"
    )
    _report_degree(records, cfg)
    _report_classes(records)
    _report_timing(records)


def _report_degree(records: list[dict], cfg) -> None:
    """그래프 차수(종점 정리 전)와 저장된 conn 수(정리 후)를 따로 본다 — 뜻이 다르다."""
    graph: Counter = Counter()
    stored: Counter = Counter()
    for record in records:
        graph.update(record["graph_degree"])
        stored.update(record["degree"])
    print(f"[그래프 차수] {_ratio_line(graph)}   (D = {cfg.data.max_degree}, 종점 정리 전)")
    print(f"[저장 conn 수] {_ratio_line(stored)}   (종점 셀은 나가는 연결이 없어 0이 된다)")
    total = sum(graph.values())
    ends = sum(r["ends"] for r in records)
    truncated = sum(r["truncated"] for r in records)
    print(f"[종점 셀]     {ends} ({ends / max(total, 1) * 100:.2f}% of nodes)")
    print(f"[차수 초과]   절단 셀 {truncated} ({truncated / max(total, 1) * 100:.3f}% of nodes)")


def _ratio_line(counter: Counter) -> str:
    total = max(sum(counter.values()), 1)
    return "  ".join(f"{k}: {v / total * 100:.2f}%" for k, v in sorted(counter.items()))


def _report_classes(records: list[dict]) -> None:
    classes: Counter = Counter()
    unknown: Counter = Counter()
    for record in records:
        classes.update(record["classes"])
        unknown.update(record["unknown"])
    print(f"[클래스 분포] {dict(sorted(classes.items()))}")
    print(f"[미등록 category_id] {dict(unknown.most_common(20)) if unknown else '없음'}")


def _report_timing(records: list[dict]) -> None:
    label = np.array([r["label_ms"] for r in records])
    encode = np.array([r["encode_ms"] for r in records])
    print(
        f"[시간] 라벨 파싱 {label.mean():.1f} ms, 인코딩 {encode.mean():.1f} ms "
        f"(p99 {_p(encode, 99):.1f} ms)"
    )


def _summary(records: list[dict]) -> dict:
    nodes = [r["nodes"] for r in records]
    return {
        "tiles": len(records),
        "nodes_mean": float(np.mean(nodes)),
        "nodes_p99": float(_p(np.array(nodes), 99)),
        "nodes_max": int(max(nodes)),
        "encode_ms_mean": float(np.mean([r["encode_ms"] for r in records])),
    }


def _p(values: np.ndarray, percent: float) -> float:
    return float(np.percentile(values, percent))


if __name__ == "__main__":
    main()

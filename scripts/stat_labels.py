"""SEED-MAP 라벨 통계 재집계 (impl_plan 6.7.5절, M10).

전체 데이터로 ① 6.7.1 표에 없는 category_id, ② `n_max`가 충분한지, ③ 새 인코더 기준
사슬 통계(사슬 길이·1셀 사슬·선 소멸·소유권 손실·건너뛴 간선), ④ 인코딩 시간을 잰다.

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
    return {
        "ends": int(target["end_map"].sum()),
        "nodes": int((target["class_map"] > 0).sum()),
        "instances": len(instances),
        "points": [int(len(item["points"])) for item in instances],
        "classes": Counter(int(item["class"]) for item in instances),
        "unknown": Counter(dataset.unknown_categories),
        "encoder": Counter(dataset.encoder.stats),
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
    _report_chains(records)
    _report_classes(records)
    _report_timing(records)


def _report_chains(records: list[dict]) -> None:
    """새 인코더(선 단위 사슬) 기준 M10 통계 — 결정 33의 '드물다' 확인도 여기서 한다."""
    enc: Counter = Counter()
    for record in records:
        enc.update(record["encoder"])
    chains = max(enc["chains"], 1)
    lines = max(enc["lines"], 1)
    edges = max(enc["edges"], 1)
    nodes = max(enc["nodes"], 1)
    ends = sum(r["ends"] for r in records)
    print(
        f"[사슬]        {enc['chains']}  평균 길이 {enc['chain_cells'] / chains:.1f}셀  "
        f"1셀 사슬 {enc['one_cell_chains'] / chains * 100:.2f}%"
    )
    print(
        f"[선 소멸]     {enc['lines_vanished']} ({enc['lines_vanished'] / lines * 100:.2f}% of "
        f"lines — 2셀 이하 또는 셀 전량 상실)"
    )
    print(
        f"[셀 손실]     소유권 {enc['cells_lost']} ({enc['cells_lost'] / nodes * 100:.2f}% of "
        f"nodes)  스침 제외 {enc['cells_scraped']}"
    )
    print(f"[건너뛴 간선] {enc['skip_edges']} ({enc['skip_edges'] / edges * 100:.2f}% of edges)")
    print(f"[끝 셀]       {ends} ({ends / nodes * 100:.2f}% of nodes)")


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

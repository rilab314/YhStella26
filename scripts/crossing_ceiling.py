"""교차하는 선과 교차하지 않는 선의 천장을 따로 잰다 — 4-5절 (GPU 불필요).

**질문**: 유도선의 정답 주입 천장이 전체보다 낮은 이유가 **다른 선과 교차하기 때문인가?**
정답을 그대로 넣었을 때(= 모델이 완벽할 때) 선 하나하나가 맞았는지를 보고, 그 선이
다른 선과 교차하는지로 두 집단을 갈라 **집단별 적중률**을 비교한다.

두 집단의 적중률이 비슷하면 교차는 원인이 아니다 → 논문의 C6 서술을 그만큼 줄인다.

"교차"의 정의: 같은 영상의 **다른 정답 선(종류 무관)과 선분이 실제로 만난다.**
끝점이 닿기만 하는 것은 교차로 치지 않는다(이어지는 선을 교차로 오인하지 않으려고).

사용:
    python scripts/crossing_ceiling.py --split val --count 0
"""

import argparse
import collections

import numpy as np
from torch.utils.data import DataLoader

from stella.builder import build_instance
from stella.config_io import load_config
from stella.data.types import CLASS_NAMES, GridDatasetBase, collate_fn
from stella.eval.ccq import InstanceCCQ, _greedy_match, _prepare
from stella.model.inject import gt_model_output

ENDPOINT_MARGIN = 1e-3  # 선분 매개변수가 이 안쪽이어야 교차 — 끝점 접촉은 뺀다


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, [])
    loader = DataLoader(
        build_instance(cfg.data, cfg, base=GridDatasetBase, split=args.split),
        batch_size=1,
        shuffle=False,
        num_workers=6,
        collate_fn=collate_fn,
    )
    decoder = build_instance(cfg.decode, cfg)
    metric = InstanceCCQ.from_cfg(cfg.eval, cfg)
    tally = collect(loader, decoder, metric, cfg, args.count)
    report(tally, cfg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs.base")
    parser.add_argument("--split", default="val")
    parser.add_argument("--count", type=int, default=0, help="0이면 전체")
    return parser.parse_args()


def collect(loader, decoder, metric, cfg, count: int) -> dict:
    """(종류, 교차여부) -> [적중 수, 전체 수]."""
    tally = collections.defaultdict(lambda: [0, 0])
    for index, batch in enumerate(loader):
        if count and index >= count:
            break
        truth = batch["instances"][0]
        output = gt_model_output(batch, cfg.data.num_classes, cfg.model.num_conn_slots)
        matched = matched_targets(metric, decoder(output[0]), truth)
        crossing = crossing_flags([np.asarray(t["points"], np.float64) for t in truth])
        for gt_index, target in enumerate(truth):
            key = (int(target["class"]), bool(crossing[gt_index]))
            tally[key][0] += int(gt_index in matched)
            tally[key][1] += 1
    return tally


def matched_targets(metric, predictions: list[dict], truth: list[dict]) -> set:
    """평가와 **같은 매칭**으로 어느 정답 선이 맞았는지 — 채점 코드를 그대로 쓴다."""
    pred = _prepare(predictions, metric.sample_step)
    gt = _prepare(truth, metric.sample_step)
    coverage, correctness = metric._score_matrices(pred, gt)
    matched_gt, _ = _greedy_match(
        gt, pred, coverage, correctness, metric.cov_thresh, metric.cor_thresh
    )
    return set(matched_gt)


def crossing_flags(lines: list[np.ndarray]) -> np.ndarray:
    """선마다 다른 선과 실제로 교차하는가."""
    flags = np.zeros(len(lines), dtype=bool)
    boxes = [_box(line) for line in lines]
    for first in range(len(lines)):
        for second in range(first + 1, len(lines)):
            if not _boxes_overlap(boxes[first], boxes[second]):
                continue
            if _polylines_cross(lines[first], lines[second]):
                flags[first] = flags[second] = True
    return flags


def _box(line: np.ndarray) -> np.ndarray:
    return np.concatenate([line.min(axis=0), line.max(axis=0)])


def _boxes_overlap(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3])


def _polylines_cross(first: np.ndarray, second: np.ndarray) -> bool:
    """선분 쌍 전부를 한꺼번에 검사한다 — 두 매개변수가 모두 (0, 1) 안쪽이면 교차."""
    p, r = first[:-1], first[1:] - first[:-1]  # (n, 2)
    q, s = second[:-1], second[1:] - second[:-1]  # (m, 2)
    denom = _cross(r[:, None], s[None])  # (n, m)
    diff = q[None] - p[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = _cross(diff, s[None]) / denom
        u = _cross(diff, r[:, None]) / denom
    inner = (ENDPOINT_MARGIN < t) & (t < 1 - ENDPOINT_MARGIN)
    inner &= (ENDPOINT_MARGIN < u) & (u < 1 - ENDPOINT_MARGIN)
    return bool((inner & (np.abs(denom) > 1e-12)).any())


def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def report(tally: dict, cfg) -> None:
    excluded = set(int(label) for label in cfg.eval.exclude_classes)
    print(f"\n{'종류':32} {'교차':>4} {'정답 선':>8} {'천장 적중률':>11}")
    for label in sorted({key[0] for key in tally}):
        if label in excluded:
            continue
        for crossing in (True, False):
            hit, total = tally.get((label, crossing), [0, 0])
            if total:
                mark = "예" if crossing else "아니오"
                print(f"{CLASS_NAMES[label]:32} {mark:>4} {total:8,} {hit / total:10.1%}")
    _report_all(tally, excluded)


def _report_all(tally: dict, excluded: set) -> None:
    print()
    for crossing in (True, False):
        hit = sum(v[0] for k, v in tally.items() if k[1] == crossing and k[0] not in excluded)
        total = sum(v[1] for k, v in tally.items() if k[1] == crossing and k[0] not in excluded)
        mark = "교차하는 선" if crossing else "교차 안 하는 선"
        print(f"  전체 · {mark:14} {total:7,}개 · 천장 적중률 {hit / max(total, 1):.1%}")


if __name__ == "__main__":
    main()

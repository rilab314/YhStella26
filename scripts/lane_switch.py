"""차선 갈아탐 진단 — 선 밖의 점을 **옆 선으로 갈아탄 것**과 **아무 데도 없는 것**으로 가른다.

`correctness`(예측 점이 GT 선 **하나** 위에 머문 비율)가 0.78 이면 22%가 밖이다. 그런데 그
22%의 처방이 둘로 갈린다 — 옆 차선으로 넘어간 것은 **디코딩(게이트·비용)** 문제이고, 아무
GT 위에도 없는 것은 **검출** 문제다. 비율만 보면 어느 쪽인지 알 수 없다.

    d_own = 가장 잘 맞는 GT 선 하나까지의 거리
    d_any = 모든 GT 선 중 가장 가까운 것까지의 거리

    갈아탐  <=>  d_own > rho  and  d_any <= rho      (옆 선 위에 있다)
    이탈    <=>  d_own > rho  and  d_any  > rho      (아무 선 위에도 없다)

사용:
    python scripts/lane_switch.py --cache <캐시> --count 80
    python scripts/lane_switch.py --cache <캐시> --set radius=6 --set align_mode=perp

`eval_decode.py`와 같은 캐시·같은 config 플러밍을 쓴다 (D 트랙, GPU 불필요).
"""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from stella.builder import build_instance
from stella.data.types import CLASS_NAMES
from stella.decode.cache import load_prediction
from stella.decode.sweep import build_cfg, list_files, read_meta, shape_of
from stella.eval import geometry
from stella.eval.ccq import _prepare


class SwitchCounter:
    """예측 점을 자기 선 위 / 옆 선으로 갈아탐 / 아무 데도 없음 셋으로 나눠 센다."""

    def __init__(self, *, buffer_rho: float, angle_gate: float, sample_step: float):
        self.buffer_rho = buffer_rho
        self.angle_cos = float(np.cos(np.deg2rad(angle_gate)))
        self.sample_step = sample_step
        self.total = Counter()
        self.by_class = {}

    def update(self, predictions: list[dict], targets: list[dict]) -> None:
        pred = _prepare(predictions, self.sample_step)
        gt = _prepare(targets, self.sample_step)
        if not pred or not gt:
            return
        for item in pred:
            self._count_one(item, gt)

    def _count_one(self, item: dict, gt: list[dict]) -> None:
        distances = self._distances(item, gt)
        if not distances:
            self._add(item["class"], stray=item["points"].shape[0])
            return
        stacked = np.stack(distances)  # (GT, P)
        own = stacked[np.argmin((stacked > self.buffer_rho).mean(axis=1))]
        nearest = stacked.min(axis=0)
        outside = own > self.buffer_rho
        self._add(
            item["class"],
            inside=int((~outside).sum()),
            switched=int((outside & (nearest <= self.buffer_rho)).sum()),
            stray=int((outside & (nearest > self.buffer_rho)).sum()),
        )

    def _distances(self, item: dict, gt: list[dict]) -> list[np.ndarray]:
        """예측 점 -> GT 선마다의 거리. 상자가 안 겹치는 GT 는 무한대로 둔다."""
        rows = []
        for other in gt:
            if not geometry.boxes_overlap(item["box"], other["box"], self.buffer_rho):
                continue
            rows.append(
                geometry.gated_distance(
                    item["points"], item["tangent"], other["points"], self.angle_cos
                )
            )
        return rows

    def _add(self, label: int, *, inside: int = 0, switched: int = 0, stray: int = 0) -> None:
        counts = {"inside": inside, "switched": switched, "stray": stray}
        self.total.update(counts)
        self.by_class.setdefault(label, Counter()).update(counts)

    def report(self) -> str:
        lines = [_header(), _row("전체", self.total)]
        for label, counts in sorted(self.by_class.items(), key=lambda kv: -kv[1].total()):
            name = CLASS_NAMES[label] if label < len(CLASS_NAMES) else str(label)
            lines.append(_row(name, counts))
        return "\n".join(lines)


def _header() -> str:
    columns = ("점 수", "자기 선 위", "갈아탐", "이탈", "밖 중 갈아탐")
    return "\n" + "종류".ljust(32) + " ".join(f"{c:>12}" for c in columns)


def _row(name: str, counts: Counter) -> str:
    total = max(counts.total(), 1)
    outside = max(counts["switched"] + counts["stray"], 1)
    values = (
        f"{counts.total():>12,}",
        f"{counts['inside'] / total:>11.1%}",
        f"{counts['switched'] / total:>11.1%}",
        f"{counts['stray'] / total:>11.1%}",
        f"{counts['switched'] / outside:>11.1%}",
    )
    return f"{name[:32]:<32}" + " ".join(values)


def main() -> None:
    args = parse_args()
    meta = read_meta(args.cache)
    files = list_files(args.cache, args.count)
    cfg = build_cfg(args.config, args.fixed, meta)
    build_instance(cfg.cpu, cfg).apply()
    decoder = build_instance(cfg.decode, cfg)
    counter = SwitchCounter(
        buffer_rho=cfg.eval.buffer_rho,
        angle_gate=cfg.eval.angle_gate,
        sample_step=cfg.eval.sample_step,
    )
    print(f"[switch] {len(files)} samples from {args.cache} (source={meta['source']})")
    for path in files:
        output, targets = load_prediction(path, shape_of(meta))
        counter.update(decoder(output), targets)
    print(counter.report())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--config", default="configs.base")
    parser.add_argument("--count", type=int, default=80, help="0이면 캐시 전체")
    parser.add_argument("--set", dest="fixed", action="append", default=[], help="파라미터 고정")
    return parser.parse_args()


if __name__ == "__main__":
    main()

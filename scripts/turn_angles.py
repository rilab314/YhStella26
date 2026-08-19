"""라벨 폴리라인의 꺾임각 분포 — `decode.max_turn_deg` 를 실측으로 정한다.

꺾임각은 **한 스텝의 길이에 따라 완전히 달라진다.** 4 px 마다 재면 격자 양자화 잡음이
각도를 부풀리고, 20 px 마다 재면 실제 곡률만 남는다. 그래서 스텝 길이별로 낸다 —
디코더의 스텝 길이는 `decode.radius` x 셀 크기(4 px)가 상한이다.

두 벌을 함께 낸다. 처방이 다르기 때문이다.

    라벨    폴리라인을 스텝 길이로 재표본 -> **실제 도로의 곡률**
    디코드  GT 주입 캐시를 **꺾임각 게이트를 끄고** 디코딩한 사슬 -> **디코더가 실제로 보는 각도**

정점은 셀 중심이 아니라 셀 안의 예측 좌표(`self_coord`)에 놓이므로, 셀 중심에 붙여 재면
격자 계단 효과로 각도가 45/90도로 튀어 **실제보다 훨씬 나쁘게 나온다**. 그래서 합성이 아니라
디코딩 결과로 잰다. `max_turn_deg` 는 이 분포의 분위수로 정해야 한다 — 라벨 쪽 분위수로
정하면 직선 위에서도 잡음에 걸려 사슬이 끊긴다(실측: 20도 상한이 GT 주입 천장을
0.981 -> 0.777 로 깎았다).

사용:
    python scripts/turn_angles.py --cache <GT 캐시> --steps 4,12,16,20   # 라벨 곡률
    python scripts/turn_angles.py --cache <GT 캐시> --decoded            # 디코더가 보는 각도
"""

import argparse
from pathlib import Path

import numpy as np

from stella.builder import build_instance
from stella.decode.cache import load_prediction
from stella.decode.sweep import build_cfg, list_files, read_meta, shape_of
from stella.eval import geometry

QUANTILES = (50, 90, 95, 99, 99.9)
CELL_PIXELS = 4.0


class TurnAngles:
    """스텝 길이 하나에 대한 꺾임각 표본을 모은다 (매끈 · 격자 스냅 두 벌)."""

    def __init__(self, *, step: float):
        self.step = step
        self.smooth: list[np.ndarray] = []

    def update(self, instances: list[dict]) -> None:
        for item in instances:
            points = np.asarray(item["points"], dtype=np.float64)
            if geometry.polyline_length(points) < 2.0 * self.step:
                continue
            sampled, _ = geometry.resample(points, self.step)
            self.smooth.append(_turn_degrees(sampled))

    def rows(self) -> list[tuple]:
        return [("라벨", _summary(self.smooth))]


def _turn_degrees(points: np.ndarray) -> np.ndarray:
    """연속한 두 스텝 사이의 방향 변화(도). 점이 3개 미만이면 빈 배열."""
    delta = np.diff(points, axis=0)
    norm = np.linalg.norm(delta, axis=1, keepdims=True)
    unit = np.divide(delta, norm, out=np.zeros_like(delta), where=norm > 1e-9)
    if unit.shape[0] < 2:
        return np.zeros(0)
    cosine = np.clip((unit[:-1] * unit[1:]).sum(axis=1), -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def _summary(chunks: list[np.ndarray]) -> dict:
    values = np.concatenate(chunks) if chunks else np.zeros(0)
    if values.size == 0:
        return {"n": 0}
    return {"n": values.size} | {q: float(np.percentile(values, q)) for q in QUANTILES}


def main() -> None:
    args = parse_args()
    meta = read_meta(args.cache)
    files = list_files(args.cache, args.count)
    print(f"[turn] {len(files)} samples from {args.cache}")
    print(_header())
    if args.decoded:
        _report_decoded(args, meta, files)
        return
    counters = [TurnAngles(step=s) for s in args.steps]
    for path in files:
        _, instances = load_prediction(path, shape_of(meta))
        for counter in counters:
            counter.update(instances)
    for counter in counters:
        for name, summary in counter.rows():
            print(_row(f"{counter.step:g} px  {name}", summary))


def _report_decoded(args, meta: dict, files: list) -> None:
    """꺾임각 게이트를 끈 디코더로 사슬을 만들어 그 사슬의 꺾임각을 잰다.

    게이트를 켠 채로 재면 게이트가 잘라 낸 분포를 다시 재는 순환이 된다 — 그래서 180도로 연다.
    """
    fixed = [*args.fixed, "max_turn_deg=180"]
    cfg = build_cfg(args.config, fixed, meta)
    build_instance(cfg.cpu, cfg).apply()
    decoder = build_instance(cfg.decode, cfg)
    chunks, lengths = [], []
    for path in files:
        output, _ = load_prediction(path, shape_of(meta))
        for item in decoder(output):
            points = np.asarray(item["points"], dtype=np.float64)
            if points.shape[0] < 3:
                continue
            chunks.append(_turn_degrees(points))
            lengths.append(np.linalg.norm(np.diff(points, axis=0), axis=1))
    print(_row(f"반경 {cfg.decode.radius}  디코드", _summary(chunks)))
    print(_row(f"반경 {cfg.decode.radius}  스텝길이 px", _summary(lengths)))


def _header() -> str:
    columns = ("표본", *(f"{q}%" for q in QUANTILES))
    return "\n" + "스텝 길이 / 좌표".ljust(24) + " ".join(f"{c:>10}" for c in columns)


def _row(label: str, summary: dict) -> str:
    if not summary.get("n"):
        return f"{label:<24}{'(없음)':>10}"
    values = f"{summary['n']:>10,}" + " ".join(f"{summary[q]:>10.1f}" for q in QUANTILES)
    return f"{label:<24}{values}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--count", type=int, default=300, help="0이면 캐시 전체")
    parser.add_argument("--steps", default="4,12,16,20", help="스텝 길이(px) 목록")
    parser.add_argument("--decoded", action="store_true", help="디코딩한 사슬의 꺾임각을 잰다")
    parser.add_argument("--config", default="configs.base")
    parser.add_argument("--set", dest="fixed", action="append", default=[], help="디코더 파라미터")
    args = parser.parse_args()
    args.steps = [float(s) for s in args.steps.split(",")]
    return args


if __name__ == "__main__":
    main()

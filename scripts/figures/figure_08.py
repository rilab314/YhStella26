"""Fig. 8 — 대표 정성 결과 (VI-G, 2단, 4~6장면 × 3열).

열: 원본 | GT | STELLA.

**"잘 된 장면만 골랐다"는 반박을 구조적으로 막는 그림이다.** 그래서 한 사분면에서만 뽑지 않고
**F1 × coverage 사분면 넷**에서 각각 후보를 만든다 — 잘 찾고 잘 덮은 장면(HH), 정확하지만 덜 덮은
장면(HL), 많이 덮었지만 틀린 장면(LH), 둘 다 낮은 장면(LL).

    python scripts/figures/figure_08.py --candidates --quadrant hh    # 사분면마다 한 번씩
    python scripts/figures/figure_08.py --quadrant hh                 # 고른 것만 조판

**후보 순위**: 그 사분면 중심에서 먼 순 — 사분면의 성격이 가장 뚜렷한 장면이 위로 온다.
사분면마다 **교차로 묶음과 그 밖의 묶음**을 따로 뽑는다(`figure_base` 의 두 묶음).
**무엇을 실을지는 사람이 고른다**(미팅 결정 09-15).

경계값은 검증셋 전량 평균(F1 0.4489 · coverage 0.6728)에서 잡았다 — 임의로 고른 값이 아니라
**이 모델의 평균**이라 "평균보다 나은/못한 장면"이라는 뜻이 분명하다.
"""

import argparse

from figure_base import SampleFigure, add_arguments

from stella.config_io import load_config

F1_SPLIT = 0.4489
COVERAGE_SPLIT = 0.6728
MIN_GT = 5  # 정답이 거의 없는 타일은 어느 사분면에서도 볼 것이 없다
QUADRANTS = ("hh", "hl", "lh", "ll")


class Qualitative(SampleFigure):
    name = "figure_08"
    split = "val"
    columns = (
        ("원본", ""),
        ("GT", ""),  # 정답 선은 저장된 예측이 아니라 데이터셋에서 읽는다
        ("STELLA", "stella"),
    )

    def __init__(self, cfg, quadrant: str):
        super().__init__(cfg)
        self.quadrant = quadrant
        self.name = f"figure_08_{quadrant}"

    def score(self, stem: str, pool: str):
        if self.scores.value("stella", stem, "n_gt") < MIN_GT:
            return None
        f1 = self.scores.value("stella", stem, "f1")
        coverage = self.scores.value("stella", stem, "coverage")
        if self._quadrant_of(f1, coverage) != self.quadrant:
            return None
        if not self.in_pool(stem, pool):
            return None
        return abs(f1 - F1_SPLIT) + abs(coverage - COVERAGE_SPLIT)

    def _quadrant_of(self, f1: float, coverage: float) -> str:
        return ("h" if f1 >= F1_SPLIT else "l") + ("h" if coverage >= COVERAGE_SPLIT else "l")


def main() -> None:
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    parser.add_argument("--quadrant", default="hh", choices=QUADRANTS, help="F1×coverage 사분면")
    args = parser.parse_args()
    figure = Qualitative(load_config(args.config, []), args.quadrant)
    if args.candidates:
        figure.write_candidates()
        return
    figure.write_selected()


if __name__ == "__main__":
    main()

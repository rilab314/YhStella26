"""Fig. 1 — 결과 쇼케이스 (I절, 1단, 2×2).

서로 다른 장면 넷에 검출된 폴리라인을 클래스 색으로 그리고 양 끝점에 흰 테두리 원을 얹는다.
독자가 첫 쪽에서 **출력물이 무엇인지** 알게 하는 그림이고, 끝점 원이 있어야 "인스턴스가 끊기지
않았다"는 주장이 그림에서 바로 읽힌다.

**후보 순위**: F1(STELLA) 내림차순.
**후보 자격**(설계 §3): 클래스 4종 이상 · 정답 인스턴스 10개 이상 — 너무 비거나 한 종류만 있는
장면은 "무엇을 검출하는지"를 보여 주지 못한다. 정답 수 상한은 묶음마다 다르다
(`figure_base` 의 두 묶음) — 교차로는 선이 많아(중앙값 56개) 80개, 그 밖의 장면은 25개.
**무엇을 실을지는 사람이 고른다**(미팅 결정 09-15) — 후보 100장을 보고 `selected.txt` 에 적는다.

사용:
    python scripts/figures/figure_01.py --candidates
    python scripts/figures/figure_01.py
"""

import argparse

from figure_base import INTERSECTION, OTHER, SampleFigure, add_arguments

from stella.config_io import load_config

GT_RANGE = {INTERSECTION: (10, 80), OTHER: (10, 25)}  # 묶음별 정답 인스턴스 수
MIN_CLASSES = 4


class Showcase(SampleFigure):
    name = "figure_01"
    split = "val"
    columns = (
        ("원본", ""),
        ("STELLA", "stella"),
    )

    def score(self, stem: str, pool: str):
        n_gt = self.scores.value("stella", stem, "n_gt")
        n_class = self.scores.value("stella", stem, "n_class")
        low, high = GT_RANGE[pool]
        if not low <= n_gt <= high or n_class < MIN_CLASSES:
            return None
        if not self.in_pool(stem, pool):
            return None
        return self.scores.value("stella", stem, "f1")


def main() -> None:
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    figure = Showcase(load_config(args.config, []))
    if args.candidates:
        figure.write_candidates()
        return
    figure.write_selected()


if __name__ == "__main__":
    main()

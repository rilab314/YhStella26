"""Fig. 1 — 결과 쇼케이스 (I절, 1단, 2×2).

서로 다른 장면 넷에 검출된 폴리라인을 클래스 색으로 그리고 양 끝점에 흰 테두리 원을 얹는다.
독자가 첫 쪽에서 **출력물이 무엇인지** 알게 하는 그림이고, 끝점 원이 있어야 "인스턴스가 끊기지
않았다"는 주장이 그림에서 바로 읽힌다.

**후보 순위**: F1(STELLA) 내림차순.
**후보 자격**(설계 §3): 클래스 4종 이상 · 정답 인스턴스 10개 이상 — 너무 비거나 한 종류만 있는
장면은 "무엇을 검출하는지"를 보여 주지 못한다. 정답 수 상한은 묶음마다 다르다
(`figure_base` 의 두 묶음) — 교차로는 선이 많아(중앙값 56개) 80개, 그 밖의 장면은 25개.
**무엇을 실을지는 사람이 고른다**(미팅 결정 09-15) — 후보 100장을 보고 `selected.txt` 에 적는다.

고른 4장은 `selected/` 에 한 장씩 남기고, **논문에 싣는 2×2 한 장**을 `figure_01.png` 로 따로
만든다. 논문 그림에는 글자를 넣지 않는다(09-23 미팅) — 그래서 이 한 장에는 STELLA 결과만 있다.

사용:
    python scripts/figures/figure_01.py --candidates
    python scripts/figures/figure_01.py
"""

import argparse

from figure_base import INTERSECTION, OTHER, SampleFigure, add_arguments, draw

from stella.config_io import load_config

GT_RANGE = {INTERSECTION: (10, 80), OTHER: (10, 25)}  # 묶음별 정답 인스턴스 수
MIN_CLASSES = 4
GRID_COLUMNS = 2


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

    def write_selected(self) -> None:
        """고른 순서대로 왼쪽 위 → 오른쪽 위 → 왼쪽 아래 → 오른쪽 아래."""
        super().write_selected()
        tiles = [
            draw(self.images(stem)[0], self.lines("stella", stem)) for stem in self.read_selected()
        ]
        rows = [tiles[start : start + GRID_COLUMNS] for start in range(0, len(tiles), GRID_COLUMNS)]
        self.write_paper_grid(rows, expected=GRID_COLUMNS)


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

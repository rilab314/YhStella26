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
만든다. 논문 그림에는 글자를 넣지 않는다(09-23 미팅) — 다만 **오른쪽에 종류 색 범례**를 붙인다
(교수 피드백 10-07 · 예시 `~/Downloads/fig_1_sample.png`). 다른 그림은 범례 없이 캡션에서
"선 색은 Fig. 1 과 같다"고만 적는다.

사용:
    python scripts/figures/figure_01.py --candidates
    python scripts/figures/figure_01.py
"""

import argparse

import numpy as np
from figure_base import INTERSECTION, OTHER, SampleFigure, add_arguments, draw
from PIL import Image, ImageDraw, ImageFont

from stella.config_io import load_config
from stella.data.types import CLASS_COLOR, CLASS_NAMES

GT_RANGE = {INTERSECTION: (10, 80), OTHER: (10, 25)}  # 묶음별 정답 인스턴스 수
MIN_CLASSES = 4
GRID_COLUMNS = 2
# 범례 — 그림에 실제로 그려지는 종류 전부(배경 0 만 뺀다). 긴 이름은 예시처럼 두 줄로 접는다.
LEGEND_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
LEGEND_FONT_PX = 34
LEGEND_WIDTH = 560
LEGEND_SWATCH = (70, 18)  # 색 막대 (가로, 세로) px
LEGEND_WRAP = 18  # 이 글자 수를 넘는 이름은 '_' 뒤에서 접는다


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
        height = sum(row[0].shape[0] for row in rows) + (len(rows) - 1) * 12
        self.write_paper_grid(rows, expected=GRID_COLUMNS, side=class_legend(height))


def class_legend(height: int) -> np.ndarray:
    """흰 바탕에 종류마다 색 막대 + 이름을 위에서부터 세로로."""
    canvas = Image.new("RGB", (LEGEND_WIDTH, height), (255, 255, 255))
    pen = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(LEGEND_FONT, LEGEND_FONT_PX)
    width, bar = LEGEND_SWATCH
    y = 10
    for label in range(1, len(CLASS_NAMES)):
        lines = _wrap(CLASS_NAMES[label])
        pen.rectangle([10, y + 12, 10 + width, y + 12 + bar], fill=tuple(CLASS_COLOR[label]))
        for number, text in enumerate(lines):
            pen.text(
                (width + 30, y + number * (LEGEND_FONT_PX + 4)), text, fill=(0, 0, 0), font=font
            )
        y += len(lines) * (LEGEND_FONT_PX + 4) + 16
    return np.asarray(canvas)


def _wrap(name: str) -> list[str]:
    """'path_change_restriction_line' -> ['path_change_', '  restriction_line'] (예시대로)."""
    if len(name) <= LEGEND_WRAP:
        return [name]
    cut = name.rfind("_", 0, LEGEND_WRAP) + 1
    return [name[:cut], "  " + name[cut:]]


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

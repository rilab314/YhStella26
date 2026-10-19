"""점검용 — **유도선만** 남겨 방법끼리 나란히 본다 (교수 미팅 09-23 지시: 눈으로 확인).

논문 그림이 아니다. "분할 방식에서 유도선이 어떻게 검출되는가"를 사람이 넘겨 보기 위한
확인 시트라, 출력도 그림 폴더가 아닌 `paper/check/guiding/` 에 따로 쌓는다.

각 패널에서 **유도선 클래스만 남기고 나머지는 지운다** — 다른 선이 섞이면 유도선이 안 보인다.
순위는 **그 타일의 정답 유도선 수**다. 많을수록 차이가 뚜렷하다.

    python scripts/figures/check_guiding.py            # 시트를 그린다
    python scripts/figures/check_guiding.py --count 20 # 장수만 바꾼다
"""

import argparse

import cv2
from figure_base import GUIDING, PAPER_ROOT, SampleFigure, draw, fresh, label

from stella.config_io import load_config
from stella.train import viz

DEFAULT_COUNT = 40
CHECK_ROOT = PAPER_ROOT / "check" / "guiding"


class GuidingCheck(SampleFigure):
    """유도선만 남긴 한 줄 시트 — 원본 · 정답 · 분할 · Sat2Graph · 우리."""

    name = "check_guiding"
    split = "val"
    # OpenCV 는 한글을 못 그린다(네모로 깨진다) — 패널 제목은 ASCII 로만 쓴다.
    columns = (
        ("image", ""),
        ("GT (guiding only)", ""),
        ("Segmentation", "seg_dp"),
        ("Sat2Graph GTE", "gte_delta"),
        ("STELLA (ours)", "stella"),
    )

    def score(self, stem: str, pool: str):
        """정답 유도선 수. 하나도 없으면 후보에서 뺀다. 시트는 묶음을 나누지 않는다."""
        count = self.guiding_count(stem)
        return float(count) if count else None

    def write_sheets(self, count: int) -> None:
        ranked = self.rank()[:count]
        out_dir = fresh(CHECK_ROOT)
        for order, (stem, lines) in enumerate(ranked, start=1):
            sheet = self.compose(stem)
            name = f"{order:03d}_{stem}_guiding{int(lines)}.png"
            cv2.imwrite(str(out_dir / name), sheet[:, :, ::-1])
        print(f"[check] 유도선 있는 타일 {len(self.rank())}장 중 상위 {len(ranked)}장 -> {out_dir}")

    def _panel(self, image, title: str, method: str, truth: list[dict], stem: str):
        if not method:
            if title == "image":
                return label(viz.to_uint8_image(image), title)
            return label(draw(image, guiding_only(truth)), title)
        return label(draw(image, guiding_only(self.lines(method, stem))), title)


def guiding_only(instances: list[dict]) -> list[dict]:
    return [item for item in instances if item["class"] == GUIDING]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs.base")
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    args = parser.parse_args()
    GuidingCheck(load_config(args.config, [])).write_sheets(args.count)


if __name__ == "__main__":
    main()

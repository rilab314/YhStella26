"""논문 Fig. 9 — 실패 사례와 한계 (VI-F · VII-A, 2단, 4행 × 4열). 파일 이름은 옛 번호(10)를 따른다.

열: 원본 | 정답 | 분할 | STELLA — 장면마다 실패 자리를 잘라 키웠다.
행: ① 교차로의 짧은 선 ② 건물 그늘 속 도로 (①·② 분할 우세, 옛 Fig. 11 에서)
    ③ 종류 혼동 ④ 정답 라벨 누락 (옛 Fig. 10 에서 — ④는 교수 미팅 09-30 지시).
옛 Fig. 10 · 11 은 그늘 장면이 겹쳐 하나로 합쳤다(사용자 10-08).
선에서 벗어남 행은 본문 근거가 없어 뺐다.

칸은 `scripts/sample_error_kind.py --class-color --with-seg` 가 그려 둔 것이다
(①·② 는 `--full --window`).
모든 선을 Fig. 1 범례의 종류 색으로 같은 굵기로 그리고 따로 표시하지 않는다(교수 지시 10-07).
여기서는 고른 칸을 쌓기만 한다.

사용:
    python scripts/figures/figure_10.py
    → paper/figure/figure_10/figure_10.png
"""

from pathlib import Path

import cv2
import numpy as np
from figure_base import FIGURE_ROOT, join_white

E5_ROOT = Path(__file__).resolve().parents[2] / "experiment" / "data" / "e5"
# (후보 폴더, 고른 번호) — 행 순서. 10-08: 옛 Fig. 10 · 11 을 하나로 — 같은 색 · 같은 굵기.
ROWS = (
    ("stella_spurious_long_cls_seg_full_short1008", "01"),  # 교차로의 짧은 선 (분할 우세)
    ("stella_spurious_long_cls_seg_full_shadow1008", "01"),  # 건물 그늘 속 도로 (분할 우세)
    ("stella_wrong_class_cls_seg_row3_1008", "03"),  # 종류 혼동 — STELLA 만 틀림(10-08)
    ("stella_spurious_long_cls_seg_1008", "03"),  # 정답 라벨 누락
)
NAME = "figure_10"


def main() -> None:
    rows = [read_row(folder, number) for folder, number in ROWS]
    path = FIGURE_ROOT / NAME / f"{NAME}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), join_white(rows, axis=0))
    print(f"[{NAME}] 논문용 그림: {path}")


def read_row(folder: str, number: str) -> np.ndarray:
    """후보 한 장(원본 | 정답 | 예측이 이미 한 줄로 붙어 있다)."""
    found = sorted((E5_ROOT / folder).glob(f"{number}_*.jpg"))
    if len(found) != 1:
        raise FileNotFoundError(f"[{NAME}] {folder} 의 {number}번을 찾지 못했다")
    return cv2.imread(str(found[0]))


if __name__ == "__main__":
    main()

"""Fig. 10 — 실패 유형 (VI-G, 2단, 4행 × 3열).

열: 원본 | 정답 | STELLA — 실패 지점만 잘라 키운 칸이다. 실패가 작아 타일 전체로는 보이지 않는다.
행: ① 옆으로 벗어남 ② 그늘에서 끊김 ③ 종류 혼동 ④ 정답이 틀린 사례(라벨 누락).
네 유형은 사용자가 고르고(10-02) ④는 교수 미팅(09-30) 지시다.

칸은 `scripts/sample_error_kind.py` 가 이미 그려 둔 것이다 — 원인 판정은 채점기와 같은 짝
(`stella/eval/error_kinds.py`), 문제의 선은 흰 테두리를 두른 빨강(1행만), 정답 선은 초록 ·
예측 선은 하늘색, 종류 혼동만 문제의 선을 종류 색으로 그린다. 여기서는 고른 칸을 쌓기만 한다.

사용:
    python scripts/figures/figure_10.py
    → paper/figure/figure_10/figure_10.png
"""

from pathlib import Path

import cv2
import numpy as np
from figure_base import FIGURE_ROOT, join_white

E5_ROOT = Path(__file__).resolve().parents[2] / "experiment" / "data" / "e5"
# (후보 폴더, 고른 번호) — 행 순서. 종류 혼동은 칸 위 글자가 없는 판(_plain)을 쓴다.
ROWS = (  # 10-05 후처리를 원고대로 바꾼 뒤 같은 장면·같은 선을 새 예측으로 다시 그린 판(_new1005)
    ("stella_drift_new1005", "02"),
    ("stella_missed_nomark_new1005", "07"),  # 2·4행은 빨강 없이(사용자 지적 10-02)
    ("stella_wrong_class_plain_new1005", "01"),
    ("stella_spurious_long_nomark_new1005", "03"),
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

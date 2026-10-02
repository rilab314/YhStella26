"""Table III — 주 비교 (VI-A, 2단) ★ 이 논문의 생사를 가르는 표.

**값의 출처는 5단계 요약 파일 하나다** — `paper/score/{방법}_{split}/summary.json`.
그 파일은 `scripts/eval_methods.py` 가 만들고, 여섯 조건이 기존 값과 소수 넷째 자리까지
일치하는 것을 확인했다(`experiment/result_0916b.md`).

행 구성(설계 `docs/paper_outline.md` §4):

    Seg + skeleton + Douglas–Peucker      분할 비교 대상
    Sat2Graph GTE (각도 섹터 + 변위 회귀)  출력 표현만 바꾼 비교 대상 — 같은 backbone/neck
    STELLA (ours), 후처리 없음
    STELLA (ours) + 후처리                 채택

**유도선 F1 열은 뺐다**(교수 지시 09-30 — 유도선을 굳이 강조할 필요 없다).
**Chamfer 거리 열을 더했다**(같은 지시) — 값은 `scripts/eval_chamfer.py` 가 남긴 `chamfer.json`.
이 열에서는 분할 방식이 이긴다(9.50 vs 14.61 px) — 통째로 놓친 선이 거리를 키운다.
본문 VI-A 에 이유를 쓴다.

**#pred 는 후처리 전/후를 나란히 본다** — "후처리가 선을 줄이기만 한다"는 서술의 수치다.

사용:
    python scripts/tables/table_03.py --split val
"""

import argparse
import json

from table_base import BLANK, SCORE_ROOT, PaperTable, number, read_summary

# (논문 행 이름, 표현, 5단계 방법 이름). 비교 대상은 둘이다 — AerialLaneNet 은 뺐다
# (교수 미팅 09-23 · `experiment/meeting_0923.md` 3절). GTE 는 **변위 회귀까지 갖춘 판**이다 —
# 각도 섹터만 가져온 판(`gte`)은 Table VI 의 한 행으로 남는다.
ROWS = (
    ("Seg + skeleton + Douglas–Peucker", "면", "seg_dp"),
    ("Sat2Graph GTE (각도 섹터 + 변위 회귀)", "셀·섹터", "gte_delta"),
    ("STELLA (ours), 후처리 없음", "셀·매칭", "stella_nopost"),
    ("STELLA (ours) + 후처리", "셀·매칭", "stella"),
)
COLUMNS = (
    "방법",
    "표현",
    "inst F1",
    "precision",
    "recall",
    "coverage",
    "frag",
    "RMS(px)",
    "Chamfer(px)",
    "#pred",
)
GSD_M = 0.255  # 픽셀 하나가 덮는 실제 거리 — RMS 를 미터로도 적을 때 쓴다


class MainComparison(PaperTable):
    name = "table_03"
    columns = COLUMNS

    def __init__(self, split: str):
        self.split = split

    def collect(self) -> list[dict]:
        return [self._row(label, kind, method) for label, kind, method in ROWS]

    def _row(self, label: str, kind: str, method: str) -> dict:
        payload = read_summary(method, self.split)
        scores = payload.get("summary", {})
        return {
            "방법": label,
            "표현": kind,
            "inst F1": number(scores.get("f1")),
            "precision": number(scores.get("precision")),
            "recall": number(scores.get("recall")),
            "coverage": number(scores.get("coverage")),
            "frag": number(scores.get("frag")),
            "RMS(px)": number(scores.get("rms"), 2),
            "Chamfer(px)": number(read_chamfer(method, self.split).get("chamfer"), 2),
            "#pred": number(payload.get("pred_per_tile"), 1),
        }


def read_chamfer(method: str, split: str) -> dict:
    """`scripts/eval_chamfer.py` 의 결과. 없으면 빈 dict — 표는 빈칸으로 남는다."""
    path = SCORE_ROOT / f"{method}_{split}" / "chamfer.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def main() -> None:
    args = parse_args()
    table = MainComparison(args.split)
    table.run()
    print(f"[table_03] split={args.split} · 빈 칸 기호 {BLANK} · RMS 는 px (× {GSD_M} = m)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="val", help="결과 표는 7단계 이후 test 로 낸다")
    return parser.parse_args()


if __name__ == "__main__":
    main()

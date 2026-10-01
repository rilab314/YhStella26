"""Table IV — 천장 분해: 남은 격차는 어느 출력에서 오는가 (VI-B, 1단).

모델 출력의 **한 묶음만 정답으로 바꿔 끼우고** 나머지는 모델 예측 그대로 둔 채 채점한다.
전부 바꿔 끼우면 표현의 천장이고, 하나도 안 바꾸면 모델 자신의 점수다. 각 묶음이 F1 을
**얼마나 올리는지**(전부 예측 대비 절대 증가량)가 이 표의 논지다.

**퍼센트로 쓰지 않는다 (교수 지시 09-30)** — "격차의 28%" 처럼 쓰면 행마다 쌓아 올린 누적값으로
읽힌다. 각 행은 서로 독립인 실험이라 전부 예측(0.4489)에서 오른 양을 그대로 적는다.

## 이 표가 말하는 것 (09-17 · val 전량 1,218장 실측)

**단일 병목이 없다.** 하나씩 따로 주면 다 합쳐도 격차의 44% 인데 전부 주면 100% 다 — 선 하나를
복원하려면 위치·종류·방향·끝 판정이 **동시에** 맞아야 한다. 가장 큰 것은 종류(28%)다 — 매칭이
종류별이라 종류를 틀리면 잘 그린 선도 통째로 오답이 된다. 다음이 정점 위치(9%)다.

(09-10 에 400장으로 잰 값은 정점 위치 2.0% · 전부 예측 0.4002 였다. 논문에는 전량 값을 쓴다.)

## 값의 출처

    python scripts/ceiling_breakdown.py --run <F06d> --count 0 --rows <행> --save
    → paper/score/ceiling_{split}/{행}.json

**행마다 한 파일**이다(전량 한 행에 30분쯤이라 나눠 돌린다). 모든 행이 같은 체크포인트·같은
디코더 설정에서 나온다.

## 두 끝값이 다른 표와 맞는지 — 맞는다

"전부 예측" 0.4489 는 Table III 의 STELLA 값과 같고, "전부 정답" 0.9402 는 Table V 의 천장(0.9404)과
넷째 자리에서만 다르다(Table V 는 정답을 캐시에서 바로 디코딩하고, 이 표는 모델 출력 틀에 정답을
얹어 디코딩한다). 표를 낼 때 이 두 값을 눈으로 대조한다.

사용:
    python scripts/tables/table_04.py --split val
"""

import argparse
import json

from table_base import BLANK, SCORE_ROOT, PaperTable, number

# (저장 이름, 논문 행 이름). 순서가 표의 행 순서다.
ROWS = (
    ("none", "학습 모델 (전부 예측)"),
    ("node", "+ 정점 위치만 정답"),
    ("class", "+ 종류만 정답"),
    ("coord", "+ 칸 안 위치만 정답"),
    ("conn", "+ 연결 방향만 정답"),
    ("end", "+ 끝 판정만 정답"),
    ("all", "전부 정답 (표현의 천장)"),
)
COLUMNS = ("조건", "inst F1", "frag", "F1 증가량")
FLOOR = "none"


class CeilingBreakdown(PaperTable):
    name = "table_04"
    columns = COLUMNS

    def __init__(self, split: str):
        self.split = split

    def collect(self) -> list[dict]:
        scores = {key: self._read(key) for key, _ in ROWS}
        return [self._row(label, scores[key], scores) for key, label in ROWS]

    def _read(self, key: str) -> dict:
        path = SCORE_ROOT / f"ceiling_{self.split}" / f"{key}.json"
        if not path.exists():
            print(f"[table_04] 행 없음: {path}")
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def _row(self, label: str, score: dict, scores: dict) -> dict:
        return {
            "조건": label,
            "inst F1": number(score.get("f1")),
            "frag": number(score.get("frag")),
            "F1 증가량": gain(score, scores[FLOOR]),
        }


def gain(score: dict, floor: dict) -> str:
    """이 행 − 전부 예측 (F1 절대값). 기준 행은 "기준", 둘 중 하나라도 없으면 빈칸."""
    if not (score and floor):
        return BLANK
    if score is floor:
        return "기준"
    return f"{score['f1'] - floor['f1']:+.4f}"


def main() -> None:
    args = parse_args()
    CeilingBreakdown(args.split).run()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="val", help="결과 표는 7단계 이후 test 로 낸다")
    return parser.parse_args()


if __name__ == "__main__":
    main()

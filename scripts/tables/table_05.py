"""Table V — 후처리 단계별 ablation (VI-C, 1단).

후처리 두 단계를 하나씩 켜 가며 이득과 대가를 격리한다. **마지막 열(정답 주입 천장)이 논지다.**

    능선만 남기기   정점 단계 — 한 차선 위에 두 줄로 선 정점을 한 줄로 줄인다 (`vertex_local_max`)
    중복 정리      선 단계 — 다른 선과 겹친 머리·꼬리를 자른다 (`dedup_high`)

## 이 표가 말하는 것 (09-11 · val 전량 실측)

1. **두 단계는 서로를 대체하지 않고 더해진다.**
2. **능선만 남기기는 천장을 전혀 깎지 않는다** — 정답에는 넓은 전경 띠가 없어 줄일 것이 없다.
3. **중복 정리의 대가는 천장 −0.3%** 다. 이 논문이 공개하는 후처리의 대가다.

"지우기만" 행은 없다 — 현행 중복 정리(`dedup_mode="ends"`)에는 **잇는 단계가 아예 없어서**
"지우기만"과 "채택"이 소수점까지 같다(`result_0910b.md` C절). 각주로 적는다.

## 열

    inst F1 · 정밀도 · 재현율     5단계 요약 파일 (`eval_methods.py`)
    ΔF1                         후처리 없음 대비
    중복 선 비율                  `eval_breakdown.py` — 같은 종류의 다른 예측선 하나와 자기 길이의
                                80% 이상이 8 px 안에서 나란히 가는 선의 비율
    #pred                       타일당 예측 선 수 — "후처리는 선을 줄이기만 한다"의 수치
    정답 주입 천장                같은 후처리 조건으로 정답을 디코딩한 F1

**미리 정한 판정 기준**: 후처리는 정밀도를 올리고 재현율을 밴드(±2%) 밖으로 떨어뜨리지 않아야
한다. 떨어뜨리면 중복이 아니라 진짜 선을 지우고 있는 것이다.

사용:
    python scripts/tables/table_05.py --split val
"""

import argparse

from table_base import PaperTable, change, number, read_breakdown, read_summary

# (논문 행 이름, 예측 방법, 천장 방법). 방법 이름은 `scripts/eval_methods.py` 의 METHODS 다.
ROWS = (
    ("후처리 없음 (디코더 출력 그대로)", "stella_nopost", "gt_nopost"),
    ("+ 능선만 남기기 (정점 단계)", "stella_ridge", "gt_ridge"),
    ("+ 중복 정리 (선 단계)", "stella_dedup", "gt_dedup"),
    ("+ 둘 다 (채택)", "stella", "gt"),
)
COLUMNS = (
    "조건",
    "inst F1",
    "ΔF1",
    "precision",
    "recall",
    "중복 선 비율",
    "#pred",
    "정답 주입 천장",
)


class PostprocessAblation(PaperTable):
    name = "table_05"
    columns = COLUMNS

    def __init__(self, split: str):
        self.split = split

    def collect(self) -> list[dict]:
        base = read_summary(ROWS[0][1], self.split).get("summary", {}).get("f1")
        rows = [self._row(label, method, ceiling, base) for label, method, ceiling in ROWS]
        if base is not None:
            rows[0]["ΔF1"] = "기준"
        return rows

    def _row(self, label: str, method: str, ceiling: str, base) -> dict:
        payload = read_summary(method, self.split)
        scores = payload.get("summary", {})
        redundancy = read_breakdown(method, self.split).get("redundancy", {})
        ceiling_scores = read_summary(ceiling, self.split).get("summary", {})
        return {
            "조건": label,
            "inst F1": number(scores.get("f1")),
            "ΔF1": change(scores.get("f1"), base),
            "precision": number(scores.get("precision")),
            "recall": number(scores.get("recall")),
            "중복 선 비율": percent(redundancy.get("redundant_ratio")),
            "#pred": number(payload.get("pred_per_tile"), 1),
            "정답 주입 천장": number(ceiling_scores.get("f1")),
        }


def percent(value) -> str:
    return number(None) if value is None else f"{float(value) * 100:.1f}%"


def main() -> None:
    args = parse_args()
    PostprocessAblation(args.split).run()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="val", help="결과 표는 7단계 이후 test 로 낸다")
    return parser.parse_args()


if __name__ == "__main__":
    main()

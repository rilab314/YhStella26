"""Table VI — 표현·구조 ablation (VI-D, 2단) ★ 가장 중요한 표.

첫 행이 **채택한 최적 조합**이고, 아래 각 행은 그 조합에서 **왼쪽 열의 인자 하나만** 바꾼
실행이다. 여섯 실행 모두 **F 규격(전체 데이터 · 40에폭) · 같은 백본·넥·디코더 설정**이다.
직전 논문 심사에서 "누적식 ablation 으로는 독립 효과를 못 본다"(R1-2)는 지적을 받아 한 번에
하나만 바꾼다.

## 행 — 무엇을 뺐고 무엇을 넣었나

- **`(dx,dy)` 회귀 행은 뺐다** (사용자 결정 09-17). 방향 대신 좌표를 회귀하게 하면 디코더까지
  고쳐야 해 "타깃의 효과"와 "디코더의 효과"를 가를 수 없다.
- **각도 순서 고정 행을 넣었다** (`paper_review_0915.md` 2.1). 사진에서 계산되는 합리적 고정
  기준이라 슬롯 수까지 같은 **가장 깨끗한 비교**다. 라벨 순서(−61%)는 기준을 잘못 고른 경우,
  각도 섹터는 슬롯 수(2 → 6)도 함께 바뀐 경우라 이 행이 없으면 "합리적으로 고정하면 된다"는
  반박에 답이 없다.

## 열 — 오른쪽 세 열이 인스턴스 점수 변화에 기전을 붙인다

    link_ok        한 칸에서 다음 칸으로 제대로 이어질 확률
    dir_err (°)    예측 연결 방향이 정답에서 벗어난 각도
    배정 모호도      두 연결 슬롯을 정답에 짝짓는 최적·차선 순열의 비용 차가 0.05 미만인 칸의 비율
                   — 학습이 끝난 에폭의 train 값. 순서를 고정한 조건은 짝짓기가 없어
                   **정의상 없다**(—)

## 값의 출처 — 학습 로그

인스턴스 점수와 칸 진단 모두 **각 실행의 최고 에폭**(val inst F1 기준) 값이다
(`stella.eval.runlog.best_epoch`). 예측 캐시·그림이 그 에폭의 체크포인트에서 나오므로 같은 에폭이다.
표가 `metrics.csv` 를 직접 파싱하지 않고 판정 스크립트와 같은 판독기를 쓴다.

**Table III 과 4째 자리가 다를 수 있다** — Table III 은 예측 캐시를 다시 디코딩한 값이고
(GTE 0.4149), 이 표는 학습 중 검증 값이다(0.4148). 7단계(test)에서는 두 표가 같은 채점 경로를
쓰도록 맞춘다 — 학습 로그에는 test 값이 없으므로 그때는 이 표도 예측 캐시를 떠서 잰다.

사용:
    python scripts/tables/table_06.py --split val
"""

import argparse

from table_base import BLANK, DATA_ROOT, UNDEFINED, PaperTable, change, number

from stella.eval.runlog import best_epoch, decode_signature, last_value

LOG_ROOT = DATA_ROOT / "log"
# (실험인자, 조건, 실행 폴더, 배정 모호도가 정의되는가). 첫 행이 대조군이다.
ROWS = (
    ("모든 인자", "최적 조합 (채택)", "260902_113253_base_F06d_cnxlarge", True),
    ("슬롯 배정", "라벨 순서 고정 (prev / next)", "260913_001811_exp_fixed_order_G03_fixed", False),
    ("슬롯 배정", "각도 순서 고정", "260913_233504_exp_canonical_G04_canon", False),
    ("슬롯 배정", "각도 섹터 고정 (GTE식 60°, 슬롯 6)", "260909_130501_exp_gte_G01_gte", False),
    ("토큰 구조", "노드당 토큰 1개", "260915_101641_exp_single_token_G05_single", True),
    ("슬롯 수", "R = 3", "260911_192050_exp_r3_G02_r3", True),
)
KEYS = {
    "inst F1": "val/inst/f1",
    "coverage": "val/inst/coverage",
    "frag": "val/inst/frag",
    "link_ok": "val/cell/link_ok",
    "dir_err (°)": "val/cell/dir_err_deg",
}
AMBIGUITY_KEY = "train/conn/match_ambiguity"
COLUMNS = ("실험인자", "조건", "inst F1", "ΔF1", *list(KEYS)[1:], "배정 모호도", "최고 에폭")
LOGGED_SPLIT = "val"


class RepresentationAblation(PaperTable):
    name = "table_06"
    columns = COLUMNS

    def __init__(self, split: str):
        self.split = split

    def collect(self) -> list[dict]:
        if self.split != LOGGED_SPLIT:
            print(f"[table_06] 학습 로그에는 {self.split} 값이 없다 — 예측 캐시를 떠서 재야 한다")
            return [self._blank_row(factor, condition) for factor, condition, _, _ in ROWS]
        check_decoders([run for *_, run, _ in ROWS])
        values = [best_epoch(LOG_ROOT / run, tuple(KEYS.values())) or {} for *_, run, _ in ROWS]
        base = values[0].get(KEYS["inst F1"])
        rows = [self._row(row, value, base) for row, value in zip(ROWS, values, strict=True)]
        rows[0]["ΔF1"] = BLANK if not values[0] else "기준"
        return rows

    def _row(self, row: tuple, value: dict, base) -> dict:
        factor, condition, run, has_ambiguity = row
        cells = {column: number(value.get(key)) for column, key in KEYS.items()}
        cells["dir_err (°)"] = number(value.get(KEYS["dir_err (°)"]), 2)
        return {
            "실험인자": factor,
            "조건": condition,
            **cells,
            "ΔF1": change(value.get(KEYS["inst F1"]), base),
            "배정 모호도": ambiguity(run) if has_ambiguity else UNDEFINED,
            "최고 에폭": str(value.get("epoch", BLANK)),
        }

    def _blank_row(self, factor: str, condition: str) -> dict:
        return {"실험인자": factor, "조건": condition}


def ambiguity(run: str) -> str:
    value = last_value(LOG_ROOT / run, AMBIGUITY_KEY)
    return BLANK if value is None else f"{value:.5f}"


def check_decoders(runs: list[str]) -> None:
    """실행마다 검증 디코더 설정이 같아야 표현의 차이로 읽힌다 — 다르면 경고만 찍는다."""
    signatures = {run: decode_signature(LOG_ROOT / run) for run in runs}
    if len({str(sorted(sig.items())) for sig in signatures.values()}) > 1:
        print(f"[table_06] ★ 디코더 설정이 실행마다 다르다: {signatures}")


def main() -> None:
    args = parse_args()
    RepresentationAblation(args.split).run()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="val")
    return parser.parse_args()


if __name__ == "__main__":
    main()

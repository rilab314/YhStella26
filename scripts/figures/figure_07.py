"""Fig. 7 — baseline 정성 비교 (VI-A, 2단, 4행 × 5열).

열: 원본 | 원본+GT | Seg+skeleton+DP | Sat2Graph GTE | STELLA.
세 예측 열은 **같은 렌더링 규칙**(클래스 색 폴리라인 + 끝점 흰 원)을 쓴다 — 렌더링을 통일해야
차이가 알고리즘에서 온 것으로 읽힌다.

**후보 순위**: ΔF1 = F1(STELLA) − max(F1(분할), F1(GTE)). 내림차순.
**후보 자격**(설계 §3): F1(STELLA) ≥ 0.5 · 정답 인스턴스 8개 이상. 상한은 묶음마다 다르다
(`figure_base` 의 두 묶음) — 교차로 80개 · 그 밖의 장면 25개. 분할 방식이 선을 가르지 못하는
곳이 교차로라 교차로 묶음을 앞 번호에 둔다.
자격을 통과한 타일만 후보가 되고, **그중 무엇을 실을지는 사람이 고른다**(미팅 결정 09-15).

사용:
    python scripts/figures/figure_07.py --candidates   # 후보 100장
    python scripts/figures/figure_07.py                # selected.txt 의 타일만 조판
"""

import argparse

from figure_base import INTERSECTION, OTHER, SampleFigure, add_arguments

from stella.config_io import load_config

MIN_F1 = 0.5  # 우리 예측이 이보다 나쁜 장면은 "비교 그림"으로 쓸 수 없다
# 묶음별 정답 인스턴스 수. 교차로는 선이 많다(중앙값 56개) — 80 을 넘으면 그림에서 안 보인다
GT_RANGE = {INTERSECTION: (8, 80), OTHER: (8, 25)}
BASELINES = ("seg_dp", "gte_delta")  # GTE 는 변위 회귀까지 갖춘 판 (Table III 와 같은 행)


class BaselineComparison(SampleFigure):
    name = "figure_07"
    split = "val"
    columns = (
        ("원본", ""),
        ("원본+GT", ""),  # 정답 선은 저장된 예측이 아니라 데이터셋에서 읽는다
        ("Seg+skeleton+DP", "seg_dp"),
        ("Sat2Graph GTE", "gte_delta"),
        ("STELLA", "stella"),
    )

    def score(self, stem: str, pool: str):
        ours = self.scores.value("stella", stem, "f1")
        n_gt = self.scores.value("stella", stem, "n_gt")
        low, high = GT_RANGE[pool]
        if ours < MIN_F1 or not low <= n_gt <= high:
            return None
        if not self.in_pool(stem, pool):
            return None
        rival = max(self.scores.value(method, stem, "f1") for method in BASELINES)
        return ours - rival


def main() -> None:
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    figure = BaselineComparison(load_config(args.config, []))
    if args.candidates:
        figure.write_candidates()
        return
    figure.write_selected()


if __name__ == "__main__":
    main()

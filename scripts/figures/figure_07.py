"""Fig. 7 — baseline 정성 비교 (VI-A, 2단, 4행 × 5열).

열: 원본 | 원본+GT | Seg+skeleton+DP | GTE head | STELLA.
세 예측 열은 **같은 렌더링 규칙**(클래스 색 폴리라인 + 끝점 흰 원)을 쓴다 — 렌더링을 통일해야
차이가 알고리즘에서 온 것으로 읽힌다.

**후보 순위**: ΔF1 = F1(STELLA) − max(F1(분할), F1(GTE)). 내림차순.
**후보 자격**(설계 §3): F1(STELLA) ≥ 0.5 · 정답 인스턴스 8~25개.
자격을 통과한 타일만 후보가 되고, **그중 무엇을 실을지는 사람이 고른다**(미팅 결정 09-15).

사용:
    python scripts/figures/figure_07.py --candidates   # 후보 100장
    python scripts/figures/figure_07.py                # selected.txt 의 타일만 조판
"""

import argparse

from figure_base import SampleFigure, add_arguments

from stella.config_io import load_config

MIN_F1 = 0.5  # 우리 예측이 이보다 나쁜 장면은 "비교 그림"으로 쓸 수 없다
MIN_GT, MAX_GT = 8, 25  # 너무 비거나 너무 붐비면 그림에서 아무것도 안 보인다
BASELINES = ("seg_dp", "gte")


class BaselineComparison(SampleFigure):
    name = "figure_07"
    split = "val"
    columns = (
        ("원본", ""),
        ("원본+GT", ""),  # 정답 선은 저장된 예측이 아니라 데이터셋에서 읽는다
        ("Seg+skeleton+DP", "seg_dp"),
        ("GTE head", "gte"),
        ("STELLA", "stella"),
    )

    def score(self, stem: str):
        ours = self.scores.value("stella", stem, "f1")
        n_gt = self.scores.value("stella", stem, "n_gt")
        if ours < MIN_F1 or not MIN_GT <= n_gt <= MAX_GT:
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

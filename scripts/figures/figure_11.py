"""Fig. 11 — 두 체제: 어디서 지고 어디서 이기나 (VII-B, 2단, 3행 × 4열).

열: 원본 | GT | Segmentation | STELLA.
행: ① 짧은 선이 붐비는 구간(seg 우세) ② 그늘·가림 구간(seg 우세) ③ 교차·중첩 구간(STELLA 우세).

**우리가 지는 장면을 먼저 놓는다** — 리뷰어가 가장 먼저 물을 "그래서 seg 보다 무조건 낫다는
거냐"에 그림이 먼저 답한다.

    python scripts/figures/figure_11.py --candidates --side seg    # 분할이 이긴 장면(1·2행용)
    python scripts/figures/figure_11.py --candidates --side ours   # 우리가 이긴 장면(3행용)

**후보 순위**: (분할 F1 − STELLA F1) 또는 그 반대. 두 방법 다 정답이 충분한 타일만 본다.
**`--side ours` 는 교차로 묶음과 그 밖의 묶음을 따로 뽑는다**(`figure_base` 의 두 묶음) — 3행이
교차·중첩 구간이다. 분할이 이긴 쪽(1·2행: 짧은 선 · 그늘)은 교차로와 무관해 한 묶음으로 둔다.

**아직 자동으로 가르지 못하는 것**: "짧은 선이 붐비는 구간"과 "그늘·가림 구간"을 나누려면
타일별 선 길이 분포와 가림 여부가 필요한데 지금 점수표에 없다. 그래서 `--side seg` 후보 안에서
1행·2행에 쓸 장면을 **사람이 눈으로 고른다**(미팅 결정 09-15와 같은 방향이다).
"""

import argparse

from figure_base import ANY, SampleFigure, add_arguments

from stella.config_io import load_config

MIN_GT = 8
SIDES = ("seg", "ours")


class TwoRegimes(SampleFigure):
    name = "figure_11"
    split = "val"
    columns = (
        ("원본", ""),
        ("GT", ""),  # 정답 선은 저장된 예측이 아니라 데이터셋에서 읽는다
        ("Segmentation", "seg_dp"),
        ("STELLA", "stella"),
    )

    def __init__(self, cfg, side: str):
        super().__init__(cfg)
        self.side = side
        self.name = f"figure_11_{side}"
        if side == "seg":
            self.pools = (ANY,)

    def score(self, stem: str, pool: str):
        if self.scores.value("stella", stem, "n_gt") < MIN_GT:
            return None
        ours = self.scores.value("stella", stem, "f1")
        seg = self.scores.value("seg_dp", stem, "f1")
        margin = seg - ours if self.side == "seg" else ours - seg
        if margin <= 0:
            return None
        if not self.in_pool(stem, pool):
            return None
        return margin


def main() -> None:
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    parser.add_argument("--side", default="seg", choices=SIDES, help="어느 쪽이 이긴 장면인가")
    args = parser.parse_args()
    figure = TwoRegimes(load_config(args.config, []), args.side)
    if args.candidates:
        figure.write_candidates()
        return
    figure.write_selected()


if __name__ == "__main__":
    main()

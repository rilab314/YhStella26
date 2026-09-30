"""연결 손실 — **각도 섹터 배정 + 변위 회귀** 판 (Sat2Graph 식 GTE 출력 표현 전체).

`stella.loss.conn_gte`(섹터 배정만)와 **딱 한 가지만 다르다**: 슬롯이 맞혀야 할 목표가
단위 방향이 아니라 **이웃 정점까지의 변위 `(dx, dy)`** 다. 슬롯 배정 규칙·존재 손실·
가중치는 그대로 둔다 — 두 판의 차이가 **"거리까지 배우는가"** 하나로 좁혀지도록 하기
위해서다 (Table III 의 비교 대상 행, 논지 C1-(2)).

단위는 **셀**이다. 목표 변위 = `conn_dirs` x `conn_lengths` 이고, 두 GT 키는 인코더가
`data.conn_store_length` 일 때 함께 낸다. 길이가 대체로 1셀 근처라 **따로 정규화하지
않는다** — 손실 크기가 기본 판의 코사인 손실과 같은 자릿수다.
"""

import torch
import torch.nn.functional as F

from stella.loss.conn_gte import ConnLossGTE

SMOOTH_L1_BETA = 1.0  # 셀 단위 변위 — 1셀 안에서는 제곱, 밖에서는 선형


class ConnLossGTEDelta(ConnLossGTE):
    """섹터 고정 배정은 그대로 쓰고, 방향 대신 변위를 회귀한다."""

    def _branch_vectors(self, targets: dict, supervised, gt_dir: torch.Tensor) -> torch.Tensor:
        """단위 방향에 분기 길이를 곱해 변위로 만든다."""
        if "conn_lengths" not in targets:
            raise KeyError("(dx,dy) 회귀는 conn_lengths 가 필요하다 — data.conn_store_length=True")
        lengths = targets["conn_lengths"][supervised].float()
        return gt_dir * lengths.unsqueeze(-1)

    def _dir_loss(self, pred, target, exist) -> torch.Tensor:
        """찬 섹터에서만 변위를 감독한다 — 빈 섹터의 변위는 뜻이 없다."""
        occupied = exist > 0
        if not bool(occupied.any()):
            return pred.sum() * 0.0
        return F.smooth_l1_loss(pred[occupied], target[occupied], beta=SMOOTH_L1_BETA)

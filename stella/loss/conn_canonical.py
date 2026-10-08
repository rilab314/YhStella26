"""연결 손실 — **각도 순서 고정 배정** 판 (Table VI · 짝 맞춤의 필요성을 재는 깨끗한 대조군).

기본 판(`ConnLoss`)과 **배정 한 단계만** 다르다. 기본 판은 예측과 정답을 **더 잘 맞는 쪽으로
짝지어**(매칭) 손실을 준다. 이 판은 짝을 찾지 않고 **정답 방향의 각도 순서**로 고정 배정한다 —
각도가 작은 분기가 0번 슬롯, 큰 쪽이 1번이다.

**왜 각도인가.** 앞선 판(`conn_fixed`)은 정답에 적힌 순서(= 라벨을 그린 방향)를 썼는데,
그 방향은 **사진에서 보이지 않고 실측상 무작위**다(반대끼리 균형 0.97). 그래서 모델이 맞힐
수가 없어 대조군으로서 힘이 약했다. 각도는 **좌표에서 계산되므로 사진에서 보이는 정보**다 —
모델이 배울 수 있는 기준이고, 그래서 "짝 맞춤이 정말 필요한가"를 공정하게 묻는다.

**라벨을 다시 만들지 않는다** — 기존 정답의 좌표에서 각도를 계산할 뿐이다.

**★ 이 규칙의 한계(각주로 적는다)**: 원 위의 어떤 고정 순서든 **이음매**가 생긴다. 각도 0도
부근에서 두 분기의 순서가 뒤집히므로, 거의 같은 방향의 선인데 배정이 갈리는 칸이 생긴다.
고정 배정 방식이 원리적으로 피할 수 없는 대가다.
"""

import torch

from stella.loss.conn import VALID_NORM_THRESH, ConnLoss
from stella.loss.matching import pad_branches

TWO_PI = 2.0 * torch.pi
ABSENT_ANGLE = 10.0  # 빈 분기는 각도 정렬에서 뒤로 밀어 둔다 (2pi 보다 크면 된다)


class ConnLossCanonical(ConnLoss):
    """슬롯을 **정답 방향의 각도 순서**로 고정 배정한다 — 매칭이 없다."""

    def forward(self, output, targets: dict) -> dict[str, torch.Tensor]:
        supervised = (targets["class_map"] > 0) & output.node_mask
        if not bool(supervised.any()):
            return self._empty(output)
        gt_dir = targets["conn_dirs"][supervised].float()
        valid = gt_dir.norm(dim=-1) > VALID_NORM_THRESH
        gt_dir, valid = pad_branches(gt_dir, valid, self.num_conn_slots)
        assignment = angle_order(gt_dir, valid)
        matched = torch.gather(valid, 1, assignment)
        pred_dir = output.conn_dir[supervised].float()
        losses = self._losses(output, supervised, matched, pred_dir, gt_dir, assignment)
        losses["match_ambiguity"] = pred_dir.sum() * 0.0  # 짝을 찾지 않으므로 흔들릴 것이 없다
        return losses


def angle_order(gt_dir: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """(P, R, 2) 정답 방향 -> (P, R) 각도가 작은 분기부터의 순서.

    빈 분기는 뒤로 민다 — 유효한 분기가 항상 앞 슬롯을 차지한다.
    """
    angle = torch.atan2(gt_dir[..., 1], gt_dir[..., 0]) % TWO_PI
    angle = torch.where(valid, angle, torch.full_like(angle, ABSENT_ANGLE))
    return angle.argsort(dim=1)

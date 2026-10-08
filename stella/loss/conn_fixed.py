"""연결 손실 — **고정 순서 배정** 판 (Table VI · "슬롯 배정 = 고정 순서" 행).

기본 판(`stella.loss.conn.ConnLoss`)을 **그대로 물려받고 배정 한 단계만** 바꾼다.
기본 판은 예측 슬롯과 정답 분기를 **방향이 가장 잘 맞는 쪽으로 짝지어**(매칭) 손실을 준다.
이 판은 짝을 찾지 않고 **정답에 적힌 순서 그대로** 슬롯 k 가 분기 k 를 맡게 한다.

정답의 분기 순서는 인코더가 정한다(`encode._chain_dirs`) — 0번 = 사슬의 **앞(head) 쪽**,
1번 = **뒤(tail) 쪽**이다. 그런데 어느 끝이 head 인지는 **라벨을 그은 방향**으로 정해질 뿐
영상에서는 보이지 않는다. 같은 모양의 선이라도 반대로 그어졌으면 두 슬롯의 뜻이 뒤바뀐다.
**이 판은 그 임의의 순서를 모델에게 맞히라고 요구하는 셈**이고, 그래서 매칭이 필요한가를 잰다.

존재·방향 손실의 형태와 가중치, `dir_loss` 모드는 **전부 기본 판에서 물려받는다** —
비교에서 배정 방식만 달라지도록 하기 위해서다. 기본 판 파일은 고치지 않았다
(학습이 도는 동안 그 파일을 편집하면 도는 학습이 죽을 수 있다 — STATE 운영 함정).
"""

import torch

from stella.loss.conn import VALID_NORM_THRESH, ConnLoss
from stella.loss.matching import pad_branches


class ConnLossFixed(ConnLoss):
    """슬롯 k 가 정답 분기 k 를 맡는다 — 매칭이 없다."""

    def forward(self, output, targets: dict) -> dict[str, torch.Tensor]:
        supervised = (targets["class_map"] > 0) & output.node_mask
        if not bool(supervised.any()):
            return self._empty(output)
        gt_dir = targets["conn_dirs"][supervised].float()
        valid = gt_dir.norm(dim=-1) > VALID_NORM_THRESH
        gt_dir, valid = pad_branches(gt_dir, valid, self.num_conn_slots)
        pred_dir = output.conn_dir[supervised].float()
        assignment = fixed_assignment(gt_dir.shape[0], self.num_conn_slots, gt_dir.device)
        matched = torch.gather(valid, 1, assignment)
        losses = self._losses(output, supervised, matched, pred_dir, gt_dir, assignment)
        losses["match_ambiguity"] = pred_dir.sum() * 0.0  # 짝을 찾지 않으므로 흔들릴 것이 없다
        return losses


def fixed_assignment(points: int, num_slots: int, device) -> torch.Tensor:
    """(P, R) — 모든 칸에서 슬롯 k 가 분기 k 를 맡는다."""
    return torch.arange(num_slots, device=device).expand(points, num_slots).contiguous()

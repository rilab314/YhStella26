"""연결 손실 — 매칭 + 존재·방향 (design 8.3~8.4절, 10차 개정).

GT가 연결 방향 2개를 직접 저장하므로(`conn_dirs`) 여기서는 아무것도 유도하지 않는다 —
저장된 방향 D개와 예측 슬롯 R개를 매칭해 손실을 줄 뿐이다. 기본 설정은 R = D = 2라
모든 순열이 모든 슬롯을 쓰고 무매칭 슬롯이 없다. 거짓 양성 셀은 전 슬롯 존재 0으로만
감독한다.
"""

import torch
import torch.nn.functional as F

from stella.loss.criterion import LossModule
from stella.loss.matching import assign_slots, pad_branches

VALID_NORM_THRESH = 0.5  # conn_dirs는 단위벡터, 빈 분기는 0 — 노름으로 유효를 가른다
DIR_LOSSES = ("cosine", "angle")
ANGLE_SCALE = 1.0 / torch.pi  # acos 를 [0, 1]로 — cosine 손실과 크기를 맞춘다


class ConnLoss(LossModule):
    @classmethod
    def from_cfg(cls, module_cfg, cfg, **kwargs) -> "ConnLoss":
        """슬롯 수는 `cfg.model`에 있다 — LossConfig에 중복 저장하지 않는다(8.0절)."""
        return cls(
            num_conn_slots=cfg.model.num_conn_slots,
            w_exist=module_cfg.w_exist,
            w_dir=module_cfg.w_dir,
            match_w_dir=module_cfg.match_w_dir,
            match_w_exist=module_cfg.match_w_exist,
            exist_pos_weight=module_cfg.exist_pos_weight,
            dir_loss=module_cfg.dir_loss,
        )

    def __init__(
        self,
        *,
        num_conn_slots: int,
        w_exist: float,
        w_dir: float,
        match_w_dir: float,
        match_w_exist: float,
        exist_pos_weight: float,
        dir_loss: str,
    ):
        super().__init__()
        if dir_loss not in DIR_LOSSES:
            raise ValueError(f"dir_loss 는 {DIR_LOSSES} 중 하나여야 한다: {dir_loss}")
        self.num_conn_slots = num_conn_slots
        self.w_exist = w_exist
        self.w_dir = w_dir
        self.match_w_dir = match_w_dir
        self.match_w_exist = match_w_exist
        self.exist_pos_weight = exist_pos_weight
        self.dir_loss = dir_loss

    def forward(self, output, targets: dict) -> dict[str, torch.Tensor]:
        supervised = (targets["class_map"] > 0) & output.node_mask
        if not bool(supervised.any()):
            return self._empty(output)
        gt_dir = targets["conn_dirs"][supervised].float()  # (P, D, 2)
        valid = gt_dir.norm(dim=-1) > VALID_NORM_THRESH  # 항상 2개지만 R > D ablation 대비
        gt_dir, valid = pad_branches(gt_dir, valid, self.num_conn_slots)
        pred_dir = output.conn_dir[supervised].float()
        pred_exist = output.exist_logit[supervised].float()
        assignment, matched, ambiguity = assign_slots(
            pred_dir, pred_exist, gt_dir, valid, self.match_w_dir, self.match_w_exist
        )
        losses = self._losses(output, supervised, matched, pred_dir, gt_dir, assignment)
        losses["match_ambiguity"] = ambiguity
        return losses

    def _losses(self, output, supervised, matched, pred_dir, gt_dir, assignment) -> dict:
        exist = self._exist_loss(output, supervised, matched)
        direction = self._direction_loss(pred_dir, gt_dir, assignment, matched)
        return {
            "exist": exist,
            "dir": direction,
            "total": self.w_exist * exist + self.w_dir * direction,
        }

    def _exist_loss(self, output, supervised, matched) -> torch.Tensor:
        """선택된 전 셀에서 계산한다 — 거짓 양성 셀은 전 슬롯 0으로 감독한다(8.4절)."""
        target = torch.zeros_like(output.exist_logit)
        target[supervised] = matched.float()
        node_mask = output.node_mask
        logit = output.exist_logit[node_mask].float()
        weight = logit.new_tensor(self.exist_pos_weight)
        return F.binary_cross_entropy_with_logits(logit, target[node_mask], pos_weight=weight)

    def _direction_loss(self, pred_dir, gt_dir, assignment, matched) -> torch.Tensor:
        """매칭된 쌍에만 방향 손실. 크기·좌표 감독 없이 방향 차이만 학습한다.

        `cosine`은 오차 0 근처에서 기울기가 죽는다 — `angle`은 그 구간을 살린다 (백로그 B4).
        """
        if not bool(matched.any()):
            return pred_dir.sum() * 0.0
        index = assignment.unsqueeze(-1).expand(-1, -1, 2)
        assigned_dir = torch.gather(gt_dir, 1, index)
        alignment = (pred_dir * assigned_dir).sum(dim=-1)[matched]
        if self.dir_loss == "angle":
            return torch.arccos(alignment.clamp(-1.0 + 1e-6, 1.0 - 1e-6)).mean() * ANGLE_SCALE
        return (1.0 - alignment).mean()

    def _empty(self, output) -> dict:
        zero = output.exist_logit.sum() * 0.0
        return {"exist": zero, "dir": zero, "match_ambiguity": zero, "total": zero}

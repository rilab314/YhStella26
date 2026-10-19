"""연결 손실 — **각도 섹터 배정** 판 (3-C 비교 대상, Sat2Graph 식 GTE 표현).

기본 판(`stella.loss.conn`)과 **딱 한 가지만 다르다**: 어느 슬롯이 어느 GT 분기를 맡는지를
**매칭으로 정하지 않고 각도로 고정 배정**한다. 방향을 R개 섹터로 나누고, 분기의 각도가
들어가는 섹터의 슬롯이 그 분기를 맡는다. 그래서 슬롯 번호에 **고정된 뜻**이 생긴다
(0번 = 동쪽, 1번 = 북동쪽, ...).

손실의 형태(존재 = BCE · 방향 = 코사인)와 가중치는 기본 판과 같게 두었다 — **비교에서
표현만 달라지도록** 하기 위해서다.

**이 표현의 약점을 함께 잰다**: 한 셀의 두 분기가 같은 섹터에 들어가면 하나는 버려진다.
`sector_collision` 으로 그 비율을 기록한다 — 섹터 수가 적을수록 커진다.
"""

import torch
import torch.nn.functional as F

from stella.loss.criterion import LossModule

VALID_NORM_THRESH = 0.5  # conn_dirs 는 단위벡터, 빈 분기는 0 — 노름으로 유효를 가른다
TWO_PI = 2.0 * torch.pi


class ConnLossGTE(LossModule):
    """슬롯을 각도 섹터로 고정 배정한다 — 매칭이 없다."""

    @classmethod
    def from_cfg(cls, module_cfg, cfg, **kwargs) -> "ConnLossGTE":
        return cls(
            num_conn_slots=cfg.model.num_conn_slots,
            w_exist=module_cfg.w_exist,
            w_dir=module_cfg.w_dir,
            exist_pos_weight=module_cfg.exist_pos_weight,
        )

    def __init__(
        self, *, num_conn_slots: int, w_exist: float, w_dir: float, exist_pos_weight: float
    ):
        super().__init__()
        self.num_conn_slots = num_conn_slots
        self.w_exist = w_exist
        self.w_dir = w_dir
        self.exist_pos_weight = exist_pos_weight

    def forward(self, output, targets: dict) -> dict[str, torch.Tensor]:
        supervised = (targets["class_map"] > 0) & output.node_mask
        if not bool(supervised.any()):
            return self._empty(output)
        gt_dir = targets["conn_dirs"][supervised].float()
        gt_vec = self._branch_vectors(targets, supervised, gt_dir)
        target_dir, target_exist, collision = self._assign_by_sector(gt_dir, gt_vec)
        pred_exist = output.exist_logit[supervised].float()
        pred_dir = output.conn_dir[supervised].float()
        exist = self._exist_loss(pred_exist, target_exist, output, supervised)
        direction = self._dir_loss(pred_dir, target_dir, target_exist)
        total = self.w_exist * exist + self.w_dir * direction
        return {"exist": exist, "dir": direction, "sector_collision": collision, "total": total}

    def _branch_vectors(self, targets: dict, supervised, gt_dir: torch.Tensor) -> torch.Tensor:
        """섹터 칸에 실어 둘 목표 벡터 — 기본 판은 단위 방향 그대로다.

        `(dx, dy)` 회귀 판(`conn_gte_delta`)이 여기만 바꿔 길이를 곱한다.
        """
        return gt_dir

    def _assign_by_sector(self, gt_dir: torch.Tensor, gt_vec: torch.Tensor) -> tuple:
        """분기 (P, D, 2) -> 섹터별 목표 벡터 (P, R, 2) · 존재 (P, R) · 충돌 비율.

        **섹터·유효 판정은 언제나 단위 방향(`gt_dir`)으로 한다** — 목표 벡터의 크기가
        달라져도 어느 슬롯이 어느 분기를 맡는지는 똑같아야 한다.
        """
        valid = gt_dir.norm(dim=-1) > VALID_NORM_THRESH  # (P, D)
        angle = torch.atan2(gt_dir[..., 1], gt_dir[..., 0]) % TWO_PI
        sector = (angle / (TWO_PI / self.num_conn_slots)).long().clamp(0, self.num_conn_slots - 1)
        points, branches = gt_dir.shape[0], gt_dir.shape[1]
        target_dir = gt_dir.new_zeros((points, self.num_conn_slots, 2))
        target_exist = gt_dir.new_zeros((points, self.num_conn_slots))
        rows = torch.arange(points, device=gt_dir.device)
        for branch in range(branches):  # 분기 수는 2뿐이라 순회가 싸다
            keep = valid[:, branch]
            target_dir[rows[keep], sector[keep, branch]] = gt_vec[keep, branch]
            target_exist[rows[keep], sector[keep, branch]] = 1.0
        filled = target_exist.sum(dim=1)
        collision = (valid.sum(dim=1).float() - filled).clamp(min=0).sum() / valid.sum().clamp(
            min=1
        )
        return target_dir, target_exist, collision

    def _exist_loss(self, pred, target, output, supervised) -> torch.Tensor:
        """양성 셀은 섹터별 목표로, 배경 셀은 전 슬롯 0 으로 감독한다 (기본 판과 같다)."""
        weight = torch.full_like(target, 1.0)
        weight[target > 0] = self.exist_pos_weight
        positive = F.binary_cross_entropy_with_logits(pred, target, weight=weight)
        background = output.node_mask & ~supervised
        if not bool(background.any()):
            return positive
        negative_pred = output.exist_logit[background].float()
        negative = F.binary_cross_entropy_with_logits(
            negative_pred, torch.zeros_like(negative_pred)
        )
        return positive + negative

    def _dir_loss(self, pred, target, exist) -> torch.Tensor:
        """찬 섹터에서만 방향을 감독한다 — 빈 섹터의 방향은 뜻이 없다."""
        occupied = exist > 0
        if not bool(occupied.any()):
            return pred.sum() * 0.0
        cosine = (pred[occupied] * target[occupied]).sum(dim=-1)
        return (1.0 - cosine).mean()

    def _empty(self, output) -> dict[str, torch.Tensor]:
        zero = output.exist_logit.sum() * 0.0
        return {"exist": zero, "dir": zero, "sector_collision": zero, "total": zero}

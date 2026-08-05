"""연결 손실 — 매칭 + 존재·방향·종점 (impl_plan 8.3~8.4절).

GT는 이웃 셀 좌표만 준다. 방향과 종점 라벨은 여기서 6.2절 식으로 유도한다:

    d_gt(a->b) = normalize(p_full(b) - o(a))
    t_gt(a->b) = 1[end(b) = 1 or Y(b) != Y(a)]
"""

import torch
import torch.nn.functional as F

from stella.loss.criterion import LossModule
from stella.loss.matching import assign_slots, pad_branches

NORMALIZE_EPS = 1e-6


class ConnLoss(LossModule):
    @classmethod
    def from_cfg(cls, module_cfg, cfg, **kwargs) -> "ConnLoss":
        """슬롯 수는 `cfg.model`에 있다 — LossConfig에 중복 저장하지 않는다(8.0절)."""
        return cls(
            num_conn_slots=cfg.model.num_conn_slots,
            w_exist=module_cfg.w_exist,
            w_dir=module_cfg.w_dir,
            w_t=module_cfg.w_t,
            match_w_dir=module_cfg.match_w_dir,
            match_w_exist=module_cfg.match_w_exist,
        )

    def __init__(
        self,
        *,
        num_conn_slots: int,
        w_exist: float,
        w_dir: float,
        w_t: float,
        match_w_dir: float,
        match_w_exist: float,
    ):
        super().__init__()
        self.num_conn_slots = num_conn_slots
        self.w_exist = w_exist
        self.w_dir = w_dir
        self.w_t = w_t
        self.match_w_dir = match_w_dir
        self.match_w_exist = match_w_exist

    def forward(self, output, targets: dict) -> dict[str, torch.Tensor]:
        supervised = (targets["class_map"] > 0) & output.node_mask
        cells = supervised.nonzero(as_tuple=False)
        if cells.shape[0] == 0:
            return self._empty(output)
        branches = self._gather_branches(output, targets, supervised, cells)
        assignment, matched, ambiguity = assign_slots(
            branches["pred_dir"],
            branches["pred_exist"],
            branches["gt_dir"],
            branches["valid"],
            self.match_w_dir,
            self.match_w_exist,
        )
        losses = self._losses(output, supervised, branches, assignment, matched)
        losses["switch_rate"] = ambiguity
        return losses

    def _gather_branches(self, output, targets: dict, supervised, cells) -> dict:
        gt_dir, gt_t, valid = derive_branches(targets, cells)
        gt_dir, gt_t, valid = pad_branches(gt_dir, gt_t, valid, self.num_conn_slots)
        return {
            "gt_dir": gt_dir,
            "gt_t": gt_t,
            "valid": valid,
            "pred_dir": output.conn_dir[supervised].float(),
            "pred_exist": output.exist_logit[supervised].float(),
            "pred_t": output.t_logit[supervised].float(),
        }

    def _losses(self, output, supervised, branches, assignment, matched) -> dict:
        exist = self._exist_loss(output, supervised, matched)
        direction, endness = self._matched_losses(branches, assignment, matched)
        total = self.w_exist * exist + self.w_dir * direction + self.w_t * endness
        return {"exist": exist, "dir": direction, "t": endness, "total": total}

    @staticmethod
    def _exist_loss(output, supervised, matched) -> torch.Tensor:
        """S 전체에서 계산한다 — 거짓 양성 셀은 전 슬롯 0으로 감독한다(8.4절)."""
        target = torch.zeros_like(output.exist_logit)
        target[supervised] = matched.float()
        node_mask = output.node_mask
        return F.binary_cross_entropy_with_logits(
            output.exist_logit[node_mask].float(), target[node_mask]
        )

    def _matched_losses(self, branches, assignment, matched) -> tuple[torch.Tensor, torch.Tensor]:
        pred_dir, pred_t = branches["pred_dir"], branches["pred_t"]
        if not bool(matched.any()):
            return pred_dir.sum() * 0.0, pred_t.sum() * 0.0
        index = assignment.unsqueeze(-1).expand(-1, -1, 2)
        assigned_dir = torch.gather(branches["gt_dir"], 1, index)
        assigned_t = torch.gather(branches["gt_t"], 1, assignment)
        alignment = (pred_dir * assigned_dir).sum(dim=-1)
        direction = (1.0 - alignment)[matched].mean()
        endness = F.binary_cross_entropy_with_logits(pred_t[matched], assigned_t[matched])
        return direction, endness

    def _empty(self, output) -> dict:
        zero = output.exist_logit.sum() * 0.0
        return {"exist": zero, "dir": zero, "t": zero, "switch_rate": zero, "total": zero}


def derive_branches(
    targets: dict, cells: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """cells (P,3) = (b, i, j) 에 대해 GT 방향·종점 라벨·유효 마스크를 유도한다 (6.2절)."""
    conn = targets["conn_cells"][cells[:, 0], cells[:, 1], cells[:, 2]]  # (P, D, 2)
    valid = conn[..., 0] >= 0
    rows, cols = conn[..., 0].clamp(min=0), conn[..., 1].clamp(min=0)
    batch = cells[:, 0:1].expand_as(rows)
    neighbor_coord = targets["coord_map"][batch, rows, cols].float()
    target_point = torch.stack(
        [cols + neighbor_coord[..., 0], rows + neighbor_coord[..., 1]], dim=-1
    )
    origin = torch.stack([cells[:, 2] + 0.5, cells[:, 1] + 0.5], dim=-1).unsqueeze(1)
    gt_dir = F.normalize(target_point - origin, dim=-1, eps=NORMALIZE_EPS)
    own_class = targets["class_map"][cells[:, 0], cells[:, 1], cells[:, 2]].unsqueeze(1)
    is_end = targets["end_map"][batch, rows, cols] > 0
    gt_t = (is_end | (targets["class_map"][batch, rows, cols] != own_class)).float()
    return gt_dir * valid.unsqueeze(-1), gt_t * valid, valid

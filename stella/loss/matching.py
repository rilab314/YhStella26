"""연결 슬롯 매칭 — 셀별 슬롯 배정 (impl_plan 8.3절).

셀의 R개 예측 슬롯 중 어느 슬롯이 어느 GT 분기를 맡는지 정한다.
셀마다 독립인 크기 R x D의 작은 배정 문제라 **순열 완전탐색을 벡터화**한다 (R <= 4 전제).
"""

import itertools
from functools import lru_cache

import torch

AMBIGUITY_MARGIN = 0.05  # 최적과 차선 순열의 총비용 차가 이보다 작으면 배정이 흔들린다


@lru_cache(maxsize=8)
def _permutations(num_slots: int, device: str) -> torch.Tensor:
    perms = list(itertools.permutations(range(num_slots)))
    return torch.tensor(perms, dtype=torch.long, device=device)


def assign_slots(
    pred_dir: torch.Tensor,
    pred_exist: torch.Tensor,
    gt_dir: torch.Tensor,
    valid: torch.Tensor,
    match_w_dir: float,
    match_w_exist: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """pred_dir (P,R,2), pred_exist (P,R), gt_dir (P,D,2), valid (P,D).

    Returns: assignment (P,R) 슬롯 k가 맡은 분기 인덱스, matched (P,R) bool, ambiguity (스칼라).
    """
    cost = _slot_cost(pred_dir, pred_exist, gt_dir, valid, match_w_dir, match_w_exist)
    perms = _permutations(cost.shape[1], str(cost.device))
    totals = _permutation_totals(cost, perms)
    order = totals.argsort(dim=1)
    assignment = perms[order[:, 0]]
    matched = torch.gather(valid, 1, assignment)
    return assignment, matched, _ambiguity(totals, order)


def _slot_cost(
    pred_dir: torch.Tensor,
    pred_exist: torch.Tensor,
    gt_dir: torch.Tensor,
    valid: torch.Tensor,
    match_w_dir: float,
    match_w_exist: float,
) -> torch.Tensor:
    """C(k, m) = lam_dir (1 - d_k . d_gt_m) - lam_e sigmoid(e_k). 무효 분기는 상수 0."""
    alignment = torch.einsum("prc,pdc->prd", pred_dir.float(), gt_dir.float())
    cost = (
        match_w_dir * (1.0 - alignment) - match_w_exist * pred_exist.float().sigmoid()[:, :, None]
    )
    return torch.where(valid[:, None, :], cost, torch.zeros_like(cost))


def _permutation_totals(cost: torch.Tensor, perms: torch.Tensor) -> torch.Tensor:
    """순열별 총비용 (P, R!)."""
    cells, slots = cost.shape[0], cost.shape[1]
    index = perms.view(1, -1, slots, 1).expand(cells, -1, slots, 1)
    picked = torch.gather(cost.unsqueeze(1).expand(-1, perms.shape[0], -1, -1), 3, index)
    return picked.squeeze(-1).sum(dim=2)


def _ambiguity(totals: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
    """최적과 차선의 총비용 차가 좁은 셀의 비율 — 매칭 불안정성 감시 지표(손실 아님)."""
    if totals.shape[1] < 2 or totals.shape[0] == 0:
        return totals.new_zeros(())
    best = torch.gather(totals, 1, order[:, :1]).squeeze(1)
    second = torch.gather(totals, 1, order[:, 1:2]).squeeze(1)
    return ((second - best) < AMBIGUITY_MARGIN).float().mean()


def pad_branches(
    gt_dir: torch.Tensor, valid: torch.Tensor, num_slots: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """D != R 이면 분기 축을 R로 맞춘다 — R > D ablation에서만 무효 분기가 생긴다 (8.3절)."""
    branches = gt_dir.shape[1]
    if branches == num_slots:
        return gt_dir, valid
    if branches > num_slots:
        return gt_dir[:, :num_slots], valid[:, :num_slots]
    pad = num_slots - branches
    cells = gt_dir.shape[0]
    return (
        torch.cat([gt_dir, gt_dir.new_zeros((cells, pad, 2))], dim=1),
        torch.cat([valid, valid.new_zeros((cells, pad))], dim=1),
    )

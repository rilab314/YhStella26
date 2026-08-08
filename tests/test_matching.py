"""순열 매칭을 scipy의 선형 배정(LSA)과 대조 검증 (design 8.3절)."""

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from stella.loss.matching import _slot_cost, assign_slots

NUM_SLOTS = 3


def random_case(seed: int, valid_count: int):
    generator = torch.Generator().manual_seed(seed)
    pred_dir = torch.nn.functional.normalize(
        torch.randn(1, NUM_SLOTS, 2, generator=generator), dim=-1
    )
    pred_exist = torch.randn(1, NUM_SLOTS, generator=generator)
    gt_dir = torch.nn.functional.normalize(
        torch.randn(1, NUM_SLOTS, 2, generator=generator), dim=-1
    )
    valid = torch.zeros(1, NUM_SLOTS, dtype=torch.bool)
    valid[0, :valid_count] = True
    return pred_dir, pred_exist, gt_dir, valid


def test_permutation_search_matches_scipy_lsa():
    for seed in range(40):
        for valid_count in (0, 1, 2, 3):
            pred_dir, pred_exist, gt_dir, valid = random_case(seed, valid_count)
            assignment, _, _ = assign_slots(pred_dir, pred_exist, gt_dir, valid, 1.0, 1.0)
            cost = _slot_cost(pred_dir, pred_exist, gt_dir, valid, 1.0, 1.0)[0].numpy()
            rows, cols = linear_sum_assignment(cost)
            ours = cost[np.arange(NUM_SLOTS), assignment[0].numpy()].sum()
            assert (
                ours == float(np.round(cost[rows, cols].sum(), 6))
                or abs(ours - cost[rows, cols].sum()) < 1e-5
            )


def test_matched_mask_follows_valid_branches():
    pred_dir, pred_exist, gt_dir, valid = random_case(7, valid_count=2)
    assignment, matched, _ = assign_slots(pred_dir, pred_exist, gt_dir, valid, 1.0, 1.0)
    assert int(matched.sum()) == 2
    assert set(assignment[0, matched[0]].tolist()) == {0, 1}


def test_no_valid_branch_gives_no_match():
    pred_dir, pred_exist, gt_dir, valid = random_case(3, valid_count=0)
    _, matched, _ = assign_slots(pred_dir, pred_exist, gt_dir, valid, 1.0, 1.0)
    assert int(matched.sum()) == 0


def test_confident_slot_takes_the_single_branch():
    """존재 확률 항의 역할: M < R 일 때 어느 슬롯이 매칭에 뽑힐지 정한다 (8.3절 2번)."""
    pred_dir = torch.zeros(1, NUM_SLOTS, 2)
    pred_dir[0, :, 0] = 1.0  # 세 슬롯의 방향이 동일 -> 방향 항으로는 구분 불가
    pred_exist = torch.tensor([[-5.0, 5.0, -5.0]])
    gt_dir = torch.zeros(1, NUM_SLOTS, 2)
    gt_dir[0, 0, 0] = 1.0
    valid = torch.tensor([[True, False, False]])
    assignment, matched, _ = assign_slots(pred_dir, pred_exist, gt_dir, valid, 1.0, 1.0)
    assert int(assignment[0, 1]) == 0 and bool(matched[0, 1])

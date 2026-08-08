"""노드 선택 (impl_plan 7.4절, 가설 백로그).

`thresh` 모드는 히트맵의 **절대 보정**에 의존한다 — 확률이 통째로 조금만 움직여도 선택 수가
수만 개씩 뒤집힌다(REF-F 실측). `topk`는 그 의존을 끊는다. 그 차이를 직접 시험한다.
"""

import torch

from stella.model.heatmap import NodeSelector

SIDE = 64
TOPK = 100


def make_selector(mode: str) -> NodeSelector:
    return NodeSelector(
        node_sampling="gt+pred",
        heatmap_thresh=0.3,
        dilate=0,
        n_max=100000,
        select_mode=mode,
        n_topk=TOPK,
    )


def ramp_logit(shift: float) -> torch.Tensor:
    """전 셀에 걸쳐 완만히 증가하는 로짓 + 전역 이동 — 보정이 흔들리는 상황의 모형."""
    base = torch.linspace(-3.0, 1.0, SIDE * SIDE).view(1, SIDE, SIDE)
    return base + shift


def test_topk_selects_a_fixed_count_regardless_of_calibration():
    selector = make_selector("topk")
    for shift in (-5.0, 0.0, 5.0):
        cells = selector(ramp_logit(shift), None, training=False)
        assert cells[0].shape[0] == TOPK, f"shift {shift}"


def test_threshold_count_swings_with_calibration():
    """대조 — 같은 순위인데도 선택 수가 0에서 전체까지 흔들린다."""
    selector = make_selector("thresh")
    counts = [selector(ramp_logit(s), None, training=False)[0].shape[0] for s in (-5.0, 0.0, 5.0)]
    assert counts[0] < counts[1] < counts[2]
    assert counts[0] <= 1 and counts[2] > SIDE * SIDE * 0.9


def test_topk_keeps_the_highest_scoring_cells():
    selector = make_selector("topk")
    cells = selector(ramp_logit(0.0), None, training=False)[0]
    flat = cells[:, 0] * SIDE + cells[:, 1]
    assert int(flat.min()) == SIDE * SIDE - TOPK  # ramp는 flat 인덱스 순으로 커진다


def test_training_still_unions_gt_cells():
    selector = make_selector("topk")
    gt = torch.zeros((1, SIDE, SIDE), dtype=torch.bool)
    gt[0, 0, 0] = True  # 점수가 가장 낮은 셀 — topk에는 절대 안 들어온다
    cells = selector(ramp_logit(0.0), gt, training=True)[0]
    assert ((cells[:, 0] == 0) & (cells[:, 1] == 0)).any()
    assert cells.shape[0] == TOPK + 1


def test_unknown_mode_is_rejected():
    import pytest

    with pytest.raises(ValueError):
        make_selector("nonsense")

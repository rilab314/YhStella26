"""각도 순서 고정 배정 검증 (Table VI · 짝 맞춤 필요성 대조군).

핵심 성질: **같은 모양의 선이면 라벨을 어느 방향으로 그었든 배정이 같아야 한다.**
그래야 모델이 사진만 보고 배울 수 있는 기준이 된다 — 앞선 판(라벨 순서)이 못 가졌던 성질이다.
"""

import numpy as np
import pytest
import torch

from stella.loss.conn_canonical import ConnLossCanonical, angle_order
from stella.loss.conn_fixed import ConnLossFixed
from stella.model.stella import ModelOutput

SIDE = 2
SLOTS = 2
NUM_CLASSES = 12


def unit(degrees: float) -> list[float]:
    radians = np.deg2rad(degrees)
    return [float(np.cos(radians)), float(np.sin(radians))]


def make(cls):
    return cls(
        num_conn_slots=SLOTS,
        w_exist=1.0,
        w_dir=1.0,
        match_w_dir=1.0,
        match_w_exist=1.0,
        exist_pos_weight=1.0,
        dir_loss="cosine",
    )


def scene(gt_branches: list[list[float]], pred_slots: list[list[float]]):
    class_map = torch.zeros((1, SIDE, SIDE), dtype=torch.long)
    class_map[0, 0, 0] = 3
    conn = torch.zeros((1, SIDE, SIDE, 2, 2))
    conn[0, 0, 0] = torch.tensor(gt_branches)
    node = torch.zeros((1, SIDE, SIDE), dtype=torch.bool)
    node[0, 0, 0] = True
    pred = torch.zeros((1, SIDE, SIDE, SLOTS, 2))
    pred[0, 0, 0] = torch.tensor(pred_slots)
    output = ModelOutput(
        heatmap_logit=torch.zeros((1, SIDE, SIDE)),
        node_mask=node,
        class_logit=torch.zeros((1, SIDE, SIDE, NUM_CLASSES)),
        self_coord=torch.zeros((1, SIDE, SIDE, 2)),
        end_logit=torch.zeros((1, SIDE, SIDE)),
        fg_logit=torch.zeros((1, SIDE, SIDE)),
        exist_logit=torch.full((1, SIDE, SIDE, SLOTS), 5.0),
        conn_dir=pred,
    )
    return output, {"class_map": class_map, "conn_dirs": conn}


def test_order_is_by_angle():
    """각도가 작은 분기가 0번 슬롯을 맡는다."""
    gt = torch.tensor([[unit(200), unit(20)]], dtype=torch.float32)
    valid = torch.ones((1, 2), dtype=torch.bool)
    assert angle_order(gt, valid).tolist() == [[1, 0]]  # 20도가 앞, 200도가 뒤


def test_absent_branch_goes_last():
    """빈 분기는 뒤로 밀려 유효한 분기가 앞 슬롯을 차지한다."""
    gt = torch.tensor([[[0.0, 0.0], unit(300)]], dtype=torch.float32)
    valid = torch.tensor([[False, True]])
    assert angle_order(gt, valid).tolist() == [[1, 0]]


@pytest.mark.parametrize("drawn", [("forward"), ("reverse")])
def test_same_line_gives_same_loss_whichever_way_it_was_drawn(drawn: str):
    """★ 핵심 — 라벨을 반대로 그어도 손실이 같아야 한다.

    라벨 순서 판(`ConnLossFixed`)은 이 성질이 없어서 모델이 맞힐 수 없었다.
    """
    branches = [unit(20), unit(200)]
    if drawn == "reverse":
        branches = branches[::-1]  # 같은 선을 반대 방향으로 그은 라벨
    prediction = [unit(20), unit(200)]
    output, targets = scene(branches, prediction)
    loss = make(ConnLossCanonical)(output, targets)
    assert float(loss["dir"]) < 1e-5  # 어느 쪽으로 그었든 예측이 정답과 맞는다


def test_label_order_version_is_not_invariant():
    """대조 — 라벨 순서 판은 그은 방향이 바뀌면 같은 예측이 틀린 답이 된다."""
    prediction = [unit(20), unit(200)]
    same = make(ConnLossFixed)(*scene([unit(20), unit(200)], prediction))
    flipped = make(ConnLossFixed)(*scene([unit(200), unit(20)], prediction))
    assert float(same["dir"]) < 1e-5
    assert float(flipped["dir"]) > 1.9  # 같은 선인데 정반대로 채점된다


def test_no_ambiguity_is_reported():
    """짝을 찾지 않으므로 배정이 흔들릴 일이 없다."""
    output, targets = scene([unit(20), unit(200)], [unit(20), unit(200)])
    assert float(make(ConnLossCanonical)(output, targets)["match_ambiguity"]) == 0.0

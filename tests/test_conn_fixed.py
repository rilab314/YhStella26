"""고정 순서 배정 연결 손실 검증 (Table VI).

기본 판과 **배정 한 단계만** 달라야 한다. 그래서 두 가지를 본다:
예측이 정답 순서와 **같으면 두 판의 손실이 똑같고**, 예측이 **순서만 뒤바뀌면**
기본 판(매칭)은 벌하지 않지만 이 판은 벌한다 — 그 차이가 이 ablation 이 재는 것이다.
"""

import torch

from stella.loss.conn import ConnLoss
from stella.loss.conn_fixed import ConnLossFixed, fixed_assignment
from stella.model.stella import ModelOutput

SIDE = 2
SLOTS = 2
NUM_CLASSES = 12
BACKWARD = [-1.0, 0.0]  # 0번 분기 — 사슬의 앞(head) 쪽
FORWARD = [1.0, 0.0]  # 1번 분기 — 뒤(tail) 쪽


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


def scene(pred_slots: list[list[float]]) -> tuple[ModelOutput, dict]:
    """한 칸만 양성. 정답은 [뒤, 앞] 순서, 예측은 인자로 준 순서."""
    class_map = torch.zeros((1, SIDE, SIDE), dtype=torch.long)
    class_map[0, 0, 0] = 3
    conn = torch.zeros((1, SIDE, SIDE, 2, 2))
    conn[0, 0, 0] = torch.tensor([BACKWARD, FORWARD])
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


def test_assignment_is_identity():
    """모든 칸에서 슬롯 k 가 분기 k 를 맡는다."""
    assignment = fixed_assignment(3, SLOTS, "cpu")
    assert assignment.tolist() == [[0, 1], [0, 1], [0, 1]]


def test_same_as_matching_when_order_is_right():
    """예측이 정답 순서와 같으면 두 판이 **완전히 같은** 손실을 낸다 — 나머지가 같다는 증거."""
    output, targets = scene([BACKWARD, FORWARD])
    fixed = make(ConnLossFixed)(output, targets)
    matched = make(ConnLoss)(output, targets)
    for key in ("exist", "dir", "total"):
        assert torch.allclose(fixed[key], matched[key]), key


def test_swapped_order_is_punished_only_without_matching():
    """순서만 뒤바뀐 예측 — 매칭은 짝을 바꿔 맞춰 주지만 고정 판은 그대로 벌한다."""
    output, targets = scene([FORWARD, BACKWARD])
    fixed = make(ConnLossFixed)(output, targets)
    matched = make(ConnLoss)(output, targets)
    assert float(matched["dir"]) < 1e-6  # 매칭은 뒤바뀐 순서를 알아서 맞춘다
    assert float(fixed["dir"]) > 1.9  # 고정 판은 정반대 방향이라 코사인 손실이 최대(2)에 가깝다


def test_no_ambiguity_is_reported():
    """짝을 찾지 않으므로 배정이 흔들릴 일이 없다."""
    output, targets = scene([FORWARD, BACKWARD])
    assert float(make(ConnLossFixed)(output, targets)["match_ambiguity"]) == 0.0

"""부분 주입 검증 (Table IV — 천장 분해).

정답을 **묶음 단위로만** 얹었는지 본다. 얹은 묶음은 정답과 같아야 하고,
**안 얹은 묶음은 예측 그대로여야 한다** — 그래야 "이 헤드만 완벽하면 얼마"가 성립한다.
"""

import pytest
import torch

from stella.model.inject import INJECTABLE_PARTS, gt_model_output, inject_parts
from stella.model.stella import ModelOutput

SIDE = 4
NUM_CLASSES = 12
NUM_SLOTS = 2


def targets() -> dict:
    class_map = torch.zeros((1, SIDE, SIDE), dtype=torch.long)
    class_map[0, 1, 1] = 3
    class_map[0, 1, 2] = 5
    conn = torch.zeros((1, SIDE, SIDE, 2, 2))
    conn[0, 1, 1] = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
    conn[0, 1, 2] = torch.tensor([[0.0, 1.0], [0.0, -1.0]])
    coord = torch.full((1, SIDE, SIDE, 2), 0.25)
    end = torch.zeros((1, SIDE, SIDE))
    end[0, 1, 1] = 1.0
    return {"class_map": class_map, "conn_dirs": conn, "coord_map": coord, "end_map": end}


def prediction() -> ModelOutput:
    """정답 칸에서 계산된 '엉뚱한 예측' — 주입 여부가 눈에 띄게 다른 값을 넣는다."""
    ones = torch.ones((1, SIDE, SIDE))
    return ModelOutput(
        heatmap_logit=ones * 7.0,
        node_mask=torch.ones((1, SIDE, SIDE), dtype=torch.bool),
        class_logit=torch.full((1, SIDE, SIDE, NUM_CLASSES), 0.5),
        self_coord=torch.full((1, SIDE, SIDE, 2), 0.9),
        end_logit=ones * 3.0,
        fg_logit=ones * 3.0,
        exist_logit=torch.full((1, SIDE, SIDE, NUM_SLOTS), 2.0),
        conn_dir=torch.full((1, SIDE, SIDE, NUM_SLOTS, 2), 0.7),
    )


def test_no_parts_leaves_prediction_untouched():
    """아무 묶음도 안 얹으면 예측 그대로여야 한다."""
    out = inject_parts(prediction(), targets(), (), NUM_CLASSES, NUM_SLOTS)
    base = prediction()
    assert torch.equal(out.self_coord, base.self_coord)
    assert torch.equal(out.class_logit, base.class_logit)


@pytest.mark.parametrize("part", INJECTABLE_PARTS)
def test_each_part_can_be_injected_alone(part: str):
    """묶음 하나만 얹으면 그 묶음만 정답이 된다."""
    tgt = targets()
    truth = gt_model_output(tgt, NUM_CLASSES, NUM_SLOTS)
    out = inject_parts(prediction(), tgt, (part,), NUM_CLASSES, NUM_SLOTS)
    positive = tgt["class_map"] > 0
    field = {
        "node": "heatmap_logit",
        "class": "class_logit",
        "coord": "self_coord",
        "conn": "conn_dir",
        "end": "end_logit",
    }[part]
    assert torch.allclose(getattr(out, field)[positive], getattr(truth, field)[positive])


def test_untouched_groups_stay_predicted():
    """좌표만 얹으면 종류와 연결 방향은 예측 그대로여야 한다."""
    tgt = targets()
    out = inject_parts(prediction(), tgt, ("coord",), NUM_CLASSES, NUM_SLOTS)
    base = prediction()
    positive = tgt["class_map"] > 0
    assert torch.allclose(out.class_logit[positive], base.class_logit[positive])
    assert torch.allclose(out.conn_dir[positive], base.conn_dir[positive])


def test_all_parts_equals_full_injection():
    """전부 얹으면 통째 주입(천장)과 같아야 한다 — 두 경로가 어긋나면 표가 틀어진다."""
    tgt = targets()
    out = inject_parts(prediction(), tgt, INJECTABLE_PARTS, NUM_CLASSES, NUM_SLOTS)
    truth = gt_model_output(tgt, NUM_CLASSES, NUM_SLOTS)
    positive = tgt["class_map"] > 0
    assert torch.equal(out.node_mask, truth.node_mask)
    for field in ("class_logit", "self_coord", "conn_dir", "exist_logit", "end_logit"):
        assert torch.allclose(getattr(out, field)[positive], getattr(truth, field)[positive])


def test_unknown_part_is_rejected():
    """오타를 조용히 넘기면 표가 거짓이 된다 — 크게 실패해야 한다."""
    with pytest.raises(ValueError):
        inject_parts(prediction(), targets(), ("heatmap",), NUM_CLASSES, NUM_SLOTS)

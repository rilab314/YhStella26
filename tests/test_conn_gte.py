"""각도 섹터 배정 연결 손실 검증 (3-C 비교 대상).

기본 판은 매칭으로 슬롯을 정하고, 이 판은 **각도로 고정 배정**한다. 그 배정이 규약대로인지
(0번 = 동쪽에서 시작해 반시계로 60도씩)와, 두 분기가 한 섹터에 겹칠 때 그 사실이
`sector_collision` 에 잡히는지를 본다.
"""

import numpy as np
import pytest
import torch

from stella.loss.conn_gte import ConnLossGTE
from stella.model.stella import ModelOutput

SECTORS = 6
SIDE = 2
NUM_CLASSES = 12


def make_loss() -> ConnLossGTE:
    return ConnLossGTE(num_conn_slots=SECTORS, w_exist=1.0, w_dir=1.0, exist_pos_weight=1.0)


def unit(degrees: float) -> list[float]:
    radians = np.deg2rad(degrees)
    return [float(np.cos(radians)), float(np.sin(radians))]


def scene(branches: list[list[float]]) -> tuple[ModelOutput, dict]:
    """한 칸만 양성인 최소 장면 — 그 칸에 주어진 분기를 넣는다."""
    class_map = torch.zeros((1, SIDE, SIDE), dtype=torch.long)
    class_map[0, 0, 0] = 3
    conn = torch.zeros((1, SIDE, SIDE, len(branches), 2))
    conn[0, 0, 0] = torch.tensor(branches, dtype=torch.float32)
    node = torch.zeros((1, SIDE, SIDE), dtype=torch.bool)
    node[0, 0, 0] = True
    output = ModelOutput(
        heatmap_logit=torch.zeros((1, SIDE, SIDE)),
        node_mask=node,
        class_logit=torch.zeros((1, SIDE, SIDE, NUM_CLASSES)),
        self_coord=torch.zeros((1, SIDE, SIDE, 2)),
        end_logit=torch.zeros((1, SIDE, SIDE)),
        fg_logit=torch.zeros((1, SIDE, SIDE)),
        exist_logit=torch.zeros((1, SIDE, SIDE, SECTORS)),
        conn_dir=torch.zeros((1, SIDE, SIDE, SECTORS, 2)),
    )
    return output, {"class_map": class_map, "conn_dirs": conn}


def assigned(branches: list[list[float]]) -> tuple[torch.Tensor, torch.Tensor, float]:
    loss = make_loss()
    _, targets = scene(branches)
    gt = targets["conn_dirs"][targets["class_map"] > 0].float()
    target_dir, target_exist, collision = loss._assign_by_sector(gt)
    return target_dir[0], target_exist[0], float(collision)


@pytest.mark.parametrize(
    "degrees,sector",
    [(0, 0), (30, 0), (61, 1), (90, 1), (150, 2), (190, 3), (250, 4), (310, 5), (359, 5)],
)
def test_angle_lands_in_expected_sector(degrees: float, sector: int):
    """0번 섹터가 동쪽(0도)에서 시작해 반시계로 60도씩 나뉜다."""
    _, exist, _ = assigned([unit(degrees), [0.0, 0.0]])
    assert int(exist.argmax()) == sector
    assert float(exist.sum()) == 1.0


def test_opposite_branches_take_opposite_sectors():
    """선 위의 셀은 앞뒤 두 방향을 가진다 — 섹터가 셋 떨어져야 한다."""
    _, exist, collision = assigned([unit(20), unit(200)])
    filled = torch.nonzero(exist).flatten().tolist()
    assert filled == [0, 3]
    assert collision == 0.0


def test_two_branches_in_one_sector_are_counted_as_collision():
    """급하게 꺾인 셀은 두 분기가 한 섹터에 들어간다 — 하나를 잃는다."""
    _, exist, collision = assigned([unit(10), unit(50)])
    assert float(exist.sum()) == 1.0  # 하나만 남았다
    assert collision == pytest.approx(0.5)  # 분기 둘 중 하나를 잃었다


def test_direction_target_is_the_branch_itself():
    """찬 섹터의 목표 방향은 그 분기의 단위벡터 그대로다."""
    target_dir, exist, _ = assigned([unit(90), [0.0, 0.0]])
    slot = int(exist.argmax())
    assert torch.allclose(target_dir[slot], torch.tensor(unit(90)), atol=1e-6)


def test_empty_sectors_are_supervised_as_absent():
    """분기가 없는 섹터는 '없음'으로 감독한다 — 그래야 빈 방향을 안 만든다."""
    _, exist, _ = assigned([unit(20), unit(200)])
    assert float(exist.sum()) == 2.0
    assert float((exist == 0).sum()) == SECTORS - 2


def test_loss_runs_and_reports_collision():
    """전체 손실이 돌고, 진단값이 함께 나온다."""
    output, targets = scene([unit(10), unit(50)])
    losses = make_loss()(output, targets)
    assert set(losses) == {"exist", "dir", "sector_collision", "total"}
    assert torch.isfinite(losses["total"])
    assert float(losses["sector_collision"]) == pytest.approx(0.5)

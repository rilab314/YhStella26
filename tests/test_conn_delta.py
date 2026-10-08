"""GTE 변위 회귀 판 검증 — GT 길이 · 손실 목표 · 디코딩 (Table III 비교 대상, 논지 C1-(2)).

기본 판은 슬롯이 **단위 방향**만 내고, 이 판은 **이웃 정점까지의 변위**를 낸다.
확인할 것은 셋이다. ① GT 길이가 실제 변위를 복원하는가 ② 손실 목표가 방향 x 길이인가
③ 디코더가 방향만이 아니라 **거리까지 써서** 다른 정점을 고르는가.
"""

import numpy as np
import pytest
import torch

from configs.base import get_config
from stella.data.encode import ChainEncoder
from stella.decode.graph import ChainDecoder
from stella.decode.graph_delta import DeltaChainDecoder
from stella.loss.conn_gte import ConnLossGTE
from stella.loss.conn_gte_delta import ConnLossGTEDelta
from stella.model.heads import ConnHead

GRID_STRIDE = 4
IMAGE_SIZE = 128
SIDE = IMAGE_SIZE // GRID_STRIDE
SECTORS = 6
NUM_CLASSES = 12


def encode_line(store_length: bool) -> dict:
    encoder = ChainEncoder(
        image_size=IMAGE_SIZE,
        grid_stride=GRID_STRIDE,
        num_classes=NUM_CLASSES,
        max_degree=2,
        store_length=store_length,
    )
    points = np.array([[20.0, 40.5], [100.0, 40.5]], dtype=np.float32)
    return encoder.encode([{"class": 3, "points": points}])


def test_lengths_are_absent_unless_asked():
    """기본 설정의 GT 는 예전과 똑같아야 한다 — 새 키가 조용히 끼어들면 안 된다."""
    assert "conn_lengths" not in encode_line(store_length=False)
    assert "conn_lengths" in encode_line(store_length=True)


def test_direction_times_length_reconstructs_the_neighbor_point():
    """방향 x 길이 = 이웃 노드 점까지의 실제 변위 (격자 단위)."""
    target = encode_line(store_length=True)
    cells = np.argwhere(target["class_map"] > 0)
    point = {(i, j): np.array([j, i]) + target["coord_map"][i, j] for i, j in cells}
    inner = [(i, j) for i, j in map(tuple, cells) if (i, j + 1) in point and (i, j - 1) in point]
    assert inner, "가로선이라 가운데 칸이 있어야 한다"
    for i, j in inner:
        delta = target["conn_dirs"][i, j] * target["conn_lengths"][i, j][:, None]
        reached = point[(i, j)] + delta
        neighbours = {tuple(np.round(point[(i, j - 1)], 4)), tuple(np.round(point[(i, j + 1)], 4))}
        assert {tuple(np.round(r, 4)) for r in reached} == neighbours


def test_end_branch_length_reaches_the_real_endpoint():
    """끝 칸의 바깥 분기는 선의 실제 끝점까지의 거리다 (끝칸 미채움 규약, 6.2절)."""
    target = encode_line(store_length=True)
    lengths = target["conn_lengths"][target["class_map"] > 0]
    assert float(lengths.min()) >= 0.0
    assert float(lengths.max()) < SIDE  # 타일 밖으로 나가는 길이는 없다


def slot_scene(branches: list[list[float]], lengths: list[float]) -> dict:
    class_map = torch.zeros((1, 2, 2), dtype=torch.long)
    class_map[0, 0, 0] = 3
    conn = torch.zeros((1, 2, 2, len(branches), 2))
    conn[0, 0, 0] = torch.tensor(branches, dtype=torch.float32)
    length = torch.zeros((1, 2, 2, len(branches)))
    length[0, 0, 0] = torch.tensor(lengths, dtype=torch.float32)
    return {"class_map": class_map, "conn_dirs": conn, "conn_lengths": length}


def assigned(loss, targets: dict) -> tuple[torch.Tensor, torch.Tensor]:
    supervised = targets["class_map"] > 0
    gt_dir = targets["conn_dirs"][supervised].float()
    gt_vec = loss._branch_vectors(targets, supervised, gt_dir)
    target_vec, target_exist, _ = loss._assign_by_sector(gt_dir, gt_vec)
    return target_vec[0], target_exist[0]


def make_loss(cls):
    return cls(num_conn_slots=SECTORS, w_exist=1.0, w_dir=1.0, exist_pos_weight=1.0)


def test_delta_target_is_direction_times_length():
    """찬 섹터의 목표는 단위 방향이 아니라 변위다."""
    targets = slot_scene([[1.0, 0.0], [-1.0, 0.0]], [2.5, 1.5])
    target_vec, exist = assigned(make_loss(ConnLossGTEDelta), targets)
    assert torch.allclose(target_vec[int(exist.argmax())], torch.tensor([2.5, 0.0]))


def test_sector_assignment_does_not_move_when_lengths_change():
    """길이를 곱해도 **어느 슬롯이 맡는가**는 그대로여야 한다 — 배정은 각도만 본다."""
    branches = [[0.0, 1.0], [0.0, -1.0]]
    _, unit_exist = assigned(make_loss(ConnLossGTE), slot_scene(branches, [1.0, 1.0]))
    _, delta_exist = assigned(make_loss(ConnLossGTEDelta), slot_scene(branches, [4.0, 0.2]))
    assert torch.equal(unit_exist, delta_exist)


def test_delta_loss_says_what_is_missing_without_lengths():
    """길이 없는 GT 로 돌리면 조용히 틀리지 않고 즉시 멈춘다."""
    targets = slot_scene([[1.0, 0.0], [-1.0, 0.0]], [1.0, 1.0])
    del targets["conn_lengths"]
    with pytest.raises(KeyError, match="conn_store_length"):
        assigned(make_loss(ConnLossGTEDelta), targets)


def test_delta_head_keeps_parameters_and_stops_normalizing():
    """파라미터 수가 같아야 비교가 공정하다. 출력 길이는 더 이상 1이 아니다."""
    torch.manual_seed(0)
    unit_head = ConnHead(d_model=32, num_slots=2)
    torch.manual_seed(0)
    delta_head = ConnHead(d_model=32, num_slots=2, target="delta")
    assert sum(p.numel() for p in unit_head.parameters()) == sum(
        p.numel() for p in delta_head.parameters()
    )
    torch.manual_seed(1)
    tokens = torch.randn(5, 2, 32)
    with torch.no_grad():
        _, unit_dir = unit_head(tokens)
        _, delta_dir = delta_head(tokens)
    assert torch.allclose(unit_dir.norm(dim=-1), torch.ones(5, 2), atol=1e-5)
    assert not torch.allclose(delta_dir.norm(dim=-1), torch.ones(5, 2), atol=1e-3)


def test_delta_head_rejects_an_unknown_target():
    with pytest.raises(ValueError, match="conn_target"):
        ConnHead(d_model=8, target="distance")


def row_vertices(xs: list[float], reach: float) -> dict:
    """가로로 늘어선 정점들. 방향은 단위벡터고, `reach` 가 예측한 거리다 (셀).

    `_best_candidate` 는 두 판 모두 **쪼개진 뒤**의 속성을 받는다 — 기본 판은 `reach` 를 쓰지
    않고, 변위 판만 쓴다.
    """
    count = len(xs)
    point = np.stack([np.array(xs, np.float64), np.zeros(count)], axis=-1)
    direction = np.zeros((count, 2, 2))
    direction[:, 0] = [-1.0, 0.0]
    direction[:, 1] = [1.0, 0.0]
    class_prob = np.zeros((count, NUM_CLASSES))
    class_prob[:, 3] = 1.0
    return {
        "point": point,
        "dir": direction,
        "reach": np.full((count, 2), reach),
        "exist": np.ones((count, 2)),
        "class_prob": class_prob,
        "neighbors": np.tile(np.arange(count), (count, 1)),
    }


def decoders() -> tuple[ChainDecoder, DeltaChainDecoder]:
    cfg = get_config()
    cfg.decode.perp_thresh = 3.0
    return ChainDecoder.from_cfg(cfg.decode, cfg), DeltaChainDecoder.from_cfg(cfg.decode, cfg)


def choose(decoder, vertices: dict) -> int:
    used = np.zeros(vertices["point"].shape[0], bool)
    slot_used = np.zeros((vertices["point"].shape[0], 2), bool)
    picked = decoder._best_candidate(vertices, used, slot_used, 0, 1, 3, None)
    assert picked is not None
    return picked[0]


def test_delta_decoder_walks_to_the_predicted_point():
    """0번이 3셀 앞을 가리키면 3셀 앞 정점으로 간다 — 기본 판은 가장 가까운 정점을 고른다."""
    base, delta = decoders()
    vertices = row_vertices([0.0, 1.0, 3.0], reach=3.0)
    assert choose(base, vertices) == 1  # 방향만 보면 가장 가까운 정점이 이긴다
    assert choose(delta, vertices) == 2  # 거리까지 보면 예측한 자리의 정점이 이긴다


def test_delta_decoder_splits_output_into_direction_and_reach():
    """상속한 규칙들이 단위 방향을 기대하므로, 정점 단계에서 방향과 거리를 쪼갠다."""
    _, delta = decoders()
    raw = row_vertices([0.0, 2.0], reach=2.0)
    raw["dir"] = raw["dir"] * 2.0  # 헤드가 내놓는 날것 — 길이가 2인 변위
    delta.extractor = lambda output: raw
    split = delta._vertices(None)
    assert np.allclose(np.linalg.norm(split["dir"], axis=-1), 1.0)
    assert np.allclose(split["reach"], 2.0)


def test_delta_decoder_survives_a_zero_length_slot():
    """길이 0 을 내놓은 슬롯이 0 나누기로 죽지 않는다 (학습 초기에 나온다)."""
    _, delta = decoders()
    raw = row_vertices([0.0, 2.0], reach=0.0)
    raw["dir"] = raw["dir"] * 0.0
    delta.extractor = lambda output: raw
    split = delta._vertices(None)
    assert np.all(np.isfinite(split["dir"]))
    assert np.allclose(split["reach"], 0.0)

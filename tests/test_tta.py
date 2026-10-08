"""추론 시각 증강(TTA)의 되돌리기 검증 (plan_0908 6절).

**되돌리기가 조금 틀려도 모델 점수는 그럴듯하게 나온다** — 그래서 점수가 아니라 기하로 잰다.

기준은 **좌표 변환 규칙 하나**다: 반시계 90도는 `(x, y) -> (y, S - x)`, 좌우 뒤집기는
`x -> S - x` (`VectorAugment._map_points` 와 같다). 먼저 사진 변환이 그 규칙과 맞는지
확인하고(`test_image_transform_matches_point_transform`), 그 규칙으로 만든 "돌아간 예측"을
`invert_output` 이 원본으로 되돌리는지 본다.

인코더로 재지 않는 이유: 폴리라인을 돌려서 다시 인코딩하면 끝점에서 **1픽셀 반올림 차이**가
나 칸 두 개가 어긋난다. 그것은 되돌리기의 잘못이 아니라 래스터화의 비대칭이다.
"""

import numpy as np
import pytest
import torch

from stella.model.stella import ModelOutput
from stella.model.tta import (
    GRID_SYMMETRIES,
    SLOT_MATCHED,
    SLOT_SINGLE,
    GridSymmetryTTA,
    invert_output,
    transform_image,
)

SIDE = 16  # 격자 한 변
IMAGE_SIZE = 64
NUM_CLASSES = 12
NUM_SLOTS = 2
# 셀 안 위치는 0.5 를 중심으로 도므로 경계에서 멀리 잡아 반올림 흔들림을 피한다.
NODES = (
    {"cell": (3, 5), "coord": (0.25, 0.75), "label": 3, "dirs": ((1.0, 0.0), (-1.0, 0.0))},
    {"cell": (11, 2), "coord": (0.6, 0.4), "label": 7, "dirs": ((0.6, 0.8), (-0.6, -0.8))},
    {"cell": (8, 13), "coord": (0.5, 0.5), "label": 5, "dirs": ((0.0, 1.0), (0.0, -1.0))},
)


def move_point(point: np.ndarray, turns: int, flip: bool, size: float) -> np.ndarray:
    """좌표 변환 규칙 — 반시계 회전 뒤 좌우 뒤집기 (`VectorAugment._map_points` 와 같다)."""
    out = np.asarray(point, np.float64).copy()
    for _ in range(turns):
        out = np.array([out[1], size - out[0]])
    if flip:
        out[0] = size - out[0]
    return out


def move_vector(vector: np.ndarray, turns: int, flip: bool) -> np.ndarray:
    """방향은 평행이동이 없으므로 규칙의 선형 부분만 적용한다."""
    out = np.asarray(vector, np.float64).copy()
    for _ in range(turns):
        out = np.array([out[1], -out[0]])
    if flip:
        out[0] = -out[0]
    return out


def build_output(nodes: tuple, turns: int, flip: bool) -> ModelOutput:
    """노드 목록을 그 방향으로 돌린 프레임의 ModelOutput 으로 만든다."""
    output = _empty_output()
    for node in nodes:
        row, col = node["cell"]
        point = move_point(
            np.array([col + node["coord"][0], row + node["coord"][1]]), turns, flip, float(SIDE)
        )
        new_col, new_row = int(np.floor(point[0])), int(np.floor(point[1]))
        _write_node(output, new_row, new_col, point, node, turns, flip)
    return output


def _empty_output() -> ModelOutput:
    zeros = lambda *shape: torch.zeros(shape)  # noqa: E731 — 시험 안의 짧은 별칭
    return ModelOutput(
        heatmap_logit=torch.full((1, SIDE, SIDE), -10.0),
        node_mask=torch.zeros((1, SIDE, SIDE), dtype=torch.bool),
        class_logit=zeros(1, SIDE, SIDE, NUM_CLASSES),
        self_coord=zeros(1, SIDE, SIDE, 2),
        end_logit=zeros(1, SIDE, SIDE),
        fg_logit=zeros(1, SIDE, SIDE),
        exist_logit=zeros(1, SIDE, SIDE, NUM_SLOTS),
        conn_dir=zeros(1, SIDE, SIDE, NUM_SLOTS, 2),
    )


def _write_node(output, row: int, col: int, point, node: dict, turns: int, flip: bool) -> None:
    output.node_mask[0, row, col] = True
    output.heatmap_logit[0, row, col] = 10.0
    output.class_logit[0, row, col, node["label"]] = 20.0
    output.end_logit[0, row, col] = 1.0
    output.fg_logit[0, row, col] = 10.0
    output.self_coord[0, row, col] = torch.tensor(point - np.floor(point), dtype=torch.float32)
    for slot, direction in enumerate(node["dirs"]):
        moved = move_vector(np.array(direction), turns, flip)
        output.conn_dir[0, row, col, slot] = torch.tensor(moved, dtype=torch.float32)
        output.exist_logit[0, row, col, slot] = 10.0


@pytest.mark.parametrize("turns,flip", GRID_SYMMETRIES)
def test_inverse_restores_maps(turns: int, flip: bool):
    """돌아간 프레임의 예측을 되돌리면 칸·종류·끝이 원본과 정확히 같다."""
    origin = build_output(NODES, 0, False)
    restored = invert_output(build_output(NODES, turns, flip), turns, flip)
    assert torch.equal(restored.node_mask, origin.node_mask)
    assert torch.equal(restored.heatmap_logit, origin.heatmap_logit)
    assert torch.equal(restored.class_logit, origin.class_logit)
    assert torch.equal(restored.end_logit, origin.end_logit)


@pytest.mark.parametrize("turns,flip", GRID_SYMMETRIES)
def test_inverse_restores_cell_points(turns: int, flip: bool):
    """셀 안 위치는 원점이 셀 좌상단이라 0.5 를 중심으로 돌려야 맞는다."""
    origin = build_output(NODES, 0, False)
    restored = invert_output(build_output(NODES, turns, flip), turns, flip)
    nodes = origin.node_mask
    assert torch.allclose(restored.self_coord[nodes], origin.self_coord[nodes], atol=1e-5)


@pytest.mark.parametrize("turns,flip", GRID_SYMMETRIES)
def test_inverse_restores_connection_directions(turns: int, flip: bool):
    """연결 방향은 원점이 노드 점이라 0.5 보정 없이 돌린다."""
    origin = build_output(NODES, 0, False)
    restored = invert_output(build_output(NODES, turns, flip), turns, flip)
    nodes = origin.node_mask
    assert torch.allclose(restored.conn_dir[nodes], origin.conn_dir[nodes], atol=1e-5)


def test_cell_point_inverse_is_not_plain_vector_inverse():
    """0.5 보정을 빼먹으면 셀 안 위치가 틀린다 — 그 실수를 시험이 잡는지 확인한다."""
    origin = build_output(NODES, 0, False)
    view = build_output(NODES, 1, False)
    nodes = origin.node_mask
    wrong = invert_output(view, 1, False).self_coord.clone()
    assert not torch.allclose(wrong[nodes] * 0.0, origin.self_coord[nodes], atol=1e-5)


@pytest.mark.parametrize("turns,flip", GRID_SYMMETRIES)
def test_image_transform_matches_point_transform(turns: int, flip: bool):
    """사진 변환이 좌표 변환 규칙과 같은지 — 한 점만 켠 사진으로 확인한다."""
    image = torch.zeros((1, 1, IMAGE_SIZE, IMAGE_SIZE))
    x, y = 30, 7
    image[0, 0, y, x] = 1.0
    moved = transform_image(image, turns, flip)
    expected = move_point(np.array([x + 0.5, y + 0.5]), turns, flip, float(IMAGE_SIZE))
    found = torch.nonzero(moved[0, 0])[0]
    assert (int(found[1]), int(found[0])) == (int(expected[0] - 0.5), int(expected[1] - 0.5))


class EquivariantStub:
    """사진에 대해 **정확히 등변인** 가짜 모델 — TTA 합치기 전체를 끝까지 검증한다.

    돌린 사진을 넣으면 출력도 그만큼 돌아간다. 그러므로 8뷰를 되돌려 합친 결과는
    원본 한 장을 넣은 결과와 **같아야** 한다. 어긋나면 되돌리기나 합치기가 틀린 것이다.
    """

    def __init__(self, *, stride: int):
        self.stride = stride

    def __call__(self, image: torch.Tensor) -> ModelOutput:
        cells = image[:, 0].unfold(1, self.stride, self.stride).unfold(2, self.stride, self.stride)
        weight = cells.flatten(-2)  # (B, L, L, stride*stride)
        side = weight.shape[1]
        mass = weight.sum(dim=-1)
        output = self._blank(weight.shape[0], side)
        output.heatmap_logit = mass
        output.node_mask = mass > 0
        output.self_coord = self._centroid(weight, mass)
        output.class_logit[..., 1] = mass
        return output

    def _blank(self, batch: int, side: int) -> ModelOutput:
        return ModelOutput(
            heatmap_logit=torch.zeros((batch, side, side)),
            node_mask=torch.zeros((batch, side, side), dtype=torch.bool),
            class_logit=torch.zeros((batch, side, side, NUM_CLASSES)),
            self_coord=torch.zeros((batch, side, side, 2)),
            end_logit=torch.zeros((batch, side, side)),
            fg_logit=torch.zeros((batch, side, side)),
            exist_logit=torch.zeros((batch, side, side, NUM_SLOTS)),
            conn_dir=torch.zeros((batch, side, side, NUM_SLOTS, 2)),
        )

    def _centroid(self, weight: torch.Tensor, mass: torch.Tensor) -> torch.Tensor:
        """셀 안 밝기 무게중심 — 픽셀 면적 중심(+0.5)을 써야 뒤집기에 대칭이 된다."""
        offset = (torch.arange(self.stride, dtype=torch.float32) + 0.5) / self.stride
        grid_y, grid_x = torch.meshgrid(offset, offset, indexing="ij")
        flat_x, flat_y = grid_x.reshape(-1), grid_y.reshape(-1)
        safe = mass.clamp(min=1e-6)
        return torch.stack([(weight * flat_x).sum(-1) / safe, (weight * flat_y).sum(-1) / safe], -1)


def test_eight_view_merge_matches_single_view_for_equivariant_model():
    """등변인 모델이면 8뷰 평균이 원본 한 장과 같아야 한다 — 되돌리기+합치기 전체 검증."""
    stride = 4
    torch.manual_seed(0)
    image = (torch.rand((1, 3, IMAGE_SIZE, IMAGE_SIZE)) > 0.9).float()
    stub = EquivariantStub(stride=stride)
    single = stub(image)
    tta = GridSymmetryTTA(symmetries=GRID_SYMMETRIES, min_views=1, slot_mode=SLOT_SINGLE)
    merged = tta.predict(stub, image)
    assert torch.equal(merged.node_mask, single.node_mask)
    assert torch.allclose(merged.heatmap_logit, single.heatmap_logit, atol=1e-4)
    nodes = single.node_mask
    assert torch.allclose(merged.self_coord[nodes], single.self_coord[nodes], atol=1e-5)


class SwappingStub(EquivariantStub):
    """뷰마다 슬롯 번호를 뒤바꿔 내놓는 가짜 모델 — 슬롯 순서가 무의미함을 흉내낸다.

    방향은 셀 안 무게중심에서 뽑아 등변이고, 두 슬롯은 서로 반대 방향이다.
    **짝을 안 맞추고 평균하면 두 방향이 상쇄돼 0 이 된다** — 방안 B 가 그것을 막는지 본다.
    """

    def __init__(self, *, stride: int):
        super().__init__(stride=stride)
        self.calls = 0

    def __call__(self, image: torch.Tensor) -> ModelOutput:
        output = super().__call__(image)
        heading = output.self_coord - 0.5
        norm = heading.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        unit = heading / norm
        slots = [unit, -unit] if self.calls % 2 == 0 else [-unit, unit]
        output.conn_dir = torch.stack(slots, dim=-2)
        output.exist_logit = torch.where(output.node_mask, 5.0, -5.0)[..., None].repeat(
            *([1] * output.node_mask.dim()), NUM_SLOTS
        )
        self.calls += 1
        return output


def _merged_directions(slot_mode: str):
    torch.manual_seed(1)
    image = (torch.rand((1, 3, IMAGE_SIZE, IMAGE_SIZE)) > 0.9).float()
    stub = SwappingStub(stride=4)
    single = stub(image)
    stub.calls = 0
    tta = GridSymmetryTTA(symmetries=GRID_SYMMETRIES, min_views=1, slot_mode=slot_mode)
    return tta.predict(stub, image), single


def test_matched_slots_survive_view_dependent_swapping():
    """슬롯 번호가 뷰마다 뒤바뀌어도 방안 B 는 방향을 지켜야 한다."""
    merged, single = _merged_directions(SLOT_MATCHED)
    nodes = single.node_mask
    assert torch.allclose(merged.conn_dir[nodes], single.conn_dir[nodes], atol=1e-4)


def test_matched_slots_keep_unit_length():
    """평균낸 방향은 길이가 1이 아니다 — 디코더 계약대로 되돌려야 한다.

    무게중심이 셀 정중앙이면 방향이 정의되지 않아 길이가 0이다(가짜 모델의 성질) — 그 칸은 뺀다.
    """
    merged, single = _merged_directions(SLOT_MATCHED)
    defined = single.conn_dir.norm(dim=-1) > 1e-4
    lengths = merged.conn_dir.norm(dim=-1)[defined & single.node_mask[..., None]]
    assert lengths.numel() > 0
    assert torch.allclose(lengths, torch.ones_like(lengths), atol=1e-4)


def test_averaging_slots_without_matching_would_cancel():
    """짝을 안 맞추면 상쇄된다는 것 — 이 시험이 방안 B 의 존재 이유다."""
    merged, single = _merged_directions(SLOT_SINGLE)
    nodes = single.node_mask
    naive = torch.stack([single.conn_dir, single.conn_dir.flip(-2)]).mean(dim=0)
    assert torch.allclose(naive[nodes], torch.zeros_like(naive[nodes]), atol=1e-6)
    assert not torch.allclose(merged.conn_dir[nodes], naive[nodes], atol=1e-4)

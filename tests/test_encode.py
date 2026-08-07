"""GT 인코더 불변식 검증 (impl_plan 6.4절 불변식 9종, 9차 개정 — 선 단위 사슬)."""

import numpy as np
import pytest

from stella.data.encode import ChainEncoder
from stella.data.synthetic import SyntheticDataset

GRID_STRIDE = 4
IMAGE_SIZE = 128
SIDE = IMAGE_SIZE // GRID_STRIDE


def encode_scene(instances, image_size: int = IMAGE_SIZE):
    encoder = ChainEncoder(
        image_size=image_size,
        grid_stride=GRID_STRIDE,
        num_classes=12,
        max_degree=2,
        supersample=1,
    )
    return encoder, encoder.encode(instances)


def horizontal_line(label: int = 3, row: float = 40.5) -> list[dict]:
    points = np.array([[20.0, row], [100.0, row]], dtype=np.float32)
    return [{"class": label, "points": points}]


def node_points(target) -> dict[tuple[int, int], np.ndarray]:
    """양성 셀 -> 절대 노드 점 (격자 단위, (x, y))."""
    points = {}
    for i, j in np.argwhere(target["class_map"] > 0):
        points[(int(i), int(j))] = np.array([j, i]) + target["coord_map"][i, j]
    return points


def test_simple_line_shapes_and_end_rule():
    _, target = encode_scene(horizontal_line())
    assert target["class_map"].shape == (SIDE, SIDE)
    assert target["coord_map"].shape == (SIDE, SIDE, 2)
    assert target["end_map"].shape == (SIDE, SIDE)
    assert target["conn_dirs"].shape == (SIDE, SIDE, 2, 2)
    row = int(40.5 // GRID_STRIDE)
    owned = np.flatnonzero(target["class_map"][row] > 0)
    # x = 20..100 -> 셀 5..25. 끝칸 미채움이라 6..24 만 채워진다 (6.2절 끝 규약)
    assert owned.min() == 6 and owned.max() == 24
    assert target["end_map"][row, 6] == 1.0 and target["end_map"][row, 24] == 1.0
    assert target["end_map"].sum() == 2.0


def test_invariant_two_unit_directions_on_every_positive_cell():
    """불변식 1: 양성 셀의 conn_dirs 는 항상 단위벡터 2개다."""
    _, target = encode_scene(horizontal_line())
    positive = target["class_map"] > 0
    norms = np.linalg.norm(target["conn_dirs"][positive], axis=-1)
    assert norms.shape[-1] == 2
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_invariant_antiparallel_between_chain_neighbors():
    """불변식 2: 이웃 a·b 에서 a의 'b 방향'과 b의 'a 방향'은 정확히 반대다."""
    _, target = encode_scene(horizontal_line())
    row = int(40.5 // GRID_STRIDE)
    owned = np.flatnonzero(target["class_map"][row] > 0)
    for a, b in zip(owned[:-1], owned[1:]):
        forward = target["conn_dirs"][row, a]  # 둘 중 하나가 b를 향한다
        backward = target["conn_dirs"][row, b]
        toward_b = forward[np.argmax(forward[:, 0])]  # +x 쪽 분기
        toward_a = backward[np.argmin(backward[:, 0])]  # -x 쪽 분기
        np.testing.assert_allclose(toward_b, -toward_a, atol=1e-6)


def test_invariant_direction_ray_hits_neighbor_point():
    """불변식 3: 저장된 방향을 따라가면 실제 이웃 점이 나온다 (인코딩·디코딩 대칭의 근거)."""
    _, target = encode_scene(horizontal_line())
    points = node_points(target)
    row = int(40.5 // GRID_STRIDE)
    owned = np.flatnonzero(target["class_map"][row] > 0)
    for a, b in zip(owned[:-1], owned[1:]):
        p_a, p_b = points[(row, int(a))], points[(row, int(b))]
        expected = (p_b - p_a) / np.linalg.norm(p_b - p_a)
        stored = target["conn_dirs"][row, a]
        best = stored[np.argmax(stored @ expected)]
        np.testing.assert_allclose(best, expected, atol=1e-5)


def test_invariant_non_positive_cells_are_empty():
    """불변식 4: coord·end·conn_dirs 는 양성 셀에서만 유효하다."""
    _, target = encode_scene(horizontal_line())
    background = target["class_map"] == 0
    assert np.all(target["coord_map"][background] == 0)
    assert np.all(target["end_map"][background] == 0)
    assert np.all(target["conn_dirs"][background] == 0)


def test_invariant_centroid_inside_cell():
    """불변식 5: 무게중심은 셀 안에 있다."""
    _, target = encode_scene(horizontal_line())
    coord = target["coord_map"][target["class_map"] > 0]
    assert coord.min() >= 0.0 and coord.max() < 1.0


def test_invariant_each_cell_owned_by_one_line():
    """불변식 6: 같은 셀에 두 선이 걸려도 픽셀이 많은 한 선만 셀을 소유한다."""
    x = np.linspace(20.0, 100.0, 40, dtype=np.float32)
    upper = np.stack([x, np.full_like(x, 41.0)], axis=-1)
    lower = np.stack([x, np.full_like(x, 42.5)], axis=-1)  # 같은 셀 행(10)에 겹친다
    encoder, target = encode_scene([{"class": 3, "points": upper}, {"class": 3, "points": lower}])
    rows = np.unique(np.argwhere(target["class_map"] > 0)[:, 0])
    assert rows.tolist() == [10]
    norms = np.linalg.norm(target["conn_dirs"][target["class_map"] > 0], axis=-1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)  # 소유 선의 사슬만 남아도 분기는 2개


def test_invariant_t_junction_keeps_main_line_intact():
    """불변식 7: B의 끝칸(A가 지나는 셀)은 A 소유로 남고, B는 물러난 끝 셀에서 접합점을 가리킨다."""
    main = np.array([[10.0, 60.5], [118.0, 60.5]], dtype=np.float32)
    stem = np.array([[64.5, 20.0], [64.5, 60.5]], dtype=np.float32)
    _, target = encode_scene([{"class": 3, "points": main}, {"class": 5, "points": stem}])
    junction_row = int(60.5 // GRID_STRIDE)
    junction_col = int(64.5 // GRID_STRIDE)
    assert target["class_map"][junction_row, junction_col] == 3  # 본선 소유, 사슬 연속
    owned_main = np.flatnonzero(target["class_map"][junction_row] == 3)
    assert np.all(np.diff(owned_main) == 1)
    stem_cells = np.argwhere(target["class_map"] == 5)
    stem_end = stem_cells[stem_cells[:, 0].argmax()]  # 접합에 가장 가까운 끝 셀
    assert target["end_map"][tuple(stem_end)] == 1.0
    dirs = target["conn_dirs"][tuple(stem_end)]
    assert dirs[:, 1].max() > 0.9  # 끝방향 분기가 아래(접합점, +y)를 향한다


def test_invariant_x_crossing_loser_skips_the_cell():
    """불변식 8: 교차 셀을 잃은 선의 사슬은 그 칸을 건너뛰어 2칸 거리 이웃을 잇는다."""
    down = np.array([[20.0, 20.0], [108.0, 108.0]], dtype=np.float32)
    up = np.array([[20.0, 108.0], [108.0, 20.0]], dtype=np.float32)
    encoder, target = encode_scene([{"class": 3, "points": down}, {"class": 5, "points": up}])
    positive = target["class_map"] > 0
    norms = np.linalg.norm(target["conn_dirs"][positive], axis=-1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)
    assert encoder.stats["cells_lost"] >= 1  # 진 선이 교차 칸을 잃었다
    assert target["end_map"].sum() == 4.0  # 선 2개 x 끝 셀 2개 — 교차가 사슬을 못 끊는다


def test_invariant_gentle_diagonal_has_no_zigzag():
    """불변식 9: 완만한 대각선이 톱니가 아니라 단조로운 셀 열로 나온다 (3x3 순위 규칙)."""
    x = np.linspace(10.0, 118.0, 60, dtype=np.float32)
    y = 40.0 + 0.18 * (x - 10.0)  # 기울기 0.18 — 행이 가끔 한 칸씩 내려간다
    _, target = encode_scene([{"class": 3, "points": np.stack([x, y], axis=-1)}])
    cells = np.argwhere(target["class_map"] > 0)
    for row in np.unique(cells[:, 0]):
        cols = np.sort(cells[cells[:, 0] == row][:, 1])
        assert np.all(np.diff(cols) == 1), f"행 {row} 안에서 셀 열이 끊긴다: {cols}"
    columns, counts = np.unique(cells[:, 1], return_counts=True)
    assert counts.max() <= 2, "한 열에 셀이 3개 이상 — 톱니가 남아 있다"


def test_three_cell_line_survives_as_one_cell_chain():
    """3칸짜리 선: 채워지는 셀 1개, 두 분기 모두 끝방향 (6.2절 — 선이 소멸하지 않는다)."""
    points = np.array([[40.0, 50.0], [51.0, 50.0]], dtype=np.float32)  # 셀 10~12
    _, target = encode_scene([{"class": 3, "points": points}])
    positive = np.argwhere(target["class_map"] > 0)
    assert len(positive) == 1
    cell = tuple(positive[0])
    assert target["end_map"][cell] == 1.0
    dirs = target["conn_dirs"][cell]
    assert dirs[0] @ dirs[1] < -0.9  # 양쪽 끝점을 향해 서로 반대


def test_synthetic_dataset_sample_contract():
    dataset = SyntheticDataset(
        split="val",
        image_size=256,
        grid_stride=4,
        num_classes=12,
        max_degree=2,
        encode_supersample=1,
        augment=False,
        limit=4,
    )
    sample = dataset[0]
    assert sample["image"].shape == (3, 256, 256)
    assert sample["image"].min() >= 0.0 and sample["image"].max() <= 1.0
    assert sample["class_map"].shape == (64, 64)
    assert sample["conn_dirs"].shape == (64, 64, 2, 2)
    positive = sample["class_map"].numpy() > 0
    assert positive.sum() > 0
    norms = np.linalg.norm(sample["conn_dirs"].numpy()[positive], axis=-1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)
    assert len(sample["instances"]) > 0


def test_encoder_rejects_wrong_degree():
    with pytest.raises(ValueError):
        ChainEncoder(
            image_size=IMAGE_SIZE, grid_stride=4, num_classes=12, max_degree=3, supersample=1
        )

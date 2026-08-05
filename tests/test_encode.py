"""GT 인코더 불변식 검증 (impl_plan 6.4절 불변식 7종)."""

import numpy as np
import pytest

from stella.data.encode import GridEncoder
from stella.data.synthetic import SyntheticDataset

GRID_STRIDE = 4
IMAGE_SIZE = 128
MAX_DEGREE = 3


def make_encoder(**kwargs) -> GridEncoder:
    params = dict(
        image_size=IMAGE_SIZE,
        grid_stride=GRID_STRIDE,
        num_classes=12,
        max_degree=MAX_DEGREE,
        supersample=1,
    )
    params.update(kwargs)
    return GridEncoder(**params)


def horizontal_line(label: int = 3, row: float = 40.5) -> list[dict]:
    points = np.array([[20.0, row], [100.0, row]], dtype=np.float32)
    return [{"class": label, "points": points}]


def encode_scene(instances, **kwargs):
    encoder = make_encoder(**kwargs)
    return encoder, encoder.encode(instances)


def neighbor_pairs(target) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    conn = target["conn_cells"]
    pairs = []
    for i, j, _ in np.argwhere(conn[..., 0] >= 0):
        for slot in range(conn.shape[2]):
            bi, bj = conn[i, j, slot]
            if bi >= 0:
                pairs.append(((i, j), (int(bi), int(bj))))
    return sorted(set(pairs))


def test_simple_line_shapes_and_ends():
    _, target = encode_scene(horizontal_line())
    side = IMAGE_SIZE // GRID_STRIDE
    assert target["class_map"].shape == (side, side)
    assert target["coord_map"].shape == (side, side, 2)
    assert target["conn_cells"].shape == (side, side, MAX_DEGREE, 2)
    row = int(40.5 // GRID_STRIDE)
    owned = np.flatnonzero(target["class_map"][row] > 0)
    assert owned.min() == 5 and owned.max() == 25  # x = 20..100 -> cell 5..25
    assert target["end_map"][row, owned.min()] == 1.0
    assert target["end_map"][row, owned.max()] == 1.0
    assert target["end_map"][row, owned].sum() == 2.0


def test_invariant_targets_positive_and_not_self():
    _, target = encode_scene(horizontal_line())
    for a, b in neighbor_pairs(target):
        assert a != b
        assert target["class_map"][b] > 0


def test_invariant_symmetry_for_same_class_non_end():
    encoder, target = encode_scene(horizontal_line())
    assert encoder.stats["truncated_cells"] == 0
    pairs = set(neighbor_pairs(target))
    for a, b in pairs:
        same_class = target["class_map"][a] == target["class_map"][b]
        if same_class and target["end_map"][b] == 0.0:
            assert (b, a) in pairs, f"{b} -> {a} 가 없다"


def test_invariant_end_cells_have_no_outgoing():
    _, target = encode_scene(horizontal_line())
    ends = np.argwhere(target["end_map"] > 0)
    for i, j in ends:
        assert (target["conn_cells"][i, j] < 0).all()


def test_invariant_degree_within_max():
    _, target = encode_scene(horizontal_line())
    used = (target["conn_cells"][..., 0] >= 0).sum(axis=-1)
    assert used.max() <= MAX_DEGREE


def test_invariant_non_positive_cells_are_empty():
    _, target = encode_scene(horizontal_line())
    background = target["class_map"] == 0
    assert np.all(target["coord_map"][background] == 0)
    assert np.all(target["end_map"][background] == 0)
    assert np.all(target["conn_cells"][background] == -1)


def test_invariant_centroid_inside_cell():
    _, target = encode_scene(horizontal_line())
    positive = target["class_map"] > 0
    coord = target["coord_map"][positive]
    assert coord.min() >= 0.0 and coord.max() < 1.0


def parallel_pair(gap: float, centre_y: float, label: int = 3) -> list[dict]:
    x = np.linspace(20.0, 100.0, 40, dtype=np.float32)
    upper = np.stack([x, np.full_like(x, centre_y - gap / 2)], axis=-1)
    lower = np.stack([x, np.full_like(x, centre_y + gap / 2)], axis=-1)
    return [{"class": label, "points": upper}, {"class": label, "points": lower}]


def test_double_line_inside_one_cell_collapses_to_one_row():
    """같은 셀에 들어오는 이중선은 노드 한 줄로 합쳐지고 무게중심이 두 선 사이에 놓인다 (불변식 7).

    실데이터의 이중 실선 간격은 15~30 cm이고 GSD가 0.1278 m/px라 1~2 px이므로
    대부분 같은 셀 안에 들어온다.
    """
    _, target = encode_scene(parallel_pair(gap=2.0, centre_y=42.0))
    rows = np.unique(np.argwhere(target["class_map"] > 0)[:, 0])
    assert rows.tolist() == [10]
    centre = target["coord_map"][10, 12]
    # 두 선의 중간 y = 42.0(라벨 = 픽셀 인덱스)이고 그 픽셀의 면적 중심은 42.5다.
    # 셀 10은 y in [40, 44)이므로 (42.5 - 40) / 4 = 0.625 — 6.4절의 +0.5 보정 그대로.
    assert centre[1] == pytest.approx((42.5 - 40.0) / GRID_STRIDE, abs=0.02)


def test_double_line_straddling_a_cell_border_stays_two_rows():
    """셀 경계를 걸치는 이중선은 두 줄로 남는다 — 무게중심은 셀 안에서만 평균되기 때문."""
    _, target = encode_scene(parallel_pair(gap=4.0, centre_y=40.0))
    rows = np.unique(np.argwhere(target["class_map"] > 0)[:, 0])
    assert rows.tolist() == [9, 10]


def test_cross_class_junction_is_one_way():
    """다른 클래스 T자 접합: 끝나는 선만 접합 셀을 가리킨다 (불변식 3)."""
    main = np.array([[10.0, 60.5], [118.0, 60.5]], dtype=np.float32)
    stem = np.array([[64.5, 20.0], [64.5, 60.5]], dtype=np.float32)
    _, target = encode_scene([{"class": 3, "points": main}, {"class": 5, "points": stem}])
    pairs = set(neighbor_pairs(target))
    cross = [(a, b) for a, b in pairs if target["class_map"][a] != target["class_map"][b]]
    assert cross, "다른 클래스 접합 간선이 하나도 없다"
    for a, b in cross:
        assert (b, a) not in pairs


def test_synthetic_dataset_sample_contract():
    dataset = SyntheticDataset(
        split="val",
        image_size=256,
        grid_stride=4,
        num_classes=12,
        max_degree=3,
        encode_supersample=1,
        augment=False,
        limit=4,
    )
    sample = dataset[0]
    assert sample["image"].shape == (3, 256, 256)
    assert sample["image"].min() >= 0.0 and sample["image"].max() <= 1.0
    assert sample["class_map"].shape == (64, 64)
    assert sample["conn_cells"].shape == (64, 64, 3, 2)
    assert int((sample["class_map"] > 0).sum()) > 0
    assert len(sample["instances"]) > 0

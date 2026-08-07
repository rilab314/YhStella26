"""시각 로그 함수의 shape·색상 규약 검증 (impl_plan 9.5절)."""

import numpy as np

from stella.data.types import CLASS_COLOR, SLOT_COLOR
from stella.train import viz

IMAGE = np.full((32, 32, 3), 0.5, dtype=np.float32)


def test_heatmap_shape_dtype_and_blend_direction():
    probability = np.zeros((8, 8), np.float32)
    probability[2, 3] = 1.0
    page = viz.draw_heatmap(IMAGE, probability, alpha=0.5)
    assert page.shape == (32, 32, 3) and page.dtype == np.uint8
    hot = page[2 * 4 + 1, 3 * 4 + 1]
    cold = page[20, 20]
    assert hot[0] > hot[2]  # 확률 1 -> 빨강 우세
    assert cold[2] > cold[0]  # 확률 0 -> 파랑 우세


def test_class_map_paints_cell_center_only():
    class_ids = np.zeros((8, 8), np.int64)
    class_ids[4, 4] = 3
    page = viz.draw_class_map(IMAGE, class_ids, class_ids > 0, stride=4)
    assert tuple(page[4 * 4 + 1, 4 * 4 + 1]) == CLASS_COLOR[3]  # 중심 2x2
    assert tuple(page[4 * 4, 4 * 4]) != CLASS_COLOR[3]  # 셀 모서리는 칠하지 않는다


def test_slots_draw_self_point_and_direction_line():
    coord = np.zeros((8, 8, 2), np.float32)
    coord[2, 2] = (0.5, 0.5)
    directions = np.zeros((8, 8, 2, 2), np.float32)
    directions[2, 2, 0] = (1.0, 0.0)
    exist = np.zeros((8, 8, 2), np.float32)
    exist[2, 2, 0] = 1.0
    mask = np.zeros((8, 8), bool)
    mask[2, 2] = True
    page = viz.draw_slots(IMAGE, coord, directions, exist, mask, stride=4)
    assert tuple(page[11, 10]) == (0, 0, 0)  # 자기 점 (10, 10) 주변의 검은 원
    assert tuple(page[10, 14]) == SLOT_COLOR[0]  # 자기 점에서 +x 방향선 (원점 규약 6.1절)
    assert tuple(page[10, 4]) != SLOT_COLOR[1]  # 존재 낮은 슬롯은 그리지 않는다


def test_draw_instances_uses_class_color():
    instances = [{"class": 5, "points": np.array([[2.0, 16.0], [30.0, 16.0]], np.float32)}]
    page = viz.draw_instances(IMAGE, instances)
    # LINE_AA가 색을 약간 섞으므로 근사 비교한다
    assert np.abs(page[16, 16].astype(int) - CLASS_COLOR[5]).max() < 30

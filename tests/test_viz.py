"""시각 로그 함수의 shape·색상 규약 검증 (design 9.5절)."""

import numpy as np
import torch

from stella.data.types import CLASS_COLOR, SLOT_COLOR
from stella.model.stella import ModelOutput
from stella.train import viz

IMAGE = np.full((32, 32, 3), 0.5, dtype=np.float32)


def _blank_output(side: int = 8, classes: int = 4, slots: int = 2) -> ModelOutput:
    return ModelOutput(
        heatmap_logit=torch.full((side, side), -10.0),
        node_mask=torch.zeros((side, side), dtype=torch.bool),
        class_logit=torch.zeros((side, side, classes)),
        self_coord=torch.zeros((side, side, 2)),
        end_logit=torch.zeros((side, side)),
        fg_logit=torch.zeros((side, side)),
        exist_logit=torch.zeros((side, side, slots)),
        conn_dir=torch.zeros((side, side, slots, 2)),
    )


def test_page_renderer_makes_one_sheet_of_six_pages():
    """콜백과 캐시 스크립트가 공유하는 진입점 — 프레임 하나가 파일 하나가 된다."""
    renderer = viz.PageRenderer(
        grid_stride=4, heat_alpha=0.5, slot_line_len=6.0, exist_thresh=0.5, class_thresh=0.5
    )
    pages = renderer.build_pages(IMAGE, _blank_output(), [], [])
    assert list(pages) == ["heat", "class", "slot", "end", "gt", "inst"]
    sheet = renderer.render(IMAGE, _blank_output(), [], [])
    assert sheet.shape == (32 * 2 + viz.TILE_GAP, 32 * 3 + viz.TILE_GAP * 2, 3)


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
    page = viz.draw_instances(IMAGE, instances, endpoint_radius=0)
    # LINE_AA가 색을 약간 섞으므로 근사 비교한다
    assert np.abs(page[16, 16].astype(int) - CLASS_COLOR[5]).max() < 30


def test_draw_instances_marks_both_endpoints():
    """선의 범위를 눈으로 끊어 주는 표시 — 끝점에만 찍히고 중간에는 없다."""
    instances = [{"class": 5, "points": np.array([[6.0, 16.0], [26.0, 16.0]], np.float32)}]
    plain = viz.draw_instances(IMAGE, instances, endpoint_radius=0)
    marked = viz.draw_instances(IMAGE, instances, endpoint_radius=4)
    for x in (6, 26):  # 끝점 위쪽 3픽셀 — 두께 1 선은 닿지 않는 자리
        assert not np.array_equal(marked[13, x], plain[13, x])
    assert np.array_equal(marked[13, 16], plain[13, 16])  # 선 중간은 그대로


def test_tile_pages_lays_six_pages_in_two_by_three():
    pages = {name: np.full((32, 32, 3), value, np.uint8) for value, name in enumerate("abcdef")}
    sheet = viz.tile_pages(pages)
    height = 32 * 2 + viz.TILE_GAP
    width = 32 * 3 + viz.TILE_GAP * 2
    assert sheet.shape == (height, width, 3)
    assert tuple(sheet[16, 32]) == viz.TILE_GAP_COLOR  # 첫 행의 첫 구분선
    assert tuple(sheet[2, 2]) == (0, 0, 0)  # 첫 페이지(value 0), 이름표가 닿지 않는 모서리


def test_tile_pages_pads_missing_cell():
    pages = {name: np.full((16, 16, 3), 7, np.uint8) for name in "abcd"}
    sheet = viz.tile_pages(pages)
    assert sheet.shape == (16 * 2 + viz.TILE_GAP, 16 * 3 + viz.TILE_GAP * 2, 3)
    assert tuple(sheet[-1, -1]) == (0, 0, 0)  # 빈 칸은 검다

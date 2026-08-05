"""폴리라인 -> 격자 GT 인코더 (impl_plan 6.4절).

두 갈래로 나뉜다. **위치·클래스는 래스터에서, 연결성은 폴리라인 위상에서** 나온다.
셀의 값은 인스턴스가 아니라 픽셀에서 정한다 — 이중선을 두 인스턴스로 라벨링한
실데이터에서 노드가 두 줄기로 갈라지지 않게 하기 위함이다.
"""

from collections import defaultdict

import cv2
import numpy as np

from stella.data.types import rarity_order

DENSIFY_STEP_PX = 1.0  # 셀 열 추출용 샘플 간격. 셀 크기보다 작아야 셀을 건너뛰지 않는다


class GridEncoder:
    def __init__(
        self,
        *,
        image_size: int,
        grid_stride: int,
        num_classes: int,
        max_degree: int,
        supersample: int = 1,
        min_cell_pixels: int = 1,
    ):
        self.min_cell_pixels = min_cell_pixels
        self.image_size = image_size
        self.grid_stride = grid_stride
        self.num_classes = num_classes
        self.max_degree = max_degree
        self.supersample = supersample
        self.grid_size = image_size // grid_stride
        self.cell_pixels = grid_stride * supersample
        self.rarity = rarity_order(num_classes)
        self.stats = defaultdict(int)

    def encode(self, instances: list[dict]) -> dict[str, np.ndarray]:
        """instances: [{"class": int(1..C-1), "points": (P,2) float32 픽셀좌표}]"""
        class_map, coord_map = self._raster_stage(instances)
        same_edges, cross_edges = self._topology_stage(instances, class_map)
        end_map, conn_cells = self._build_conn_arrays(class_map, same_edges, cross_edges)
        self.stats["nodes"] += int((class_map > 0).sum())
        return {
            "class_map": class_map,
            "coord_map": coord_map,
            "end_map": end_map,
            "conn_cells": conn_cells,
        }

    # --- (A) 래스터 단계 -> class_map, coord_map ---------------------------------

    def _raster_stage(self, instances: list[dict]) -> tuple[np.ndarray, np.ndarray]:
        n_cells = self.grid_size**2
        counts = np.zeros((len(self.rarity), n_cells), dtype=np.float32)
        sum_x = np.zeros_like(counts)
        sum_y = np.zeros_like(counts)
        for k, label in enumerate(self.rarity):
            polylines = [i["points"] for i in instances if i["class"] == label]
            if not polylines:
                continue
            mask = self._draw_class_mask(polylines)
            counts[k], sum_x[k], sum_y[k] = self._cell_statistics(mask)
        return self._resolve_ownership(counts, sum_x, sum_y)

    def _draw_class_mask(self, polylines: list[np.ndarray]) -> np.ndarray:
        """클래스 c의 모든 폴리라인을 인스턴스 구분 없이 캔버스 하나에 두께 1로 그린다."""
        size = self.image_size * self.supersample
        mask = np.zeros((size, size), dtype=np.uint8)
        for points in polylines:
            if len(points) < 2:
                continue
            canvas_pts = np.round(np.asarray(points, np.float64) * self.supersample)
            cv2.polylines(mask, [canvas_pts.astype(np.int32)], False, 1, 1)
        return mask

    def _cell_statistics(self, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """셀별 픽셀 수와, 셀 내 오프셋 합(무게중심 계산용)."""
        n_cells = self.grid_size**2
        flat_idx = np.flatnonzero(mask)
        if flat_idx.size == 0:
            zeros = np.zeros(n_cells, dtype=np.float32)
            return zeros, zeros.copy(), zeros.copy()
        py, px = np.divmod(flat_idx, mask.shape[1])
        sb = self.cell_pixels
        cell = (py // sb) * self.grid_size + (px // sb)
        offset_x = (px % sb + 0.5) / sb
        offset_y = (py % sb + 0.5) / sb
        count = np.bincount(cell, minlength=n_cells).astype(np.float32)
        acc_x = np.bincount(cell, weights=offset_x, minlength=n_cells).astype(np.float32)
        acc_y = np.bincount(cell, weights=offset_y, minlength=n_cells).astype(np.float32)
        return count, acc_x, acc_y

    def _resolve_ownership(
        self, counts: np.ndarray, sum_x: np.ndarray, sum_y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """지배 클래스 = 셀에 픽셀이 더 많은 클래스. 동점이면 희소 클래스가 이긴다."""
        winner = counts.argmax(axis=0)  # rarity 순 배열이라 첫 최대값 = 희소 클래스
        cells = np.arange(counts.shape[1])
        top = counts[winner, cells]
        # 셀을 스치기만 한 선(1~2 px)은 위상 단계의 셀 열에 안 잡혀 고립 노드가 된다.
        # 그래서 최소 픽셀 수로 걸러 봤지만 **오히려 나빠졌다** (SEED-MAP val 6장, GT 주입):
        #   min_px=1: F1 0.686, 조각/GT 1.69, 고립 7.8%
        #   min_px=2: F1 0.557, 조각/GT 2.07, 고립 4.8%
        #   min_px=3: F1 0.393, 조각/GT 2.87, 고립 2.2%
        # 걸러진 셀 상당수가 선의 끝이나 사슬 중간이라 연결이 더 끊긴다. 기본값은 1로 둔다.
        positive = top >= self.min_cell_pixels * self.supersample**2
        label = np.where(positive, self.rarity[winner], 0).astype(np.int64)
        denom = np.where(positive, top, 1.0)
        coord = np.stack([sum_x[winner, cells] / denom, sum_y[winner, cells] / denom], axis=-1)
        coord = np.where(positive[:, None], coord, 0.0).astype(np.float32)
        side = self.grid_size
        return label.reshape(side, side), np.clip(coord, 0.0, 0.999999).reshape(side, side, 2)

    # --- (B) 위상 단계 -> conn_cells, end_map ------------------------------------

    def _topology_stage(
        self, instances: list[dict], class_map: np.ndarray
    ) -> tuple[set, dict[int, set]]:
        """같은 클래스 간선(합집합)과 다른 클래스 종단 간선(단방향)을 모은다."""
        flat_class = class_map.reshape(-1)
        same_edges: set[tuple[int, int]] = set()
        cross_edges: dict[int, set[int]] = defaultdict(set)
        for inst in instances:
            raw = self._cell_sequence(inst["points"])
            if raw.size == 0:
                continue
            kept = _dedup_consecutive(raw[flat_class[raw] == inst["class"]])
            for a, b in zip(kept[:-1], kept[1:]):
                same_edges.add((int(min(a, b)), int(max(a, b))))
            self._add_cross_edges(raw, kept, inst["class"], flat_class, cross_edges)
        return same_edges, cross_edges

    def _cell_sequence(self, points: np.ndarray) -> np.ndarray:
        """폴리라인이 지나는 셀을 순서대로. 연속 중복은 제거한다."""
        dense = _densify(np.asarray(points, np.float64), DENSIFY_STEP_PX)
        if dense.shape[0] == 0:
            return np.zeros(0, dtype=np.int64)
        cell = np.floor(dense / self.grid_stride).astype(np.int64)
        np.clip(cell, 0, self.grid_size - 1, out=cell)
        return _dedup_consecutive(cell[:, 1] * self.grid_size + cell[:, 0])

    def _add_cross_edges(
        self,
        raw: np.ndarray,
        kept: np.ndarray,
        label: int,
        flat_class: np.ndarray,
        cross_edges: dict[int, set[int]],
    ) -> None:
        """끝나는 선의 마지막 소유 셀 -> 다른 클래스 접합 셀 (단방향, 6.4절 위상 10)."""
        if kept.size == 0:
            return
        owned = flat_class[raw] == label
        positions = np.flatnonzero(owned)
        tail = _first_other_class(raw[positions[-1] + 1 :], flat_class, label)
        if tail is not None:
            cross_edges[int(kept[-1])].add(tail)
        head = _first_other_class(raw[: positions[0]][::-1], flat_class, label)
        if head is not None:
            cross_edges[int(kept[0])].add(head)

    def _build_conn_arrays(
        self, class_map: np.ndarray, same_edges: set, cross_edges: dict[int, set]
    ) -> tuple[np.ndarray, np.ndarray]:
        adjacency = defaultdict(list)
        for a, b in same_edges:
            adjacency[a].append(b)
            adjacency[b].append(a)
        side, degree = self.grid_size, self.max_degree
        end_map = np.zeros(side * side, dtype=np.float32)
        conn = np.full((side * side, degree, 2), -1, dtype=np.int64)
        for cell in np.flatnonzero(class_map.reshape(-1) > 0):
            neighbors, is_end = self._cell_neighbors(int(cell), adjacency, cross_edges)
            end_map[cell] = float(is_end)
            for slot, target in enumerate(neighbors):
                conn[cell, slot] = (target // side, target % side)
        return end_map.reshape(side, side), conn.reshape(side, side, degree, 2)

    def _cell_neighbors(
        self, cell: int, adjacency: dict[int, list], cross_edges: dict[int, set]
    ) -> tuple[list[int], bool]:
        """종점 셀은 나가는 연결이 없다. 다른 클래스로 종단하는 셀은 종점이 아니다."""
        same = adjacency.get(cell, [])
        cross = sorted(cross_edges.get(cell, ()))
        is_end = len(same) == 1 and not cross
        neighbors = ([] if is_end else list(same)) + cross
        if len(neighbors) > self.max_degree:
            neighbors = self._nearest(cell, neighbors)
            self.stats["truncated_cells"] += 1
        self.stats[f"degree_{min(len(same) + len(cross), 9)}"] += 1
        return neighbors, is_end

    def _nearest(self, cell: int, neighbors: list[int]) -> list[int]:
        side = self.grid_size
        origin = np.array([cell // side, cell % side])
        coords = np.array([[n // side, n % side] for n in neighbors])
        order = np.argsort(np.abs(coords - origin).max(axis=1), kind="stable")
        return [neighbors[k] for k in order[: self.max_degree]]


def _densify(points: np.ndarray, step: float) -> np.ndarray:
    """폴리라인을 일정 간격으로 촘촘히 샘플한다 (마지막 점 포함)."""
    if points.shape[0] < 2:
        return points.copy()
    delta = np.diff(points, axis=0)
    counts = np.maximum(np.ceil(np.linalg.norm(delta, axis=1) / step), 1).astype(np.int64)
    seg = np.repeat(np.arange(delta.shape[0]), counts)
    starts = np.repeat(np.cumsum(counts) - counts, counts)
    ratio = (np.arange(seg.shape[0]) - starts) / counts[seg]
    dense = points[seg] + delta[seg] * ratio[:, None]
    return np.concatenate([dense, points[-1:]], axis=0)


def _dedup_consecutive(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    keep = np.ones(values.shape[0], dtype=bool)
    keep[1:] = values[1:] != values[:-1]
    return values[keep]


def _first_other_class(cells: np.ndarray, flat_class: np.ndarray, label: int) -> int | None:
    """셀 열에서 처음 만나는 '다른 클래스의 양성 셀'."""
    if cells.size == 0:
        return None
    other = flat_class[cells]
    hit = np.flatnonzero((other > 0) & (other != label))
    return int(cells[hit[0]]) if hit.size else None

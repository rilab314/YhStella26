"""폴리라인 -> 격자 GT 인코더 (design 6.4절, 9·10차 개정).

**선 하나 = 사슬 하나.** 선마다 따로 그려 셀 열을 만들고(래스터 단계), 소유권 경쟁에서
남은 셀만 순서대로 이어(위상 단계) **연결 방향 2개를 직접 저장**한다. 셀의 소유권 다툼은
선 사이에서만 일어나고, 사슬 위상은 각 선이 독자적으로 갖는다.

끝 규약(6.2절): 선이 지나는 셀 열의 양 끝 칸은 채우지 않는다. 사슬 첫·끝 셀이 끝 셀이고
(`end_map = 1`), 그 둘째 분기는 이웃이 아니라 **선의 실제 끝점**을 향한다.
"""

from collections import defaultdict

import cv2
import numpy as np

from stella.data.types import rarity_order

DENSIFY_STEP_PX = 1.0  # 셀 열 추출용 샘플 간격. 셀 크기보다 작아야 셀을 건너뛰지 않는다
RANK_RULE_LIMIT = 3  # 3x3 이웃 중 픽셀 수가 더 많은 셀이 이 수 이상이면 지운다 (= 4등 이하)
NORM_EPS = 1e-6


class ChainEncoder:
    """입력: 폴리라인 인스턴스 목록(픽셀 좌표 + 클래스). 출력: 6.2절 targets."""

    def __init__(
        self,
        *,
        image_size: int,
        grid_stride: int,
        num_classes: int,
        max_degree: int,
        supersample: int = 1,
        conn_lookahead: int = 1,
    ):
        if max_degree != 2:
            raise ValueError(f"선 단위 사슬은 분기가 항상 2다 (6.4절): max_degree={max_degree}")
        if conn_lookahead < 1:
            raise ValueError(f"conn_lookahead 는 1 이상이어야 한다: {conn_lookahead}")
        self.image_size = image_size
        self.grid_stride = grid_stride
        self.num_classes = num_classes
        self.supersample = supersample
        self.conn_lookahead = conn_lookahead
        self.grid_size = image_size // grid_stride
        self.cell_pixels = grid_stride * supersample
        self.rarity_position = _rarity_position(num_classes)
        self.stats = defaultdict(int)

    def encode(self, instances: list[dict]) -> dict[str, np.ndarray]:
        """instances: [{"class": int(1..C-1), "points": (P,2) float32 픽셀좌표}]"""
        traces = [t for t in map(self._trace_line, instances) if t is not None]
        owner, coord = self._resolve_ownership(traces)
        class_map = self._class_map(traces, owner)
        length_map = self._length_map(traces, owner)
        end_map, conn_dirs = self._topology_stage(traces, owner, coord)
        side = self.grid_size
        self.stats["lines"] += len(instances)
        self.stats["nodes"] += int((class_map > 0).sum())
        return {
            "class_map": class_map.reshape(side, side),
            "coord_map": np.clip(coord, 0.0, 0.999999).reshape(side, side, 2),
            "end_map": end_map.reshape(side, side),
            "conn_dirs": conn_dirs.reshape(side, side, 2, 2),
            "length_map": length_map.reshape(side, side),
        }

    # --- (A) 선별 래스터 -> 셀 통계·셀 열, 끝칸·잉여 셀 제거 --------------------------

    def _trace_line(self, inst: dict) -> dict | None:
        points = np.asarray(inst["points"], np.float64)
        if points.shape[0] < 2:
            return None
        full_sequence = self._cell_sequence(points)
        if full_sequence.size <= 2:  # 끝칸을 지우면 남는 게 없다 — 선이 소멸한다
            self.stats["lines_vanished"] += 1
            return None
        sequence = _dedup_keep_first(full_sequence[1:-1])  # 끝칸 미채움 + 자기 교차 무시
        cells, counts, sums = self._pixel_statistics(points)
        live = ~np.isin(cells, full_sequence[[0, -1]])  # 끝칸 픽셀은 경쟁 불참 (A-2)
        cells, counts, sums = cells[live], counts[live], sums[live]
        kept = self._rank_rule_mask(cells, counts)
        claim = kept & np.isin(cells, sequence)  # 셀 열에 없는 스침 셀은 소유권을 주장하지 않는다
        self.stats["cells_scraped"] += int((kept & ~np.isin(cells, sequence)).sum())
        return {
            "label": int(inst["class"]),
            "cells": cells[claim],
            "counts": counts[claim],
            "sums": sums[claim],
            "sequence": sequence[np.isin(sequence, cells[claim])],
            "head": (points[0] + 0.5) / self.grid_stride,
            "tail": (points[-1] + 0.5) / self.grid_stride,
        }

    def _cell_sequence(self, points: np.ndarray) -> np.ndarray:
        """폴리라인이 지나는 셀을 순서대로 (연속 중복 제거). 래스터와 같은 픽셀 반올림을 쓴다."""
        dense = _densify(points, DENSIFY_STEP_PX)
        pixel = np.round(dense * self.supersample).astype(np.int64)
        np.clip(pixel, 0, self.image_size * self.supersample - 1, out=pixel)
        cell = pixel // self.cell_pixels
        return _dedup_consecutive(cell[:, 1] * self.grid_size + cell[:, 0])

    def _pixel_statistics(self, points: np.ndarray) -> tuple[np.ndarray, ...]:
        """선 하나를 두께 1로 그려 셀별 픽셀 수와 셀 내 오프셋 합을 센다 (bbox 크롭)."""
        canvas_pts = np.round(points * self.supersample).astype(np.int64)
        limit = self.image_size * self.supersample - 1
        np.clip(canvas_pts, 0, limit, out=canvas_pts)
        origin = canvas_pts.min(axis=0)
        mask = np.zeros(tuple(canvas_pts.max(axis=0) - origin + 1)[::-1], dtype=np.uint8)
        cv2.polylines(mask, [(canvas_pts - origin).astype(np.int32)], False, 1, 1)
        ys, xs = np.nonzero(mask)
        px, py = xs + origin[0], ys + origin[1]
        cells, inverse = np.unique(
            (py // self.cell_pixels) * self.grid_size + (px // self.cell_pixels),
            return_inverse=True,
        )
        counts = np.bincount(inverse).astype(np.float32)
        sums = np.stack(
            [
                np.bincount(inverse, weights=(px % self.cell_pixels + 0.5) / self.cell_pixels),
                np.bincount(inverse, weights=(py % self.cell_pixels + 0.5) / self.cell_pixels),
            ],
            axis=-1,
        ).astype(np.float32)
        return cells, counts, sums

    def _rank_rule_mask(self, cells: np.ndarray, counts: np.ndarray) -> np.ndarray:
        """3x3 순위 규칙 — 이웃한 자기 셀 중 픽셀이 더 많은 셀이 3개 이상이면 지운다 (A-3).

        모서리를 1~2 px 스치는 잉여 셀(고립 노드의 원인)과 지그재그 사슬을 함께 없앤다.
        제거는 원래 픽셀 수 기준으로 동시에 판정한다(순차 제거 아님).
        """
        if cells.size == 0:
            return np.zeros(0, dtype=bool)
        rows, cols = np.divmod(cells, self.grid_size)
        row0, col0 = rows.min() - 1, cols.min() - 1
        grid = np.zeros((rows.max() - row0 + 2, cols.max() - col0 + 2), np.float32)
        grid[rows - row0, cols - col0] = counts
        greater = np.zeros(cells.size, np.int64)
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di or dj:
                    greater += grid[rows - row0 + di, cols - col0 + dj] > counts
        return greater < RANK_RULE_LIMIT

    # --- 셀 소유권 (선 사이) -> class_map, coord_map --------------------------------

    def _resolve_ownership(self, traces: list[dict]) -> tuple[np.ndarray, np.ndarray]:
        """픽셀이 더 많은 선이 소유한다. 동점은 희소 클래스 -> 앞선 인스턴스 순 (6.4절 A-4).

        희소 우선순위 순서로 돌며 엄격 부등호로 갱신하면 동점 규칙이 그대로 나온다.
        """
        n_cells = self.grid_size**2
        owner = np.full(n_cells, -1, np.int64)
        best = np.zeros(n_cells, np.float32)
        coord = np.zeros((n_cells, 2), np.float32)
        priority = sorted(
            range(len(traces)), key=lambda k: (self.rarity_position[traces[k]["label"]], k)
        )
        for index in priority:
            trace = traces[index]
            win = trace["counts"] > best[trace["cells"]]
            chosen = trace["cells"][win]
            owner[chosen] = index
            best[chosen] = trace["counts"][win]
            coord[chosen] = trace["sums"][win] / trace["counts"][win, None]
        return owner, coord

    def _class_map(self, traces: list[dict], owner: np.ndarray) -> np.ndarray:
        labels = np.array([t["label"] for t in traces] + [0], dtype=np.int64)
        return np.where(owner >= 0, labels[owner], 0)

    def _length_map(self, traces: list[dict], owner: np.ndarray) -> np.ndarray:
        """셀이 속한 **선의 길이**(소유 셀 수). 배경은 0.

        손실은 셀 단위인데 지표는 선 단위다 — 100칸 선은 100표, 7칸 선은 7표를 갖는다.
        실측(08-20): 20칸 미만 선이 정답 선의 46%인데 셀로는 13.6% 뿐이고, 그 구간의
        정점 검출률이 0.339 로 70칸 이상(0.595)의 57% 에 그친다. 손실 쪽에서 그 불균형을
        보려면 셀마다 자기 선의 길이를 알아야 하므로 여기서 실어 보낸다.
        """
        counts = np.array([len(t["cells"]) for t in traces] + [0], dtype=np.float32)
        return np.where(owner >= 0, counts[owner], 0.0)

    # --- (B) 선별 위상 -> conn_dirs, end_map ----------------------------------------

    def _topology_stage(
        self, traces: list[dict], owner: np.ndarray, coord: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        n_cells = self.grid_size**2
        end_map = np.zeros(n_cells, np.float32)
        conn_dirs = np.zeros((n_cells, 2, 2), np.float32)
        for index, trace in enumerate(traces):
            chain = trace["sequence"][owner[trace["sequence"]] == index]
            self.stats["cells_lost"] += int(trace["sequence"].size - chain.size)
            if chain.size == 0:
                self.stats["lines_vanished"] += 1
                continue
            rows, cols = np.divmod(chain, self.grid_size)
            points = np.stack([cols, rows], axis=-1) + coord[chain]
            conn_dirs[chain] = self._chain_dirs(points, trace["head"], trace["tail"])
            end_map[chain[0]] = end_map[chain[-1]] = 1.0
            self._record_chain_stats(chain, rows, cols)
        return end_map, conn_dirs

    def _chain_dirs(self, points: np.ndarray, head: np.ndarray, tail: np.ndarray) -> np.ndarray:
        """(n, 2) 점 열 -> (n, 2, 2) 분기 방향 — 자기 점에서 `conn_lookahead`칸 앞뒤로.

        n = 1(3칸짜리 선)이면 두 분기가 모두 끝점 방향이 된다 — 일반식이 그대로 처리한다.

        **왜 lookahead가 있나.** 이웃 칸으로의 접선은 스텝마다 흔들린다 — 실측(val 40장 ·
        사슬 1,475개) 평균 5.9도 · 90%분위 14.0도 · **>20도가 4.4%**. 점이 서브셀 좌표를
        갖기 때문에 격자 양자화만큼 심하지는 않다. 그래도 디코더는 이 방향을 반경 끝까지
        외삽하고, 스텝당 4.4%라도 길이 45 사슬에서는 사슬당 평균 2회 끊긴다.
        4셀이면 평균 1.6도 · >20도 0.5%.
        """
        dirs = np.zeros((points.shape[0], 2, 2), np.float32)
        unit = _unit_rows(np.diff(points, axis=0))
        forward, backward = self._branch_targets(points)
        dirs[:-1, 1] = _unit_rows(points[forward[:-1]] - points[:-1])
        dirs[1:, 0] = _unit_rows(points[backward[1:]] - points[1:])
        if points.shape[0] >= 2:
            head_fallback, tail_fallback = -unit[0], unit[-1]
        else:
            axis = _unit_one(tail - head, np.array([1.0, 0.0]))
            head_fallback, tail_fallback = -axis, axis
        dirs[0, 0] = _unit_one(head - points[0], head_fallback)
        dirs[-1, 1] = _unit_one(tail - points[-1], tail_fallback)
        return dirs

    def _branch_targets(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """각 점이 앞·뒤로 바라볼 상대 점의 인덱스. 사슬 끝에서는 끝 점으로 잘린다."""
        count = points.shape[0]
        step = min(self.conn_lookahead, max(count - 1, 1))
        index = np.arange(count)
        return np.minimum(index + step, count - 1), np.maximum(index - step, 0)

    def _record_chain_stats(self, chain: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> None:
        """M10 통계 — 사슬 길이, 1셀 사슬, 건너뛴 간선(체비셰프 거리 >= 2)."""
        self.stats["chains"] += 1
        self.stats["chain_cells"] += int(chain.size)
        self.stats["one_cell_chains"] += int(chain.size == 1)
        if chain.size >= 2:
            steps = np.maximum(np.abs(np.diff(rows)), np.abs(np.diff(cols)))
            self.stats["edges"] += int(steps.size)
            self.stats["skip_edges"] += int((steps >= 2).sum())


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


def _dedup_keep_first(values: np.ndarray) -> np.ndarray:
    """자기 교차 선 — 같은 셀의 뒤 등장은 무시한다 (design 14절)."""
    _, first = np.unique(values, return_index=True)
    return values[np.sort(first)]


def _unit_rows(vectors: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return (vectors / np.maximum(norm, NORM_EPS)).astype(np.float32)


def _unit_one(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    """단위벡터. 끝점이 노드 점과 겹치면(사실상 없음) 직진 연장 방향으로 대체한다."""
    norm = float(np.linalg.norm(vector))
    if norm > NORM_EPS:
        return (vector / norm).astype(np.float32)
    return np.asarray(fallback, np.float32)


def _rarity_position(num_classes: int) -> np.ndarray:
    """클래스 -> 희소 우선순위 (0 = 가장 희소). 소유권 동점 판정용."""
    order = rarity_order(num_classes)
    position = np.zeros(num_classes, dtype=np.int64)
    position[order] = np.arange(order.size)
    return position

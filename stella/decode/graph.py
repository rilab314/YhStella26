"""객체 생성(디코딩) — 셀 단위 예측을 폴리라인 객체로 (impl_plan 10절).

    ① 정점 추출 -> ② 간선 후보 -> ③ 양방향 확인 -> ④ 경로 추출

좌표 규약: 내부 계산은 전부 **격자 단위**로 한다(학습 때 GT를 만든 공간과 같다).
반환 직전에만 픽셀로 바꾼다 — 격자 좌표 p는 픽셀 인덱스 p*s - 0.5 에 대응한다
(인코더가 픽셀의 면적 중심 +0.5 를 쓰기 때문, 6.4절 A-4).
"""

from collections import defaultdict

import numpy as np
import torch

PIXEL_CENTER_SHIFT = 0.5


class GraphDecoder:
    @classmethod
    def from_cfg(cls, module_cfg, cfg, **kwargs) -> "GraphDecoder":
        fields = {
            f: getattr(module_cfg, f)
            for f in (
                "heatmap_thresh",
                "exist_thresh",
                "t_thresh",
                "max_conn_dist",
                "cos_thresh",
                "w_cos",
                "w_dist",
                "w_class",
                "mutual",
                "min_points",
                "simplify_tol",
            )
        }
        return cls(grid_stride=cfg.data.grid_stride, grid_size=cfg.data.grid_size, **fields)

    def __init__(
        self,
        *,
        grid_stride: int,
        grid_size: int,
        heatmap_thresh: float,
        exist_thresh: float,
        t_thresh: float,
        max_conn_dist: float,
        cos_thresh: float,
        w_cos: float,
        w_dist: float,
        w_class: float,
        mutual: bool,
        min_points: int,
        simplify_tol: float,
    ):
        self.grid_stride = grid_stride
        self.grid_size = grid_size
        self.heatmap_thresh = heatmap_thresh
        self.exist_thresh = exist_thresh
        self.t_thresh = t_thresh
        self.max_conn_dist = max_conn_dist
        self.cos_thresh = cos_thresh
        self.w_cos = w_cos
        self.w_dist = w_dist
        self.w_class = w_class
        self.mutual = mutual
        self.min_points = min_points
        self.simplify_tol = simplify_tol

    def __call__(self, output) -> list[dict]:
        vertices = self._extract_vertices(output)
        if vertices["cells"].shape[0] == 0:
            return []
        directed = self._edge_candidates(vertices, output)
        edges, terminal = self._clean_graph(directed, vertices)
        return self._extract_paths(vertices, edges, terminal)

    # --- ① 정점 추출 -------------------------------------------------------------

    def _extract_vertices(self, output) -> dict:
        arrays = {k: _to_numpy(v) for k, v in vars(output).items()}
        probability = _sigmoid(arrays["heatmap_logit"])
        label = arrays["class_logit"].argmax(axis=-1)
        keep = arrays["node_mask"] & (probability > self.heatmap_thresh) & (label > 0)
        cells = np.argwhere(keep)
        rows, cols = cells[:, 0], cells[:, 1]
        coord = arrays["self_coord"][rows, cols]
        class_prob = _softmax(arrays["class_logit"][rows, cols]).max(axis=-1)
        return {
            "cells": cells,
            "point": np.stack([cols + coord[:, 0], rows + coord[:, 1]], axis=-1),
            "origin": np.stack([cols + 0.5, rows + 0.5], axis=-1),
            "label": label[rows, cols],
            "score": probability[rows, cols] * class_prob,
            "exist": _sigmoid(arrays["exist_logit"][rows, cols]),
            "dir": arrays["conn_dir"][rows, cols],
            "terminal": _sigmoid(arrays["t_logit"][rows, cols]) > self.t_thresh,
        }

    # --- ② 간선 후보 생성 --------------------------------------------------------

    def _edge_candidates(self, vertices: dict, output) -> dict[tuple[int, int], bool]:
        """정점 a의 슬롯 k가 가리키는 상대를 방향+거리+클래스 비용으로 고른다."""
        candidates = self._neighbor_index(vertices["cells"])
        unit, distance, usable = self._candidate_geometry(vertices, candidates)
        cost = self._candidate_cost(vertices, candidates, unit, distance, usable)
        directed: dict[tuple[int, int], bool] = {}
        best = cost.argmin(axis=-1)
        for node in range(cost.shape[0]):
            for slot in range(cost.shape[1]):
                column = best[node, slot]
                if not np.isfinite(cost[node, slot, column]):
                    continue
                target = int(candidates[node, column])
                is_terminal = bool(vertices["terminal"][node, slot])
                directed[(node, target)] = directed.get((node, target), False) or is_terminal
        return directed

    def _neighbor_index(self, cells: np.ndarray) -> np.ndarray:
        radius = int(np.ceil(self.max_conn_dist))
        grid = np.full((self.grid_size, self.grid_size), -1, dtype=np.int64)
        grid[cells[:, 0], cells[:, 1]] = np.arange(cells.shape[0])
        padded = np.pad(grid, radius, constant_values=-1)
        span = np.arange(2 * radius + 1)
        rows = cells[:, 0:1] + span
        cols = cells[:, 1:2] + span
        return padded[rows[:, :, None], cols[:, None, :]].reshape(cells.shape[0], -1)

    @staticmethod
    def _candidate_geometry(vertices: dict, candidates: np.ndarray):
        index = np.maximum(candidates, 0)
        delta = vertices["point"][index] - vertices["origin"][:, None, :]
        distance = np.linalg.norm(delta, axis=-1)
        unit = delta / np.maximum(distance, 1e-6)[..., None]
        self_index = np.arange(candidates.shape[0])[:, None]
        return unit, distance, (candidates >= 0) & (candidates != self_index)

    def _candidate_cost(self, vertices, candidates, unit, distance, usable) -> np.ndarray:
        index = np.maximum(candidates, 0)
        alignment = np.einsum("nkc,noc->nko", vertices["dir"], unit)
        mismatch = (
            vertices["label"][index][:, None, :] != vertices["label"][:, None, None]
        ).astype(np.float32)
        charged = mismatch * (~vertices["terminal"])[:, :, None]
        cost = (
            self.w_cos * (1.0 - alignment)
            + self.w_dist * distance[:, None, :]
            + self.w_class * charged
        )
        allowed = (
            usable[:, None, :]
            & (distance[:, None, :] <= self.max_conn_dist)
            & (alignment >= self.cos_thresh)
            & (vertices["exist"] > self.exist_thresh)[:, :, None]
        )
        return np.where(allowed, cost, np.inf)

    # --- ③ 그래프 정리 -----------------------------------------------------------

    def _clean_graph(self, directed: dict, vertices: dict) -> tuple[set, set]:
        """a->b 와 b->a 가 둘 다 있을 때만 무향 간선으로 채택한다 (종점·다른 클래스는 예외)."""
        edges, terminal_nodes = set(), set()
        for (source, target), is_terminal in directed.items():
            cross_class = vertices["label"][source] != vertices["label"][target]
            if is_terminal and not cross_class:
                # t = 1 은 "종점" 과 "다른 클래스 접합" 을 함께 뜻한다(6.2절). 후자에서
                # 상대를 종점으로 표시하면, 그 위를 지나가던 본선이 접합 셀에서 잘린다.
                terminal_nodes.add(target)
            confirmed = (target, source) in directed or is_terminal or cross_class
            if confirmed or not self.mutual:
                edges.add((min(source, target), max(source, target)))
        return edges, terminal_nodes

    # --- ④ 경로 추출 -------------------------------------------------------------

    def _extract_paths(self, vertices: dict, edges: set, terminal: set) -> list[dict]:
        adjacency = defaultdict(list)
        for a, b in edges:
            if (
                vertices["label"][a] == vertices["label"][b]
            ):  # 다른 클래스 간선은 접합 표시로만 쓴다
                adjacency[a].append(b)
                adjacency[b].append(a)
        walker = _PathWalker(vertices, adjacency, terminal)
        paths = walker.run()
        return [self._to_instance(vertices, path) for path in paths if len(path) >= self.min_points]

    def _to_instance(self, vertices: dict, path: list[int]) -> dict:
        nodes = np.array(path, dtype=np.int64)
        points = vertices["point"][nodes] * self.grid_stride - PIXEL_CENTER_SHIFT
        labels = vertices["label"][nodes]
        if self.simplify_tol > 0:
            points = _simplify(points, self.simplify_tol)
        return {
            # 폴리라인 클래스는 정점 다수결이다 — 종점 셀에는 클래스 손실을 주지 않으므로(8.2절)
            # 양 끝 정점의 예측을 그대로 믿을 수 없다.
            "class": int(np.bincount(labels).argmax()),
            "points": points.astype(np.float32),
            "score": float(vertices["score"][nodes].mean()),
        }


class _PathWalker:
    """차수 2 정점을 따라가며 폴리라인을 자른다. 분기점은 방향 연속성으로 통과한다 (10.5절)."""

    def __init__(self, vertices: dict, adjacency: dict, terminal: set):
        self.point = vertices["point"]
        self.adjacency = adjacency
        self.terminal = terminal
        self.used: set[tuple[int, int]] = set()

    def run(self) -> list[list[int]]:
        paths = []
        for node in self._start_order():
            for neighbor in self.adjacency[node]:
                if self._edge_key(node, neighbor) in self.used:
                    continue
                paths.append(self._walk(node, neighbor))
        return paths

    def _start_order(self) -> list[int]:
        nodes = sorted(self.adjacency)
        cuts = [n for n in nodes if len(self.adjacency[n]) != 2 or n in self.terminal]
        return cuts + nodes  # 자르는 지점 먼저, 남은 고리는 아무 데서나

    def _walk(self, start: int, first: int) -> list[int]:
        path, previous, current = [start], start, first
        self.used.add(self._edge_key(start, first))
        while True:
            path.append(current)
            if current in self.terminal or len(self.adjacency[current]) == 1:
                return path
            nxt = self._next_node(previous, current)
            if nxt is None:
                return path
            self.used.add(self._edge_key(current, nxt))
            previous, current = current, nxt

    def _next_node(self, previous: int, current: int) -> int | None:
        options = [
            n
            for n in self.adjacency[current]
            if n != previous and self._edge_key(current, n) not in self.used
        ]
        if not options:
            return None
        if len(options) == 1:
            return options[0]
        incoming = _unit(self.point[current] - self.point[previous])
        scores = [float(incoming @ _unit(self.point[n] - self.point[current])) for n in options]
        return options[int(np.argmax(scores))]

    @staticmethod
    def _edge_key(a: int, b: int) -> tuple[int, int]:
        return (min(a, b), max(a, b))


def _simplify(points: np.ndarray, tolerance: float) -> np.ndarray:
    """Ramer-Douglas-Peucker 단순화 (10.5절 5번). 반복 스택으로 구현해 깊은 재귀를 피한다."""
    if points.shape[0] < 3:
        return points
    keep = np.zeros(points.shape[0], dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, points.shape[0] - 1)]
    while stack:
        start, end = stack.pop()
        if end <= start + 1:
            continue
        offset = _perpendicular_distance(points[start + 1 : end], points[start], points[end])
        far = int(offset.argmax())
        if offset[far] <= tolerance:
            continue
        split = start + 1 + far
        keep[split] = True
        stack += [(start, split), (split, end)]
    return points[keep]


def _perpendicular_distance(points: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    segment = end - start
    length = float(np.linalg.norm(segment))
    if length < 1e-9:
        return np.linalg.norm(points - start, axis=1)
    offset = points - start
    cross = segment[0] * offset[:, 1] - segment[1] * offset[:, 0]
    return np.abs(cross) / length


def _unit(vector: np.ndarray) -> np.ndarray:
    return vector / max(float(np.linalg.norm(vector)), 1e-6)


def _to_numpy(value) -> np.ndarray:
    if not isinstance(value, torch.Tensor):
        return value
    tensor = value.detach().cpu()
    return tensor.numpy() if tensor.dtype == torch.bool else tensor.float().numpy()


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _softmax(x: np.ndarray) -> np.ndarray:
    shifted = np.exp(x - x.max(axis=-1, keepdims=True))
    return shifted / shifted.sum(axis=-1, keepdims=True)

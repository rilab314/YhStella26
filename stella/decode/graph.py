"""객체 생성(디코딩) — 셀 단위 예측을 폴리라인 객체로 (impl_plan 10절, 9·10차 개정).

    ① 정점 추출 -> ② 사슬 확장 (클래스 확률 국소 피크 시드, 양방향) -> ③ 후처리

핵심 확인은 **"서로가 서로의 점을 향하는가"**다: 내 확장 슬롯 방향 c와 후보의 되가리킴
슬롯 방향 n이 마주보면 c . n -> -1. 인코딩(6.4절 사슬)과 같은 모양으로 한 노드씩
확장하며, 구 GraphDecoder의 전역 그래프·상호 최선 확인·경로 절단은 폐기했다 —
간선 소실 2.3%가 성분 수 1.8배로 증폭되던 구조였다.

좌표 규약: 내부 계산은 전부 **격자 단위**다. 반환 직전에만 픽셀로 바꾼다 —
격자 좌표 p는 픽셀 p * s - 0.5 에 대응한다(인코더가 픽셀 면적 중심 +0.5를 쓰기 때문).
"""

from dataclasses import fields as dataclass_fields

import numpy as np
import torch

PIXEL_CENTER_SHIFT = 0.5
PEAK_EPS = 1e-12  # 국소 피크 판정에서 자기 자신과의 부동소수 비교 여유
# 동률 해소용 미세 거리 항. 일직선 위에서는 한 칸 뒤와 두 칸 뒤가 정렬·마주봄 모두
# 동률이라(둘 다 같은 선의 셀이라 되가리킴 슬롯도 있다) 가까운 쪽을 골라야 정점을
# 건너뛰지 않는다. 계획이 배제한 것은 "정렬 나쁜 가까운 셀을 끌어들이는" 크기의
# 거리 항(w_dist = 0.3)이고, 이 값은 반경 2에서 최대 0.004라 동률만 가른다 (10.3절).
DISTANCE_TIEBREAK = 1e-3


class ChainDecoder:
    @classmethod
    def from_cfg(cls, module_cfg, cfg, **kwargs) -> "ChainDecoder":
        params = {
            f.name: getattr(module_cfg, f.name)
            for f in dataclass_fields(module_cfg)
            if f.name not in ("path", "name")
        }
        return cls(grid_stride=cfg.data.grid_stride, grid_size=cfg.data.grid_size, **params)

    def __init__(
        self,
        *,
        grid_stride: int,
        grid_size: int,
        heatmap_thresh: float,
        exist_thresh: float,
        end_thresh: float,
        radius: int,
        align_thresh: float,
        opp_thresh: float,
        w_opp: float,
        min_class_prob: float,
        purity_thresh: float,
        end_extend: float,
        min_points: int,
        simplify_tol: float,
    ):
        self.grid_stride = grid_stride
        self.grid_size = grid_size
        self.heatmap_thresh = heatmap_thresh
        self.exist_thresh = exist_thresh
        self.end_thresh = end_thresh
        self.radius = radius
        self.align_thresh = align_thresh
        self.opp_thresh = opp_thresh
        self.w_opp = w_opp
        self.min_class_prob = min_class_prob
        self.purity_thresh = purity_thresh
        self.end_extend = end_extend
        self.min_points = min_points
        self.simplify_tol = simplify_tol

    def __call__(self, output) -> list[dict]:
        vertices = self._extract_vertices(output)
        if vertices["point"].shape[0] == 0:
            return []
        instances = [self._to_instance(vertices, *chain) for chain in self._grow_chains(vertices)]
        return [item for item in instances if item is not None]

    # --- ① 정점 추출 (10.2절) ----------------------------------------------------

    def _extract_vertices(self, output) -> dict:
        """학습 dilation 없이 노드 셀을 고르고 정점 속성을 모은다."""
        arrays = {k: _to_numpy(v) for k, v in vars(output).items()}
        heat = _sigmoid(arrays["heatmap_logit"])
        label = arrays["class_logit"].argmax(axis=-1)
        keep = arrays["node_mask"] & (heat > self.heatmap_thresh) & (label > 0)
        cells = np.argwhere(keep)
        rows, cols = cells[:, 0], cells[:, 1]
        coord = arrays["self_coord"][rows, cols]
        class_prob = _softmax(arrays["class_logit"][rows, cols])
        return {
            "cells": cells,
            "point": np.stack([cols + coord[:, 0], rows + coord[:, 1]], axis=-1),
            "label": label[rows, cols],
            "class_prob": class_prob,
            "score": heat[rows, cols] * class_prob.max(axis=-1),
            "end_prob": _sigmoid(arrays["end_logit"][rows, cols]),
            "exist": _sigmoid(arrays["exist_logit"][rows, cols]),
            "dir": arrays["conn_dir"][rows, cols],
            "neighbors": self._neighbor_table(cells, self.radius),
        }

    def _neighbor_table(self, cells: np.ndarray, radius: int) -> np.ndarray:
        """각 정점의 체비셰프 반경 안 정점 인덱스 (V, (2r+1)^2). 빈 칸은 -1."""
        if cells.shape[0] == 0:  # 학습 초기엔 임계값을 넘는 정점이 없을 수 있다
            return np.zeros((0, (2 * radius + 1) ** 2), dtype=np.int64)
        grid = np.full((self.grid_size, self.grid_size), -1, dtype=np.int64)
        grid[cells[:, 0], cells[:, 1]] = np.arange(cells.shape[0])
        padded = np.pad(grid, radius, constant_values=-1)
        span = np.arange(2 * radius + 1)
        rows = cells[:, 0:1] + span
        cols = cells[:, 1:2] + span
        return padded[rows[:, :, None], cols[:, None, :]].reshape(cells.shape[0], -1)

    # --- ② 사슬 확장 (10.3절) ----------------------------------------------------

    def _grow_chains(self, vertices: dict) -> list[tuple]:
        """시드마다 양방향으로 확장하고, 순도 검사를 통과한 사슬만 남긴다."""
        total, slots = vertices["exist"].shape
        used = np.zeros(total, dtype=bool)
        slot_used = np.zeros((total, slots), dtype=bool)
        failed = np.zeros(total, dtype=bool)
        chains = []
        for seed in self._seed_order(vertices):
            if used[seed] or failed[seed]:
                continue
            chain = self._grow_from_seed(vertices, used, slot_used, failed, int(seed))
            if chain is not None:
                chains.append(chain)
        return chains

    def _seed_order(self, vertices: dict) -> np.ndarray:
        """클래스 확률 국소 피크(정점 3x3 이웃 중 최대)를 확률 내림차순으로,
        소진되면 남은 정점을 확률 내림차순으로 (안전망)."""
        best = vertices["class_prob"].max(axis=-1)
        table = self._neighbor_table(vertices["cells"], radius=1)
        around = np.where(table >= 0, best[np.maximum(table, 0)], -1.0)
        peak = best >= around.max(axis=1) - PEAK_EPS
        ids = np.arange(best.shape[0])
        peaks = ids[peak][np.argsort(-best[peak], kind="stable")]
        rest = ids[~peak][np.argsort(-best[~peak], kind="stable")]
        return np.concatenate([peaks, rest])

    def _grow_from_seed(self, vertices, used, slot_used, failed, seed) -> tuple | None:
        """시드의 슬롯 두 개를 따라 양방향 확장 -> 순도 검사 -> 끝 연장.

        R > 2 ablation에서는 존재 확률 상위 2개 슬롯만 시드 방향으로 쓴다 —
        사슬은 정의상 양방향뿐이다.
        """
        label = int(vertices["label"][seed])
        used[seed] = True
        touched = [seed]
        forward, backward = np.argsort(-vertices["exist"][seed], kind="stable")[:2]
        head = self._expand(vertices, used, slot_used, seed, int(forward), label, touched)
        tail = self._expand(vertices, used, slot_used, seed, int(backward), label, touched)
        chain = [*reversed(tail), seed, *head]
        purity = float(np.mean(vertices["label"][chain] == label))
        if purity <= self.purity_thresh:  # 정점·슬롯을 되돌리고 시드만 실패로 남긴다
            used[touched] = False
            slot_used[touched] = False
            failed[seed] = True
            return None
        head_ext, tail_ext = self._extensions(vertices, slot_used, chain)
        return chain, head_ext, tail_ext, label

    def _expand(self, vertices, used, slot_used, start, slot, label, touched) -> list[int]:
        """한 노드씩 단방향 확장. 정지: 끝 확률·후보 없음·비활성 슬롯.

        고리 폐쇄(시작 정점 복귀)는 시작 정점이 이미 사용 상태라 후보에서 빠져
        "후보 없음" 정지에 흡수된다. 스텝마다 미사용 정점을 하나 소비하므로 무한 루프가 없다.
        """
        path: list[int] = []
        vertex, k = start, slot
        while k is not None and not slot_used[vertex, k]:
            if vertices["exist"][vertex, k] <= self.exist_thresh:
                break
            found = self._best_candidate(vertices, used, slot_used, vertex, k, label)
            if found is None:
                break
            vertex, k = self._step(vertices, used, slot_used, (vertex, k), found, touched, path)
            if vertices["end_prob"][vertex] > self.end_thresh:
                break
        return path

    def _step(self, vertices, used, slot_used, current, found, touched, path):
        """후보를 사슬에 붙이고 (다음 정점, 계속 확장할 슬롯)을 돌려준다."""
        vertex, k = current
        target, back = found
        slot_used[vertex, k] = slot_used[target, back] = True
        used[target] = True
        touched.append(target)
        path.append(target)
        return target, self._next_slot(vertices, slot_used, target, back)

    def _next_slot(self, vertices, slot_used, vertex, back) -> int | None:
        """되가리킴 슬롯의 반대쪽 활성 슬롯 — 되가리킴 방향과 가장 반대인 것 (R = 2면 남은 하나)."""
        usable = ~slot_used[vertex] & (vertices["exist"][vertex] > self.exist_thresh)
        if not usable.any():
            return None
        dots = vertices["dir"][vertex] @ vertices["dir"][vertex, back]
        return int(np.where(usable, dots, np.inf).argmin())

    def _best_candidate(self, vertices, used, slot_used, vertex, k, label) -> tuple | None:
        """반경 안 미사용 정점 중 게이트(정렬·마주봄·사슬 클래스 확률)를 통과한 비용 최소."""
        nearby = vertices["neighbors"][vertex]
        nearby = nearby[nearby >= 0]
        nearby = nearby[~used[nearby]]
        nearby = nearby[vertices["class_prob"][nearby, label] >= self.min_class_prob]
        if nearby.size == 0:
            return None
        heading = vertices["dir"][vertex, k]
        delta = vertices["point"][nearby] - vertices["point"][vertex]
        distance = np.linalg.norm(delta, axis=-1)
        unit = delta / np.maximum(distance, 1e-9)[:, None]
        align = unit @ heading
        opposite, back = self._facing_slots(vertices, slot_used, nearby, heading)
        cost = (1.0 - align) + self.w_opp * (1.0 + opposite) + DISTANCE_TIEBREAK * distance
        allowed = (align >= self.align_thresh) & (-opposite >= self.opp_thresh)
        cost = np.where(allowed, cost, np.inf)
        best = int(cost.argmin())
        if not np.isfinite(cost[best]):
            return None
        return int(nearby[best]), int(back[best])

    def _facing_slots(self, vertices, slot_used, nearby, heading) -> tuple[np.ndarray, np.ndarray]:
        """후보별 되가리킴 슬롯: 활성·미사용 슬롯 중 c . n 이 가장 작은 것 (마주봄 최대)."""
        dots = vertices["dir"][nearby] @ heading  # (M, R)
        usable = (vertices["exist"][nearby] > self.exist_thresh) & ~slot_used[nearby]
        dots = np.where(usable, dots, np.inf)  # 쓸 슬롯이 없으면 inf -> 마주봄 게이트에서 탈락
        back = dots.argmin(axis=1)
        return dots[np.arange(nearby.size), back], back

    def _extensions(self, vertices, slot_used, chain) -> tuple[list, list]:
        """끝 셀에서 남은 활성 슬롯(끝방향) 쪽으로 연장점을 하나 추가한다 (결정 31).

        1셀 사슬(3칸짜리 선)은 남은 슬롯 두 개로 양쪽에 연장해 3점 폴리라인이 된다.
        """
        if len(chain) == 1:
            points = self._extend_points(vertices, slot_used, chain[0])
            return points[:1], points[1:2]
        head = self._extend_points(vertices, slot_used, chain[0])
        tail = self._extend_points(vertices, slot_used, chain[-1])
        return head[:1], tail[:1]

    def _extend_points(self, vertices, slot_used, vertex) -> list[np.ndarray]:
        if vertices["end_prob"][vertex] <= self.end_thresh:
            return []
        usable = ~slot_used[vertex] & (vertices["exist"][vertex] > self.exist_thresh)
        origin = vertices["point"][vertex]
        slots = np.flatnonzero(usable)
        return [origin + vertices["dir"][vertex, k] * self.end_extend for k in slots]

    # --- ③ 후처리 (10.4절) -------------------------------------------------------

    def _to_instance(self, vertices, chain, head_ext, tail_ext, label) -> dict | None:
        """폴리라인 클래스 = 사슬 클래스(시드의 클래스) — 순도 검사가 다수결 일치를 보증한다."""
        points = np.array([*head_ext, *vertices["point"][chain], *tail_ext])
        if points.shape[0] < self.min_points:
            return None
        pixels = points * self.grid_stride - PIXEL_CENTER_SHIFT
        if self.simplify_tol > 0:
            pixels = _simplify(pixels, self.simplify_tol)
        return {
            "class": label,
            "points": pixels.astype(np.float32),
            "score": float(vertices["score"][chain].mean()),
        }


def _simplify(points: np.ndarray, tolerance: float) -> np.ndarray:
    """Ramer-Douglas-Peucker 단순화. 반복 스택으로 구현해 깊은 재귀를 피한다."""
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

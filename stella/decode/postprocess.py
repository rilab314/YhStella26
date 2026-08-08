"""디코딩 후처리 — 조각 병합과 폴리라인 단순화 (impl_plan 10.4, 가설 백로그).

사슬 확장이 멈춘 자리에는 "이어야 했던 조각"이 남는다. 병합은 **사슬 확장과 같은 규칙**
(가깝다 + 서로를 향한다)을 폴리라인 끝점 수준에서 한 번 더 적용하는 것이다.
셀 격자가 아니라 픽셀 좌표에서 동작하므로 디코더 본체와 독립적으로 켜고 끌 수 있다.
"""

import numpy as np
from scipy.spatial import cKDTree

TANGENT_EPS = 1e-9


class ChainMerger:
    """끝점이 서로 가깝고 마주보는 두 폴리라인을 하나로 잇는다. `gap <= 0`이면 무동작."""

    def __init__(self, *, gap: float, align_cos: float):
        self.gap = gap
        self.align_cos = align_cos

    def __call__(self, instances: list[dict]) -> tuple[list[dict], int]:
        if self.gap <= 0.0 or len(instances) < 2:
            return instances, 0
        ends = _endpoints(instances)
        links = _accept_links(self._candidate_pairs(ends), len(instances))
        merged = _rebuild(instances, links)
        return merged, len(instances) - len(merged)

    def _candidate_pairs(self, ends: dict) -> list[tuple]:
        """반경 안 끝점 쌍 중 게이트를 통과한 것을 비용 오름차순으로."""
        tree = cKDTree(ends["position"])
        pairs = []
        for first, second in tree.query_pairs(self.gap):
            cost = self._pair_cost(ends, first, second)
            if cost is not None:
                pairs.append((cost, first, second))
        return sorted(pairs)

    def _pair_cost(self, ends: dict, first: int, second: int) -> float | None:
        if ends["owner"][first] == ends["owner"][second]:
            return None
        if ends["label"][first] != ends["label"][second]:
            return None
        delta = ends["position"][second] - ends["position"][first]
        distance = float(np.linalg.norm(delta))
        if distance < TANGENT_EPS:
            return 0.0
        unit = delta / distance
        forward = float(unit @ ends["tangent"][first])
        backward = float(-unit @ ends["tangent"][second])
        if forward < self.align_cos or backward < self.align_cos:
            return None
        return distance / self.gap + (1.0 - forward) + (1.0 - backward)


def _endpoints(instances: list[dict]) -> dict:
    """인스턴스 k의 끝점 두 개를 인덱스 2k(시작)·2k+1(끝)에 편다. 접선은 바깥을 향한다."""
    position = np.zeros((2 * len(instances), 2), dtype=np.float64)
    tangent = np.zeros_like(position)
    for index, item in enumerate(instances):
        points = np.asarray(item["points"], dtype=np.float64)
        position[2 * index] = points[0]
        position[2 * index + 1] = points[-1]
        if points.shape[0] >= 2:  # 점 하나짜리는 접선이 0 -> 게이트에서 탈락한다
            tangent[2 * index] = _unit(points[0] - points[1])
            tangent[2 * index + 1] = _unit(points[-1] - points[-2])
    return {
        "position": position,
        "tangent": tangent,
        "label": np.repeat([item["class"] for item in instances], 2),
        "owner": np.repeat(np.arange(len(instances)), 2),
    }


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > TANGENT_EPS else np.zeros(2)


def _accept_links(pairs: list[tuple], count: int) -> np.ndarray:
    """끝점당 최대 1개, 고리가 생기지 않게 탐욕적으로 채택한다."""
    link = np.full(2 * count, -1, dtype=np.int64)
    parent = np.arange(count)
    for _, first, second in pairs:
        if link[first] >= 0 or link[second] >= 0:
            continue
        roots = _find(parent, first // 2), _find(parent, second // 2)
        if roots[0] == roots[1]:
            continue
        parent[roots[0]] = roots[1]
        link[first], link[second] = second, first
    return link


def _find(parent: np.ndarray, node: int) -> int:
    while parent[node] != node:
        parent[node] = parent[parent[node]]
        node = int(parent[node])
    return node


def _rebuild(instances: list[dict], link: np.ndarray) -> list[dict]:
    """자유 끝에서 출발해 연결을 따라가며 폴리라인을 이어 붙인다."""
    visited = np.zeros(len(instances), dtype=bool)
    result = []
    for index in range(len(instances)):
        for side in (0, 1):
            if not visited[index] and link[2 * index + side] < 0:
                result.append(_walk(instances, link, visited, index, side))
    for index in range(len(instances)):  # 남은 것은 고리 — 원본 그대로 둔다
        if not visited[index]:
            visited[index] = True
            result.append(instances[index])
    return result


def _walk(instances: list[dict], link: np.ndarray, visited: np.ndarray, start: int, side: int):
    pieces, scores, current, entry = [], [], start, side
    while current >= 0 and not visited[current]:
        visited[current] = True
        points = np.asarray(instances[current]["points"], dtype=np.float32)
        pieces.append(points if entry == 0 else points[::-1])
        scores.append(float(instances[current]["score"]))
        following = int(link[2 * current + (1 - entry)])
        current, entry = (following // 2, following % 2) if following >= 0 else (-1, 0)
    return {
        "class": int(instances[start]["class"]),
        "points": np.concatenate(pieces, axis=0).astype(np.float32),
        "score": float(np.mean(scores)),
    }


def simplify_polyline(points: np.ndarray, tolerance: float) -> np.ndarray:
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

"""1픽셀 뼈대를 **순서 있는 점 목록**으로 바꾼다 (3-A 비교 대상).

덩어리 하나가 폴리라인 하나가 된다. 직전 연구(LaneStitch)의 `skeleton_trace.py` 를 옮겨 온
것으로, 알고리즘은 같고 주석만 우리 규약으로 고쳤다.

덩어리마다:
  - 이웃이 하나뿐인 픽셀이 **끝점**, 셋 이상이면 **갈림길**이다.
  - 끝점에서 끝점까지 **가장 긴 경로**만 남기고 나머지 가지는 버린다.
  - 끝점이 없는 고리는 아무 데서나 한 바퀴 돈다.

끝점이 `EXACT_ENDPOINTS` 이하면 모든 끝점에서 너비우선 탐색을 돌려 정확한 최장 경로를 얻고,
그보다 많으면 두 번 훑기로 근사한다(뼈대는 거의 트리라 트리에서는 정확하다).
"""

from collections import deque

import cv2
import numpy as np

NEIGHBOR_OFFSETS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
EXACT_ENDPOINTS = 16
MIN_PATH_POINTS = 2


class SkeletonTracer:
    """뼈대 덩어리 하나를 순서 있는 폴리라인 하나로 바꾼다."""

    def trace(self, skeleton: np.ndarray) -> list[np.ndarray]:
        """덩어리마다 (N, 2) `(x, y)` 점 배열 하나씩."""
        count, labels = cv2.connectedComponents(skeleton.astype(np.uint8), connectivity=8)
        paths = []
        for label in range(1, count):
            points = self._longest_path((labels == label).astype(np.uint8))
            if points is not None and len(points) >= MIN_PATH_POINTS:
                paths.append(points)
        return paths

    def _longest_path(self, component: np.ndarray) -> np.ndarray | None:
        pixels, neighbors = _pixel_graph(component)
        if len(pixels) < MIN_PATH_POINTS:
            return None
        endpoints = np.flatnonzero((neighbors >= 0).sum(axis=1) == 1)
        order = (
            _walk_loop(neighbors) if len(endpoints) == 0 else self._between(neighbors, endpoints)
        )
        return pixels[order][:, ::-1].astype(np.float64)  # (행, 열) -> (x, y)

    def _between(self, neighbors: np.ndarray, endpoints: np.ndarray) -> np.ndarray:
        """가장 멀리 떨어진 두 끝점 사이의 경로."""
        if len(endpoints) <= EXACT_ENDPOINTS:
            starts = endpoints
        else:
            starts = [_farthest(neighbors, int(endpoints[0]))[0]]
        best_distance, best_path = -1, None
        for start in starts:
            target, parents, distance = _farthest(neighbors, int(start))
            if distance > best_distance:
                best_distance, best_path = distance, _backtrack(parents, target)
        return best_path


def _pixel_graph(component: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """켜진 픽셀 목록과, 픽셀마다 이웃 8칸의 번호(없으면 -1)."""
    pixels = np.argwhere(component > 0)
    index = -np.ones(np.array(component.shape) + 2, dtype=np.int64)
    index[pixels[:, 0] + 1, pixels[:, 1] + 1] = np.arange(len(pixels))
    neighbors = np.stack(
        [index[pixels[:, 0] + 1 + dy, pixels[:, 1] + 1 + dx] for dy, dx in NEIGHBOR_OFFSETS],
        axis=1,
    )
    return pixels, neighbors


def _farthest(neighbors: np.ndarray, start: int) -> tuple[int, np.ndarray, int]:
    """너비우선 탐색 — (가장 먼 점, 부모 배열, 그 거리)."""
    parents = -np.ones(len(neighbors), dtype=np.int64)
    distances = -np.ones(len(neighbors), dtype=np.int64)
    parents[start], distances[start] = start, 0
    queue, farthest = deque([start]), start
    while queue:
        node = queue.popleft()
        if distances[node] > distances[farthest]:
            farthest = node
        for neighbor in neighbors[node]:
            if neighbor >= 0 and distances[neighbor] < 0:
                parents[neighbor] = node
                distances[neighbor] = distances[node] + 1
                queue.append(int(neighbor))
    return farthest, parents, int(distances[farthest])


def _backtrack(parents: np.ndarray, node: int) -> np.ndarray:
    """탐색 시작점에서 `node` 까지의 경로."""
    path = [node]
    while parents[path[-1]] != path[-1]:
        path.append(int(parents[path[-1]]))
    return np.array(path[::-1], dtype=np.int64)


def _walk_loop(neighbors: np.ndarray) -> np.ndarray:
    """끝점이 없는 덩어리 — 아무 데서나 한 바퀴 돈다."""
    order, visited, node = [], set(), 0
    while node is not None and node not in visited:
        order.append(node)
        visited.add(node)
        node = next((int(n) for n in neighbors[node] if n >= 0 and int(n) not in visited), None)
    return np.array(order, dtype=np.int64)

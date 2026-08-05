"""폴리라인 기하 헬퍼 — 재샘플·접선·점-폴리라인 거리 (impl_plan 11절 계산 기반).

지표는 폴리라인을 일정 간격으로 샘플해 점-폴리라인 거리로 계산한다.
그래서 성긴 점이 대각선 거리를 재는 문제가 없다.
"""

import numpy as np

EPS = 1e-9


def resample(points: np.ndarray, step: float) -> tuple[np.ndarray, np.ndarray]:
    """폴리라인을 일정 간격으로 샘플하고 각 샘플의 단위 접선을 함께 낸다."""
    poly = np.asarray(points, dtype=np.float64)
    if poly.shape[0] < 2:
        return poly.copy(), np.zeros_like(poly)
    delta = np.diff(poly, axis=0)
    lengths = np.linalg.norm(delta, axis=1)
    total = float(lengths.sum())
    if total < EPS:
        return poly[:1].copy(), np.zeros((1, 2))
    count = max(int(np.ceil(total / step)) + 1, 2)
    positions = np.linspace(0.0, total, count)
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    segment = np.clip(np.searchsorted(cumulative, positions, side="right") - 1, 0, len(delta) - 1)
    ratio = (positions - cumulative[segment]) / np.maximum(lengths[segment], EPS)
    sampled = poly[segment] + delta[segment] * ratio[:, None]
    tangent = delta[segment] / np.maximum(lengths[segment], EPS)[:, None]
    return sampled, tangent


def polyline_length(points: np.ndarray) -> float:
    poly = np.asarray(points, dtype=np.float64)
    if poly.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(poly, axis=0), axis=1).sum())


def bounding_box(points: np.ndarray) -> np.ndarray:
    poly = np.asarray(points, dtype=np.float64)
    return np.concatenate([poly.min(axis=0), poly.max(axis=0)])


def boxes_overlap(first: np.ndarray, second: np.ndarray, margin: float) -> bool:
    return bool(
        first[0] - margin <= second[2]
        and second[0] - margin <= first[2]
        and first[1] - margin <= second[3]
        and second[1] - margin <= first[3]
    )


def gated_distance(
    points: np.ndarray, tangents: np.ndarray, polyline: np.ndarray, angle_cos: float
) -> np.ndarray:
    """각 샘플점에서 폴리라인까지의 최소 거리. 접선 방향 차가 게이트를 넘는 구간은 제외한다."""
    poly = np.asarray(polyline, dtype=np.float64)
    if poly.shape[0] < 2:
        return np.full(points.shape[0], np.inf)
    start, end = poly[:-1], poly[1:]
    distance = _point_segment_distance(points, start, end)
    direction = end - start
    norm = np.linalg.norm(direction, axis=1, keepdims=True)
    unit = direction / np.maximum(norm, EPS)
    aligned = np.abs(tangents @ unit.T) >= angle_cos
    return np.where(aligned, distance, np.inf).min(axis=1)


def _point_segment_distance(points: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """(M, 2) 점들과 (S, 2) 선분들 사이의 거리 (M, S)."""
    segment = end - start
    length_sq = np.maximum((segment**2).sum(axis=1), EPS)
    offset = points[:, None, :] - start[None, :, :]
    ratio = np.clip((offset * segment[None, :, :]).sum(axis=2) / length_sq[None, :], 0.0, 1.0)
    closest = start[None, :, :] + ratio[:, :, None] * segment[None, :, :]
    return np.linalg.norm(points[:, None, :] - closest, axis=2)

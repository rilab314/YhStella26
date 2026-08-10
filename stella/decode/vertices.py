"""정점 추출과 시드 순서 (design 10.2절).

디코더의 ① 단계만 떼어 낸다 — "어느 셀이 정점인가"와 "어디서부터 사슬을 시작하는가"는
사슬 확장 규칙과 독립이라 따로 바꿔 볼 수 있어야 한다 (가설 백로그).
"""

import numpy as np
import torch

PEAK_EPS = 1e-12  # 국소 피크 판정에서 자기 자신과의 부동소수 비교 여유
SEED_MODES = ("class_peak", "end_peak")


class VertexExtractor:
    """모델 출력 -> 정점 속성 dict. 반환 dict가 사슬 확장이 읽는 유일한 자료구조다."""

    def __init__(
        self,
        *,
        grid_size: int,
        heatmap_thresh: float,
        radius: int,
        seed_mode: str,
        end_thresh: float,
        fg_thresh: float,
    ):
        if seed_mode not in SEED_MODES:
            raise ValueError(f"seed_mode 는 {SEED_MODES} 중 하나여야 한다: {seed_mode}")
        self.grid_size = grid_size
        self.heatmap_thresh = heatmap_thresh
        self.radius = radius
        self.seed_mode = seed_mode
        self.end_thresh = end_thresh
        self.fg_thresh = fg_thresh

    def __call__(self, output) -> dict:
        """학습 dilation 없이 노드 셀을 고르고 정점 속성을 모은다."""
        arrays = {k: _to_numpy(v) for k, v in vars(output).items()}
        heat = _sigmoid(arrays["heatmap_logit"])
        label = arrays["class_logit"].argmax(axis=-1)
        keep = arrays["node_mask"] & (heat > self.heatmap_thresh) & self._foreground(arrays, label)
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
            "neighbors": self.neighbor_table(cells, self.radius),
        }

    def _foreground(self, arrays: dict, label: np.ndarray) -> np.ndarray:
        """전경 판정. `fg_thresh < 0`이면 기존 `argmax != 0`, 아니면 이진 로짓 (E12)."""
        if self.fg_thresh < 0.0:
            return label > 0
        return _sigmoid(arrays["fg_logit"]) > self.fg_thresh

    def neighbor_table(self, cells: np.ndarray, radius: int) -> np.ndarray:
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

    def seed_order(self, vertices: dict) -> np.ndarray:
        """시드 순서. 앞의 시드가 먼저 정점을 소비하므로 순서가 곧 사슬 모양을 정한다."""
        best = vertices["class_prob"].max(axis=-1)
        if self.seed_mode == "end_peak":
            return self._end_first_order(vertices, best)
        return self._peak_first_order(vertices, best)

    def _peak_first_order(self, vertices: dict, best: np.ndarray) -> np.ndarray:
        """클래스 확률 국소 피크(정점 3x3 이웃 중 최대) 우선, 소진되면 남은 정점 (안전망)."""
        peak = self._local_peak(vertices, best)
        return _rank_by(best, peak)

    def _local_peak(self, vertices: dict, best: np.ndarray) -> np.ndarray:
        table = self.neighbor_table(vertices["cells"], radius=1)
        around = np.where(table >= 0, best[np.maximum(table, 0)], -1.0)
        return best >= around.max(axis=1) - PEAK_EPS

    def _end_first_order(self, vertices: dict, best: np.ndarray) -> np.ndarray:
        """선의 실제 끝(끝 확률 상위)에서 출발한다 — 한 번에 선 전체를 훑는 것이 목표다."""
        is_end = vertices["end_prob"] > self.end_thresh
        ends = _rank_by(vertices["end_prob"], is_end)[: int(is_end.sum())]
        rest = _rank_by(best, ~is_end)[: int((~is_end).sum())]
        return np.concatenate([ends, rest])


def _rank_by(score: np.ndarray, first: np.ndarray) -> np.ndarray:
    """`first`가 참인 정점을 점수 내림차순으로 앞에, 나머지를 뒤에 놓은 인덱스 배열."""
    ids = np.arange(score.shape[0])
    head = ids[first][np.argsort(-score[first], kind="stable")]
    tail = ids[~first][np.argsort(-score[~first], kind="stable")]
    return np.concatenate([head, tail])


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

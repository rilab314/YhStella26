"""시각 로그 그리기 — Lightning을 모르는 순수 함수 모음 (design 9.5절).

`np.ndarray` in -> `np.ndarray` out 이라 단위 테스트가 가능하고,
GT와 예측이 같은 격자 형태라 인자만 바꿔 넣으면 **GT도 같은 함수로 그릴 수 있다**.
"""

import cv2
import numpy as np

from stella.data.types import CLASS_COLOR, SLOT_COLOR


def to_uint8_image(image: np.ndarray) -> np.ndarray:
    """(3, H, W) 또는 (H, W, 3) float [0,1] -> (H, W, 3) uint8 RGB."""
    array = np.asarray(image, dtype=np.float32)
    if array.ndim == 3 and array.shape[0] == 3:
        array = array.transpose(1, 2, 0)
    return np.ascontiguousarray(np.clip(array * 255.0, 0, 255).astype(np.uint8))


def draw_heatmap(image: np.ndarray, probability: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """히트맵 확률을 파랑(0)->빨강(1)으로 칠하고 원본과 블렌딩한다."""
    canvas = to_uint8_image(image)
    prob = _upsample_nearest(np.asarray(probability, np.float32), canvas.shape[0])
    heat = np.stack([prob, np.zeros_like(prob), 1.0 - prob], axis=-1) * 255.0
    blended = (1.0 - alpha) * canvas.astype(np.float32) + alpha * heat
    return np.clip(blended, 0, 255).astype(np.uint8)


def draw_class_map(
    image: np.ndarray, class_ids: np.ndarray, draw_mask: np.ndarray, stride: int
) -> np.ndarray:
    """4x4 셀마다 중심 2x2 픽셀을 클래스 색으로 칠한다."""
    canvas = to_uint8_image(image)
    colors = np.array(CLASS_COLOR, dtype=np.uint8)
    margin = max(stride // 4, 1)
    for i, j in np.argwhere(draw_mask & (class_ids > 0)):
        y, x = i * stride + margin, j * stride + margin
        canvas[y : y + stride - 2 * margin, x : x + stride - 2 * margin] = colors[class_ids[i, j]]
    return canvas


def draw_slots(
    image: np.ndarray,
    self_coord: np.ndarray,
    conn_dir: np.ndarray,
    exist_prob: np.ndarray,
    draw_mask: np.ndarray,
    stride: int,
    exist_thresh: float = 0.5,
    line_len: float = 6.0,
) -> np.ndarray:
    """self 좌표 = 검은 점, 연결 슬롯 방향 = R/G 선 — 둘 다 자기 점에서 (6.1절 원점 규약)."""
    canvas = to_uint8_image(image)
    for i, j in np.argwhere(draw_mask):
        point = (np.array([j, i]) + self_coord[i, j]) * stride
        cv2.circle(canvas, tuple(point.astype(int)), 1, (0, 0, 0), -1)
        _draw_slot_lines(canvas, point, conn_dir[i, j], exist_prob[i, j], exist_thresh, line_len)
    return canvas


def _draw_slot_lines(
    canvas: np.ndarray,
    origin: np.ndarray,
    directions: np.ndarray,
    exist: np.ndarray,
    threshold: float,
    line_len: float,
) -> None:
    for slot in range(directions.shape[0]):
        if exist[slot] < threshold:
            continue
        tip = origin + directions[slot] * line_len
        color = SLOT_COLOR[slot % len(SLOT_COLOR)]
        cv2.line(canvas, tuple(origin.astype(int)), tuple(tip.astype(int)), color, 1)


def draw_instances(image: np.ndarray, instances: list[dict], thickness: int = 1) -> np.ndarray:
    """폴리라인 인스턴스를 클래스 색으로 그린다 (GT·디코딩 결과 공용)."""
    canvas = to_uint8_image(image)
    for inst in instances:
        points = np.round(np.asarray(inst["points"], np.float32)).astype(np.int32)
        color = tuple(int(v) for v in CLASS_COLOR[inst["class"] % len(CLASS_COLOR)])
        cv2.polylines(canvas, [points], False, color, thickness, cv2.LINE_AA)
    return canvas


def _upsample_nearest(array: np.ndarray, size: int) -> np.ndarray:
    factor = size // array.shape[0]
    return np.repeat(np.repeat(array, factor, axis=0), factor, axis=1)

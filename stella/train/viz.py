"""시각 로그 그리기 — Lightning을 모르는 순수 함수 모음 (design 9.5절).

`np.ndarray` in -> `np.ndarray` out 이라 단위 테스트가 가능하고,
GT와 예측이 같은 격자 형태라 인자만 바꿔 넣으면 **GT도 같은 함수로 그릴 수 있다**.

한 프레임의 여섯 페이지는 `PageRenderer`가 **2x3 시트 한 장**으로 붙인다 — 학습 콜백과
캐시 시각화 스크립트가 같은 렌더러를 써서 그림의 규칙이 갈라지지 않는다.
"""

import cv2
import numpy as np

from stella.data.types import CLASS_COLOR, SLOT_COLOR

TILE_COLUMNS = 3  # 2행 x 3열 — 페이지 여섯 장
TILE_GAP = 4  # 패널 사이 구분선 두께(픽셀)
TILE_GAP_COLOR = (24, 24, 24)
LABEL_ORIGIN = (12, 34)
LABEL_SCALE = 1.0
LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
LABEL_COLOR = (255, 255, 255)
LABEL_EDGE_COLOR = (0, 0, 0)
ENDPOINT_RADIUS = 4  # 폴리라인 양 끝 표시 원의 반지름(픽셀)
ENDPOINT_EDGE_COLOR = (255, 255, 255)


class PageRenderer:
    """모델 출력 한 샘플 -> 여섯 페이지 -> 이름표 달린 2x3 시트 한 장.

    임계값은 전부 `LogConfig`에서 온다. 이 클래스는 config를 모른다 (조립 규칙 1).
    """

    def __init__(
        self,
        *,
        grid_stride: int,
        heat_alpha: float,
        slot_line_len: float,
        exist_thresh: float,
        class_thresh: float,
    ):
        self.grid_stride = grid_stride
        self.heat_alpha = heat_alpha
        self.slot_line_len = slot_line_len
        self.exist_thresh = exist_thresh
        self.class_thresh = class_thresh

    def render(self, image, output, decoded: list[dict], instances: list[dict]) -> np.ndarray:
        """(3, H, W) 영상 + 배치 차원 없는 `ModelOutput` -> 시트 (H*2, W*3, 3) RGB."""
        return tile_pages(self.build_pages(image, output, decoded, instances))

    def build_pages(self, image, output, decoded: list[dict], instances: list[dict]) -> dict:
        """삽입 순서가 곧 시트의 배치다 — 윗줄은 셀 단위 예측, 아랫줄은 폴리라인."""
        probability = _sigmoid(output.heatmap_logit.numpy())
        node_mask = output.node_mask.numpy()
        end_prob = _sigmoid(output.end_logit.numpy()) * node_mask
        return {
            "heat": draw_heatmap(image, probability, self.heat_alpha),
            "class": self._class_page(image, output, probability, node_mask),
            "slot": self._slot_page(image, output, node_mask),
            "end": draw_heatmap(image, end_prob, self.heat_alpha),
            "gt": draw_instances(image, instances),
            "inst": draw_instances(image, decoded),
        }

    def _class_page(self, image, output, probability, node_mask) -> np.ndarray:
        class_ids = output.class_logit.numpy().argmax(axis=-1)
        draw = node_mask & (probability > self.class_thresh)
        return draw_class_map(image, class_ids, draw, self.grid_stride)

    def _slot_page(self, image, output, node_mask) -> np.ndarray:
        return draw_slots(
            image,
            output.self_coord.numpy(),
            output.conn_dir.numpy(),
            _sigmoid(output.exist_logit.numpy()),
            node_mask,
            self.grid_stride,
            self.exist_thresh,
            self.slot_line_len,
        )


def tile_pages(pages: dict[str, np.ndarray], columns: int = TILE_COLUMNS) -> np.ndarray:
    """페이지마다 이름표를 달아 격자 한 장으로 붙인다 (부족한 칸은 검게 남긴다)."""
    labeled = [_label_page(page, name) for name, page in pages.items()]
    if not labeled:
        raise ValueError("붙일 페이지가 없다")
    rows = [labeled[start : start + columns] for start in range(0, len(labeled), columns)]
    return _concat([_join_row(row, columns, labeled[0].shape) for row in rows], axis=0)


def _label_page(page: np.ndarray, name: str) -> np.ndarray:
    """어두운 위성영상 위에서도 읽히도록 검은 외곽선 위에 흰 글자를 얹는다."""
    canvas = page.copy()
    for color, width in ((LABEL_EDGE_COLOR, 4), (LABEL_COLOR, 2)):
        cv2.putText(canvas, name, LABEL_ORIGIN, LABEL_FONT, LABEL_SCALE, color, width, cv2.LINE_AA)
    return canvas


def _join_row(row: list[np.ndarray], columns: int, shape: tuple) -> np.ndarray:
    filled = row + [np.zeros(shape, np.uint8)] * (columns - len(row))
    return _concat(filled, axis=1)


def _concat(pieces: list[np.ndarray], axis: int) -> np.ndarray:
    """조각 사이에 구분선을 끼워 이어 붙인다 (axis 0 = 세로, 1 = 가로)."""
    gap_shape = list(pieces[0].shape)
    gap_shape[axis] = TILE_GAP
    gap = np.full(gap_shape, TILE_GAP_COLOR, np.uint8)
    stacked = [gap if index % 2 else pieces[index // 2] for index in range(2 * len(pieces) - 1)]
    return np.concatenate(stacked, axis=axis)


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


def draw_instances(
    image: np.ndarray,
    instances: list[dict],
    thickness: int = 1,
    endpoint_radius: int = ENDPOINT_RADIUS,
) -> np.ndarray:
    """폴리라인 인스턴스를 클래스 색으로 그린다 (GT·디코딩 결과 공용)."""
    canvas = to_uint8_image(image)
    for inst in instances:
        points = np.round(np.asarray(inst["points"], np.float32)).astype(np.int32)
        color = tuple(int(v) for v in CLASS_COLOR[inst["class"] % len(CLASS_COLOR)])
        cv2.polylines(canvas, [points], False, color, thickness, cv2.LINE_AA)
        _draw_endpoints(canvas, points, color, endpoint_radius)
    return canvas


def _draw_endpoints(canvas: np.ndarray, points: np.ndarray, color: tuple, radius: int) -> None:
    """선의 시작·끝을 원으로 찍는다 — 어디서 끊겼는지가 조각남 진단의 핵심이다."""
    if radius <= 0 or len(points) == 0:
        return
    for end in (points[0], points[-1]):
        center = (int(end[0]), int(end[1]))
        cv2.circle(canvas, center, radius, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, center, radius, ENDPOINT_EDGE_COLOR, 1, cv2.LINE_AA)


def _upsample_nearest(array: np.ndarray, size: int) -> np.ndarray:
    factor = size // array.shape[0]
    return np.repeat(np.repeat(array, factor, axis=0), factor, axis=1)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

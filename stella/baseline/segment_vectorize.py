"""분할 결과(색으로 칠한 PNG)를 폴리라인으로 바꾼다 — 3-A 비교 대상의 본체.

**칠하기 → 뼈대 → 꼭짓점 줄이기** 세 단계다. 종류마다 따로 처리하므로 서로 다른 종류가
한 덩어리로 합쳐지지는 않는다. **같은 종류끼리 붙어 있으면 합쳐진다 — 그것이 이 방식의
한계이고, 이 논문이 보이려는 지점이다.** 그래서 씨앗 기반 분리(watershed)를 넣지 않는다.

색 → 종류 번호 표는 직전 연구(LaneStitch `config.METAINFO`)의 것을 그대로 쓴다.
예측 PNG 가 그 표로 칠해져 있기 때문이다.
"""

import cv2
import numpy as np

from stella.baseline.skeleton_trace import SkeletonTracer
from stella.baseline.thinning import thin

# LaneStitch `config.METAINFO` 의 RGB — 예측 PNG 가 이 색으로 칠해져 있다.
CLASS_RGB = {
    1: (77, 77, 255),
    2: (77, 178, 255),
    3: (77, 255, 77),
    4: (255, 153, 77),
    5: (255, 77, 77),
    6: (178, 77, 255),
    7: (77, 255, 178),
    8: (255, 178, 77),
    9: (77, 102, 255),
    10: (255, 77, 128),
    11: (128, 255, 77),
}
MIN_COMPONENT_PIXELS = 20  # 이보다 작은 덩어리는 잡음으로 본다
MIN_POLYLINE_POINTS = 2


class SegmentVectorizer:
    """분할 마스크 한 장 -> 폴리라인 인스턴스 목록."""

    def __init__(self, *, epsilon: float, min_pixels: int, classes: tuple):
        """epsilon: 꼭짓점 줄이기 허용 오차(픽셀). classes: 뽑아낼 종류 번호."""
        self.epsilon = epsilon
        self.min_pixels = min_pixels
        self.classes = classes
        self.tracer = SkeletonTracer()

    def __call__(self, rgb: np.ndarray) -> list[dict]:
        """rgb: (H, W, 3) uint8 RGB. 반환은 `{"class", "points"}` 목록이다."""
        instances = []
        for label in self.classes:
            mask = self._class_mask(rgb, label)
            if mask.sum() < self.min_pixels:
                continue
            for points in self.tracer.trace(thin(mask)):
                simplified = self._simplify(points)
                if len(simplified) >= MIN_POLYLINE_POINTS:
                    instances.append({"class": label, "points": simplified})
        return instances

    def _class_mask(self, rgb: np.ndarray, label: int) -> np.ndarray:
        color = np.array(CLASS_RGB[label], dtype=np.uint8)
        return np.all(rgb == color, axis=-1)

    def _simplify(self, points: np.ndarray) -> np.ndarray:
        """Douglas-Peucker — 모양을 유지하면서 꼭짓점 수를 줄인다."""
        contour = points.astype(np.float32).reshape(-1, 1, 2)
        reduced = cv2.approxPolyDP(contour, self.epsilon, closed=False)
        return reduced.reshape(-1, 2).astype(np.float64)


def read_prediction(path: str) -> np.ndarray:
    """예측 PNG 를 RGB 로 읽는다 — cv2 는 BGR 로 읽으므로 뒤집는다."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"예측 그림을 못 읽었다: {path}")
    return image[:, :, ::-1]

"""개발용 합성 데이터셋 (design 6.6절).

Y자 분기·다른 클래스 T자 접합·교차·이중선·파선을 의도적으로 섞어 생성한다 —
6.4절의 끝칸 미채움·셀 소유권·건너뛰기·3x3 순위 규칙을 실전에서 검증하기 위함이다.
곡선이 경계에서 경계로 가로지르므로 X자 교차는 곡선 사이에서 저절로 생긴다.
"""

import cv2
import numpy as np

from stella.builder import Buildable
from stella.data.augment import VectorAugment
from stella.data.encode import ChainEncoder
from stella.data.types import GridDatasetBase, make_sample

SPLIT_SEED = {"train": 1, "val": 2, "test": 3}
DEFAULT_LENGTH = {"train": 256, "val": 32, "test": 32}
CURVE_SAMPLES = 24
DASH_SAMPLES = 200  # 파선용 촘촘한 곡선 샘플 수 — 인접 샘플 간격이 수 픽셀이 되게
DASH_KEEP_RANGE = (8, 14)  # dash 하나의 샘플 수 (수십 픽셀 길이)
DASH_GAP_RANGE = (3, 7)  # dash 사이 건너뛰는 샘플 수
LINE_WIDTH = 2
DOUBLE_LINE_GAP = 4.0
BACKGROUND_LEVEL = 0.35
LINE_LEVEL = 0.85


class SyntheticDataset(GridDatasetBase, Buildable):
    def __init__(
        self,
        *,
        split: str,
        image_size: int,
        grid_stride: int,
        num_classes: int,
        max_degree: int,
        encode_supersample: int,
        augment: bool,
        limit: int,
    ):
        self.split = split
        self.image_size = image_size
        self.num_classes = num_classes
        self.length = limit if limit > 0 else DEFAULT_LENGTH[split]
        self.encoder = ChainEncoder(
            image_size=image_size,
            grid_stride=grid_stride,
            num_classes=num_classes,
            max_degree=max_degree,
            supersample=encode_supersample,
        )
        use_aug = augment and split == "train"
        self.augment = VectorAugment(image_size=image_size) if use_aug else None

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict:
        rng = np.random.default_rng([SPLIT_SEED[self.split], idx])
        instances = self._make_instances(rng)
        image = self._render(instances, rng)
        if self.augment is not None:
            image, instances = self.augment(image, instances, rng)
        meta = {"filename": f"synth_{self.split}_{idx:05d}"}
        return make_sample(image, instances, self.encoder.encode(instances), meta)

    def _make_instances(self, rng: np.random.Generator) -> list[dict]:
        instances: list[dict] = []
        for _ in range(int(rng.integers(3, 7))):
            label = int(rng.integers(1, self.num_classes))
            points = self._random_curve(rng)
            roll = rng.random()
            if roll < 0.2:
                instances.extend(self._dashes(label, rng))
            elif roll < 0.5:
                instances.extend(self._double_line(points, label))
            else:
                instances.append({"class": label, "points": points})
            if rng.random() < 0.6:
                instances.append(self._branch(points, label, rng))
            if rng.random() < 0.7:
                instances.append(self._junction(points, label, rng))
        return instances

    def _random_curve(self, rng: np.random.Generator, samples: int = CURVE_SAMPLES) -> np.ndarray:
        """양 끝이 이미지 경계에 붙은 2차 베지어 곡선."""
        start, end = self._border_point(rng), self._border_point(rng)
        while np.linalg.norm(end - start) < self.image_size * 0.5:
            end = self._border_point(rng)
        middle = (start + end) / 2 + rng.normal(0, self.image_size * 0.12, size=2)
        t = np.linspace(0.0, 1.0, samples)[:, None]
        curve = (1 - t) ** 2 * start + 2 * (1 - t) * t * middle + t**2 * end
        return np.clip(curve, 0, self.image_size - 1).astype(np.float32)

    def _dashes(self, label: int, rng: np.random.Generator) -> list[dict]:
        """파선 — dash마다 별도 인스턴스 = 별도 사슬. 짧은 사슬·끝 연장을 훈련시킨다."""
        dense = self._random_curve(rng, samples=DASH_SAMPLES)
        pieces: list[dict] = []
        cursor = 0
        while cursor + DASH_KEEP_RANGE[0] <= len(dense):
            keep = int(rng.integers(*DASH_KEEP_RANGE))
            pieces.append({"class": label, "points": dense[cursor : cursor + keep]})
            cursor += keep + int(rng.integers(*DASH_GAP_RANGE))
        return pieces

    def _border_point(self, rng: np.random.Generator) -> np.ndarray:
        size = self.image_size - 1
        along = rng.uniform(0, size)
        side = int(rng.integers(4))
        return np.array([[along, 0], [along, size], [0, along], [size, along]][side], np.float32)

    def _double_line(self, points: np.ndarray, label: int) -> list[dict]:
        """같은 클래스의 평행선 두 개 = 이중선. 별도 인스턴스로 라벨링한다."""
        normal = _unit_normals(points) * (DOUBLE_LINE_GAP / 2)
        return [
            {"class": label, "points": (points + normal).astype(np.float32)},
            {"class": label, "points": (points - normal).astype(np.float32)},
        ]

    def _branch(self, points: np.ndarray, label: int, rng: np.random.Generator) -> dict:
        """같은 클래스 Y자 분기 — 줄기 중간에서 갈라져 경계로 나간다."""
        anchor = points[int(rng.integers(len(points) // 3, 2 * len(points) // 3))]
        target = self._border_point(rng)
        t = np.linspace(0.0, 1.0, CURVE_SAMPLES)[:, None]
        bend = (anchor + target) / 2 + rng.normal(0, self.image_size * 0.08, size=2)
        curve = (1 - t) ** 2 * anchor + 2 * (1 - t) * t * bend + t**2 * target
        return {"class": label, "points": np.clip(curve, 0, self.image_size - 1).astype(np.float32)}

    def _junction(self, points: np.ndarray, label: int, rng: np.random.Generator) -> dict:
        """다른 클래스 T자 접합 — 줄기에 닿고 끝난다."""
        other = 1 + (label - 1 + int(rng.integers(1, self.num_classes - 1))) % (
            self.num_classes - 1
        )
        index = int(rng.integers(1, len(points) - 1))
        anchor = points[index]
        direction = _unit_normals(points)[index]
        length = rng.uniform(self.image_size * 0.1, self.image_size * 0.3)
        far = self._clip_inside(anchor, direction, length)
        segment = np.linspace(anchor, far, CURVE_SAMPLES).astype(np.float32)
        return {"class": other, "points": segment}

    def _clip_inside(self, anchor: np.ndarray, direction: np.ndarray, length: float) -> np.ndarray:
        """이미지 안으로 실제 길이가 남는 방향을 고른다 — 길이 0인 접합선을 만들지 않는다."""
        limit = self.image_size - 1
        forward = np.clip(anchor + direction * length, 0, limit)
        backward = np.clip(anchor - direction * length, 0, limit)
        forward_len = float(np.linalg.norm(forward - anchor))
        backward_len = float(np.linalg.norm(backward - anchor))
        return forward if forward_len >= backward_len else backward

    def _render(self, instances: list[dict], rng: np.random.Generator) -> np.ndarray:
        size = self.image_size
        image = np.full((size, size, 3), BACKGROUND_LEVEL, dtype=np.float32)
        image += rng.normal(0, 0.03, size=image.shape).astype(np.float32)
        for inst in instances:
            pts = np.round(inst["points"]).astype(np.int32)
            level = LINE_LEVEL - 0.05 * (inst["class"] % 3)
            cv2.polylines(image, [pts], False, (level, level, level), LINE_WIDTH, cv2.LINE_AA)
        return np.clip(cv2.GaussianBlur(image, (3, 3), 0.6), 0.0, 1.0)


def _unit_normals(points: np.ndarray) -> np.ndarray:
    """각 점에서의 폴리라인 법선(단위벡터). 이중선 오프셋·T접합 방향에 쓴다."""
    tangent = np.gradient(np.asarray(points, np.float64), axis=0)
    norm = np.linalg.norm(tangent, axis=1, keepdims=True)
    tangent = tangent / np.maximum(norm, 1e-6)
    return np.stack([-tangent[:, 1], tangent[:, 0]], axis=-1)

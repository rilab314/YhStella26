"""증강 — 기하 변환은 인코딩 전 벡터 단계에서, 색상 변환은 이미지에만 (impl_plan 6.4·6.7.6절).

기본은 격자 대칭(90도 회전·반전)뿐이다. 이것만으로는 **차선 방향 분포가 8방향에 갇힌다** —
임의 각도 회전(`rotate_deg`)과 스케일 지터(`scale_jitter`)는 그 가정을 시험하기 위한 선택지다
(가설 백로그). 둘 다 기본값 0이라 켜지 않으면 기존 동작 그대로다.

회전·스케일은 타일 밖을 **검은색으로** 채운다. 반사(reflect)로 채우면 라벨 없는 차선이
화면에 들어와 거짓 음성을 학습시키기 때문이다.
"""

import cv2
import numpy as np

BRIGHTNESS_LIMIT = 0.2
CONTRAST_LIMIT = 0.2
GAMMA_RANGE = (0.8, 1.2)
NOISE_STD_RANGE = (0.0, 0.04)
GAMMA_SKIP = 0.02  # 이보다 1에 가까우면 거듭제곱을 건너뛴다 (비싸고 효과가 없다)
NOISE_SKIP = 0.005
AFFINE_SKIP = 1e-3  # 이보다 작은 회전·스케일은 적용하지 않는다


class VectorAugment:
    """좌우/상하 반전 + 90도 회전 (폴리라인 좌표 변환만) + 임의 아핀 + 색상 증강."""

    def __init__(
        self,
        *,
        image_size: int,
        geometric: bool = True,
        color: bool = True,
        rotate_deg: float = 0.0,
        scale_jitter: float = 0.0,
    ):
        self.image_size = image_size
        self.geometric = geometric
        self.color = color
        self.rotate_deg = rotate_deg
        self.scale_jitter = scale_jitter

    @property
    def reshapes(self) -> bool:
        """참이면 증강 뒤 폴리라인을 타일 경계로 다시 잘라야 한다 (호출부가 읽는다)."""
        return self.rotate_deg > 0.0 or self.scale_jitter > 0.0

    def __call__(
        self, image: np.ndarray, instances: list[dict], rng: np.random.Generator
    ) -> tuple[np.ndarray, list[dict]]:
        if self.geometric:
            image, instances = self._geometric(image, instances, rng)
        if self.reshapes:
            image, instances = self._affine(image, instances, rng)
        if self.color:
            image = self._color(image, rng)
        return image, instances

    def _geometric(
        self, image: np.ndarray, instances: list[dict], rng: np.random.Generator
    ) -> tuple[np.ndarray, list[dict]]:
        turns = int(rng.integers(4))
        flip_x = bool(rng.integers(2))
        flip_y = bool(rng.integers(2))
        if turns:
            image = np.rot90(image, turns, axes=(0, 1))
        if flip_x:
            image = image[:, ::-1]
        if flip_y:
            image = image[::-1]
        points = [self._map_points(i["points"], turns, flip_x, flip_y) for i in instances]
        moved = [{**inst, "points": pts} for inst, pts in zip(instances, points)]
        return np.ascontiguousarray(image), moved

    def _map_points(self, points: np.ndarray, turns: int, flip_x: bool, flip_y: bool) -> np.ndarray:
        size = float(self.image_size)
        out = np.asarray(points, np.float32).copy()
        for _ in range(turns):  # np.rot90 CCW: (x, y) -> (y, S - x)
            out = np.stack([out[:, 1], size - out[:, 0]], axis=-1)
        if flip_x:
            out[:, 0] = size - out[:, 0]
        if flip_y:
            out[:, 1] = size - out[:, 1]
        return np.clip(out, 0.0, size).astype(np.float32)

    def _affine(
        self, image: np.ndarray, instances: list[dict], rng: np.random.Generator
    ) -> tuple[np.ndarray, list[dict]]:
        """중심 기준 회전 + 등방 스케일. 이미지와 폴리라인에 **같은 행렬**을 쓴다."""
        angle = float(rng.uniform(-self.rotate_deg, self.rotate_deg))
        scale = float(1.0 + rng.uniform(-self.scale_jitter, self.scale_jitter))
        if abs(angle) < AFFINE_SKIP and abs(scale - 1.0) < AFFINE_SKIP:
            return image, instances
        center = (self.image_size / 2.0, self.image_size / 2.0)
        matrix = cv2.getRotationMatrix2D(center, angle, scale)
        warped = cv2.warpAffine(
            image, matrix, (self.image_size, self.image_size), flags=cv2.INTER_LINEAR
        )
        moved = [{**inst, "points": _apply_affine(inst["points"], matrix)} for inst in instances]
        return warped, moved

    def _color(self, image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """768^2 x 3 위의 연산이라 자잘한 임시 배열이 곧 비용이다 — in-place로 처리한다."""
        out = np.asarray(image, dtype=np.float32).copy()
        out *= np.float32(1.0 + rng.uniform(-CONTRAST_LIMIT, CONTRAST_LIMIT))
        out += np.float32(rng.uniform(-BRIGHTNESS_LIMIT, BRIGHTNESS_LIMIT))
        np.clip(out, 0.0, 1.0, out=out)
        gamma = rng.uniform(*GAMMA_RANGE)
        if abs(gamma - 1.0) > GAMMA_SKIP:
            np.power(out, np.float32(gamma), out=out)
        noise_std = rng.uniform(*NOISE_STD_RANGE)
        if noise_std > NOISE_SKIP:
            out += rng.standard_normal(out.shape, dtype=np.float32) * np.float32(noise_std)
            np.clip(out, 0.0, 1.0, out=out)
        return out


def _apply_affine(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate(
        [np.asarray(points, np.float64), np.ones((len(points), 1))], axis=-1
    )
    return (homogeneous @ matrix.T).astype(np.float32)

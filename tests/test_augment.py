"""기하 증강 — 이미지와 폴리라인이 **같은 변환**을 받는지 (improve_plan §7 D1·D2).

증강에서 가장 무서운 버그는 "이미지만 돌고 라벨은 안 도는 것"이다. 조용히 학습만 망가진다.
그래서 밝은 선을 그린 합성 이미지로 **변환 후에도 폴리라인이 선 위에 있는지**를 직접 본다.
"""

import numpy as np

from stella.data.augment import VectorAugment

SIZE = 128


def line_image(points: np.ndarray) -> np.ndarray:
    import cv2

    canvas = np.zeros((SIZE, SIZE, 3), np.float32)
    cv2.polylines(canvas, [points.astype(np.int32)], False, (1.0, 1.0, 1.0), 3)
    return canvas


def brightness_along(image: np.ndarray, points: np.ndarray) -> float:
    """폴리라인을 촘촘히 샘플해 그 자리의 밝기 평균. 라벨이 선 위에 있으면 1에 가깝다."""
    start, end = points[0], points[-1]
    ratio = np.linspace(0.1, 0.9, 40)[:, None]  # 끝 부근은 잘림 영향이 있어 뺀다
    sampled = start + (end - start) * ratio
    inside = sampled[(sampled.min(axis=1) >= 0) & (sampled.max(axis=1) < SIZE)]
    if inside.shape[0] == 0:
        return 0.0
    rows = np.round(inside[:, 1]).astype(int)
    cols = np.round(inside[:, 0]).astype(int)
    return float(image[rows, cols, 0].mean())


def test_rotation_moves_image_and_label_together():
    points = np.array([[20.0, 64.0], [108.0, 64.0]], dtype=np.float32)
    augment = VectorAugment(
        image_size=SIZE, geometric=False, color=False, rotate_deg=25.0, scale_jitter=0.0
    )
    for seed in range(6):
        rng = np.random.default_rng(seed)
        image, moved = augment(line_image(points), [{"class": 3, "points": points}], rng)
        assert brightness_along(image, moved[0]["points"]) > 0.8, f"seed {seed}"


def test_scale_jitter_moves_image_and_label_together():
    points = np.array([[20.0, 40.0], [108.0, 96.0]], dtype=np.float32)
    augment = VectorAugment(
        image_size=SIZE, geometric=False, color=False, rotate_deg=0.0, scale_jitter=0.15
    )
    for seed in range(6):
        rng = np.random.default_rng(seed)
        image, moved = augment(line_image(points), [{"class": 3, "points": points}], rng)
        assert brightness_along(image, moved[0]["points"]) > 0.8, f"seed {seed}"


def test_grid_symmetry_only_is_the_default():
    augment = VectorAugment(image_size=SIZE)
    assert augment.reshapes is False  # 기본값에서는 재클리핑이 필요 없다


def test_affine_flag_requests_reclipping():
    assert VectorAugment(image_size=SIZE, rotate_deg=10.0).reshapes is True
    assert VectorAugment(image_size=SIZE, scale_jitter=0.1).reshapes is True


def test_rotation_fills_outside_with_black():
    """반사로 채우면 라벨 없는 차선이 들어와 거짓 음성이 된다 — 검은색이어야 한다."""
    augment = VectorAugment(
        image_size=SIZE, geometric=False, color=False, rotate_deg=45.0, scale_jitter=0.0
    )
    image, _ = augment(np.ones((SIZE, SIZE, 3), np.float32), [], np.random.default_rng(0))
    assert image[0, 0].max() < 0.01  # 모서리는 회전으로 비는 자리다

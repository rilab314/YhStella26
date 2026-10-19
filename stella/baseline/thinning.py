"""이진 마스크를 1픽셀 폭 뼈대로 깎는다 — Zhang-Suen 세선화 (3-A 비교 대상).

`cv2.ximgproc` 는 contrib 빌드에만 있고 이 환경에는 없다. scikit-image 도 넣지 않는다
(잠금 환경을 건드리지 않는다). 그래서 numpy 로 직접 구현한다 — 한 번에 한 픽셀씩 도는
대신 **전체 격자에 조건을 한꺼번에 걸어** 반복마다 두 번의 벡터 연산으로 끝낸다.

규칙은 원논문 그대로다: 이웃 8개 중 켜진 수가 2~6개이고, 시계 방향으로 돌 때 0→1 전이가
정확히 한 번이며, 방향 조건 두 개를 만족하는 픽셀을 지운다. 두 단계가 번갈아 돌면서
선의 굵기만 깎고 **연결은 끊지 않는다.**
"""

import numpy as np

MIN_NEIGHBORS = 2
MAX_NEIGHBORS = 6
MAX_ITERATIONS = 200  # 768x768 에서 실제로는 30회 안쪽에 멈춘다 — 무한 반복 안전장치
# 단계마다 확인하는 이웃 세 쌍 (P2..P9 를 0..7 로 센다) — 원논문의 방향 조건이다.
TRIPLES = (((0, 2, 4), (2, 4, 6)), ((0, 2, 6), (0, 4, 6)))


def thin(mask: np.ndarray) -> np.ndarray:
    """(H, W) 이진 마스크 -> 1픽셀 폭 뼈대. 입력은 건드리지 않는다."""
    image = (np.asarray(mask) > 0).astype(np.uint8)
    for _ in range(MAX_ITERATIONS):
        removed = 0
        for step in (0, 1):
            doomed = _condemned(image, step)
            if doomed.any():
                image[doomed] = 0
                removed += int(doomed.sum())
        if removed == 0:
            break
    return image.astype(bool)


def _condemned(image: np.ndarray, step: int) -> np.ndarray:
    """이번 단계에서 지울 픽셀 — 네 조건을 모두 만족하는 곳."""
    p = _ring(image)
    count = sum(p)
    first, second = TRIPLES[step]
    return (
        (image > 0)
        & (count >= MIN_NEIGHBORS)
        & (count <= MAX_NEIGHBORS)
        & (_transitions(p) == 1)
        & (_product(p, first) == 0)
        & (_product(p, second) == 0)
    )


def _product(p: list[np.ndarray], triple: tuple) -> np.ndarray:
    """이웃 셋을 곱한다 — 하나라도 꺼져 있으면 0 이다."""
    return p[triple[0]] * p[triple[1]] * p[triple[2]]


def _ring(image: np.ndarray) -> list[np.ndarray]:
    """이웃 8개를 북쪽부터 시계 방향으로 — P2..P9 (원논문 번호)."""
    padded = np.pad(image, 1)
    h, w = image.shape
    shifts = ((0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0), (1, 0), (0, 0))
    return [padded[dy : dy + h, dx : dx + w] for dy, dx in shifts]


def _transitions(p: list[np.ndarray]) -> np.ndarray:
    """시계 방향으로 한 바퀴 돌 때의 0 -> 1 전이 횟수."""
    ring = p + [p[0]]
    return sum(((ring[i] == 0) & (ring[i + 1] == 1)).astype(np.int32) for i in range(8))

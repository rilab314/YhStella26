"""예측 인스턴스 입출력 — 방법이 달라도 **같은 형식**으로 남긴다 (5단계).

논문 그림은 타일 하나에 여러 방법의 결과를 나란히 그린다. 그런데 방법마다 출력이 다르다 —
STELLA·GTE 는 희소 예측 캐시(`decode/cache.py`)를 디코딩해야 나오고, 분할 비교 대상은 색칠된
PNG 를 벡터화해야 나온다. **그림이 그 차이를 알 필요는 없다.** 그래서 채점 직전의 모양
(`{"class": int, "points": (N,2)}`)을 그대로 파일로 떨궈 두고, 그림은 이 파일만 읽는다.

형식은 GT 를 캐시에 넣을 때 쓰던 것과 **같다** — 길이가 제각각인 폴리라인 여러 개를 점 배열
하나로 이어 붙이고 경계를 offset 으로 적는다. 그래서 npz 하나가 타일 하나다.

    inst_points  float32 (M, 2)   모든 인스턴스의 (x, y) 를 이어 붙인 것
    inst_offset  int64   (N+1,)   k 번째 인스턴스는 points[offset[k]:offset[k+1]]
    inst_class   int64   (N,)     인스턴스별 클래스 번호
"""

from pathlib import Path

import numpy as np

EMPTY_POINTS = np.zeros((0, 2), np.float32)


def save_instances(path: Path, instances: list[dict]) -> None:
    """인스턴스 목록을 npz 하나로 떨군다. 폴더는 미리 있어야 한다."""
    np.savez_compressed(path, **pack_instances(instances))


def pack_instances(instances: list[dict]) -> dict[str, np.ndarray]:
    """가변 길이 폴리라인 목록 → 고정 배열 셋."""
    lengths = [len(item["points"]) for item in instances]
    points = (
        np.concatenate([np.asarray(item["points"]) for item in instances]).astype(np.float32)
        if instances
        else EMPTY_POINTS
    )
    return {
        "inst_points": points,
        "inst_offset": np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64),
        "inst_class": np.array([item["class"] for item in instances], dtype=np.int64),
    }


def load_instances(path: Path) -> list[dict]:
    with np.load(path) as data:
        return unpack_instances(data)


def unpack_instances(data) -> list[dict]:
    """`pack_instances` 의 역이다. 점 배열은 원본의 뷰라 복사하지 않는다."""
    points, offset, labels = data["inst_points"], data["inst_offset"], data["inst_class"]
    return [
        {"class": int(labels[k]), "points": points[offset[k] : offset[k + 1]]}
        for k in range(len(labels))
    ]

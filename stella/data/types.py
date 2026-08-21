"""데이터 출력 계약과 공통 상수 (design 6.1~6.2, 6.5, 9.5절).

좌표 규약 (전 코드 공통)
------------------------
- 이미지 픽셀: x 오른쪽+, y 아래쪽+. H = W = `image_size`.
- 격자: 배율 s = `grid_stride`, 한 변 L = image_size // s.
  **셀 인덱스만 (i, j) = (행, 열) = (y, x) 순서**이고,
  **그 외 모든 2차원 벡터 텐서는 (x, y) 순서**로 저장한다.
- 셀 내 좌표(`coord_map`)의 원점 = 셀 좌상단. 노드의 절대 위치(격자 단위)는
  p_full = (j + c_x, i + c_y).
- 연결 방향의 원점 = **자기 노드 점 p_full** (9차 개정 — 셀 중심에서 변경, 6.1절).
"""

from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

# 6.7.1절 — 학습 라벨 0(배경) + category_id 11종
CLASS_NAMES = (
    "background",
    "center_line",
    "u_turn_zone_line",
    "lane_line",
    "bus_only_lane",
    "edge_line",
    "path_change_restriction_line",
    "no_parking_stopping_line",
    "guiding_line",
    "stop_line",
    "safety_zone",
    "bicycle_lane",
)

CATEGORY_ID_TO_LABEL = {
    "501": 1,
    "502": 2,
    "503": 3,
    "504": 4,
    "505": 5,
    "506": 6,
    "515": 7,
    "525": 8,
    "530": 9,
    "531": 10,
    "535": 11,
}

# 논문 Table V의 전체 인스턴스 수. 셀 소유권 동점일 때 희소 클래스가 이긴다 (6.4절 A-3).
CLASS_INSTANCE_COUNT = (
    0,
    136182,
    27381,
    196614,
    9186,
    75276,
    95188,
    212132,
    31455,
    48353,
    10145,
    14309,
)

CLASS_COLOR = (
    (0, 0, 0),  # 0  background/ignore — 칠하지 않는다
    (77, 77, 255),  # 1  center_line
    (77, 178, 255),  # 2  u_turn_zone_line
    (77, 255, 77),  # 3  lane_line
    (255, 153, 77),  # 4  bus_only_lane
    (255, 77, 77),  # 5  edge_line
    (178, 77, 255),  # 6  path_change_restriction_line
    (77, 255, 178),  # 7  no_parking_stopping_line
    (255, 178, 77),  # 8  guiding_line
    (255, 215, 0),  # 9  stop_line
    (255, 77, 128),  # 10 safety_zone
    (0, 139, 139),  # 11 bicycle_lane
)

SLOT_COLOR = ((255, 0, 0), (0, 255, 0), (0, 0, 255))

# collate에서 stack하지 않고 list로 유지하는 키
LIST_KEYS = ("instances", "meta")


def rarity_order(num_classes: int) -> np.ndarray:
    """희소 클래스가 앞에 오는 클래스 인덱스 배열 (1..num_classes-1). 동점 소유권 판정용."""
    labels = np.arange(1, num_classes)
    counts = np.array([CLASS_INSTANCE_COUNT[c] for c in labels])
    return labels[np.argsort(counts, kind="stable")]


class GridDatasetBase(Dataset):
    """`__getitem__(idx)` -> dict. 아래 계약을 반드시 따른다 (design 6.2절).

    | key         | dtype   | shape        | 의미                                   |
    | ----------- | ------- | ------------ | ------------------------------------ |
    | `image`     | float32 | (3, H, W)    | RGB [0, 1]. 정규화는 백본이 한다             |
    | `class_map` | int64   | (L, L)       | 셀 소유 선의 클래스. 0 = 배경                 |
    | `coord_map` | float32 | (L, L, 2)    | 소유 선 픽셀의 무게중심, 원점 = 셀 좌상단          |
    | `end_map`   | float32 | (L, L)       | 이 셀이 사슬의 끝 셀인지 — 직접 감독 대상 (8.2절)    |
    | `conn_dirs` | float32 | (L, L, D, 2) | 연결 방향 2개 — 자기 점 -> 사슬 이웃 점 단위벡터    |
    | `length_map`| float32 | (L, L)       | 셀이 속한 선의 길이(소유 셀 수). 배경 0 (8.2절)     |
    | `instances` | list    | —            | 평가용 원본 폴리라인. 학습 미사용                 |
    | `meta`      | dict    | —            | filename 등                           |

    히트맵 GT는 `class_map > 0`으로 유도한다(별도 키 없음).
    모든 양성 셀의 분기는 정확히 2개다 — 중간 셀은 앞·뒤 이웃 방향, 끝 셀은 안쪽 이웃 방향 +
    끝점 방향. `conn_dirs`의 칸 순서는 무의미하다(매칭이 배정한다, 8.3절).
    """

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> dict[str, Any]:
        raise NotImplementedError


def make_sample(
    image: np.ndarray, instances: list[dict], target: dict[str, np.ndarray], meta: dict
) -> dict[str, Any]:
    """계약에 맞는 한 샘플을 만든다. image는 (H, W, 3) float32 [0, 1]."""
    chw = np.ascontiguousarray(image.transpose(2, 0, 1), dtype=np.float32)
    sample: dict[str, Any] = {"image": torch.from_numpy(chw)}
    for key in ("class_map", "coord_map", "end_map", "conn_dirs", "length_map"):
        sample[key] = torch.from_numpy(np.ascontiguousarray(target[key]))
    sample["instances"] = instances
    sample["meta"] = meta
    return sample


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """모든 GT 키가 고정 크기라서 전부 그대로 stack한다 (design 6.5절)."""
    out: dict[str, Any] = {}
    for key in batch[0]:
        if key in LIST_KEYS:
            out[key] = [sample[key] for sample in batch]
        else:
            out[key] = torch.stack([sample[key] for sample in batch], dim=0)
    return out

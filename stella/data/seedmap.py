"""SEED-MAP v1.1 실데이터 로더 (impl_plan 6.7절).

원본 배포 구조가 `dataset.json` + 평평한 `image/`·`label/` 폴더이므로 split 폴더 변환 없이
`dataset.json`을 그대로 읽는다(계획서 6.7.2의 폴더 구조는 이 저장본에 적용되어 있지 않다).

로딩 규칙
1. `geometry_type == "LINE_STRING"` 만 사용한다 (POLYGON = 노면 기호).
2. `category_id`를 6.7.1 표로 라벨에 매핑하고, 표에 없으면 버리고 카운트한다.
3. 연속 중복 점을 제거한다 — 그대로 두면 방향 계산에서 0 벡터가 나온다.
4. 이미지 경계로 자른다. 교점을 새 정점으로 넣으므로 잘린 끝이 정확히 가장자리에 놓인다.
"""

import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from stella.builder import Buildable
from stella.data.augment import VectorAugment
from stella.data.encode import GridEncoder
from stella.data.types import CATEGORY_ID_TO_LABEL, GridDatasetBase, make_sample

SPLIT_KEYS = {"train": "train", "val": "validation", "test": "test"}
CACHED_SPLITS = {"val_test": ("val", "test"), "all": ("train", "val", "test"), "none": ()}
TARGET_KEYS = ("class_map", "coord_map", "end_map", "conn_cells")


class SeedMapDataset(GridDatasetBase, Buildable):
    def __init__(
        self,
        *,
        split: str,
        root: str,
        image_size: int,
        grid_stride: int,
        num_classes: int,
        max_degree: int,
        encode_supersample: int,
        augment: bool,
        limit: int,
        cache_gt: str,
        cache_dir: str,
    ):
        self.split = split
        self.root = Path(root)
        self.image_size = image_size
        self.stems = self._load_split(split, limit)
        self.encoder = GridEncoder(
            image_size=image_size,
            grid_stride=grid_stride,
            num_classes=num_classes,
            max_degree=max_degree,
            supersample=encode_supersample,
        )
        use_aug = augment and split == "train"
        self.augment = VectorAugment(image_size=image_size) if use_aug else None
        self.cache_dir = self._cache_directory(cache_gt, cache_dir, split)
        self.unknown_categories: Counter = Counter()

    def _load_split(self, split: str, limit: int) -> list[str]:
        with open(self.root / "dataset.json", encoding="utf-8") as handle:
            splits = json.load(handle)
        stems = sorted(splits[SPLIT_KEYS[split]])
        return stems[:limit] if limit > 0 else stems

    def _cache_directory(self, cache_gt: str, cache_dir: str, split: str) -> Path | None:
        if split not in CACHED_SPLITS[cache_gt] or self.augment is not None:
            return None
        base = Path(cache_dir) if cache_dir else self.root / "gt_cache"
        path = base / split
        path.mkdir(parents=True, exist_ok=True)
        return path

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, index: int) -> dict:
        stem = self.stems[index]
        image = self._load_image(stem)
        cached = self._read_cache(stem)
        if cached is not None:
            return make_sample(image, cached[1], cached[0], {"filename": stem})
        instances = self._load_instances(stem)
        if self.augment is not None:
            rng = np.random.default_rng([hash(stem) % (2**31), index])
            image, instances = self.augment(image, instances, rng)
        target = self.encoder.encode(instances)
        self._write_cache(stem, target, instances)
        return make_sample(image, instances, target, {"filename": stem})

    def _load_image(self, stem: str) -> np.ndarray:
        path = self.root / "image" / f"{stem}.png"
        raw = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if raw is None:
            raise FileNotFoundError(f"이미지를 읽지 못했다: {path}")
        return cv2.cvtColor(raw, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    def _load_instances(self, stem: str) -> list[dict]:
        with open(self.root / "label" / f"{stem}.json", encoding="utf-8") as handle:
            records = json.load(handle)
        instances: list[dict] = []
        for record in records:
            label = self._record_label(record)
            if label is None:
                continue
            points = _dedup_points(np.asarray(record["image_points"], dtype=np.float32))
            for piece in clip_polyline(points, self.image_size - 1):
                instances.append({"class": label, "points": piece})
        return instances

    def _record_label(self, record: dict) -> int | None:
        if record.get("class") != "RoadObject" or record.get("geometry_type") != "LINE_STRING":
            return None
        category = str(record.get("category_id", ""))
        if category not in CATEGORY_ID_TO_LABEL:
            self.unknown_categories[category] += 1
            return None
        return CATEGORY_ID_TO_LABEL[category]

    def _read_cache(self, stem: str) -> tuple[dict, list[dict]] | None:
        if self.cache_dir is None:
            return None
        path = self.cache_dir / f"{stem}.npz"
        if not path.exists():
            return None
        with np.load(path) as data:
            target = {key: data[key] for key in TARGET_KEYS}
            instances = _unpack_instances(data)
        return target, instances

    def _write_cache(self, stem: str, target: dict, instances: list[dict]) -> None:
        if self.cache_dir is None:
            return
        path = self.cache_dir / f"{stem}.npz"
        if path.exists():
            return
        np.savez(path, **target, **_pack_instances(instances))


def clip_polyline(points: np.ndarray, limit: float) -> list[np.ndarray]:
    """이미지 경계를 넘는 구간을 테두리와의 교점을 새 정점으로 넣어 자른다 (6.7.4절 4)."""
    pieces: list[list[np.ndarray]] = []
    current: list[np.ndarray] = []
    for start, end in zip(points[:-1], points[1:]):
        span = _segment_inside(start, end, limit)
        if span is None:
            current = _flush(pieces, current)
            continue
        entry = start + (end - start) * span[0]
        exit_point = start + (end - start) * span[1]
        if current and not np.allclose(current[-1], entry, atol=1e-4):
            current = _flush(pieces, current)
        if not current:
            current = [entry]
        current.append(exit_point)
    _flush(pieces, current)
    return [np.asarray(piece, dtype=np.float32) for piece in pieces]


def _flush(pieces: list, current: list) -> list:
    if len(current) >= 2:
        pieces.append(current)
    return []


def _segment_inside(start: np.ndarray, end: np.ndarray, limit: float):
    """Liang-Barsky — 선분이 [0, limit]^2 안에 있는 매개변수 구간 (t0, t1)."""
    delta = end - start
    low, high = 0.0, 1.0
    for axis in range(2):
        for sign, bound in ((-1.0, 0.0), (1.0, limit)):
            direction = sign * delta[axis]
            offset = sign * start[axis] - bound
            if abs(direction) < 1e-12:
                if offset > 0:
                    return None
                continue
            ratio = -offset / direction
            if direction < 0:
                low = max(low, ratio)
            else:
                high = min(high, ratio)
    return (low, high) if low < high else None


def _dedup_points(points: np.ndarray) -> np.ndarray:
    if points.shape[0] < 2:
        return points
    keep = np.ones(points.shape[0], dtype=bool)
    keep[1:] = np.any(points[1:] != points[:-1], axis=1)
    return points[keep]


def _pack_instances(instances: list[dict]) -> dict[str, np.ndarray]:
    lengths = [len(item["points"]) for item in instances]
    points = (
        np.concatenate([item["points"] for item in instances]).astype(np.float32)
        if instances
        else np.zeros((0, 2), np.float32)
    )
    return {
        "inst_points": points,
        "inst_offset": np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64),
        "inst_class": np.array([item["class"] for item in instances], dtype=np.int64),
    }


def _unpack_instances(data) -> list[dict]:
    points, offset, labels = data["inst_points"], data["inst_offset"], data["inst_class"]
    return [
        {"class": int(labels[k]), "points": points[offset[k] : offset[k + 1]]}
        for k in range(len(labels))
    ]

"""SEED-MAP v1.2 실데이터 로더 (design 6.7절, M13).

`{root}/{train,val,test}/{image,label}` 재정리 사본(6.7.2절)을 읽는다 —
`label/*.json`을 glob 해서 인덱스를 만들므로 `dataset.json` 파싱이 없다.

로딩 규칙
1. `geometry_type == "LINE_STRING"` 만 사용한다 (POLYGON = 노면 기호).
2. `category_id`를 6.7.1 표로 라벨에 매핑하고, 표에 없으면 버리고 카운트한다.
3. 연속 중복 점을 제거한다 — 그대로 두면 방향 계산에서 0 벡터가 나온다.
4. 이미지 경계로 자른다. 교점을 새 정점으로 넣으므로 잘린 끝이 정확히 가장자리에 놓인다.
"""

import json
import os
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from stella.builder import Buildable
from stella.data.augment import VectorAugment
from stella.data.encode import ChainEncoder
from stella.data.types import CATEGORY_ID_TO_LABEL, GridDatasetBase, make_sample

DEFAULT_GRID_STRIDE = 4  # 이 값일 때만 캐시 폴더 이름에 접미사가 붙지 않는다
SPLITS = ("train", "val", "test")
CACHED_SPLITS = {"val_test": ("val", "test"), "all": ("train", "val", "test"), "none": ()}
TARGET_KEYS = ("class_map", "coord_map", "end_map", "conn_dirs", "length_map")


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
        conn_lookahead: int,
        augment: bool,
        aug_rotate_deg: float,
        aug_scale_jitter: float,
        limit: int,
        cache_gt: str,
        cache_dir: str,
    ):
        self.split = split
        self.root = Path(root)
        self.image_size = image_size
        self.stems = self._load_split(split, limit)
        self.encoder = ChainEncoder(
            image_size=image_size,
            grid_stride=grid_stride,
            num_classes=num_classes,
            max_degree=max_degree,
            supersample=encode_supersample,
            conn_lookahead=conn_lookahead,
        )
        use_aug = augment and split == "train"
        self.augment = (
            VectorAugment(
                image_size=image_size,
                rotate_deg=aug_rotate_deg,
                scale_jitter=aug_scale_jitter,
            )
            if use_aug
            else None
        )
        self.cache_dir = self._cache_directory(cache_gt, cache_dir, split)
        self.unknown_categories: Counter = Counter()

    def _load_split(self, split: str, limit: int) -> list[str]:
        if split not in SPLITS:
            raise ValueError(f"split 은 {SPLITS} 중 하나여야 한다: {split}")
        label_dir = self.root / split / "label"
        stems = sorted(path.stem for path in label_dir.glob("*.json"))
        if not stems:
            raise FileNotFoundError(f"라벨이 하나도 없다: {label_dir} (6.7.2절 splits 구조 확인)")
        return _subsample(stems, limit)

    def _cache_directory(self, cache_gt: str, cache_dir: str, split: str) -> Path | None:
        """인코딩 설정이 기본값이 아니면 **폴더를 나눈다** — 안 그러면 낡은 GT를 조용히 읽는다.

        기본값일 때 이름을 그대로 두어 이미 떠 둔 캐시가 살아 있게 한다.
        **격자 간격(`grid_stride`)도 키에 들어간다** — GT 맵의 크기 자체가 달라지는데
        이름이 같으면 stride 8 실행이 stride 4로 인코딩된 캐시를 조용히 읽는다
        (08-24, 논문의 stride ablation 을 띄우기 직전에 발견).
        """
        if split not in CACHED_SPLITS[cache_gt] or self.augment is not None:
            return None
        base = Path(cache_dir) if cache_dir else self.root / "gt_cache"
        path = base / (split + self._cache_suffix())
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _cache_suffix(self) -> str:
        """기본값(lookahead 1 · stride 4)이면 빈 문자열 — 기존 캐시가 그대로 살아 있어야 한다."""
        parts = []
        if self.encoder.conn_lookahead != 1:
            parts.append(f"look{self.encoder.conn_lookahead}")
        if self.encoder.grid_stride != DEFAULT_GRID_STRIDE:
            parts.append(f"s{self.encoder.grid_stride}")
        return "".join(f"_{p}" for p in parts)

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
            instances = self._reclip(instances)
        target = self.encoder.encode(instances)
        self._write_cache(stem, target, instances)
        return make_sample(image, instances, target, {"filename": stem})

    def _reclip(self, instances: list[dict]) -> list[dict]:
        """회전·스케일이 타일 밖으로 밀어낸 구간을 다시 자른다. 격자 대칭 증강은 대상이 아니다."""
        if not self.augment.reshapes:
            return instances
        clipped = []
        for inst in instances:
            for piece in clip_polyline(inst["points"], self.image_size - 1):
                clipped.append({**inst, "points": piece})
        return clipped

    def _load_image(self, stem: str) -> np.ndarray:
        path = self.root / self.split / "image" / f"{stem}.png"
        raw = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if raw is None:
            raise FileNotFoundError(f"이미지를 읽지 못했다: {path}")
        return cv2.cvtColor(raw, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    def _load_instances(self, stem: str) -> list[dict]:
        with open(self.root / self.split / "label" / f"{stem}.json", encoding="utf-8") as handle:
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
        try:
            with np.load(path) as data:
                if any(key not in data for key in TARGET_KEYS):
                    return None  # 구 인코더 캐시 — 미스로 취급하고 새 형식으로 덮어쓴다
                target = {key: data[key] for key in TARGET_KEYS}
                instances = _unpack_instances(data)
        except (OSError, ValueError, EOFError):
            return None  # 손상된 캐시는 미스로 취급하고 다시 만든다 (학습을 죽이지 않는다)
        return target, instances

    def _write_cache(self, stem: str, target: dict, instances: list[dict]) -> None:
        """캐시 미스일 때만 불린다 — 구 형식 파일은 여기서 새 형식으로 갱신된다.

        **원자적으로 쓴다.** 여러 실험 arm이 같은 캐시 폴더를 동시에 채우므로, 직접 쓰면
        다른 프로세스가 절반만 쓰인 파일을 읽는다. 임시 파일에 쓰고 rename 하면
        같은 파일시스템 안에서 rename이 원자적이라 그 창이 사라진다.
        """
        if self.cache_dir is None:
            return
        final = self.cache_dir / f"{stem}.npz"
        temporary = self.cache_dir / f".{stem}.{os.getpid()}.tmp.npz"
        np.savez(temporary, **target, **_pack_instances(instances))
        temporary.replace(final)


def _subsample(stems: list[str], limit: int) -> list[str]:
    """부분집합은 **균등 간격**으로 뽑는다 — 파일명이 지역 순이라 앞에서 자르면 편향된다.

    단위 실험(research 스킬 · U 규격)이 전체 분포를 대표해야 순위가 전체 학습과 어긋나지 않는다.
    """
    if limit <= 0 or limit >= len(stems):
        return stems
    index = np.linspace(0, len(stems) - 1, limit).round().astype(np.int64)
    return [stems[i] for i in np.unique(index)]


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

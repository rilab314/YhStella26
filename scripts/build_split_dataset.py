"""평평한 SEED-MAP 원본을 `{train,val,test}/{image,label}` 구조로 재정리한다 (design 6.7.2절).

원본은 `image/`·`label/`이 평평하고 split은 `dataset.json`(`train`/`validation`/`test` 키)에만
있다. 로더(`stella/data/seedmap.py`)는 `label/*.json`을 glob 해서 인덱스를 만들므로 split이
**폴더로 갈려 있어야** 한다. 이 스크립트가 그 사본을 만든다 — 원본은 건드리지 않는다.

라벨 JSON은 **변환 없이 그대로 복사**한다. 원본 형식(`RoadObject` 리스트 + `image_points`)을
로더가 직접 읽기 때문이다. 재정리는 파일 배치만 바꾼다.

    python scripts/build_split_dataset.py \
        --src .../2026_LaneStitch_revision/SEED_MAP_v1.2 \
        --dst .../2026_stella/SEED_MAP_v1.2_splits
"""

import argparse
import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# dataset.json 의 split 키 -> 만들 폴더 이름. "validation" 을 "val" 로 줄인다 (6.7.2절).
SPLIT_FOLDERS = {"train": "train", "validation": "val", "test": "test"}
KINDS = {"image": ".png", "label": ".json"}
DEFAULT_JOBS = 8


class SplitDatasetBuilder:
    """`dataset.json`이 지정한 split 대로 이미지·라벨을 복사해 폴더 구조를 만든다."""

    def __init__(self, *, src: Path, dst: Path, jobs: int, overwrite: bool):
        self.src = src
        self.dst = dst
        self.jobs = jobs
        self.overwrite = overwrite

    def build(self) -> int:
        splits = self.read_splits()
        self.check_sources(splits)
        copied = {folder: self.build_split(folder, stems) for folder, stems in splits.items()}
        self.report(copied)
        return 0

    def read_splits(self) -> dict[str, list[str]]:
        """`dataset.json`을 읽어 {폴더 이름: 타일 id 목록}으로 바꾼다."""
        path = self.src / "dataset.json"
        if not path.exists():
            raise FileNotFoundError(f"dataset.json 이 없다: {path}")
        with open(path, encoding="utf-8") as handle:
            raw = json.load(handle)
        unknown = set(raw) - set(SPLIT_FOLDERS)
        if unknown:
            raise ValueError(f"모르는 split 키다: {sorted(unknown)}")
        return {SPLIT_FOLDERS[key]: sorted(raw[key]) for key in raw}

    def check_sources(self, splits: dict[str, list[str]]) -> None:
        """복사 전에 원본 파일이 전부 있는지 확인한다 — 절반만 복사된 사본을 만들지 않는다."""
        missing = [
            str(self.source_path(kind, stem))
            for stems in splits.values()
            for stem in stems
            for kind in KINDS
            if not self.source_path(kind, stem).exists()
        ]
        if missing:
            raise FileNotFoundError(f"원본이 {len(missing)}개 빠졌다. 예: {missing[:3]}")

    def source_path(self, kind: str, stem: str) -> Path:
        return self.src / kind / f"{stem}{KINDS[kind]}"

    def build_split(self, folder: str, stems: list[str]) -> int:
        """한 split 을 복사한다. 반환값은 실제로 쓴 파일 수."""
        for kind in KINDS:
            (self.dst / folder / kind).mkdir(parents=True, exist_ok=True)
        jobs = [(kind, stem) for stem in stems for kind in KINDS]
        with ThreadPoolExecutor(max_workers=self.jobs) as pool:
            written = list(pool.map(lambda job: self.copy_one(folder, *job), jobs))
        print(f"[{folder}] 타일 {len(stems)}장 · 파일 {sum(written)}개 복사")
        return sum(written)

    def copy_one(self, folder: str, kind: str, stem: str) -> int:
        """파일 하나를 복사한다. 크기가 같은 사본이 이미 있으면 건너뛴다 (재실행 가능)."""
        source = self.source_path(kind, stem)
        target = self.dst / folder / kind / source.name
        if not self.overwrite and target.exists():
            if target.stat().st_size == source.stat().st_size:
                return 0
        shutil.copy2(source, target)
        return 1

    def report(self, copied: dict[str, int]) -> None:
        print(f"\n완료 -> {self.dst}")
        for folder in sorted(copied):
            tiles = len(list((self.dst / folder / "label").glob("*.json")))
            images = len(list((self.dst / folder / "image").glob("*.png")))
            print(f"  {folder:5s} 라벨 {tiles:6,d} · 이미지 {images:6,d}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="평평한 SEED-MAP -> split 폴더 사본")
    parser.add_argument("--src", required=True, help="평평한 원본 (image/ label/ dataset.json)")
    parser.add_argument("--dst", required=True, help="만들 사본 루트")
    parser.add_argument("--jobs", type=int, default=DEFAULT_JOBS, help="동시 복사 스레드 수")
    parser.add_argument("--overwrite", action="store_true", help="이미 있는 파일도 다시 쓴다")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    builder = SplitDatasetBuilder(
        src=Path(args.src), dst=Path(args.dst), jobs=args.jobs, overwrite=args.overwrite
    )
    return builder.build()


if __name__ == "__main__":
    sys.exit(main())

"""방법마다 다른 출력을 **하나의 인터페이스**로 묶는다 (5단계).

    tiles() -> (타일 이름, 예측 인스턴스, 정답 인스턴스)

이 세 값만 있으면 채점도 그림도 방법을 가리지 않는다. 지금 방법은 두 갈래다.

| 갈래 | 무엇에서 나오나 | 쓰는 방법 |
| --- | --- | --- |
| `CacheMethod` | 희소 예측 캐시 + 사슬 디코더 | STELLA(후처리 유무) · GTE 대조군 · 정답 주입 천장 |
| `SegmentMethod` | 색칠된 예측 PNG + 벡터화 | 분할 + 세선화 + Douglas–Peucker |

**정답(GT)의 출처가 갈래마다 다르다** — 캐시에는 GT 가 함께 들어 있고(떨굴 때 같이 저장했다),
PNG 쪽은 데이터셋에서 읽어야 한다. 그 차이를 여기서 흡수해 부르는 쪽은 모르게 한다.
"""

from collections.abc import Iterator
from pathlib import Path

from torch.utils.data import DataLoader

from stella.baseline.segment_vectorize import read_prediction
from stella.builder import build_instance
from stella.data.types import GridDatasetBase, collate_fn
from stella.decode.cache import load_prediction
from stella.decode.sweep import list_files

TileResult = tuple[str, list[dict], list[dict]]


class CacheMethod:
    """예측 캐시를 디코딩해 인스턴스를 낸다. 디코더 설정을 바꾸면 후처리 유무가 갈린다."""

    def __init__(self, *, cache: Path, cfg, decode_cfg, shape: dict, limit: int = 0):
        self.files = list_files(Path(cache), limit)
        self.shape = shape
        self.decoder = build_instance(decode_cfg, cfg)

    def __len__(self) -> int:
        return len(self.files)

    def tiles(self) -> Iterator[TileResult]:
        for path in self.files:
            output, target = load_prediction(path, self.shape)
            yield path.stem, self.decoder(output), target


class SegmentMethod:
    """색칠된 예측 PNG 를 폴리라인으로 바꾼다. 정답은 데이터셋에서 읽는다."""

    def __init__(self, *, pred_dir: Path, cfg, vectorizer, split: str, limit: int = 0):
        self.pred_dir = Path(pred_dir)
        self.vectorizer = vectorizer
        self.loader = _build_loader(cfg, split)
        self.limit = limit
        self.missing = 0

    def __len__(self) -> int:
        return self.limit or len(self.loader)

    def tiles(self) -> Iterator[TileResult]:
        produced = 0
        for batch in self.loader:
            if self.limit and produced >= self.limit:
                return
            tile = _tile_name(batch)
            path = self.pred_dir / f"{tile}.png"
            if not path.exists():
                self.missing += 1
                continue
            yield tile, self.vectorizer(read_prediction(path)), batch["instances"][0]
            produced += 1


def _build_loader(cfg, split: str) -> DataLoader:
    dataset = build_instance(cfg.data, cfg, base=GridDatasetBase, split=split)
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn)


def _tile_name(batch: dict) -> str:
    """캐시 파일 이름(`dump_predictions._stem`)과 같은 규칙이라 두 갈래의 타일 이름이 맞는다."""
    return str(batch["meta"][0].get("filename", "")).replace("/", "_")

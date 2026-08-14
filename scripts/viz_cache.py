"""예측 캐시를 진단 시트(2x3 한 장)로 그린다 — D 트랙 3단계. **GPU를 쓰지 않는다.**

캐시가 있으면 **어떤 디코더 설정으로든 다시 그릴 수 있다.** 수치를 내는 `eval_decode.py`와
같은 캐시·같은 config 경로를 쓰므로, 표에서 본 f1을 그대로 눈으로 확인하는 짝이 된다.
영상은 캐시에 없으므로 파일 이름(stem)으로 데이터셋에서 다시 읽는다.

사용:
    python scripts/dump_predictions.py --run <실행> --split val --count 80 --out <캐시>
    python scripts/viz_cache.py --cache <캐시> --split val --out <폴더> \
        --override decode.min_points=6 decode.merge_gap=32
"""

import argparse
import time
from pathlib import Path

import numpy as np

from stella.builder import build_instance
from stella.data.types import GridDatasetBase
from stella.decode.cache import load_prediction
from stella.decode.sweep import build_cfg, list_files, read_meta, shape_of
from stella.train import viz
from stella.train.callbacks import write_sheet


def main() -> None:
    args = parse_args()
    cache = Path(args.cache)
    meta = read_meta(cache)
    cfg = build_cfg(args.config, args.override, meta)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = list_files(cache, args.count)
    start = time.perf_counter()
    draw_all(files, cfg, shape_of(meta), args.split, out_dir)
    print(f"[viz] {len(files)} sheets -> {out_dir}  ({time.perf_counter() - start:.1f} s)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs.base")
    parser.add_argument("--cache", required=True, help="dump_predictions.py가 만든 캐시 폴더")
    parser.add_argument("--split", default="val", help="영상을 읽어 올 split — 캐시와 같아야 한다")
    parser.add_argument("--count", type=int, default=0, help="0이면 캐시 전부")
    parser.add_argument("--out", required=True)
    parser.add_argument("--override", nargs="*", default=[], help="점 없는 이름은 decode 섹션")
    return parser.parse_args()


def draw_all(files: list[Path], cfg, shape: dict, split: str, out_dir: Path) -> None:
    decoder = build_instance(cfg.decode, cfg)
    renderer = build_renderer(cfg)
    images = ImageSource(cfg, split)
    for path in files:
        output, instances = load_prediction(path, shape)
        sheet = renderer.render(images(path.stem), output, decoder(output), instances)
        write_sheet(out_dir / f"{path.stem}.png", sheet)


def build_renderer(cfg) -> viz.PageRenderer:
    """학습 콜백과 같은 임계값(`cfg.log`)을 쓴다 — 그림이 갈라지면 비교가 안 된다."""
    return viz.PageRenderer(
        grid_stride=cfg.data.grid_stride,
        heat_alpha=cfg.log.heat_alpha,
        slot_line_len=cfg.log.slot_line_len,
        exist_thresh=cfg.log.exist_thresh,
        class_thresh=cfg.log.class_thresh,
    )


class ImageSource:
    """캐시 파일 이름으로 원본 영상을 찾아 준다 — 캐시는 예측만 담고 영상은 담지 않는다."""

    def __init__(self, cfg, split: str):
        self.dataset = build_instance(cfg.data, cfg, base=GridDatasetBase, split=split)
        stems = getattr(self.dataset, "stems", None)
        if stems is None:
            raise SystemExit(f"{type(self.dataset).__name__}에 stems가 없다 — split을 확인한다")
        self.index = {stem: position for position, stem in enumerate(stems)}

    def __call__(self, stem: str) -> np.ndarray:
        if stem not in self.index:
            raise SystemExit(f"{stem}이 split에 없다 — 캐시를 뜬 split과 --split이 다르다")
        return self.dataset[self.index[stem]]["image"].numpy()


if __name__ == "__main__":
    main()

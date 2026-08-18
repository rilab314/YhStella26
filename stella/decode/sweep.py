"""캐시된 예측으로 디코더를 평가하는 공통 코어 (research 스킬 · D 트랙).

`scripts/eval_decode.py`(단발·한 축 스윕)와 `scripts/tune_decoder.py`(좌표 하강)가
같은 함수를 쓴다. 워커 초기화 함수를 패키지 안에 두는 편이 pickle에도 안전하다.
"""

import json
from dataclasses import fields
from multiprocessing import Pool
from pathlib import Path

from stella.builder import build_instance
from stella.config_io import apply_override, load_config
from stella.decode.cache import load_prediction
from stella.decode.stats import ChainStats

REPORT_KEYS = (
    "f1",
    "f1_macro",
    "precision",
    "recall",
    "coverage",
    "correctness",
    "rms",
    "frag",
    "frag_strict",
    # 허위 양성을 둘로 가른다. `redundant`는 진짜 선 위에 정확히 얹혔는데 그 선을 남이 이미
    # 차지했거나 너무 짧아 주장하지 못한 것, `spurious`는 아무 GT 위에도 없는 것이다.
    # 처방이 완전히 다르다 — 전자는 엮기(디코딩), 후자는 검출이다. 비율만 보면 분모를 놓치므로
    # 영상당 값으로 낸다 (`chains_per_img`와 같은 단위여야 뺄셈이 된다).
    "fp_redundant_per_img",
    "fp_spurious_per_img",
)
_SUM_KEYS = ("fp_redundant", "fp_spurious")
_CONTEXT: dict = {}


def evaluate_decode(cfg, decode_cfg, files: list[Path], shape: dict, workers: int) -> dict:
    """한 설정으로 전 샘플을 디코딩하고 인스턴스 지표 + 디코더 진단을 합친 dict를 낸다."""
    metric = build_instance(cfg.eval, cfg)
    stats = ChainStats()
    payload = (cfg, decode_cfg, shape)
    for prediction, target, counts in _decode_all(payload, files, workers):
        metric.update(prediction, target)
        stats.counter.update(counts)
    computed = {key: float(value) for key, value in metric.compute().items()}
    scores = {key: value for key, value in computed.items() if key in REPORT_KEYS}
    return scores | _per_image_counts(computed, len(files)) | stats.summary()


def _decode_all(payload, files: list[Path], workers: int) -> list:
    if workers <= 1:
        _init_worker(payload)
        return [_decode_file(path) for path in files]
    with Pool(workers, initializer=_init_worker, initargs=(payload,)) as pool:
        return pool.map(_decode_file, files, chunksize=4)


def _init_worker(payload) -> None:
    cfg, decode_cfg, shape = payload
    _CONTEXT["decoder"] = build_instance(decode_cfg, cfg)
    _CONTEXT["shape"] = shape


def _decode_file(path: Path):
    decoder = _CONTEXT["decoder"]
    decoder.stats.reset()
    output, instances = load_prediction(path, _CONTEXT["shape"])
    return decoder(output), instances, dict(decoder.stats.counter)


def _per_image_counts(computed: dict, images: int) -> dict:
    """합계로 나오는 지표를 영상당 값으로 — 다른 열과 단위를 맞춘다."""
    divisor = float(max(images, 1))
    return {f"{key}_per_img": computed[key] / divisor for key in _SUM_KEYS}


# --- 캐시·config 플러밍 (두 스크립트가 공유한다) ---------------------------------


def read_meta(cache: Path) -> dict:
    return json.loads((cache / "meta.json").read_text(encoding="utf-8"))


def list_files(cache: Path, count: int) -> list[Path]:
    return sorted(cache.glob("*.npz"))[: count or None]


def shape_of(meta: dict) -> dict:
    return {key: meta[key] for key in ("grid_size", "num_classes", "num_slots")}


def build_cfg(config: str, fixed: list[str], meta: dict):
    """캐시를 만든 격자 규격을 config에 되돌려 넣는다 — 디코더가 격자 크기를 쓰기 때문.

    `fixed`의 항목은 점이 없으면 `decode` 섹션, 있으면 전체 경로로 해석한다.
    """
    cfg = load_config(config, [])
    cfg.data.grid_stride = meta["grid_stride"]
    cfg.data.image_size = meta["grid_size"] * meta["grid_stride"]
    cfg.data.num_classes = meta["num_classes"]
    cfg.model.num_conn_slots = meta["num_slots"]
    for item in fixed:
        name, _, raw = item.partition("=")
        apply_override(cfg, name.split(".") if "." in name else ["decode", name], raw)
    return cfg


def changed_params(decode_cfg) -> dict:
    """기본 DecodeConfig와 다른 값만 남긴다 — 표의 라벨이 된다."""
    base = load_config("configs.base", []).decode
    return {
        f.name: getattr(decode_cfg, f.name)
        for f in fields(decode_cfg)
        if getattr(decode_cfg, f.name) != getattr(base, f.name)
    }


def short(value) -> str:
    return f"{value:g}" if isinstance(value, float) else str(value)

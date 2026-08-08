"""디코더 전용 평가·스윕 — 학습 없이 디코딩 성능만 잰다 (improve_plan 2.1절).

`dump_predictions.py`가 만든 희소 캐시를 읽어 **CPU만으로** 디코딩 → 인스턴스 지표 +
디코더 진단을 낸다. 학습이 GPU를 다 쓰는 동안에도 돌릴 수 있는 것이 요점이다.

사용:
    python scripts/eval_decode.py --cache <캐시>                       # 현재 기본값으로 1회
    python scripts/eval_decode.py --cache <캐시> --sweep exist_thresh=0.3,0.5,0.7
    python scripts/eval_decode.py --cache <캐시> --set merge_gap=6 --set end_thresh=0.7
    python scripts/eval_decode.py --cache <캐시> --set eval.buffer_rho=8   # 엄격 검사

한 축을 눈으로 보는 도구다. 여러 축을 자동으로 훑으려면 `tune_decoder.py`(좌표 하강)를 쓴다.
학습이 도는 중에는 `--workers 4~5`로 제한한다 (16이면 학습이 1.5배 느려진다).
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

from stella.config_io import cast_like
from stella.decode.sweep import (
    REPORT_KEYS,
    build_cfg,
    changed_params,
    evaluate_decode,
    list_files,
    read_meta,
    shape_of,
    short,
)

DECODE_KEYS = (
    "chains_per_img",
    "chain_len",
    "stop_end",
    "stop_nocand",
    "stop_exist",
    "merged_per_img",
)


def main() -> None:
    args = parse_args()
    cache = Path(args.cache)
    meta = read_meta(cache)
    files = list_files(cache, args.count)
    cfg = build_cfg(args.config, args.fixed, meta)
    print(f"[decode] {len(files)} samples from {cache} (source={meta['source']})")
    rows = []
    for variant in variants(cfg, args.sweep):
        scores = evaluate_decode(cfg, variant, files, shape_of(meta), args.workers)
        rows.append({"params": changed_params(variant)} | scores)
    print_table(rows)
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--config", default="configs.base")
    parser.add_argument("--count", type=int, default=0, help="0이면 캐시 전체")
    parser.add_argument("--workers", type=int, default=8)
    # append 라 `--set a=1 --set b=2` 처럼 여러 번 줄 수 있다 (nargs면 뒤엣것이 앞을 덮는다).
    parser.add_argument("--set", dest="fixed", action="append", default=[], help="파라미터 고정")
    parser.add_argument("--sweep", default="", help="이름=값,값,... 한 축만")
    parser.add_argument("--out", default="", help="결과 JSON 저장 경로")
    return parser.parse_args()


def variants(cfg, sweep: str) -> list:
    """스윕 축이 없으면 현재 설정 하나만, 있으면 그 축의 값마다 하나씩."""
    if not sweep:
        return [cfg.decode]
    name, _, values = sweep.partition("=")
    current = getattr(cfg.decode, name)
    return [replace(cfg.decode, **{name: cast_like(current, value)}) for value in values.split(",")]


def print_table(rows: list[dict]) -> None:
    columns = [*REPORT_KEYS, *DECODE_KEYS]
    print("\n" + "params".ljust(34) + " ".join(f"{c[:8]:>9}" for c in columns))
    for row in rows:
        label = ",".join(f"{k}={short(v)}" for k, v in row["params"].items()) or "(base)"
        values = " ".join(f"{row.get(c, float('nan')):>9.4f}" for c in columns)
        print(f"{label[:34]:<34}{values}")


if __name__ == "__main__":
    main()

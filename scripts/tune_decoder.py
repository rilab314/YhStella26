"""디코더 파라미터 좌표 하강 — M14 "디코더 임계값 스윕" (가설 백로그).

축을 하나씩 훑어 최선값을 고정하고 다음 축으로 간다. grid search가 아니라서 비용이
축의 값 수의 **합**이지 곱이 아니고, 무엇보다 **어느 축이 얼마나 중요한지**가 기록에 남는다.

사용:
    python scripts/tune_decoder.py --cache <캐시> --workers 8
    python scripts/tune_decoder.py --cache <캐시> --axes exist_thresh,end_thresh,radius
    python scripts/tune_decoder.py --cache <캐시> --rounds 2 --objective f1

결과는 `--out`(JSON)으로 남기고, 이긴 값은 `configs/schema.py`의 `DecodeConfig` 기본값에
반영한 뒤 근거를 결과 문서에 적는다.
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

from stella.config_io import cast_like
from stella.decode.sweep import (
    build_cfg,
    changed_params,
    evaluate_decode,
    list_files,
    read_meta,
    shape_of,
    short,
)

# 축과 후보값. 기하 추정으로 정한 기본값들이라 전부 근거가 약하다 (impl_plan 13절 "남은 확인").
DEFAULT_AXES = {
    "exist_thresh": (0.2, 0.3, 0.4, 0.5, 0.6),
    "end_thresh": (0.3, 0.5, 0.7, 0.9),
    "align_thresh": (0.5, 0.6, 0.7, 0.8),
    "opp_thresh": (0.3, 0.5, 0.7, 0.85),
    "radius": (2, 3),
    # 0.0 = 클래스 게이트 해제. REF-F에서 clsAcc가 0.04까지 떨어져 있어, `stop_nocand`가
    # 기하 게이트가 아니라 **클래스 게이트** 때문일 가능성이 크다. 그것을 가르는 값이다.
    "min_class_prob": (0.0, 0.02, 0.1, 0.3),
    "purity_thresh": (0.0, 0.4, 0.6, 0.8),
    "end_extend": (0.5, 1.0, 1.5),
    "merge_gap": (0.0, 4.0, 8.0, 12.0),
}


def main() -> None:
    args = parse_args()
    cache = Path(args.cache)
    meta, files = read_meta(cache), list_files(cache, args.count)
    cfg = build_cfg(args.config, args.fixed, meta)
    axes = {name: DEFAULT_AXES[name] for name in _axis_names(args.axes)}
    print(f"[tune] {len(files)}장 · 축 {len(axes)}개 x {args.rounds}바퀴 · 목표 {args.objective}")
    best, trace = descend(cfg, axes, files, meta, args)
    report(best, trace, args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--config", default="configs.base")
    parser.add_argument("--count", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--set", dest="fixed", action="append", default=[])
    parser.add_argument("--axes", default="", help="쉼표 구분. 비우면 전 축")
    parser.add_argument("--rounds", type=int, default=1, help="전 축을 몇 바퀴 돌 것인가")
    parser.add_argument("--objective", default="f1", help="f1 | coverage | ...")
    parser.add_argument("--out", default="")
    return parser.parse_args()


def _axis_names(raw: str) -> list[str]:
    return [n.strip() for n in raw.split(",") if n.strip()] if raw else list(DEFAULT_AXES)


def descend(cfg, axes: dict, files: list[Path], meta: dict, args) -> tuple:
    """축을 순서대로 돌며 목표값을 최대화하는 값으로 하나씩 고정한다."""
    current = cfg.decode
    shape = shape_of(meta)
    score = evaluate_decode(cfg, current, files, shape, args.workers)[args.objective]
    trace = [{"axis": "(start)", "value": None, "score": score, "params": changed_params(current)}]
    print(f"[tune] 시작 {args.objective} = {score:.4f}")
    for _ in range(args.rounds):
        for name, values in axes.items():
            current, score = _best_on_axis(cfg, current, score, name, values, files, shape, args)
            trace.append({"axis": name, "value": getattr(current, name), "score": score})
    return current, trace


def _best_on_axis(cfg, current, score, name, values, files, shape, args):
    """한 축의 후보를 다 재보고, 현재보다 나으면 갱신한다."""
    best_value, best_score = getattr(current, name), score
    for raw in values:
        candidate = replace(current, **{name: cast_like(getattr(current, name), str(raw))})
        if getattr(candidate, name) == getattr(current, name):
            continue
        value = evaluate_decode(cfg, candidate, files, shape, args.workers)[args.objective]
        print(f"    {name}={short(raw):<6} {args.objective}={value:.4f}")
        if value > best_score:
            best_value, best_score = getattr(candidate, name), value
    marker = "  <= 채택" if best_score > score else ""
    print(f"[tune] {name}: {short(best_value)}  {args.objective}={best_score:.4f}{marker}")
    return replace(current, **{name: best_value}), best_score


def report(best, trace: list[dict], args) -> None:
    print("\n[tune] 최종 파라미터 (기본값과 다른 것만)")
    for key, value in changed_params(best).items():
        print(f"  {key} = {short(value)}")
    print(f"[tune] 최종 {args.objective} = {trace[-1]['score']:.4f} (시작 {trace[0]['score']:.4f})")
    if args.out:
        payload = {"best": changed_params(best), "trace": trace}
        Path(args.out).write_text(json.dumps(payload, indent=2, ensure_ascii=False), "utf-8")


if __name__ == "__main__":
    main()

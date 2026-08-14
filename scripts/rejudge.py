"""여러 캐시를 **같은 디코더 설정으로** 디코딩해 한 표에 세운다 (improve-loop 스킬 · D 트랙).

디코더 기본값이 바뀌면 실행마다 다른 설정으로 채점된 값이 남는다 — 훈련 중 검증값을 그대로
비교하면 디코더 차이를 모델 차이로 착각한다(08-11에 실제로 백본이 +64%로 보였다).
이 스크립트는 모든 캐시를 **현재 config**로 다시 디코딩하므로 그 착각이 원천적으로 없다.

사용:
    python scripts/rejudge.py --control U1_ref_val --count 200
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

from configs.base import get_config

# `correctness`(예측이 **GT 선 하나** 위에 머무는 비율)를 뺀 채로 재채점하면 차선 갈아탐이
# 표에 안 나온다 — 08-14에 그 지표를 판정에서 빼 둔 탓에 성적이 두 배로 부풀려져 있었다.
METRICS = (
    "f1",
    "precision",
    "recall",
    "coverage",
    "correctness",
    "chains_per_img",
    "chain_len",
    "frag",
)
POLL_SECONDS = 3
NEUTRAL_BAND = 0.10


def main() -> None:
    args = parse_args()
    caches = sorted(p for p in Path(args.root).iterdir() if p.is_dir())
    scores = measure_all(caches, args)
    report(scores, args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="")
    parser.add_argument("--control", required=True, help="대조군 캐시 이름")
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--workers", type=int, default=2, help="캐시 1개당 디코딩 워커")
    parser.add_argument("--jobs", type=int, default=3, help="동시에 돌릴 캐시 수")
    parser.add_argument("--set", dest="fixed", action="append", default=[])
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    root = Path(get_config().train.output_root).parent / "pred_cache"
    args.root = args.root or str(root)
    return args


def measure_all(caches: list[Path], args: argparse.Namespace) -> dict:
    queue, running, scores = list(caches), [], {}
    while queue or running:
        running = harvest(running, scores)
        while queue and len(running) < args.jobs:
            running.append(launch(queue.pop(0), args))
        time.sleep(POLL_SECONDS)
    return scores


def harvest(running: list, scores: dict) -> list:
    alive = []
    for name, process, out in running:
        if process.poll() is None:
            alive.append((name, process, out))
            continue
        scores[name] = read_scores(out)
        print(f"[rejudge] {name}  f1={scores[name].get('f1', float('nan')):.4f}", flush=True)
    return alive


def read_scores(out: Path) -> dict:
    if not out.exists():
        return {}
    rows = json.loads(out.read_text())
    row = rows[-1] if isinstance(rows, list) else rows
    return {key: float(row[key]) for key in METRICS if key in row}


def launch(cache: Path, args: argparse.Namespace) -> tuple:
    out = Path(args.root).parent / "_rejudge" / f"{cache.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ".venv/bin/python", "scripts/eval_decode.py", "--cache", str(cache),
        "--count", str(args.count), "--workers", str(args.workers), "--out", str(out),
    ]  # fmt: skip
    for item in args.fixed:
        cmd += ["--set", item]
    return (cache.name, subprocess.Popen(cmd, stdout=subprocess.DEVNULL), out)


def report(scores: dict, args: argparse.Namespace) -> None:
    control = scores.get(args.control, {})
    base = control.get("f1")
    print(f"\n[rejudge] 대조군 {args.control}  f1={base}")
    header = f"{'캐시':32}" + "".join(f"{m[:9]:>11}" for m in METRICS) + f"{'판정':>10}"
    print(header)
    for name, row in sorted(scores.items(), key=lambda kv: -kv[1].get("f1", -1)):
        cells = "".join(f"{row.get(m, float('nan')):>11.4f}" for m in METRICS)
        print(f"{name[:32]:32}{cells}{verdict(row.get('f1'), base):>10}")


def verdict(value: float | None, base: float | None) -> str:
    """SKILL 10절의 +-10% 기준. 대조군 자신은 '대조군'으로 표시한다."""
    if value is None or not base:
        return "-"
    if abs(value - base) < 1e-12:
        return "대조군"
    ratio = value / base - 1.0
    if ratio > NEUTRAL_BAND:
        return f"채택 {ratio:+.1%}"
    if ratio < -NEUTRAL_BAND:
        return f"기각 {ratio:+.1%}"
    return f"무효 {ratio:+.1%}"


if __name__ == "__main__":
    main()

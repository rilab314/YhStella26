"""실험 arm을 GPU에 배치해 동시에 돌린다 (improve_plan 4절 ④).

한 라운드 = 가설 하나 + arm 여러 개. arm은 GPU 하나씩 차지하고, 비는 GPU가 생기면
대기 중인 arm이 들어간다. **arm 하나가 죽어도 나머지는 계속 간다** — 무인 실행이 전제다.

사용:
    python scripts/run_experiments.py --round E07 --arms \
        "poswt2:loss.self_slot.end_pos_weight=2" \
        "poswt5:loss.self_slot.end_pos_weight=5"

arm 형식은 `이름:override override ...` 이고, `@config=configs.exp_x` 토큰으로
그 arm만 다른 config 모듈을 쓸 수 있다.
"""

import argparse
import os
import re
import subprocess
import time
from pathlib import Path

from configs.base import get_config

CONFIG_TOKEN = "@config="
OUTPUT_PATTERN = re.compile(r"\[stella\] output -> (\S+)")
POLL_SECONDS = 20


def main() -> None:
    args = parse_args()
    arms = [parse_arm(spec, args.config) for spec in args.arms]
    log_dir = Path(get_config().train.output_root) / "_runner" / args.round
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"[runner] {args.round}: {len(arms)} arms on GPU {args.gpus}  logs -> {log_dir}")
    results = schedule(arms, args, log_dir)
    report(results)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", required=True, help="실험 번호 (결과 문서와 1:1)")
    parser.add_argument("--config", default="configs.unit")
    parser.add_argument("--arms", nargs="+", required=True)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--timeout", type=int, default=21600, help="arm 하나의 상한(초)")
    return parser.parse_args()


def parse_arm(spec: str, default_config: str) -> dict:
    name, _, rest = spec.partition(":")
    tokens = rest.split()
    config = next((t[len(CONFIG_TOKEN) :] for t in tokens if t.startswith(CONFIG_TOKEN)), "")
    overrides = [t for t in tokens if not t.startswith(CONFIG_TOKEN)]
    return {"name": name, "config": config or default_config, "overrides": overrides}


def schedule(arms: list[dict], args: argparse.Namespace, log_dir: Path) -> list[dict]:
    """비는 GPU에 대기 arm을 하나씩 넣는다. 모두 끝날 때까지 블록한다."""
    free = [g.strip() for g in args.gpus.split(",")]
    pending, running, done = list(arms), [], []
    while pending or running:
        while pending and free:
            running.append(launch(pending.pop(0), free.pop(0), args, log_dir))
        time.sleep(POLL_SECONDS)
        for job in list(running):
            if reap(job, args.timeout):
                running.remove(job)
                free.append(job["gpu"])
                done.append(job)
    return done


def launch(arm: dict, gpu: str, args: argparse.Namespace, log_dir: Path) -> dict:
    tag = f"{args.round}_{arm['name']}"
    command = [
        ".venv/bin/python",
        "-m",
        "stella.train.train",
        "--config",
        arm["config"],
        "--tag",
        tag,
    ]
    if arm["overrides"]:
        command += ["--override", *arm["overrides"]]
    log_path = log_dir / f"{arm['name']}.log"
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
    environment.pop("STELLA_RUN_DIR", None)  # arm마다 새 출력 폴더여야 한다
    handle = open(log_path, "w", encoding="utf-8")
    process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, env=environment)
    print(f"[runner] GPU{gpu} <- {tag}  ({' '.join(arm['overrides']) or 'no override'})")
    return {
        **arm,
        "gpu": gpu,
        "tag": tag,
        "log": log_path,
        "handle": handle,
        "process": process,
        "start": time.time(),
    }


def reap(job: dict, timeout: int) -> bool:
    """끝났으면 True. 상한을 넘으면 죽이고 True."""
    code = job["process"].poll()
    if code is None and time.time() - job["start"] < timeout:
        return False
    if code is None:
        job["process"].kill()
        job["process"].wait()
        code = -9
    job["handle"].close()
    job["code"] = code
    job["run_dir"] = find_run_dir(job["log"])
    minutes = (time.time() - job["start"]) / 60
    print(f"[runner] done {job['tag']} code={code} ({minutes:.0f} min) -> {job['run_dir']}")
    return True


def find_run_dir(log_path: Path) -> str:
    match = OUTPUT_PATTERN.search(log_path.read_text(encoding="utf-8", errors="ignore"))
    return match.group(1) if match else ""


def report(results: list[dict]) -> None:
    print("\n[runner] 요약")
    for job in results:
        status = "ok" if job["code"] == 0 else f"FAIL({job['code']})"
        print(f"  {job['tag']:<28} {status:<10} {job['run_dir']}")
    print("\n다음: .venv/bin/python scripts/summarize_runs.py --last 8")


if __name__ == "__main__":
    main()

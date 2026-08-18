"""여러 실행의 예측을 GPU에 나눠 한꺼번에 캐시로 뽑는다 (research 스킬 · D 트랙).

디코더 기본값이 바뀌면 **이전 실행의 훈련 중 검증값은 쓸 수 없다** — 실행 폴더가 시작 시
config를 고정하기 때문이다. 예측 자체는 디코더와 무관하므로, 캐시를 떠 두면 모든 실행을
같은 디코더 설정으로 다시 판정할 수 있다. 그 캐시를 뽑는 것이 이 스크립트다.

사용:
    python scripts/dump_many.py --runs 260810_..._E09_p05 260811_..._E14_swin --gpus 0,1,2,3
"""

import argparse
import subprocess
import time
from pathlib import Path

from configs.base import get_config

POLL_SECONDS = 5


def main() -> None:
    args = parse_args()
    jobs = [Job(run=name, out=Path(args.out) / cache_name(name)) for name in args.runs]
    pending = [job for job in jobs if args.force or not job.out.exists()]
    print(f"[dump] {len(jobs)}개 중 {len(pending)}개 필요 (나머지는 캐시 있음)")
    results = schedule(pending, args)
    report(results)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True, help="실행 폴더 이름 또는 절대경로")
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--workers", type=int, default=2, help="덤프 1개당 데이터로더 워커")
    parser.add_argument("--log-root", default="")
    parser.add_argument("--out", default="")
    parser.add_argument("--force", action="store_true", help="캐시가 있어도 다시 뽑는다")
    args = parser.parse_args()
    root = Path(get_config().train.output_root)
    args.log_root = args.log_root or str(root)
    args.out = args.out or str(root.parent / "pred_cache")
    return args


def cache_name(run: str) -> str:
    """실행 폴더 이름에서 캐시 이름을 만든다 — 날짜·규격을 떼고 실험 ID만 남긴다."""
    stem = Path(run).name
    parts = stem.split("_unit_", 1)
    return (parts[1] if len(parts) == 2 else stem) + "_val"


class Job:
    def __init__(self, *, run: str, out: Path):
        self.run = run
        self.out = out
        self.process: subprocess.Popen | None = None
        self.gpu = -1


def schedule(jobs: list[Job], args: argparse.Namespace) -> list[Job]:
    gpus = [int(g) for g in args.gpus.split(",") if g != ""]
    queue, running, done = list(jobs), [], []
    while queue or running:
        running, done = collect(running, done)
        while queue and len(running) < len(gpus):
            free = next(g for g in gpus if g not in [job.gpu for job in running])
            running.append(launch(queue.pop(0), free, args))
        time.sleep(POLL_SECONDS)
    return done


def collect(running: list[Job], done: list[Job]) -> tuple[list[Job], list[Job]]:
    alive = []
    for job in running:
        if job.process.poll() is None:
            alive.append(job)
        else:
            print(f"[dump] 끝 {job.out.name}  code={job.process.returncode}", flush=True)
            done.append(job)
    return alive, done


def launch(job: Job, gpu: int, args: argparse.Namespace) -> Job:
    job.gpu = gpu
    run_dir = job.run if Path(job.run).is_absolute() else str(Path(args.log_root) / job.run)
    cmd = [
        ".venv/bin/python", "scripts/dump_predictions.py",
        "--run", run_dir, "--count", str(args.count), "--out", str(job.out),
        "--override", f"data.num_workers={args.workers}",
    ]  # fmt: skip
    env = {"CUDA_VISIBLE_DEVICES": str(gpu)}
    print(f"[dump] GPU{gpu} <- {job.out.name}", flush=True)
    job.process = subprocess.Popen(cmd, env={**os_environ(), **env})
    return job


def os_environ() -> dict:
    import os

    return dict(os.environ)


def report(jobs: list[Job]) -> None:
    failed = [job for job in jobs if job.process.returncode != 0]
    print(f"\n[dump] 완료 {len(jobs) - len(failed)} · 실패 {len(failed)}")
    for job in failed:
        print(f"  FAIL {job.out.name}")


if __name__ == "__main__":
    main()

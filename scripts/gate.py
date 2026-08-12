"""PR 전에 반드시 통과해야 하는 단일 관문 (improve-loop · 게이트).

**문서에 적힌 규칙은 지켜지지 않고, 실행되는 스크립트는 지켜진다.** 실제로 "변형을 넣었으면
GT 주입으로 회귀 확인부터" 라는 규칙이 SKILL에 있었지만 E06에서 건너뛰어졌다. 그래서
검사를 파일 하나로 모았다.

검사 목록과 하한값은 `gate_baseline.json` 에 있다 — **코드가 아니라 데이터다.** 새로운
실패 양식을 만나면 코드를 고치지 않고 JSON에 항목을 하나 더한다.

사용:
    python scripts/gate.py                       # 전부
    python scripts/gate.py --only pytest,ruff    # 빠른 것만
    python scripts/gate.py --update              # 통과한 실측으로 하한을 래칫
    python scripts/gate.py --workers 2           # 학습이 도는 중이면 낮춘다
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from configs.base import get_config
from stella.builder import check_all
from stella.config_io import load_config
from stella.decode.sweep import build_cfg, evaluate_decode, list_files, read_meta, shape_of

REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINE_PATH = REPO_ROOT / "gate_baseline.json"
CONFIG_DIR = REPO_ROOT / "configs"
PYTHON_BIN = str(REPO_ROOT / ".venv/bin/python")
CACHE_DIRNAME = "pred_cache"
RATCHET_MARGIN = 0.10  # 실측이 하한을 이만큼 웃돌면 하한을 끌어올린다
LOAD_LIMIT = 16.0  # 32코어의 절반. 학습 중에 게이트를 돌리면 부하가 31까지 뛴다(실측)
STATUS_MARK = {True: "PASS", False: "FAIL", None: "SKIP"}


def main() -> None:
    args = parse_args()
    refuse_if_busy(args)
    spec = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    checks = [item for item in spec["checks"] if selected(item, args)]
    results = [run_check(item, args) for item in checks]
    print_report(results)
    if args.update:
        ratchet(spec, results)
    sys.exit(exit_code(results, args.strict))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="", help="쉼표로 구분한 검사 id")
    parser.add_argument("--skip", default="", help="쉼표로 구분한 검사 id")
    parser.add_argument("--workers", type=int, default=4, help="디코딩 워커 (학습 중이면 낮춘다)")
    parser.add_argument("--update", action="store_true", help="통과한 실측으로 하한 래칫")
    parser.add_argument("--strict", action="store_true", help="SKIP도 실패로 센다")
    parser.add_argument("--max-load", type=float, default=LOAD_LIMIT, help="이 부하 위면 거부")
    parser.add_argument("--force", action="store_true", help="부하 가드를 무시하고 강행")
    return parser.parse_args()


def refuse_if_busy(args: argparse.Namespace) -> None:
    """부하가 높으면 아예 시작하지 않는다.

    실측: 학습 4 arm(부하 13)이 도는 중에 게이트를 돌렸더니 **부하가 31까지 뛰었다.**
    pytest와 디코딩은 둘 다 CPU를 많이 쓴다. 게이트는 급한 일이 아니므로 자원이 빌 때
    돌리는 것이 맞다 — 사람이 같은 CPU로 편집기를 쓴다.
    """
    load = os.getloadavg()[0]
    if load <= args.max_load or args.force:
        return
    print(f"[gate] 부하 {load:.1f} > 상한 {args.max_load} — 시작하지 않는다.")
    print("[gate] 자원이 빈 뒤 다시 돌리거나, 정말 지금 해야 하면 --force")
    sys.exit(2)


def selected(item: dict, args: argparse.Namespace) -> bool:
    only = [s for s in args.only.split(",") if s]
    skip = [s for s in args.skip.split(",") if s]
    return (not only or item["id"] in only) and item["id"] not in skip


def run_check(item: dict, args: argparse.Namespace) -> dict:
    """검사 종류별 실행기를 부르고 소요 시간을 붙인다."""
    runners = {
        "command": run_command,
        "configs": run_configs,
        "invariant": run_invariant,
        "install": run_install,
        "decode": run_decode,
    }
    started = time.time()
    print(f"[gate] {item['id']} ...", flush=True)
    outcome = runners[item["kind"]](item, args)
    return {**item, **outcome, "seconds": time.time() - started}


def run_command(item: dict, args: argparse.Namespace) -> dict:
    completed = subprocess.run(item["cmd"], capture_output=True, text=True)
    tail = (completed.stdout + completed.stderr).strip().splitlines()[-3:]
    return {"ok": completed.returncode == 0, "message": " / ".join(tail)[:160], "observed": {}}


def run_configs(item: dict, args: argparse.Namespace) -> dict:
    """모든 config 모듈을 실제로 해석해 본다 — path+name 오타를 학습 전에 잡는다."""
    modules = sorted(
        p.stem for p in CONFIG_DIR.glob("*.py") if p.stem not in ("schema", "__init__")
    )
    broken = []
    for name in modules:
        try:
            check_all(load_config(f"configs.{name}", []))
        except Exception as error:  # noqa: BLE001 — 어떤 예외든 게이트 실패로 본다
            broken.append(f"{name}: {type(error).__name__} {error}")
    return {
        "ok": not broken,
        "message": " / ".join(broken)[:160] or f"{len(modules)}개 해석",
        "observed": {},
    }


def run_invariant(item: dict, args: argparse.Namespace) -> dict:
    """config 값들 **사이의 관계**를 모든 config 모듈에서 검사한다.

    개별 값이 아니라 관계를 보는 이유는, 한 값을 실측으로 고치면 그 값을 전제로 정해진
    다른 값이 조용히 틀리기 때문이다 — `radius`를 2에서 24로 키운 뒤 `w_dist`가 그대로라
    디코더가 정점을 건너뛰던 것이 실제 사례다.
    """
    modules = sorted(
        p.stem for p in CONFIG_DIR.glob("*.py") if p.stem not in ("schema", "__init__")
    )
    broken = []
    for name in modules:
        cfg = load_config(f"configs.{name}", [])
        try:
            if not eval(item["assert"], {"__builtins__": {}}, {"cfg": cfg}):  # noqa: S307
                broken.append(name)
        except Exception as error:  # noqa: BLE001 — 식이 틀린 것도 게이트 실패다
            broken.append(f"{name}: {type(error).__name__} {error}")
    message = f"{item['assert']} — 위반 {', '.join(broken)}" if broken else item["assert"]
    return {"ok": not broken, "message": message[:160], "observed": {}}


def run_install(item: dict, args: argparse.Namespace) -> dict:
    """editable 설치가 **이 저장소**를 가리키는지 본다.

    `.venv`는 worktree와 공유된다. worktree에서 `uv sync`가 한 번 돌면 포인터가 그쪽으로
    옮겨가고, 그 뒤 `python scripts/*.py`는 **남의 코드**로 돈다(`-m` 실행은 cwd가 우선해
    멀쩡하므로 학습만 보면 눈치채지 못한다). 저장소 밖에서 import 해야 진짜로 잡힌다.
    """
    probe = "import stella, configs; print(stella.__file__); print(configs.__file__)"
    completed = subprocess.run(
        [PYTHON_BIN, "-c", probe], capture_output=True, text=True, cwd="/tmp", check=False
    )
    paths = [Path(line) for line in completed.stdout.strip().splitlines()]
    stray = [str(p.parent) for p in paths if REPO_ROOT not in p.resolve().parents]
    if completed.returncode != 0 or not paths:
        return {"ok": False, "message": completed.stderr.strip()[-160:], "observed": {}}
    message = f"딴 곳을 가리킨다: {stray}" if stray else f"{REPO_ROOT} 확인"
    return {"ok": not stray, "message": message, "observed": {}}


def run_decode(item: dict, args: argparse.Namespace) -> dict:
    """캐시된 예측을 현재 기본 설정으로 디코딩해 지표가 하한 위인지 본다."""
    cache = cache_root() / item["cache"]
    if not cache.exists():
        return {"ok": None, "message": f"캐시 없음: {cache}", "observed": {}}
    scores = decode_scores(cache, item, args.workers)
    failures = [
        f"{key}={scores.get(key, float('nan')):.4f} < {bound['min']}"
        for key, bound in item["bounds"].items()
        if scores.get(key) is None or scores[key] < bound["min"]
    ]
    observed = {key: scores.get(key) for key in item["bounds"]}
    detail = " ".join(f"{k}={v:.4f}" for k, v in observed.items() if v is not None)
    return {"ok": not failures, "message": " / ".join(failures) or detail, "observed": observed}


def decode_scores(cache: Path, item: dict, workers: int) -> dict:
    meta = read_meta(cache)
    files = list_files(cache, item.get("count", 0))
    cfg = build_cfg(item.get("config", "configs.base"), item.get("set", []), meta)
    return evaluate_decode(cfg, cfg.decode, files, shape_of(meta), workers)


def cache_root() -> Path:
    """예측 캐시는 로그 루트 옆에 둔다 (기계마다 경로가 달라 config에서 끌어온다)."""
    return Path(get_config().train.output_root).parent / CACHE_DIRNAME


def print_report(results: list[dict]) -> None:
    print("\n[gate] 결과")
    for row in results:
        mark = STATUS_MARK[row["ok"]]
        print(f"  {mark:<5} {row['id']:<16} {row['seconds']:6.1f}s  {row['message']}")
    failed = [row["id"] for row in results if row["ok"] is False]
    skipped = [row["id"] for row in results if row["ok"] is None]
    print(
        f"\n[gate] 실패 {len(failed)} · 건너뜀 {len(skipped)}" + (f" -> {failed}" if failed else "")
    )


def ratchet(spec: dict, results: list[dict]) -> None:
    """좋아진 만큼 하한을 끌어올린다. 내리는 방향으로는 절대 자동으로 움직이지 않는다."""
    lifted = []
    for row in results:
        if row["ok"] is not True or not row.get("observed"):
            continue
        target = next(item for item in spec["checks"] if item["id"] == row["id"])
        for key, value in row["observed"].items():
            floor = value * (1 - RATCHET_MARGIN)
            if value is not None and floor > target["bounds"][key]["min"]:
                lifted.append(f"{row['id']}.{key} {target['bounds'][key]['min']:.4f}->{floor:.4f}")
                target["bounds"][key]["min"] = round(floor, 4)
    BASELINE_PATH.write_text(
        json.dumps(spec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"[gate] 하한 갱신 {len(lifted)}건 " + (" · ".join(lifted) if lifted else "(변화 없음)"))


def exit_code(results: list[dict], strict: bool) -> int:
    bad = [row for row in results if row["ok"] is False or (strict and row["ok"] is None)]
    return 1 if bad else 0


if __name__ == "__main__":
    main()

"""대기열의 다음 라운드를 자원이 비는 대로 실행하고 판정까지 한다 (research · 디스패처).

`watch.sh` 는 "자원이 비었다"를 알리고 끝난다 — 누군가 깨어나 다음을 올려야 했다.
무인 운영에서 GPU가 노는 가장 흔한 원인은 잘못된 실험이 아니라 **끝난 줄 모름**이고,
그 다음이 **깨어날 사람이 없음**이다. 이 프로세스는 큐가 있는 한 스스로 채운다.

**하지 않는 것**: 커밋·푸시·PR·병합·삭제·코드 수정. 실험 실행과 기록까지다.
판단이 필요한 사건(채택 발생 · 큐 소진 · 연속 무소득 · 가드 장기 위반)에서는 멈추고
종료한다 — 그 종료가 곧 알림이다.

사용:
    python scripts/dispatch.py                     # experiment/queue.json 을 소비
    python scripts/dispatch.py --keep-going        # 채택이 나와도 멈추지 않는다
    python scripts/dispatch.py --dry-run           # 무엇을 실행할지만 찍는다

큐 형식 (`experiment/queue.json`) — 한 항목이 라운드 하나다:

    {"items": [{
        "id": "E10",                       # 실행 태그이자 판정 키워드
        "arms": ["w9:model.window_size=9", "rot10:data.aug_rotate_deg=10"],
        "config": "configs.unit",          # 생략하면 configs.unit
        "control": "U1_ref",               # 대조군 폴더명 조각 (다른 라운드 재사용 가능)
        "watch": "val/cell/vertex_recall", # 가설이 지목한 지표
        "status": "pending"                # pending|running|done|failed
    }]}

**4 GPU를 채우도록 arm을 4개 안팎으로 설계한다** — 항목은 순차 소비라 arm이 적으면 슬롯이 논다.
"""

import argparse
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

PYTHON = ".venv/bin/python"
QUEUE_DEFAULT = Path("experiment/queue.json")
JOURNAL_DIR = Path("experiment")
GPU_FREE_MIB = 2000  # 이보다 적게 쓰면 빈 슬롯으로 본다
LOAD_LIMIT = 16.0  # 32코어의 절반 — 사람이 같은 CPU를 쓴다
GUARD_PATIENCE = 30  # 가드 위반이 이만큼 연속되면 사람을 부른다
BARREN_LIMIT = 2  # 연속 이 횟수만큼 채택 0이면 그 축은 소진됐다


def main() -> None:
    args = parse_args()
    if args.validate:
        report_queue(args.queue)
        return
    dispatcher = Dispatcher(args=args)
    reason = dispatcher.serve()
    print(f"\n=== dispatch 종료: {reason} ===")
    print(f"시각 {datetime.now():%Y-%m-%d %H:%M:%S} · 큐 {args.queue}")
    print("다음: 판정 결과를 확인하고 STATE.md를 갱신하라")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", type=Path, default=QUEUE_DEFAULT)
    parser.add_argument("--journal", type=Path, default=None, help="기본은 result_{mmdd}.md")
    parser.add_argument("--interval", type=int, default=60, help="자원 확인 간격(초)")
    parser.add_argument("--max-load", type=float, default=LOAD_LIMIT)
    parser.add_argument("--keep-going", action="store_true", help="채택이 나와도 멈추지 않는다")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate", action="store_true", help="큐 형식만 검사하고 끝낸다")
    return parser.parse_args()


def report_queue(path: Path) -> None:
    """큐를 열어 대기 항목을 찍는다. 형식이 틀리면 RoundQueue가 예외로 알린다."""
    queue = RoundQueue(path=path)
    counts: dict[str, int] = {}
    for item in queue.items:
        counts[item["status"]] = counts.get(item["status"], 0) + 1
    pending = queue.next_pending()
    print(f"[queue] {path} · 항목 {len(queue.items)}개 · {counts}")
    print(f"[queue] 다음 대기: {pending['id'] if pending else '(없음 — 큐 소진)'}")


class Dispatcher:
    """큐 → 자원 대기 → 실행 → 판정 → 기록 을 반복한다. 판단이 필요하면 멈춘다."""

    def __init__(self, *, args: argparse.Namespace) -> None:
        self.args = args
        self.queue = RoundQueue(path=args.queue)
        self.watcher = SlotWatcher(max_load=args.max_load)
        self.journal = args.journal or JOURNAL_DIR / f"result_{datetime.now():%m%d}.md"
        self.barren = 0

    def serve(self) -> str:
        while True:
            item = self.queue.next_pending()
            if item is None:
                return "큐 소진 — 새 가설이 필요하다"
            stop = self.serve_one(item)
            if stop:
                return stop

    def serve_one(self, item: dict) -> str:
        """라운드 하나를 끝까지 본다. 멈춰야 할 사유가 생기면 그 문자열을 낸다."""
        need = len(item["arms"])
        if self.args.dry_run:  # 자원을 기다리지 않는다 — 무엇을 실행할지만 보여 주는 모드다
            print(f"[dispatch] (dry-run) {item['id']} · arm {need}개 · {item['arms']}")
            self.queue.mark(item, "dry-run")
            return "dry-run 완료"
        gpus = self.await_slots(need)
        if gpus is None:
            return f"가드 위반이 {GUARD_PATIENCE}회 연속 — 부하를 사람이 확인해야 한다"
        print(f"[dispatch] {item['id']} 실행 · GPU {gpus} · arm {need}개")
        self.queue.mark(item, "running")
        code = self.launch(item, gpus)
        self.queue.mark(item, "done" if code == 0 else f"failed({code})")
        if code != 0:
            return f"{item['id']} 실행 실패 (code={code})"
        return self.settle(item)

    def await_slots(self, need: int) -> str | None:
        """필요한 만큼 GPU가 비고 부하가 상한 아래일 때까지 기다린다."""
        violations = 0
        while True:
            free = self.watcher.free_gpus()
            if len(free) >= need and self.watcher.load_ok():
                return ",".join(free)
            violations = violations + 1 if len(free) >= need else 0
            if violations >= GUARD_PATIENCE:
                return None
            time.sleep(self.args.interval)

    def launch(self, item: dict, gpus: str) -> int:
        command = [
            PYTHON,
            "scripts/run_experiments.py",
            "--round",
            item["id"],
            "--config",
            item.get("config", "configs.unit"),
            "--gpus",
            gpus,
            "--arms",
            *item["arms"],
        ]
        return subprocess.run(command, check=False).returncode

    def settle(self, item: dict) -> str:
        """판정 → 일지 기록 → 정지 조건 확인. 이 셋은 한 묶음이다."""
        verdict = self.judge(item)
        adopted = verdict.get("adopted", [])
        self.barren = 0 if adopted else self.barren + 1
        self.queue.mark(item, "done", adopted=adopted)
        if adopted and not self.args.keep_going:
            return f"{item['id']} 채택 {len(adopted)}건 — 반영 여부를 사람이 정한다"
        if self.barren >= BARREN_LIMIT:
            return f"연속 {self.barren}라운드 채택 0 — 이 축은 소진됐다"
        return ""

    def judge(self, item: dict) -> dict:
        out = Path(f"/tmp/judge_{item['id']}.json")
        command = [
            PYTHON,
            "scripts/judge_round.py",
            "--round",
            item["id"],
            "--control",
            item.get("control", ""),
            "--watch",
            item.get("watch", "val/cell/vertex_recall"),
            "--md",
            str(self.journal),
            "--json",
            str(out),
        ]
        if subprocess.run(command, check=False).returncode != 0 or not out.exists():
            print("[dispatch] 판정 실패 — 수치를 사람이 봐야 한다")
            return {}
        return json.loads(out.read_text(encoding="utf-8"))


class RoundQueue:
    """`queue.json` 하나가 대기열의 단일 출처다. 상태 전이를 그 파일에 즉시 쓴다."""

    STATUSES = ("pending", "running", "done", "failed", "dry-run")

    def __init__(self, *, path: Path) -> None:
        self.path = path
        self.data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"items": []}
        self._check_shape()

    def _check_shape(self) -> None:
        """형식 오류를 여기서 이름 붙여 던진다. 안 그러면 배정 도중 TypeError로 죽고
        **GPU가 밤새 논다** — 실제로 큐가 bare list라 배정기가 즉시 죽어 있었다."""
        if not isinstance(self.data, dict) or not isinstance(self.data.get("items"), list):
            raise ValueError(f"{self.path}: 최상위는 {{'items': [...]}} 여야 한다")
        for index, item in enumerate(self.data["items"]):
            self._check_item(index, item)

    def _check_item(self, index: int, item: dict) -> None:
        """`arms`는 배정 대상에만 요구한다 — 지나간 기록과 D 규격 항목에는 arm이 없다."""
        where = f"{self.path}[{index}]"
        for key in ("id", "status"):
            if key not in item:
                raise ValueError(f"{where}: 필수 키 '{key}'가 없다")
        if item["status"].split("(")[0] not in self.STATUSES:
            raise ValueError(f"{where}: status '{item['status']}' 는 {self.STATUSES} 중 하나다")
        if item["status"] != "pending":
            return
        if not isinstance(item.get("arms"), list) or not item["arms"]:
            raise ValueError(f"{where}: pending 항목의 arms 는 비어 있지 않은 리스트여야 한다")

    @property
    def items(self) -> list[dict]:
        return self.data["items"]

    def next_pending(self) -> dict | None:
        return next((item for item in self.items if item.get("status") == "pending"), None)

    def mark(self, item: dict, status: str, adopted: list | None = None) -> None:
        item["status"] = status
        item["at"] = f"{datetime.now():%Y-%m-%d %H:%M}"
        if adopted is not None:
            item["adopted"] = adopted
        self.path.write_text(
            json.dumps(self.data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )


class SlotWatcher:
    """자원 관측만 한다 — 무엇을 할지는 Dispatcher가 정한다."""

    def __init__(self, *, max_load: float) -> None:
        self.max_load = max_load

    def free_gpus(self) -> list[str]:
        query = "--query-gpu=index,memory.used"
        output = subprocess.run(
            ["nvidia-smi", query, "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout
        rows = [line.split(",") for line in output.strip().splitlines() if line.strip()]
        return [index.strip() for index, used in rows if int(used) < GPU_FREE_MIB]

    def load_ok(self) -> bool:
        return os.getloadavg()[0] <= self.max_load


if __name__ == "__main__":
    main()

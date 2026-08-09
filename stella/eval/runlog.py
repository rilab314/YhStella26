"""학습 실행 폴더의 `metrics.csv`를 읽어 에폭 평균 지표를 낸다 (improve-loop · 관측·판정 공용).

`scripts/summarize_runs.py`(사람이 보는 표)와 `scripts/judge_round.py`(자동 판정)가 같은
함수를 쓴다. **읽는 곳이 하나여야 판정이 사람 눈과 스크립트에서 갈라지지 않는다.**

마지막 에폭 하나는 크게 튀므로 비교는 항상 **마지막 N 에폭 평균**으로 한다 (판정 규칙 2).
"""

import csv
from pathlib import Path

EPOCH_KEY = "epoch"
PRESENCE_KEY = "val/inst/f1"  # 이 값이 있어야 "평가가 끝난 에폭"이다


def latest_runs(root: Path, count: int) -> list[Path]:
    """로그 루트에서 최근 실행 N개. 폴더명이 `YYMMDD_HHMMSS_...` 라 이름순 = 시간순."""
    return sorted(finished_runs(root), key=lambda p: p.name)[-count:]


def find_runs(root: Path, keyword: str) -> list[Path]:
    """폴더명에 keyword가 든 실행 전부 (라운드 태그로 arm을 모을 때 쓴다)."""
    return sorted((p for p in finished_runs(root) if keyword in p.name), key=lambda p: p.name)


def finished_runs(root: Path):
    return (p for p in root.iterdir() if (p / "metrics.csv").exists())


def tail_mean(run: Path, tail: int, keys: tuple[str, ...]) -> dict | None:
    """마지막 `tail` 에폭의 평균. 평가된 에폭이 하나도 없으면 None."""
    merged = merge_by_epoch(run / "metrics.csv")
    epochs = [e for e in sorted(merged) if PRESENCE_KEY in merged[e]]
    if not epochs:
        return None
    window = epochs[-tail:]
    values = {key: mean_of(merged, window, key) for key in keys}
    return {"name": run.name, "path": str(run), "epochs": len(epochs), **values}


def merge_by_epoch(path: Path) -> dict[int, dict]:
    """Lightning은 train/val 스칼라를 다른 행에 쓴다 — 에폭 기준으로 합친다."""
    merged: dict[int, dict] = {}
    with open(path, encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            target = merged.setdefault(int(row[EPOCH_KEY]), {})
            target.update({k: v for k, v in row.items() if v not in ("", None)})
    return merged


def mean_of(merged: dict, epochs: list[int], key: str) -> float | None:
    values = [float(merged[e][key]) for e in epochs if key in merged[e]]
    return sum(values) / len(values) if values else None


def relative_change(value: float | None, base: float | None) -> float | None:
    """대조군 대비 상대 변화. 대조군이 0이거나 값이 없으면 None."""
    if value is None or base is None or base == 0:
        return None
    return (value - base) / abs(base)

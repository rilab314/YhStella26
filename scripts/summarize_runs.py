"""여러 실행을 한 표로 비교한다 — 개선 루프의 ① 관측 단계 (research 스킬 · 관측).

마지막 에폭 하나는 크게 튀므로 **마지막 N 에폭 평균**으로 비교한다(판정 규칙 2).
표는 두 블록으로 나눠 찍는다 — 인스턴스 지표(무엇이 나쁜가)와 셀·디코더 진단(왜 나쁜가).

사용:
    python scripts/summarize_runs.py --last 8
    python scripts/summarize_runs.py --runs <폴더> <폴더> --tail 3
"""

import argparse
from pathlib import Path

from configs.base import get_config
from stella.eval.runlog import latest_runs, tail_mean

INSTANCE_COLUMNS = (
    ("val/inst/f1", "f1"),
    ("val/inst/f1_macro", "f1_mac"),
    ("val/inst/precision", "prec"),
    ("val/inst/recall", "recall"),
    ("val/inst/coverage", "cover"),
    ("val/inst/correctness", "correct"),
    ("val/inst/frag", "frag"),
    ("val/inst/frag_strict", "fragS"),
    ("val/inst/rms", "rms"),
    ("val/total", "vloss"),
)
DIAGNOSTIC_COLUMNS = (
    ("val/cell/heat_recall", "heatR"),
    ("val/cell/heat_precision", "heatP"),
    ("val/cell/heat_pos", "heatPos"),
    ("val/cell/heat_neg", "heatNeg"),
    ("val/cell/class_acc", "clsAcc"),
    ("val/cell/class_fg", "clsFg"),
    ("val/cell/class_recall", "clsRec"),
    ("val/cell/vertex_recall", "vtxRec"),
    ("val/cell/class_bg_recall", "bgAcc"),
    ("val/cell/coord_err_px", "coordPx"),
    ("val/cell/end_recall", "endR"),
    ("val/cell/end_precision", "endP"),
    ("val/cell/end_pos", "endPos"),
    ("val/cell/end_neg", "endNeg"),
    ("val/cell/dir_err_deg", "dirDeg"),
    ("val/cell/link_ok", "linkOk"),
    ("val/cell/chain_expect", "chainEx"),
    ("val/dec/chain_len", "decLen"),
    ("val/dec/stop_end", "stpEnd"),
    ("val/dec/stop_nocand", "stpNo"),
    ("val/dec/vertex_used", "vtxUsed"),
)


def main() -> None:
    args = parse_args()
    root = Path(get_config().train.output_root)
    runs = [Path(p) for p in args.runs] if args.runs else latest_runs(root, args.last)
    rows = [tail_mean(run, args.tail, all_keys()) for run in runs]
    rows = [row for row in rows if row]
    print_block("인스턴스 지표", rows, INSTANCE_COLUMNS)
    print_block("셀·디코더 진단", rows, DIAGNOSTIC_COLUMNS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--last", type=int, default=8, help="최근 실행 N개")
    parser.add_argument("--runs", nargs="*", default=[])
    parser.add_argument("--tail", type=int, default=3, help="마지막 N 에폭 평균")
    return parser.parse_args()


def all_keys() -> tuple[str, ...]:
    return tuple(key for key, _ in INSTANCE_COLUMNS + DIAGNOSTIC_COLUMNS)


def print_block(title: str, rows: list[dict], columns: tuple) -> None:
    print(f"\n[{title}]  (마지막 에폭 평균)")
    print("run".ljust(42) + "ep " + " ".join(f"{label:>8}" for _, label in columns))
    for row in rows:
        cells = " ".join(f"{_fmt(row.get(key)):>8}" for key, _ in columns)
        print(f"{row['name'][-42:]:<42}{row['epochs']:<3}" + cells)


def _fmt(value) -> str:
    if value is None:
        return "-"
    return f"{value:.4f}" if abs(value) < 100 else f"{value:.1f}"


if __name__ == "__main__":
    main()

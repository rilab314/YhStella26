"""학습 결과 폴더의 metrics.csv를 표로 보여준다.

사용: python scripts/show_run.py                    # 가장 최근 실행
      python scripts/show_run.py --run <경로> --classes
"""

import argparse
import csv
from pathlib import Path

from configs.base import get_config

LOSS_COLUMNS = (
    ("train/total", "total"),
    ("train/heatmap/focal", "heat"),
    ("train/self_slot/class", "cls"),
    ("train/self_slot/coord", "coord"),
    ("train/self_slot/end", "end"),
    ("train/conn/exist", "exist"),
    ("train/conn/dir", "dir"),
    ("train/conn/match_ambiguity", "ambig"),
)
VAL_COLUMNS = (
    ("val/total", "vtotal"),
    ("val/inst/f1", "f1"),
    ("val/inst/f1_macro", "f1_mac"),
    ("val/inst/coverage", "cover"),
    ("val/inst/correctness", "correct"),
    ("val/inst/rms", "rms"),
    ("val/inst/frag", "frag"),
)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run) if args.run else latest_run()
    print(f"[run] {run_dir}")
    rows = list(csv.DictReader(open(run_dir / "metrics.csv", encoding="utf-8")))
    print_table(merge_by_epoch(rows))
    if args.classes:
        print_class_scores(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="", help="비우면 output_root 에서 가장 최근 폴더")
    parser.add_argument("--classes", action="store_true", help="클래스별 F1도 출력")
    return parser.parse_args()


def latest_run() -> Path:
    root = Path(get_config().train.output_root)
    runs = sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.name)
    if not runs:
        raise SystemExit(f"{root} 에 실행 폴더가 없다")
    return runs[-1]


def merge_by_epoch(rows: list[dict]) -> dict[int, dict]:
    """Lightning은 train/val 스칼라를 다른 행에 쓴다 — 에폭 기준으로 합친다."""
    merged: dict[int, dict] = {}
    for row in rows:
        epoch = int(row["epoch"])
        target = merged.setdefault(epoch, {})
        target.update({k: v for k, v in row.items() if v not in ("", None)})
    return merged


def print_table(merged: dict[int, dict]) -> None:
    columns = LOSS_COLUMNS + VAL_COLUMNS
    print("ep  " + " ".join(f"{label:>7}" for _, label in columns))
    for epoch in sorted(merged):
        values = (_fmt(merged[epoch].get(key)) for key, _ in columns)
        print(f"{epoch:<3} " + " ".join(f"{v:>7}" for v in values))


def print_class_scores(rows: list[dict]) -> None:
    merged = merge_by_epoch(rows)
    last = merged[max(merged)]
    names = sorted(k for k in last if k.startswith("val/inst/f1/"))
    if not names:
        return
    print("\n[마지막 에폭 클래스별 F1]")
    for key in names:
        print(f"  {key.split('/')[-1]:<32} {_fmt(last[key])}")


def _fmt(value) -> str:
    if value in (None, ""):
        return "-"
    return f"{float(value):.4f}"


if __name__ == "__main__":
    main()

"""라운드 하나를 판정한다 — SKILL 10절 판정 규칙을 그대로 코드로 옮긴 것.

사람이 표를 눈으로 읽고 "채택/기각"을 말하면 세션마다 미묘하게 달라진다. 규칙이 이미
기계적이므로(마지막 3에폭 평균 · 대조군 대비 ±10% · 지목 지표가 실제로 움직였는가)
스크립트가 판정하고 사람은 해석만 한다.

사용:
    python scripts/judge_round.py --round E08 --control U1_ref
    python scripts/judge_round.py --round E08 --control U1_ref --watch val/cell/vertex_recall \
        --md experiment/result_0810.md --json /tmp/e08.json

`--md` 는 마크다운 표를 그 파일에 덧붙인다 (일지에 바로 들어가는 형식).
`--json` 은 dispatch가 정지 조건을 판단할 때 읽는다.
"""

import argparse
import json
from pathlib import Path

from configs.base import get_config
from stella.eval.runlog import find_runs, relative_change, tail_mean

PRIMARY = "val/inst/f1"
DEFAULT_WATCH = "val/cell/vertex_recall"
REPORT_KEYS = (
    PRIMARY,
    "val/inst/coverage",
    "val/inst/frag",
    "val/cell/heat_recall",
    "val/cell/class_fg",
    DEFAULT_WATCH,
)
SHORT_LABEL = {
    PRIMARY: "f1",
    "val/inst/coverage": "coverage",
    "val/inst/frag": "frag",
    "val/cell/heat_recall": "heatR",
    "val/cell/class_fg": "clsFg",
    DEFAULT_WATCH: "vtxRec",
}
ADOPT_MARGIN = 0.10  # 상대 +10% 이상 채택 (규칙 3)
REJECT_MARGIN = -0.10  # 상대 −10% 이하 기각
MOVE_MARGIN = 0.05  # 지목 지표가 "움직였다"의 하한 (규칙 4). 측정 잡음보다 크게 잡았다
VERDICT_NOTE = {
    "adopt": "채택",
    "hold": "보류 — f1은 올랐으나 지목 지표가 안 움직였다. 재현 필요",
    "neutral": "무효",
    "reject": "기각",
    "control": "대조군",
}


def main() -> None:
    args = parse_args()
    rows = collect_rows(args)
    if not rows:
        raise SystemExit(f"[judge] '{args.round}' 로 찾은 실행이 없다")
    control = pick_control(rows, args.control)
    judged = [judge_row(row, control, args.watch) for row in rows]
    report = render(judged, args, control)
    print(report)
    emit(report, judged, args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", required=True, help="실행 폴더명에 든 라운드 키워드 (예 E08)")
    parser.add_argument("--control", default="", help="대조군 폴더명 조각. 없으면 라운드 안 첫 arm")
    parser.add_argument("--watch", default=DEFAULT_WATCH, help="가설이 지목한 셀 지표")
    parser.add_argument("--tail", type=int, default=3, help="마지막 N 에폭 평균")
    parser.add_argument("--md", default="", help="마크다운을 덧붙일 파일")
    parser.add_argument("--json", dest="json_out", default="", help="판정 결과 JSON 경로")
    return parser.parse_args()


def collect_rows(args: argparse.Namespace) -> list[dict]:
    root = Path(get_config().train.output_root)
    keys = tuple(dict.fromkeys([*REPORT_KEYS, args.watch]))
    runs = find_runs(root, args.round)
    if args.control:
        runs += [r for r in find_runs(root, args.control) if r not in runs]
    rows = [tail_mean(run, args.tail, keys) for run in runs]
    return [row for row in rows if row]


def pick_control(rows: list[dict], wanted: str) -> dict:
    """대조군을 못 찾으면 판정 자체가 무의미하므로 조용히 넘어가지 않는다."""
    if not wanted:
        return rows[0]
    matched = [row for row in rows if wanted in row["name"]]
    if not matched:
        raise SystemExit(f"[judge] 대조군 '{wanted}' 을(를) 찾지 못했다")
    return matched[0]


def judge_row(row: dict, control: dict, watch: str) -> dict:
    primary_rel = relative_change(row.get(PRIMARY), control.get(PRIMARY))
    watch_rel = relative_change(row.get(watch), control.get(watch))
    tracked = control.get(watch) is not None  # 그 지표가 로그에 찍히기는 하는가
    verdict = (
        "control" if row["name"] == control["name"] else verdict_of(primary_rel, watch_rel, tracked)
    )
    return {**row, "primary_rel": primary_rel, "watch_rel": watch_rel, "verdict": verdict}


def verdict_of(primary_rel: float | None, watch_rel: float | None, tracked: bool) -> str:
    """규칙 3(±10%) 위에 규칙 4(지목 지표 확인)를 얹는다.

    지목 지표가 **로그에 없는 것**과 **있는데 안 움직인 것**은 다르다. 전자는 규칙 4를
    적용할 수 없다는 뜻이라 f1만으로 판정하고(리포트에 경고를 띄운다), 후자만 보류다.
    """
    if primary_rel is None:
        return "neutral"
    if primary_rel >= ADOPT_MARGIN:
        if not tracked:
            return "adopt"
        return "adopt" if watch_rel is not None and abs(watch_rel) >= MOVE_MARGIN else "hold"
    return "reject" if primary_rel <= REJECT_MARGIN else "neutral"


def render(judged: list[dict], args: argparse.Namespace, control: dict) -> str:
    columns = tuple(dict.fromkeys([*REPORT_KEYS, args.watch]))
    lines = [
        f"### {args.round} 자동 판정 (마지막 {args.tail}에폭 평균, 대조군 `{control['name']}`)",
        "",
        header_line(columns),
        divider_line(columns),
    ]
    lines += [body_line(row, columns) for row in judged]
    lines += ["", summary_line(judged)]
    if control.get(args.watch) is None:
        lines += [
            "",
            f"> 경고: 지목 지표 `{args.watch}` 가 로그에 없다 — 규칙 4를 적용하지 못했다."
            " f1만으로 판정했으므로 재현 확인이 필요하다.",
        ]
    return "\n".join(lines)


def header_line(columns: tuple[str, ...]) -> str:
    labels = " | ".join(SHORT_LABEL.get(key, key.split("/")[-1]) for key in columns)
    return f"| arm | ep | {labels} | f1 Δ% | 지목 Δ% | 판정 |"


def divider_line(columns: tuple[str, ...]) -> str:
    return "| --- | --- | " + " | ".join(["---"] * len(columns)) + " | --- | --- | --- |"


def body_line(row: dict, columns: tuple[str, ...]) -> str:
    values = " | ".join(fmt_value(row.get(key)) for key in columns)
    verdict = VERDICT_NOTE[row["verdict"]]
    return (
        f"| {row['name']} | {row['epochs']} | {values} | "
        f"{fmt_percent(row['primary_rel'])} | {fmt_percent(row['watch_rel'])} | **{verdict}** |"
    )


def fmt_value(value) -> str:
    return "-" if value is None else f"{value:.4f}"


def fmt_percent(value) -> str:
    return "-" if value is None else f"{value * 100:+.1f}"


def summary_line(judged: list[dict]) -> str:
    adopted = [row["name"] for row in judged if row["verdict"] == "adopt"]
    held = [row["name"] for row in judged if row["verdict"] == "hold"]
    if adopted:
        return f"**채택 {len(adopted)}건**: {', '.join(adopted)}" + (
            f" · 보류 {len(held)}건" if held else ""
        )
    return "**채택 0건** — 이 축은 소진됐을 수 있다 (연속 2라운드면 정지 조건)"


def emit(report: str, judged: list[dict], args: argparse.Namespace) -> None:
    if args.md:
        with open(args.md, "a", encoding="utf-8") as handle:
            handle.write("\n\n" + report + "\n")
    if args.json_out:
        payload = {
            "round": args.round,
            "watch": args.watch,
            "adopted": [row["name"] for row in judged if row["verdict"] == "adopt"],
            "rows": [{k: v for k, v in row.items() if k != "path"} for row in judged],
        }
        Path(args.json_out).write_text(json.dumps(payload, indent=2, ensure_ascii=False), "utf-8")


if __name__ == "__main__":
    main()

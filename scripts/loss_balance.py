"""손실 균형 진단 — 가중치 제안까지 (improve_plan 3.5절).

두 단계다.

**A. 스케일 균형** — 손실 종류마다 계산식이 달라 스케일이 크게 다르다. 한 항이 지배하면
나머지가 학습되지 않으므로, 안정화된 뒤 **가중 손실의 최대/최소가 10배 이내**여야 한다.

**B. 진단 기반 가중** — 그 다음엔 셀 단위 지표를 보고 **학습이 덜 된 항의 가중치를 올린다.**

작은 손실이 항상 굶주린 것은 아니다 — **이미 수렴해서** 작을 수도 있다. 그래서 각 항을
자기 셀 지표와 함께 보고 `수렴`/`미달`을 함께 찍는다. 미달인데 작으면 굶주린 것이고,
수렴인데 작으면 그대로 둬도 된다(그래도 A의 하한은 채운다).

사용:
    python scripts/loss_balance.py --run <실행폴더>
    python scripts/loss_balance.py --run <실행폴더> --tail 3 --band 10
"""

import argparse
import csv
import json
from pathlib import Path

# 손실 항목 -> (원시 손실 로그 키, 가중치 config 경로, 그 항을 평가하는 셀 지표)
TERMS = {
    "heatmap": ("train/heatmap/focal", "loss.heatmap.w_heatmap", "heat_recall"),
    "class": ("train/self_slot/class", "loss.self_slot.w_class", "class_acc"),
    "coord": ("train/self_slot/coord", "loss.self_slot.w_coord", "coord_err_px"),
    "end": ("train/self_slot/end", "loss.self_slot.w_end", "end_recall"),
    "exist": ("train/conn/exist", "loss.conn.w_exist", "exist_gap"),
    "dir": ("train/conn/dir", "loss.conn.w_dir", "link_ok"),
}
# 셀 지표의 목표치. `lower`면 작을수록 좋다. 목표는 디코더가 요구하는 수준에서 왔다
# (link_ok 0.99는 E03의 사슬 평균 48.2셀에서 역산한 값이다).
MAX_BOOST = 3.0  # 진단 기반 가중의 상한 — 한 번에 그 이상 흔들지 않는다
TARGETS = {
    "heat_recall": (0.90, False),
    "class_acc": (0.80, False),
    "coord_err_px": (1.00, True),
    "end_recall": (0.70, False),
    "exist_gap": (0.50, False),
    "link_ok": (0.99, False),
}


def main() -> None:
    args = parse_args()
    run = Path(args.run)
    weights = read_weights(run)
    raw = read_last_epochs(run, args.tail)
    report(raw, weights, args.band)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--tail", type=int, default=3, help="마지막 N 에폭 평균")
    parser.add_argument("--band", type=float, default=10.0, help="허용 최대/최소 배수")
    return parser.parse_args()


def read_weights(run: Path) -> dict[str, float]:
    saved = json.loads((run / "config.json").read_text(encoding="utf-8"))
    return {name: _dig(saved, path) for name, (_, path, _) in TERMS.items()}


def _dig(node: dict, dotted: str):
    for key in dotted.split("."):
        node = node[key]
    return float(node)


def read_last_epochs(run: Path, tail: int) -> dict[str, float]:
    merged: dict[int, dict] = {}
    with open(run / "metrics.csv", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            target = merged.setdefault(int(row["epoch"]), {})
            target.update({k: v for k, v in row.items() if v not in ("", None)})
    epochs = [e for e in sorted(merged) if "train/total" in merged[e]][-tail:]
    values = {name: _mean(merged, epochs, key) for name, (key, _, _) in TERMS.items()}
    values |= {f"cell:{m}": _cell(merged, epochs, m) for _, (_, _, m) in TERMS.items()}
    return values


def _mean(merged: dict, epochs: list[int], key: str) -> float:
    got = [float(merged[e][key]) for e in epochs if key in merged[e]]
    return sum(got) / len(got) if got else float("nan")


def _cell(merged: dict, epochs: list[int], metric: str) -> float:
    """`exist_gap`은 저장된 지표가 아니라 두 지표의 차(판별력)다."""
    if metric != "exist_gap":
        return _mean(merged, epochs, f"val/cell/{metric}")
    pos = _mean(merged, epochs, "val/cell/exist_pos")
    neg = _mean(merged, epochs, "val/cell/exist_neg")
    return pos - neg


def report(raw: dict, weights: dict, band: float) -> None:
    weighted = {name: raw[name] * weights[name] for name in TERMS}
    largest = max(weighted.values())
    floor = largest / band
    ratio = largest / min(weighted.values())
    verdict = "OK" if ratio <= band else "위반"
    print(f"\n[A. 스케일 균형] 최대/최소 = {ratio:.1f}배 (목표 {band:.0f}배 이내) [{verdict}]")
    header = f"{'항목':<8}{'원시':>9}{'가중':>7}{'가중손실':>10}{'최대대비':>9}  셀 지표"
    print(header)
    for name in sorted(TERMS, key=lambda n: -weighted[n]):
        _print_term(name, raw, weights[name], weighted[name], largest)
    print(_suggestion(raw, weights, floor))


def _print_term(name: str, raw: dict, weight: float, value: float, largest: float) -> None:
    metric = TERMS[name][2]
    score = raw[f"cell:{metric}"]
    attainment = _attainment(metric, score)
    state = "수렴" if attainment >= 1.0 else f"목표의 {attainment:.0%}"
    print(
        f"{name:<8}{raw[name]:>9.4f}{weight:>7.1f}{value:>10.4f}{largest / value:>8.1f}x  "
        f"{metric}={score:<10.4f} {state}"
    )


def _attainment(metric: str, score: float) -> float:
    """목표 대비 달성도. 1.0 이상이면 수렴, 0.5면 목표의 절반에 그쳤다는 뜻."""
    target, lower = TARGETS[metric]
    if lower:
        return target / score if score > 0 else 1.0
    return score / target if target > 0 else 1.0


def _suggestion(raw: dict, weights: dict, floor: float) -> str:
    """B(달성도에 반비례해 가중) 후 A(하한 채우기)를 적용한 가중치를 제안한다.

    전 항목을 똑같이 2배 하는 것은 **아무 일도 하지 않는 것과 같다**(균일 스케일).
    그래서 목표에 얼마나 못 미치는지에 비례해서만 올린다 — 목표의 50%면 2배, 99%면 1.01배.
    """
    proposed = {}
    for name in TERMS:
        attainment = min(_attainment(TERMS[name][2], raw[f"cell:{TERMS[name][2]}"]), 1.0)
        weight = weights[name] * min(1.0 / max(attainment, 1e-6), MAX_BOOST)
        proposed[name] = round(max(weight, floor / raw[name] if raw[name] > 0 else weight), 2)
    changed = {n: w for n, w in proposed.items() if abs(w - weights[n]) > 0.05}
    if not changed:
        return "\n[B. 진단 기반 가중] 조정 제안 없음 — 균형·성능 모두 기준을 만족한다."
    args = " ".join(f"{TERMS[n][1]}={w:g}" for n, w in changed.items())
    return (
        "\n[B. 진단 기반 가중] 달성도에 반비례해 올린 뒤, A의 하한을 채운 값이다.\n"
        f"  --override {args}"
    )


if __name__ == "__main__":
    main()

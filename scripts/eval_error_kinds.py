"""틀린 예측·놓친 정답을 원인별로 센다 — 방법마다 (E2 · E4, 교수 미팅 09-30).

    paper/pred/{방법}/{split}/{타일}.npz   예측 (scripts/eval_methods.py 가 떨군 것)
    데이터셋 라벨                          정답
    → experiment/data/e2/{방법}_{split}.json

원인의 정의는 `stella/eval/error_kinds.py`. 짝은 채점기와 같아서 TP·FP·FN 합계가
`paper/score/{방법}_{split}/summary.json` 과 맞아야 한다 — 끝에 대조해 찍는다.

사용:
    python scripts/eval_error_kinds.py --method stella --split val
"""

import argparse
import json
from pathlib import Path

from stella.config_io import load_config
from stella.eval.error_kinds import ErrorKinds
from stella.eval.instance_io import load_instances
from stella.eval.methods import target_tiles

PAPER_ROOT = Path("/data/ongoing/2026_stella/paper")
OUT_ROOT = Path(__file__).resolve().parents[1] / "experiment" / "data" / "e2"


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, [])
    kinds = ErrorKinds(cfg)
    for count, (tile, target) in enumerate(target_tiles(cfg, args.split), start=1):
        kinds.update(read_prediction(args.method, args.split, tile), target)
        if count % 200 == 0:
            print(f"  [{args.method}] ... {count}장")
    result = kinds.compute()
    save(args.method, args.split, result)
    check(args.method, args.split, result)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--config", default="configs.base")
    return parser.parse_args()


def read_prediction(method: str, split: str, tile: str) -> list[dict]:
    path = PAPER_ROOT / "pred" / method / split / f"{tile}.npz"
    return load_instances(path) if path.exists() else []


def save(method: str, split: str, result: dict) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out = OUT_ROOT / f"{method}_{split}.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[{method}] 저장 {out}")


def check(method: str, split: str, result: dict) -> None:
    """채점기의 정밀도·재현율을 이 개수로 다시 계산해 대조한다."""
    path = PAPER_ROOT / "score" / f"{method}_{split}" / "summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))["summary"]
    tp, fp, fn = result["tp"], result["fp_total"], result["fn_total"]
    print(
        f"[{method}] 정밀도 {tp / max(tp + fp, 1):.4f} (채점기 {summary['precision']:.4f}) · "
        f"재현율 {tp / max(tp + fn, 1):.4f} (채점기 {summary['recall']:.4f})"
    )


if __name__ == "__main__":
    main()

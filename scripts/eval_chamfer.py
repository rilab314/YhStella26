"""Table III 의 Chamfer 거리 열을 잰다 — 5단계가 떨군 예측 폴리라인을 다시 읽어서 (교수 지시 09-30).

    paper/pred/{방법}/{split}/{타일}.npz   예측 (scripts/eval_methods.py 가 떨군 것)
    데이터셋 라벨                          정답 — 모든 방법이 같은 정답과 견준다
    → paper/score/{방법}_{split}/chamfer.json

모델도 GPU 도 쓰지 않는다. 정답을 한 번 읽으며 네 방법을 함께 잰다.
**`eval_methods.py` 를 다시 돌리면 그 폴더가 지워진다** — 그 뒤에는 이것도 다시 돌린다.

사용:
    python scripts/eval_chamfer.py --split val
"""

import argparse
import json
from pathlib import Path

from stella.config_io import load_config
from stella.eval.chamfer import ChamferDistance
from stella.eval.instance_io import load_instances
from stella.eval.methods import target_tiles

PAPER_ROOT = Path("/data/ongoing/2026_stella/paper")
METHODS = ("seg_dp", "gte_delta", "stella_nopost", "stella")  # Table III 의 네 행
STEP = 1.0  # 점을 찍는 간격(px) — 거리 오차가 0.5 px 안이 되는 촘촘함


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, [])
    metrics = {
        method: ChamferDistance(step=STEP, exclude_classes=cfg.eval.exclude_classes)
        for method in METHODS
    }
    for count, (tile, target) in enumerate(target_tiles(cfg, args.split), start=1):
        for method, metric in metrics.items():
            metric.update(read_prediction(method, args.split, tile), target)
        if count % 200 == 0:
            print(f"  ... {count}장")
    for method, metric in metrics.items():
        save(method, args.split, metric.compute())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="val")
    parser.add_argument("--config", default="configs.base")
    return parser.parse_args()


def read_prediction(method: str, split: str, tile: str) -> list[dict]:
    path = PAPER_ROOT / "pred" / method / split / f"{tile}.npz"
    return load_instances(path) if path.exists() else []


def save(method: str, split: str, result: dict) -> None:
    out = PAPER_ROOT / "score" / f"{method}_{split}" / "chamfer.json"
    payload = {"method": method, "split": split, "step_px": STEP, **result}
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"[{method}] chamfer {result['chamfer']:.2f} px "
        f"(예측→정답 {result['pred_to_gt']:.2f} · 정답→예측 {result['gt_to_pred']:.2f}) "
        f"· 중앙값 {result['chamfer_median']:.2f} "
        f"· 타일 {result['tiles']} · 뺀 타일 {result['skipped']}"
    )


if __name__ == "__main__":
    main()

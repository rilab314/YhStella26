"""천장 분해 — 어느 헤드가 남은 격차를 만드는가 (Table IV · 4-3절, GPU 한 번만 쓴다).

**정답을 통째로 넣으면 0.94, 모델 예측만 쓰면 0.45** 다. 그 사이 격차가 어느 헤드에서
생기는지를 보려고, **정답을 묶음 하나씩만 얹어** 가며 점수를 잰다.

모델은 **정답 칸에서 계산**하게 한다(`gt_positive`). 그래야 정답 칸마다 예측값이 들어 있어
묶음을 바꿔 끼울 수 있다 — 그러지 않으면 모델이 안 고른 칸의 값이 0 이라 비교가 거짓이 된다.

사용:
    python scripts/ceiling_breakdown.py --run <실행폴더> --split val --count 400
    python scripts/ceiling_breakdown.py --run <실행폴더> --count 0 --rows class --save  # Table IV

`--save` 는 행마다 `paper/score/ceiling_{split}/{행}.json` 을 남긴다 — `tables/table_04.py` 가
읽는 유일한 출처다. 행 하나가 전량(1,218장)에 30분쯤 걸려 **행마다 따로 띄워** 나눠 돌린다.
"""

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from configs.schema import DATA_ROOT as _DATA_ROOT
from stella.builder import build_instance
from stella.config_io import apply_override, apply_saved_config, load_config
from stella.data.types import GridDatasetBase, collate_fn
from stella.eval.ccq import InstanceCCQ
from stella.model.checkpoint import best_checkpoint, load_model
from stella.model.inject import INJECTABLE_PARTS, inject_parts
from stella.model.stella import ModelOutput

# 표의 행 — STATE 4-3 대로 **하나씩 따로** 얹는다(누적이 아니다).
# 누적으로 하면 순서에 따라 기여가 달라져 "어느 헤드 탓인가"를 말할 수 없다.
# (저장 이름, 화면 이름, 얹는 묶음)
ROWS = (
    ("none", "전부 예측", ()),
    ("node", "정점 위치만", ("node",)),
    ("class", "종류만", ("class",)),
    ("coord", "셀 안 위치만", ("coord",)),
    ("conn", "연결 방향만", ("conn",)),
    ("end", "끝 판정만", ("end",)),
    ("all", "전부 주입 (천장)", INJECTABLE_PARTS),
)
# 표가 읽는 지표. `eval_methods.py` 의 요약 파일과 같은 이름을 쓴다.
SAVED_METRICS = ("f1", "precision", "recall", "coverage", "frag")
SCORE_ROOT = Path(_DATA_ROOT) / "paper" / "score"  # STELLA_DATA_ROOT 로 바꾼다


def main() -> None:
    args = parse_args()
    cfg = build_cfg(args)
    loader = build_loader(cfg, args.split)
    model = load_model(cfg, best_checkpoint(args.run), args.device)
    scores = {}
    for key, name, parts in selected_rows(args.rows):
        start = time.perf_counter()
        scores[name] = measure(model, loader, cfg, parts, args)
        print(
            f"  {name:20} f1={scores[name]['f1']:.4f}  ({time.perf_counter() - start:.0f} s)",
            flush=True,
        )
        if args.save:
            save_row(args, key, scores[name])
    report(scores)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs.base")
    parser.add_argument("--run", required=True, help="실행 폴더 — 체크포인트와 설정의 출처")
    parser.add_argument("--split", default="val")
    parser.add_argument("--count", type=int, default=400, help="0이면 전부")
    parser.add_argument("--rows", nargs="*", default=[], help="저장 이름 목록 — 비우면 전부")
    parser.add_argument("--save", action="store_true", help="행마다 요약 json 을 남긴다")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--override", nargs="*", default=[])
    return parser.parse_args()


def selected_rows(keys: list[str]) -> tuple:
    unknown = set(keys) - {key for key, _, _ in ROWS}
    if unknown:
        raise SystemExit(f"없는 행: {sorted(unknown)}")
    return tuple(row for row in ROWS if not keys or row[0] in keys)


def save_row(args: argparse.Namespace, key: str, scores: dict) -> None:
    """행 하나 = 파일 하나. 행을 나눠 돌려도 서로의 파일을 덮지 않는다."""
    out_dir = SCORE_ROOT / f"ceiling_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"row": key, "split": args.split, "run": args.run, "count": args.count, **scores}
    (out_dir / f"{key}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_cfg(args: argparse.Namespace):
    cfg = load_config(args.config, [])
    saved = json.loads((Path(args.run) / "config.json").read_text(encoding="utf-8"))
    # 디코더는 **지금 기본값**을 쓴다 — 실행 폴더의 옛 디코더를 그대로 읽어 후처리를 바꾼 뒤에도
    # 표 IV 가 옛 방식으로 나왔다(10-05). 모델·데이터 설정만 실행 폴더에서 가져온다.
    saved.pop("decode", None)
    apply_saved_config(cfg, saved)
    for item in args.override:
        path, _, raw = item.partition("=")
        apply_override(cfg, path.split("."), raw)
    return cfg


def build_loader(cfg, split: str) -> DataLoader:
    dataset = build_instance(cfg.data, cfg, base=GridDatasetBase, split=split)
    print(f"[천장분해] {split} {len(dataset)} samples")
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=6, collate_fn=collate_fn)


def measure(model, loader, cfg, parts: tuple, args: argparse.Namespace) -> float:
    metric = InstanceCCQ.from_cfg(cfg.eval, cfg)
    decoder = build_instance(cfg.decode, cfg)
    done = 0
    with torch.no_grad():
        for batch in loader:
            if args.count and done >= args.count:
                break
            output = predict(model, batch, cfg, args.device, use_gt_nodes="node" in parts)
            mixed = inject_parts(
                output, batch, parts, cfg.data.num_classes, cfg.model.num_conn_slots
            )
            metric.update(decoder(mixed[0]), batch["instances"][0])
            done += 1
    computed = metric.compute()
    return {key: float(computed[key]) for key in SAVED_METRICS} | {"tiles": done}


def predict(model, batch: dict, cfg, device: str, use_gt_nodes: bool) -> ModelOutput:
    """정점 묶음을 얹는 행만 **정답 칸에서 계산**하게 한다.

    ★ 이것이 "정점 위치" 행의 실체다 — 모델이 스스로 고른 칸 대신 정답 칸에서 헤드를 돌린다.
    첫 행(아무것도 안 얹음)까지 정답 칸을 쓰면 **정점 위치의 효과가 0 으로 보인다**(09-10 실측).
    """
    image = batch["image"].to(device)
    kwargs = {"gt_positive": (batch["class_map"] > 0).to(device)} if use_gt_nodes else {}
    with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16):
        output = model(image, **kwargs)
    return ModelOutput(
        **{
            k: v.detach().cpu().float() if v.dtype != torch.bool else v.detach().cpu()
            for k, v in vars(output).items()
        }
    )


def report(scores: dict) -> None:
    """각 헤드를 **혼자** 완벽하게 만들었을 때 남은 격차의 몇 %를 회복하는지."""
    names = [name for _, name, _ in ROWS]
    if any(name not in scores for name in names):
        return  # 행을 나눠 돌린 경우 — 회복 비율은 표 스크립트가 계산한다
    floor, ceiling = scores[names[0]]["f1"], scores[names[-1]]["f1"]
    span = ceiling - floor
    print(f"\n{'행':22} {'f1':>8} {'격차 중 회복':>12}")
    for name in names:
        share = (scores[name]["f1"] - floor) / span * 100 if span else 0.0
        print(f"  {name:20} {scores[name]['f1']:8.4f} {share:11.1f} %")
    print(f"\n  (격차 = 천장 {ceiling:.4f} - 전부예측 {floor:.4f} = {span:.4f})")


if __name__ == "__main__":
    main()

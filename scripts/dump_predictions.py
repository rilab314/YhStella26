"""모델 예측(또는 GT 주입)을 희소 캐시로 떨군다 — 디코더 전용 트랙 1단계 (improve_plan 2.1절).

한 번 떨궈 두면 `eval_decode.py`가 GPU 없이 디코더 파라미터를 몇 초 만에 스윕한다.

사용:
    # 학습된 체크포인트로
    python scripts/dump_predictions.py --run <실행폴더> --split val --count 320 --out <캐시>
    # GT 주입 상한 (GPU 불필요)
    python scripts/dump_predictions.py --source gt --split val --count 320 --out <캐시>

`--override model.heatmap_thresh=0.1` 로 노드 선택을 넓게 잡아 두면 나중에 스윕에서
`decode.heatmap_thresh`를 그 위로 올려 가며 비교할 수 있다 (노드 선택은 모델이 하므로
캐시보다 낮은 임계값은 사후에 복원되지 않는다).
"""

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from stella.builder import build_instance
from stella.config_io import apply_override, apply_saved_config, load_config
from stella.data.types import GridDatasetBase, collate_fn
from stella.decode.cache import save_prediction
from stella.model.inject import gt_model_output

MODEL_PREFIX = "model."


def main() -> None:
    args = parse_args()
    cfg = build_cfg(args)
    loader = build_loader(cfg, args.split)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    count = dump(args, cfg, loader, out_dir)
    write_meta(out_dir, cfg, args, count)
    print(f"[dump] {count} samples -> {out_dir}  ({time.perf_counter() - start:.1f} s)")


def dump(args: argparse.Namespace, cfg, loader: DataLoader, out_dir: Path) -> int:
    if args.source == "gt":
        return dump_gt(loader, cfg, out_dir, args.count)
    cap_memory(args.memory_fraction, args.device)
    ckpt = args.ckpt or _latest_ckpt(args.run)
    return dump_model(loader, cfg, out_dir, args.count, ckpt, args.device, args.allow_missing)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs.base")
    parser.add_argument("--run", default="", help="실행 폴더 (config.json·체크포인트 출처)")
    parser.add_argument("--ckpt", default="", help="체크포인트 직접 지정 (--run보다 우선)")
    parser.add_argument("--source", default="ckpt", choices=("ckpt", "gt"))
    parser.add_argument("--split", default="val")
    parser.add_argument("--count", type=int, default=320)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--override", nargs="*", default=[])
    # 학습이 도는 GPU에 얹을 때 쓴다. 상한을 넘으면 **이 프로세스만** 죽고 학습은 무사하다.
    parser.add_argument("--memory-fraction", type=float, default=0.0, help="0이면 제한 없음")
    parser.add_argument("--allow-missing", action="store_true", help="가중치 불일치를 허용")
    return parser.parse_args()


def build_cfg(args: argparse.Namespace):
    """실행 폴더가 있으면 그 config를 그대로 살리고, `--override`를 마지막에 얹는다."""
    cfg = load_config(args.config, [])
    if args.run:
        saved = json.loads((Path(args.run) / "config.json").read_text(encoding="utf-8"))
        apply_saved_config(cfg, saved)
    for item in args.override:
        path, _, raw = item.partition("=")
        apply_override(cfg, path.split("."), raw)
    return cfg


def build_loader(cfg, split: str) -> DataLoader:
    dataset = build_instance(cfg.data, cfg, base=GridDatasetBase, split=split)
    print(f"[dump] {split} {len(dataset)} samples")
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        collate_fn=collate_fn,
    )


def dump_gt(loader: DataLoader, cfg, out_dir: Path, count: int) -> int:
    """GT 주입 — 파이프라인의 천장. 모델도 GPU도 쓰지 않는다."""
    written = 0
    for batch in loader:
        if written >= count:
            break
        output = gt_model_output(batch, cfg.data.num_classes, cfg.model.num_conn_slots)
        save_prediction(out_dir / f"{_stem(batch, written)}.npz", output[0], batch["instances"][0])
        written += 1
    return written


def dump_model(loader, cfg, out_dir: Path, count: int, ckpt: str, device: str, allow: bool) -> int:
    model = load_model(cfg, ckpt, device, allow)
    written = 0
    with torch.no_grad():
        for batch in loader:
            if written >= count:
                break
            image = batch["image"].to(device)
            with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16):
                output = model(image)
            single = output[0].detach_cpu()
            save_prediction(out_dir / f"{_stem(batch, written)}.npz", single, batch["instances"][0])
            written += 1
    return written


def cap_memory(fraction: float, device: str) -> None:
    """학습 중인 GPU에 얹을 때의 안전장치 — 상한을 넘으면 이 프로세스만 OOM으로 죽는다."""
    if fraction > 0 and device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(fraction)
        print(f"[dump] GPU 메모리 상한 {fraction:.0%}")


def load_model(cfg, ckpt: str, device: str, allow_missing: bool = False):
    """가중치가 하나라도 안 붙으면 **크게 실패한다.**

    조용히 넘어가면 그 부품만 무작위 초기값인 채로 평가가 돌아 결과가 통째로 거짓이 된다
    (실제로 헤드 이름을 바꾼 뒤 `conn_head`가 무작위로 평가될 뻔했다).
    """
    if not ckpt:
        raise SystemExit("체크포인트가 없다 — --ckpt 또는 --run 을 주거나 --source gt 를 쓴다")
    model = build_instance(cfg.model, cfg)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
    weights = {k[len(MODEL_PREFIX) :]: v for k, v in state.items() if k.startswith(MODEL_PREFIX)}
    weights = _rename_legacy(weights, set(model.state_dict()))
    missing, unexpected = model.load_state_dict(weights, strict=False)
    print(f"[dump] {ckpt}  (missing {len(missing)}, unexpected {len(unexpected)})")
    if (missing or unexpected) and not allow_missing:
        raise SystemExit(
            f"가중치 불일치 — missing={list(missing)[:6]} unexpected={list(unexpected)[:6]}"
        )
    return model.to(device).eval()


def _rename_legacy(weights: dict, expected: set) -> dict:
    """옛 체크포인트의 키를 현재 구조로 옮긴다.

    `ConnHead`가 슬롯별 가중치를 지원하면서 `mlp` -> `mlps.0`이 됐다 (§7 C5).
    """
    renamed = {}
    for key, value in weights.items():
        new_key = key.replace("conn_head.mlp.", "conn_head.mlps.0.")
        renamed[new_key if new_key in expected else key] = value
    return renamed


def _latest_ckpt(run: str) -> str:
    if not run:
        return ""
    directory = Path(run) / "checkpoints"
    last = directory / "last.ckpt"
    if last.exists():
        return str(last)
    files = sorted(directory.glob("*.ckpt"), key=lambda p: p.stat().st_mtime)
    return str(files[-1]) if files else ""


def _stem(batch: dict, index: int) -> str:
    return str(batch["meta"][0].get("filename", f"{index:05d}")).replace("/", "_")


def write_meta(out_dir: Path, cfg, args: argparse.Namespace, count: int) -> None:
    meta = {
        "grid_size": cfg.data.grid_size,
        "grid_stride": cfg.data.grid_stride,
        "num_classes": cfg.data.num_classes,
        "num_slots": cfg.model.num_conn_slots,
        "split": args.split,
        "source": args.source,
        "run": args.run,
        "count": count,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

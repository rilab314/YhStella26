"""학습 진입점 + 최상위 조립 배선 (impl_plan 5.3·9.3절).

모든 부품이 `build_instance` 한 줄로 만들어져 "이 부품은 어떻게 만들더라"를 다시 볼 일이 없다.
진입 직후 `check_all(cfg)`를 불러 **무거운 초기화 전에** 모든 클래스 참조를 검증한다.
"""

import argparse
import dataclasses
import importlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader

from stella.builder import build_instance, check_all
from stella.data.types import GridDatasetBase, collate_fn

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_IGNORE = ("__pycache__", ".git", ".venv", "results", "data", "*.pyc", "viz_gt_out", "docs")
RUN_DIR_ENV = "STELLA_RUN_DIR"


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.override)
    check_all(cfg)  # 오타는 여기서 전부 걸린다 — 가중치 다운로드·CUDA 초기화 전에
    pl.seed_everything(cfg.train.seed, workers=True)
    out_dir = prepare_output_dir(cfg, args)
    print(f"[stella] output -> {out_dir}")
    module, loaders = build_everything(cfg, out_dir)
    trainer = build_trainer(cfg, out_dir)
    trainer.fit(module, loaders[0], loaders[1], ckpt_path=args.resume or None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs.base")
    parser.add_argument("--resume", default="")
    parser.add_argument("--tag", default="", help="출력 폴더 이름 뒤에 붙는 실험 메모")
    parser.add_argument(
        "--override", nargs="*", default=[], help="data.batch_size=2 처럼 점 경로로 덮어쓴다"
    )
    return parser.parse_args()


def load_config(module_name: str, overrides: list[str]):
    cfg = importlib.import_module(module_name).get_config()
    for item in overrides:
        path, _, raw = item.partition("=")
        apply_override(cfg, path.split("."), raw)
    return cfg


def apply_override(node, path: list[str], raw: str) -> None:
    for key in path[:-1]:
        node = getattr(node, key)
    current = getattr(node, path[-1])
    setattr(node, path[-1], _cast_like(current, raw))


def _cast_like(current, raw: str):
    if isinstance(current, bool):
        return raw.lower() in ("1", "true", "yes")
    if isinstance(current, int):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    if isinstance(current, tuple):
        return tuple(raw.split(","))
    return raw


def prepare_output_dir(cfg, args: argparse.Namespace) -> Path:
    """실행마다 날짜·시각 폴더를 만들고 config·소스 전체·git sha를 남긴다 (4.3절).

    DDP는 같은 스크립트를 rank 수만큼 다시 실행한다. 폴더를 만드는 것은 rank 0 뿐이고,
    자식 프로세스는 환경변수로 물려받은 같은 경로를 그대로 쓴다.
    """
    inherited = os.environ.get(RUN_DIR_ENV)
    if inherited:
        return Path(inherited)
    out_dir = _create_output_dir(cfg, args)
    os.environ[RUN_DIR_ENV] = str(out_dir)
    return out_dir


def _create_output_dir(cfg, args: argparse.Namespace) -> Path:
    stamp = time.strftime("%y%m%d_%H%M%S")
    suffix = f"_{args.tag}" if args.tag else ""
    out_dir = Path(cfg.train.output_root) / f"{stamp}_{args.config.split('.')[-1]}{suffix}"
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(
        json.dumps(dataclasses.asdict(cfg), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    shutil.copytree(REPO_ROOT, out_dir / "src", ignore=shutil.ignore_patterns(*SRC_IGNORE))
    (out_dir / "git_sha.txt").write_text(_git_state(), encoding="utf-8")
    return out_dir


def _git_state() -> str:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True
        ).stdout.strip()
        return f"{sha}\ndirty={bool(dirty)}\n"
    except Exception as error:  # git 저장소가 아닐 수도 있다
        return f"unavailable: {error}\n"


def build_everything(cfg, out_dir: Path) -> tuple[pl.LightningModule, tuple[DataLoader, ...]]:
    model = build_instance(cfg.model, cfg)
    criterion = build_instance(cfg.loss, cfg)
    decoder = build_instance(cfg.decode, cfg)
    metric = build_instance(cfg.eval, cfg)
    module = build_instance(
        cfg.train, cfg, model=model, criterion=criterion, decoder=decoder, metric=metric
    )
    train_set = build_instance(cfg.data, cfg, base=GridDatasetBase, split="train")
    val_set = build_instance(cfg.data, cfg, base=GridDatasetBase, split="val")
    print(f"[stella] train {len(train_set)} / val {len(val_set)} samples")
    return module, (make_loader(cfg, train_set, True), make_loader(cfg, val_set, False))


def make_loader(cfg, dataset: GridDatasetBase, shuffle: bool) -> DataLoader:
    workers = cfg.data.num_workers
    return DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=shuffle,
        num_workers=workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=workers > 0,
        drop_last=shuffle,
    )


def build_trainer(cfg, out_dir: Path) -> pl.Trainer:
    checkpoint = ModelCheckpoint(
        dirpath=out_dir / "checkpoints",
        monitor="val/total",
        mode="min",
        save_top_k=5,
        save_last=True,
        filename="epoch{epoch:03d}-val{val/total:.4f}",
        auto_insert_metric_name=False,
    )
    viz_callback = build_instance(
        cfg.log, cfg, out_dir=str(out_dir / "viz"), grid_stride=cfg.data.grid_stride
    )
    return pl.Trainer(
        max_epochs=cfg.train.epochs,
        check_val_every_n_epoch=1,
        precision=cfg.train.precision,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=_devices(cfg),
        strategy="ddp_find_unused_parameters_true" if _multi_gpu(cfg) else "auto",
        gradient_clip_val=cfg.train.grad_clip,
        accumulate_grad_batches=cfg.train.accumulate,
        limit_val_batches=cfg.train.limit_val_batches,
        callbacks=[checkpoint, viz_callback, TQDMProgressBar(refresh_rate=20)],
        logger=CSVLogger(save_dir=str(out_dir), name="", version=""),
        log_every_n_steps=10,
    )


def _devices(cfg):
    value = cfg.train.devices
    return int(value) if str(value).isdigit() else value


def _multi_gpu(cfg) -> bool:
    devices = _devices(cfg)
    count = torch.cuda.device_count() if devices == "auto" else devices
    return isinstance(count, int) and count > 1


if __name__ == "__main__":
    main()

import os
import shutil
from datetime import datetime

import torch
torch.set_float32_matmul_precision('medium')

import settings
from configs.config import CfgNode
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger

from pipeline.dataloader import create_dataloader
from util.misc import build_instance


TRAIN_WRAPPER_PREFIX = 'train_stella_cfg_gc'


def _cleanup_snapshot_train_wrappers(snapshot_pipeline_dir, keep_train_script_name):
    if not keep_train_script_name or not os.path.isdir(snapshot_pipeline_dir):
        return

    for filename in os.listdir(snapshot_pipeline_dir):
        if not filename.startswith(TRAIN_WRAPPER_PREFIX) or not filename.endswith('.py'):
            continue
        if filename == keep_train_script_name:
            continue
        os.remove(os.path.join(snapshot_pipeline_dir, filename))


def train_with_config(cfg_name, keep_train_script_name=None, resume=False):
    torch.use_deterministic_algorithms(False)

    cfg = CfgNode.from_file(cfg_name)
    cfg.runtime.output_dir = os.path.join(
        cfg.runtime.output_root,
        'log_' + datetime.now().strftime('%y%m%d_%H%M'),
    )
    os.makedirs(cfg.runtime.output_dir, exist_ok=True)

    project_root = os.path.dirname(os.path.dirname(__file__))
    snapshot_root = os.path.join(cfg.runtime.output_dir, 'src')
    shutil.copytree(
        project_root,
        snapshot_root,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns('*.pyc', '__pycache__', '.git'),
    )
    _cleanup_snapshot_train_wrappers(
        os.path.join(snapshot_root, 'pipeline'),
        keep_train_script_name,
    )

    pl.seed_everything(cfg.runtime.seed, workers=True)
    csv_logger = CSVLogger(save_dir=cfg.runtime.output_dir, name=cfg.runtime.logger_name)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(cfg.runtime.output_dir, 'checkpoints'),
        filename='{epoch:02d}-{val_total_loss:.4f}',
        monitor='val_total_loss',
        mode='min',
        save_top_k=10,
        save_last=True,
    )

    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor='val_total_loss',
        patience=10,
        mode='min',
        verbose=True,
    )

    progress_bar = pl.callbacks.TQDMProgressBar(refresh_rate=10)
    train_dataset = build_instance(cfg.dataset.module_name, cfg.dataset.class_name, cfg, split='train')
    train_loader = create_dataloader(cfg, train_dataset, 'train')
    val_dataset = build_instance(cfg.dataset.module_name, cfg.dataset.class_name, cfg, split='validation')
    val_loader = create_dataloader(cfg, val_dataset, 'validation', persistent_workers=True)
    model = build_instance(cfg.lightning_model.module_name, cfg.lightning_model.class_name, cfg)

    trainer = pl.Trainer(
        max_epochs=cfg.training.epochs,
        logger=[csv_logger],
        callbacks=[checkpoint_callback, early_stop_callback, progress_bar],
        accelerator='gpu',
        devices=torch.cuda.device_count(),
        strategy='ddp_find_unused_parameters_false',
        precision=cfg.training.get('precision', 32),
        gradient_clip_val=cfg.training.get('gradient_clip_val', 0.0),
        accumulate_grad_batches=cfg.training.get('accumulate_grad_batches', 8),
        log_every_n_steps=50,
        val_check_interval=1.0,
        check_val_every_n_epoch=1,
        deterministic=False,
    )

    if resume:
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path='last')
    else:
        trainer.fit(model, train_loader, val_loader)

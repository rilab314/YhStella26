import multiprocessing
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)

import os
import torch
torch.set_float32_matmul_precision('medium')

import settings
from configs.config import CfgNode
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pipeline.dataloader import create_dataloader
from util.misc import build_instance


def train(resume=False):
    torch.use_deterministic_algorithms(False)
    cfg = CfgNode.from_file('satellite_detr')
    pl.seed_everything(cfg.runtime.seed, workers=True)
    tb_logger = TensorBoardLogger(save_dir=cfg.runtime.output_dir, name=cfg.runtime.logger_name)
    csv_logger = CSVLogger(save_dir=cfg.runtime.output_dir, name=cfg.runtime.logger_name)
    
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(cfg.runtime.output_dir, 'checkpoints'),
        filename='{epoch:02d}-{val_loss:.4f}',
        monitor='train_total_loss',
        mode='min',
        save_top_k=10,
        save_last=True,
    )
    
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor='train_loss_total',
        patience=5,
        mode='min',
        verbose=True
    )
    
    progress_bar = pl.callbacks.TQDMProgressBar(refresh_rate=10)
    train_dataset = build_instance(cfg.dataset.module_name, cfg.dataset.class_name, cfg, split='train')
    train_loader = create_dataloader(cfg, train_dataset, 'train')
    val_dataset = build_instance(cfg.dataset.module_name, cfg.dataset.class_name, cfg, split='validation')
    val_loader = create_dataloader(cfg, val_dataset, 'validation', persistent_workers=True)
    model = build_instance(cfg.lightning_model.module_name, cfg.lightning_model.class_name, cfg)
    
    trainer = pl.Trainer(
        max_epochs=cfg.training.epochs,
        logger=[tb_logger, csv_logger],
        callbacks=[checkpoint_callback, progress_bar],
        accelerator='gpu',
        devices=2,
        strategy='ddp_find_unused_parameters_false',
        precision=cfg.training.get('precision', 32),
        gradient_clip_val=cfg.training.get('gradient_clip_val', 0.0),
        accumulate_grad_batches=cfg.training.get('accumulate_grad_batches', 8),
        log_every_n_steps=50,
        val_check_interval=1.,
        check_val_every_n_epoch=1,
        deterministic=False,
    )

    if resume == True:
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path="last")
    else:
        trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    train()

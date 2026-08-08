"""StellaTrainModule — LightningModule은 얇게 유지한다 (impl_plan 9.1·9.4절).

받는 것은 model·criterion·decoder·metric과 옵티마이저 값 몇 개뿐이다.
**전역 cfg를 들고 다니지 않는다.** 시각 로그는 module이 아니라 callback이 맡는다(9.5절).
"""

import pytorch_lightning as pl
import torch

from stella.train.optim import build_optimizer, build_scheduler


class StellaTrainModule(pl.LightningModule):
    @classmethod
    def from_cfg(cls, module_cfg, cfg, **kwargs) -> "StellaTrainModule":
        return cls(
            lr=module_cfg.lr,
            weight_decay=module_cfg.weight_decay,
            warmup_steps=module_cfg.warmup_steps,
            backbone_lr_mult=cfg.model.backbone.lr_mult,
            batch_size=cfg.data.batch_size,
            **kwargs,
        )

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        decoder,
        metric,
        cell_diag,
        lr: float,
        weight_decay: float,
        warmup_steps: int,
        backbone_lr_mult: float,
        batch_size: int,
    ):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.decoder = decoder
        self.metric = metric
        self.cell_diag = cell_diag
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.backbone_lr_mult = backbone_lr_mult
        self.batch_size = batch_size

    def forward(self, image: torch.Tensor) -> object:
        return self.model(image)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        output = self.model(batch["image"], gt_positive=batch["class_map"] > 0)
        losses = self.criterion(output, batch)
        self._log_losses("train", losses)
        return losses["total"]

    def validation_step(self, batch: dict, batch_idx: int) -> dict:
        output = self.model(batch["image"])
        losses = self.criterion(output, batch)
        self._log_losses("val", losses)
        self.cell_diag.update(output, batch)
        decoded = [self.decoder(output[index]) for index in range(batch["image"].shape[0])]
        for index, prediction in enumerate(decoded):
            self.metric.update(prediction, batch["instances"][index])
        return {"output": output, "decoded": decoded}

    def on_validation_epoch_start(self) -> None:
        self.decoder.stats.reset()  # 디코더 카운터는 에폭 단위 (improve_plan 3절 층 3)

    def on_validation_epoch_end(self) -> None:
        self._log_scores("val/inst", self.metric.compute(), sync_dist=False)
        self._log_scores("val/cell", self.cell_diag.compute(), sync_dist=False)
        self._log_scores("val/dec", self.decoder.stats.summary(), sync_dist=True)
        self.metric.reset()
        self.cell_diag.reset()

    def _log_scores(self, prefix: str, scores: dict, sync_dist: bool) -> None:
        named = {f"{prefix}/{key}": value for key, value in scores.items()}
        self.log_dict(named, sync_dist=sync_dist)

    def on_train_epoch_start(self) -> None:
        for group in self.optimizers().param_groups:
            self.log(f"lr/{group['name']}", group["lr"], on_step=False, on_epoch=True)

    def _log_losses(self, stage: str, losses: dict[str, torch.Tensor]) -> None:
        self.log_dict(
            {f"{stage}/{key}": value for key, value in losses.items()},
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=self.batch_size,
        )

    def configure_optimizers(self) -> dict:
        optimizer = build_optimizer(
            self.model,
            lr=self.lr,
            weight_decay=self.weight_decay,
            backbone_lr_mult=self.backbone_lr_mult,
        )
        scheduler = build_scheduler(
            optimizer,
            warmup_steps=self.warmup_steps,
            total_steps=int(self.trainer.estimated_stepping_batches),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

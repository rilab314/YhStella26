"""시각 로그 콜백 (impl_plan 9.5절).

검증 **배치마다 첫 번째 샘플 하나만** 그린다(전부 그리면 느리다).
학습 로직과 그리기 로직을 섞지 않기 위해 module이 아니라 callback이 맡는다.
"""

from pathlib import Path

import cv2
import numpy as np
import pytorch_lightning as pl

from stella.builder import Buildable
from stella.train import viz


class VizCallback(pl.Callback, Buildable):
    def __init__(
        self,
        *,
        out_dir: str,
        grid_stride: int,
        every_n_epochs: int,
        max_batches: int,
        heat_alpha: float,
        slot_line_len: float,
        exist_thresh: float,
        class_thresh: float,
    ):
        super().__init__()
        self.out_dir = Path(out_dir)
        self.grid_stride = grid_stride
        self.every_n_epochs = every_n_epochs
        self.max_batches = max_batches
        self.heat_alpha = heat_alpha
        self.slot_line_len = slot_line_len
        self.exist_thresh = exist_thresh
        self.class_thresh = class_thresh

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ) -> None:
        if not self._should_draw(trainer, batch_idx, outputs):
            return
        epoch_dir = self.out_dir / f"epoch{trainer.current_epoch:03d}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        sample_id = str(batch["meta"][0].get("filename", batch_idx)).replace("/", "_")
        pages = self._render(outputs, batch)
        for name, page in pages.items():
            cv2.imwrite(str(epoch_dir / f"{sample_id}_{name}.png"), page[..., ::-1])

    def _should_draw(self, trainer, batch_idx: int, outputs) -> bool:
        if outputs is None or batch_idx >= self.max_batches or trainer.sanity_checking:
            return False
        return trainer.current_epoch % self.every_n_epochs == 0

    def _render(self, outputs, batch: dict) -> dict[str, np.ndarray]:
        single = outputs["output"][0].detach_cpu()
        image = batch["image"][0].detach().cpu().numpy()
        probability = _sigmoid(single.heatmap_logit.numpy())
        node_mask = single.node_mask.numpy()
        return {
            "heat": viz.draw_heatmap(image, probability, self.heat_alpha),
            "class": self._class_page(image, single, probability, node_mask),
            "slot": self._slot_page(image, single, node_mask),
            "end": viz.draw_heatmap(
                image, _sigmoid(single.end_logit.numpy()) * node_mask, self.heat_alpha
            ),
            "inst": viz.draw_instances(image, outputs["decoded"][0]),
            "gt": viz.draw_instances(image, batch["instances"][0]),
        }

    def _class_page(self, image, single, probability, node_mask) -> np.ndarray:
        class_ids = single.class_logit.numpy().argmax(axis=-1)
        draw = node_mask & (probability > self.class_thresh)
        return viz.draw_class_map(image, class_ids, draw, self.grid_stride)

    def _slot_page(self, image, single, node_mask) -> np.ndarray:
        return viz.draw_slots(
            image,
            single.self_coord.numpy(),
            single.conn_dir.numpy(),
            _sigmoid(single.exist_logit.numpy()),
            node_mask,
            self.grid_stride,
            self.exist_thresh,
            self.slot_line_len,
        )


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

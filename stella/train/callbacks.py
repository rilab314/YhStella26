"""시각 로그 콜백 (design 9.5절).

검증 **배치마다 첫 번째 샘플 하나만** 그린다(전부 그리면 느리다).
학습 로직과 그리기 로직을 섞지 않기 위해 module이 아니라 callback이 맡는다.
그리는 규칙 자체는 `viz.PageRenderer`에 있다 — 이 클래스는 **언제·어디에 쓸지만** 정한다.
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
        self.every_n_epochs = every_n_epochs
        self.max_batches = max_batches
        self.renderer = viz.PageRenderer(
            grid_stride=grid_stride,
            heat_alpha=heat_alpha,
            slot_line_len=slot_line_len,
            exist_thresh=exist_thresh,
            class_thresh=class_thresh,
        )

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ) -> None:
        if not self._should_draw(trainer, batch_idx, outputs):
            return
        epoch_dir = self.out_dir / f"epoch{trainer.current_epoch:03d}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        sample_id = str(batch["meta"][0].get("filename", batch_idx)).replace("/", "_")
        write_sheet(epoch_dir / f"{sample_id}.png", self._render(outputs, batch))

    def _should_draw(self, trainer, batch_idx: int, outputs) -> bool:
        if outputs is None or batch_idx >= self.max_batches or trainer.sanity_checking:
            return False
        return trainer.current_epoch % self.every_n_epochs == 0

    def _render(self, outputs, batch: dict) -> np.ndarray:
        return self.renderer.render(
            batch["image"][0].detach().cpu().numpy(),
            outputs["output"][0].detach_cpu(),
            outputs["decoded"][0],
            batch["instances"][0],
        )


def write_sheet(path: Path, sheet: np.ndarray) -> None:
    """RGB 시트를 파일로 — cv2는 BGR로 쓰므로 채널을 뒤집는다."""
    cv2.imwrite(str(path), sheet[..., ::-1])

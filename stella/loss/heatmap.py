"""히트맵 focal BCE (impl_plan 8.1절).

히트맵 GT는 1[Y > 0]이다. 배경 셀이 압도적으로 많아 그대로 BCE를 쓰면 배경이 학습을
지배하므로 focal 항으로 쉬운 배경의 기여를 줄인다. 모든 셀에서 계산하고 양성 수로 정규화한다.
"""

import torch
import torch.nn.functional as F

from stella.loss.criterion import LossModule


class HeatmapLoss(LossModule):
    def __init__(self, *, w_heatmap: float, focal_alpha: float, focal_gamma: float):
        super().__init__()
        self.w_heatmap = w_heatmap
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    def forward(self, output, targets: dict) -> dict[str, torch.Tensor]:
        logit = output.heatmap_logit.float()
        positive = (targets["class_map"] > 0).float()
        loss = self._focal(logit, positive) / positive.sum().clamp(min=1.0)
        return {"focal": loss, "total": self.w_heatmap * loss}

    def _focal(self, logit: torch.Tensor, positive: torch.Tensor) -> torch.Tensor:
        probability = logit.sigmoid()
        bce = F.binary_cross_entropy_with_logits(logit, positive, reduction="none")
        focus = torch.where(positive > 0, 1.0 - probability, probability) ** self.focal_gamma
        alpha = torch.where(positive > 0, self.focal_alpha, 1.0 - self.focal_alpha)
        return (alpha * focus * bce).sum()

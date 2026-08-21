"""히트맵 focal BCE (design 8.1절).

히트맵 GT는 1[Y > 0]이다. 배경 셀이 압도적으로 많아 그대로 BCE를 쓰면 배경이 학습을
지배하므로 focal 항으로 쉬운 배경의 기여를 줄인다. 모든 셀에서 계산하고 양성 수로 정규화한다.
"""

import torch
import torch.nn.functional as F

from stella.loss.criterion import LossModule


class HeatmapLoss(LossModule):
    def __init__(
        self,
        *,
        w_heatmap: float,
        focal_alpha: float,
        focal_gamma: float,
        length_power: float,
    ):
        super().__init__()
        self.w_heatmap = w_heatmap
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.length_power = length_power

    def forward(self, output, targets: dict) -> dict[str, torch.Tensor]:
        logit = output.heatmap_logit.float()
        positive = (targets["class_map"] > 0).float()
        weight = self._length_weight(targets.get("length_map"), positive)
        loss = self._focal(logit, positive, weight) / positive.sum().clamp(min=1.0)
        return {"focal": loss, "total": self.w_heatmap * loss}

    def _length_weight(self, length, positive: torch.Tensor) -> torch.Tensor:
        """양성 셀의 가중을 **그 셀이 속한 선의 길이에 반비례**하게 (0이면 무동작).

        히트맵도 셀 단위라 100칸 선이 100표를 갖는다. 분류 손실에 같은 처방을 걸어 짧은 선
        정답률이 +20.5% 였다(08-21). 검출의 다른 단계에도 같은 불균형이 있는지 본다.
        **배경은 건드리지 않는다** — 배경까지 흔들면 focal 균형이 깨진다.
        """
        if self.length_power <= 0.0 or length is None:
            return torch.ones_like(positive)
        weight = torch.ones_like(positive)
        mask = (length > 0) & (positive > 0)
        if not bool(mask.any()):
            return weight
        raw = (length[mask].median() / length[mask]) ** self.length_power
        weight[mask] = (raw / raw.mean()).clamp(0.25, 4.0)
        return weight

    def _focal(
        self, logit: torch.Tensor, positive: torch.Tensor, weight: torch.Tensor
    ) -> torch.Tensor:
        probability = logit.sigmoid()
        bce = F.binary_cross_entropy_with_logits(logit, positive, reduction="none")
        focus = torch.where(positive > 0, 1.0 - probability, probability) ** self.focal_gamma
        alpha = torch.where(positive > 0, self.focal_alpha, 1.0 - self.focal_alpha)
        return (weight * alpha * focus * bce).sum()

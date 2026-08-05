"""보조 히트맵 헤드 + 노드 선택 (impl_plan 7.4절).

선택은 하드 연산이라 그래디언트가 없다. 단, gather된 임베딩을 통해 neck까지는 그래디언트가 흐른다.
"""

import torch
import torch.nn.functional as F
from torch import nn

FOCAL_PRIOR_BIAS = -4.595  # sigmoid(-4.595) ~= 0.01
GT_SCORE_BONUS = 10.0  # 상한 절단 시 GT 셀은 무조건 살아남게 하는 가산점


class HeatmapHead(nn.Module):
    """셀이 노면 표시 위인지 이진 판단. focal 손실로만 학습된다."""

    def __init__(self, *, d_model: int):
        super().__init__()
        self.conv = nn.Conv2d(d_model, 1, kernel_size=1)
        nn.init.constant_(self.conv.bias, FOCAL_PRIOR_BIAS)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.conv(feature).squeeze(1)


class NodeSelector:
    """학습: GT 양성 ∪ 예측 마스크. 추론: 예측 마스크만."""

    def __init__(self, *, node_sampling: str, heatmap_thresh: float, dilate: int, n_max: int):
        if node_sampling not in ("gt+pred", "gt"):
            raise ValueError(f"node_sampling 은 'gt+pred' | 'gt' 여야 한다: {node_sampling}")
        self.node_sampling = node_sampling
        self.heatmap_thresh = heatmap_thresh
        self.dilate = dilate
        self.n_max = n_max

    def __call__(
        self,
        heatmap_logit: torch.Tensor,
        gt_positive: torch.Tensor | None,
        training: bool,
    ) -> list[torch.Tensor]:
        probability = heatmap_logit.detach().float().sigmoid()
        predicted = self._predicted_mask(probability)
        selected = []
        for index in range(probability.shape[0]):
            gt = None if gt_positive is None else gt_positive[index]
            mask = self._combine(predicted[index], gt, training)
            selected.append(self._to_cells(mask, probability[index], gt))
        return selected

    def _predicted_mask(self, probability: torch.Tensor) -> torch.Tensor:
        mask = probability > self.heatmap_thresh
        if self.dilate <= 0:
            return mask
        pooled = F.max_pool2d(
            mask.float().unsqueeze(1), self.dilate, stride=1, padding=self.dilate // 2
        )
        return pooled.squeeze(1) > 0

    def _combine(
        self, predicted: torch.Tensor, gt: torch.Tensor | None, training: bool
    ) -> torch.Tensor:
        if not training or gt is None:
            return predicted
        if self.node_sampling == "gt":
            return gt
        return predicted | gt

    def _to_cells(
        self, mask: torch.Tensor, probability: torch.Tensor, gt: torch.Tensor | None
    ) -> torch.Tensor:
        cells = mask.nonzero(as_tuple=False)
        if cells.shape[0] == 0:  # 최소 1노드 보장 — 뒤 모듈이 항상 돌아야 DDP가 안 죽는다
            flat = int(probability.argmax())
            return torch.tensor([[flat // mask.shape[1], flat % mask.shape[1]]], device=mask.device)
        if cells.shape[0] <= self.n_max:
            return cells
        return cells[self._priority(cells, probability, gt).topk(self.n_max).indices]

    @staticmethod
    def _priority(
        cells: torch.Tensor, probability: torch.Tensor, gt: torch.Tensor | None
    ) -> torch.Tensor:
        score = probability[cells[:, 0], cells[:, 1]]
        if gt is None:
            return score
        return score + gt[cells[:, 0], cells[:, 1]].float() * GT_SCORE_BONUS

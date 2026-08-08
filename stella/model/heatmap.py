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


SELECT_MODES = ("thresh", "topk")


class NodeSelector:
    """학습: GT 양성 ∪ 예측 마스크. 추론: 예측 마스크만.

    `select_mode`가 핵심 손잡이다 (improve_plan §7 C8).
    - `thresh` — 확률 > tau_h. **히트맵의 절대 보정에 전적으로 의존한다.**
    - `topk`   — 확률 상위 K개. 보정과 무관하게 **선택 수가 고정**된다.

    REF-F 실측에서 `thresh`의 `heat_recall`이 에폭마다 0.0001 ~ 0.75로 요동쳤다.
    임계 근처에 확률 질량이 몰려 있으면 미세한 이동이 수만 개 셀을 뒤집기 때문이다.
    """

    def __init__(
        self,
        *,
        node_sampling: str,
        heatmap_thresh: float,
        dilate: int,
        n_max: int,
        select_mode: str = "thresh",
        n_topk: int = 6000,
    ):
        if node_sampling not in ("gt+pred", "gt"):
            raise ValueError(f"node_sampling 은 'gt+pred' | 'gt' 여야 한다: {node_sampling}")
        if select_mode not in SELECT_MODES:
            raise ValueError(f"select_mode 는 {SELECT_MODES} 중 하나여야 한다: {select_mode}")
        self.node_sampling = node_sampling
        self.heatmap_thresh = heatmap_thresh
        self.dilate = dilate
        self.n_max = n_max
        self.select_mode = select_mode
        self.n_topk = n_topk

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
        if self.select_mode == "topk":
            return self._topk_mask(probability)
        mask = probability > self.heatmap_thresh
        if self.dilate <= 0:
            return mask
        pooled = F.max_pool2d(
            mask.float().unsqueeze(1), self.dilate, stride=1, padding=self.dilate // 2
        )
        return pooled.squeeze(1) > 0

    def _topk_mask(self, probability: torch.Tensor) -> torch.Tensor:
        """샘플마다 확률 상위 K개 셀. 팽창은 하지 않는다 — K가 이미 여유를 담고 있다."""
        flat = probability.flatten(1)
        chosen = flat.topk(min(self.n_topk, flat.shape[1]), dim=1).indices
        mask = torch.zeros_like(flat, dtype=torch.bool)
        mask.scatter_(1, chosen, True)
        return mask.view_as(probability)

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

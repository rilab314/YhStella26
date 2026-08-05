"""출력 헤드 (impl_plan 7.7절).

최종 토큰 (N, K, D)에서 토큰별 작은 MLP를 거친다. self 슬롯(k=0)과 연결 슬롯(k>=1)이
각각 다른 헤드를 쓰고, 연결 슬롯은 R개가 **가중치를 공유**한다.
"""

import torch
import torch.nn.functional as F
from torch import nn

NORMALIZE_EPS = 1e-6
CONN_OUTPUT_DIM = 4  # exist 1 + dir 2 + t 1


class SelfHead(nn.Module):
    """self 슬롯 -> 클래스 로짓 + 셀 내 좌표."""

    def __init__(self, *, d_model: int, num_classes: int):
        super().__init__()
        self.class_mlp = _two_layer_mlp(d_model, num_classes)
        self.coord_mlp = _two_layer_mlp(d_model, 2)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """tokens: (N, D) -> class_logit (N, C), self_coord (N, 2) in [0, 1]."""
        return self.class_mlp(tokens), self.coord_mlp(tokens).sigmoid()


class ConnHead(nn.Module):
    """연결 슬롯 -> 존재 로짓 + 단위 방향 + 종점 로짓. 슬롯 간 가중치 공유."""

    def __init__(self, *, d_model: int):
        super().__init__()
        self.mlp = _two_layer_mlp(d_model, CONN_OUTPUT_DIM)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """tokens: (N, R, D) -> exist (N, R), dir (N, R, 2), t (N, R)."""
        raw = self.mlp(tokens)
        direction = F.normalize(raw[..., 1:3], dim=-1, eps=NORMALIZE_EPS)
        return raw[..., 0], direction, raw[..., 3]


def _two_layer_mlp(d_model: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, out_dim))

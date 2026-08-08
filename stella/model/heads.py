"""출력 헤드 (impl_plan 7.7절).

최종 토큰 (N, K, D)에서 토큰별 작은 MLP를 거친다. self 슬롯(k=0)과 연결 슬롯(k>=1)이
각각 다른 헤드를 쓰고, 연결 슬롯은 기본적으로 R개가 **가중치를 공유**한다.

끝 판정은 자기 셀의 end_logit이 담당한다 (9차 개정 — 구 설계의 슬롯별 t_logit 폐기).
연결 슬롯은 방향만 예측한다 — 상대까지의 거리·좌표는 예측하지 않는다.

`hidden_layers`(MLP 깊이)와 `share_slots`(슬롯 간 가중치 공유)는 아키텍처 탐색용
손잡이다 (improve_plan §7 C5). 기본값은 계획서 원안 그대로다.
"""

import torch
import torch.nn.functional as F
from torch import nn

NORMALIZE_EPS = 1e-6
CONN_OUTPUT_DIM = 3  # exist 1 + dir 2


class SelfHead(nn.Module):
    """self 슬롯 -> 클래스 로짓 + 셀 내 좌표 + 끝 로짓 (end_map 직접 감독, 8.2절)."""

    def __init__(self, *, d_model: int, num_classes: int, hidden_layers: int = 1):
        super().__init__()
        self.class_mlp = build_mlp(d_model, num_classes, hidden_layers)
        self.coord_mlp = build_mlp(d_model, 2, hidden_layers)
        self.end_mlp = build_mlp(d_model, 1, hidden_layers)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """tokens: (N, D) -> class_logit (N, C), self_coord (N, 2) in [0, 1], end_logit (N,)."""
        return (
            self.class_mlp(tokens),
            self.coord_mlp(tokens).sigmoid(),
            self.end_mlp(tokens).squeeze(-1),
        )


class ConnHead(nn.Module):
    """연결 슬롯 -> 존재 로짓 + 단위 방향 (원점 = 자기 노드 점, 6.1절)."""

    def __init__(
        self, *, d_model: int, num_slots: int = 2, hidden_layers: int = 1, share_slots: bool = True
    ):
        super().__init__()
        self.share_slots = share_slots
        count = 1 if share_slots else num_slots
        self.mlps = nn.ModuleList(
            build_mlp(d_model, CONN_OUTPUT_DIM, hidden_layers) for _ in range(count)
        )

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """tokens: (N, R, D) -> exist (N, R), dir (N, R, 2)."""
        raw = self._project(tokens)
        direction = F.normalize(raw[..., 1:3], dim=-1, eps=NORMALIZE_EPS)
        return raw[..., 0], direction

    def _project(self, tokens: torch.Tensor) -> torch.Tensor:
        """이름에 주의 — `_apply`는 `nn.Module`의 예약 메서드다(`.to()`가 내부에서 부른다)."""
        if self.share_slots:
            return self.mlps[0](tokens)
        parts = [mlp(tokens[:, index]) for index, mlp in enumerate(self.mlps)]
        return torch.stack(parts, dim=1)


def build_mlp(d_model: int, out_dim: int, hidden_layers: int) -> nn.Sequential:
    """`hidden_layers = 1`이면 계획서 원안의 2층 MLP(Linear-GELU-Linear)."""
    layers: list[nn.Module] = []
    for _ in range(max(hidden_layers, 1)):
        layers += [nn.Linear(d_model, d_model), nn.GELU()]
    layers.append(nn.Linear(d_model, out_dim))
    return nn.Sequential(*layers)

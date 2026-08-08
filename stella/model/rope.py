"""2D axial RoPE (design 7.6절, RoPE-ViT 방식).

head 차원을 반으로 나눠 앞쪽은 x(=열 j), 뒤쪽은 y(=행 i) 좌표로 회전한다.
cross-attn의 q·k 양쪽에 적용하며, 위치는 셀 정수 좌표를 쓴다.
같은 노드의 K개 슬롯은 같은 위치를 공유한다.
"""

import torch
from torch import nn

ROPE_BASE = 100.0


class AxialRoPE(nn.Module):
    def __init__(self, *, head_dim: int, base: float = ROPE_BASE):
        super().__init__()
        if head_dim % 4 != 0:
            raise ValueError(f"head_dim 은 4의 배수여야 한다 (axial x2, pair x2): {head_dim}")
        quarter = head_dim // 4
        inv_freq = base ** (-torch.arange(quarter, dtype=torch.float32) / quarter)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """x: (B, H, L, head_dim), positions: (B, L, 2) — (x, y) = (열, 행) 순서."""
        cos, sin = self._angles(positions)
        even, odd = x[..., 0::2], x[..., 1::2]
        rotated = torch.stack([even * cos - odd * sin, even * sin + odd * cos], dim=-1)
        return rotated.flatten(-2).to(x.dtype)

    def _angles(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        coords = positions.float().unsqueeze(-1) * self.inv_freq  # (B, L, 2, quarter)
        angles = torch.cat([coords[..., 0, :], coords[..., 1, :]], dim=-1)  # (B, L, head_dim/2)
        return angles.cos().unsqueeze(1), angles.sin().unsqueeze(1)

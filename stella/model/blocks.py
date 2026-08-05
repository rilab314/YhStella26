"""어텐션 블록 (impl_plan 7.6절).

블록 구성(pre-LN, residual):
    [슬롯 간 self-attn] -> [cross-attn: kv = 선택된 노드 임베딩 z] -> [FFN]

MHA는 직접 구현한다 — `nn.MultiheadAttention`은 RoPE를 끼워 넣을 수 없다.

윈도우 층은 N x N 마스크 대신 **이웃 gather** 방식을 쓴다(9.6절이 예고한 교체안).
셀당 노드가 최대 하나이므로 w x w 격자 오프셋을 그대로 gather 하면 정확히 같은 결과가
나오면서 어텐션 행렬이 (N, K, w^2)로 줄어든다.
"""

import torch
import torch.nn.functional as F
from torch import nn

from stella.model.rope import AxialRoPE


class MultiHeadAttention(nn.Module):
    """선형 qkv + scaled_dot_product_attention + 출력 프로젝션."""

    def __init__(self, *, d_model: int, num_heads: int, dropout: float, use_rope: bool):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = dropout
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.rope = AxialRoPE(head_dim=self.head_dim) if use_rope else None

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        query_pos: torch.Tensor | None = None,
        key_pos: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """query: (B, Lq, D), key_value: (B, Lk, D), mask: (B, 1, 1, Lk) bool = 참여 여부."""
        q = self._split_heads(self.q_proj(query))
        k = self._split_heads(self.k_proj(key_value))
        v = self._split_heads(self.v_proj(key_value))
        if self.rope is not None and query_pos is not None:
            q, k = self.rope(q, query_pos), self.rope(k, key_pos)
        dropout = self.dropout if self.training else 0.0
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dropout)
        merged = attended.transpose(1, 2).reshape(query.shape[0], query.shape[1], -1)
        return self.out_proj(merged)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        return x.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)


class FeedForward(nn.Module):
    def __init__(self, *, d_model: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StellaBlock(nn.Module):
    """한 층. `kind`가 'global'이면 전 노드를, 'window'면 w x w 이웃만 본다."""

    def __init__(self, *, kind: str, d_model: int, num_heads: int, ffn_dim: int, dropout: float):
        super().__init__()
        if kind not in ("global", "window"):
            raise ValueError(f"layer kind 는 'global' | 'window' 여야 한다: {kind}")
        self.kind = kind
        self.norm_slot = nn.LayerNorm(d_model)
        self.norm_cross = nn.LayerNorm(d_model)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.slot_attn = MultiHeadAttention(
            d_model=d_model, num_heads=num_heads, dropout=dropout, use_rope=False
        )
        self.cross_attn = MultiHeadAttention(
            d_model=d_model, num_heads=num_heads, dropout=dropout, use_rope=True
        )
        self.ffn = FeedForward(d_model=d_model, ffn_dim=ffn_dim, dropout=dropout)

    def forward(
        self,
        tokens: torch.Tensor,
        memory: torch.Tensor,
        positions: torch.Tensor,
        neighbors: torch.Tensor,
    ) -> torch.Tensor:
        """tokens: (N, K, D), memory: (N, D), positions: (N, 2), neighbors: (N, w*w)."""
        normed = self.norm_slot(tokens)
        tokens = tokens + self.slot_attn(normed, normed)
        tokens = tokens + self._cross(self.norm_cross(tokens), memory, positions, neighbors)
        return tokens + self.ffn(self.norm_ffn(tokens))

    def _cross(
        self,
        tokens: torch.Tensor,
        memory: torch.Tensor,
        positions: torch.Tensor,
        neighbors: torch.Tensor,
    ) -> torch.Tensor:
        if self.kind == "global":
            return self._cross_global(tokens, memory, positions)
        return self._cross_window(tokens, memory, positions, neighbors)

    def _cross_global(
        self, tokens: torch.Tensor, memory: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        nodes, slots, dim = tokens.shape
        query = tokens.reshape(1, nodes * slots, dim)
        query_pos = positions.repeat_interleave(slots, dim=0).unsqueeze(0)
        attended = self.cross_attn(query, memory.unsqueeze(0), query_pos, positions.unsqueeze(0))
        return attended.reshape(nodes, slots, dim)

    def _cross_window(
        self,
        tokens: torch.Tensor,
        memory: torch.Tensor,
        positions: torch.Tensor,
        neighbors: torch.Tensor,
    ) -> torch.Tensor:
        valid = neighbors >= 0
        index = neighbors.clamp(min=0)
        key_value = memory[index]  # (N, w*w, D)
        key_pos = positions[index]  # (N, w*w, 2)
        query_pos = positions.unsqueeze(1).expand(-1, tokens.shape[1], -1)
        mask = valid[:, None, None, :]
        return self.cross_attn(tokens, key_value, query_pos, key_pos, mask)


def window_neighbors(cells: torch.Tensor, grid_size: int, window: int) -> torch.Tensor:
    """각 노드의 w x w 이웃 셀에 있는 노드 인덱스. 빈 셀은 -1 (impl_plan 7.6절 윈도우 마스크)."""
    index_grid = torch.full((grid_size, grid_size), -1, dtype=torch.long, device=cells.device)
    index_grid[cells[:, 0], cells[:, 1]] = torch.arange(cells.shape[0], device=cells.device)
    half = window // 2
    padded = F.pad(index_grid, (half, half, half, half), value=-1)
    offsets = torch.arange(window, device=cells.device)
    rows = cells[:, 0:1] + offsets  # padded 좌표계에서 i + k (좌상단 half 만큼 이동 상쇄)
    cols = cells[:, 1:2] + offsets
    return padded[rows[:, :, None], cols[:, None, :]].reshape(cells.shape[0], -1)

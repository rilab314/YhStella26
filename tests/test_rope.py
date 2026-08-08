"""RoPE 상대위치 성질과 윈도우 이웃 gather 검증 (impl_plan M4)."""

import torch

from stella.model.blocks import window_neighbors
from stella.model.rope import AxialRoPE

HEAD_DIM = 32
NUM_HEADS = 2
LENGTH = 16


def attention_logits(positions: torch.Tensor) -> torch.Tensor:
    torch.manual_seed(0)
    rope = AxialRoPE(head_dim=HEAD_DIM)
    q = torch.randn(1, NUM_HEADS, LENGTH, HEAD_DIM)
    k = torch.randn(1, NUM_HEADS, LENGTH, HEAD_DIM)
    return rope(q, positions) @ rope(k, positions).transpose(-1, -2)


def test_rope_is_translation_invariant():
    positions = torch.randint(0, 50, (1, LENGTH, 2)).float()
    shift = torch.tensor([[[7.0, -3.0]]])
    base = attention_logits(positions)
    moved = attention_logits(positions + shift)
    assert torch.allclose(base, moved, atol=1e-4)


def test_rope_changes_with_relative_position():
    positions = torch.zeros(1, LENGTH, 2)
    spread = positions.clone()
    spread[0, :, 0] = torch.arange(LENGTH).float()
    assert not torch.allclose(attention_logits(positions), attention_logits(spread), atol=1e-3)


def test_window_neighbors_matches_chebyshev_range():
    grid, window = 20, 5
    cells = torch.tensor([[2, 2], [3, 4], [10, 10], [2, 9]])
    neighbors = window_neighbors(cells, grid, window)
    assert neighbors.shape == (4, window * window)
    half = window // 2
    for node in range(cells.shape[0]):
        found = {int(v) for v in neighbors[node] if v >= 0}
        expected = {
            other
            for other in range(cells.shape[0])
            if int((cells[other] - cells[node]).abs().max()) <= half
        }
        assert found == expected


def test_window_neighbors_always_contains_self():
    cells = torch.tensor([[0, 0], [19, 19], [7, 3]])
    neighbors = window_neighbors(cells, 20, 9)
    for node in range(cells.shape[0]):
        assert node in {int(v) for v in neighbors[node] if v >= 0}

from dataclasses import dataclass
from typing import List
import torch


@dataclass
class LineString:
    class_id: int
    points: torch.Tensor
    scores: torch.Tensor = None

@dataclass
class LaneDetOutput:
    segm_logit: torch.Tensor  # (B, H, W, K), K=number of classes
    side_logits: List[torch.Tensor]  # [(B, H, W, 1), (B, H, W, 1)], endness of two sides
    center_point: torch.Tensor  # (B, H, W, 2), point on line in this grid
    side_points: List[torch.Tensor]  # [(B, H, W, 2), (B, H, W, 2)], point on line in side grids
    line_strings: List[List[LineString]] = None

@dataclass
class LineNode:
    id: int
    cls: int
    score: float

    grid_y: int
    grid_x: int

    x: float
    y: float

    left_x: float
    left_y: float
    right_x: float
    right_y: float

    is_left_end: bool
    is_right_end: bool
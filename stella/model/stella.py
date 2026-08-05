"""StellaModel — 전체 forward 흐름과 출력 계약 (impl_plan 7.1·7.5절).

내부 계산은 선택된 셀만 희소하게(토큰 단위) 하고, 반환 직전에 결과를 격자에 scatter해서
dense로 되돌린다. GT의 self 맵과 같은 격자 좌표계를 쓰기 위함이다 —
criterion이 셀 인덱싱만으로 GT와 예측을 짝짓는다.
"""

from dataclasses import dataclass, fields

import torch
from torch import nn

from stella.builder import build_instance
from stella.model.backbone import Backbone
from stella.model.blocks import StellaBlock, window_neighbors
from stella.model.heads import ConnHead, SelfHead
from stella.model.heatmap import HeatmapHead, NodeSelector
from stella.model.neck import Neck


@dataclass
class ModelOutput:
    """`node_mask`가 거짓인 셀의 값은 전부 0이며 의미가 없다 — 반드시 걸러 쓴다."""

    heatmap_logit: torch.Tensor  # (B, L, L)
    node_mask: torch.Tensor  # (B, L, L) bool
    class_logit: torch.Tensor  # (B, L, L, C)
    self_coord: torch.Tensor  # (B, L, L, 2) in [0, 1], 원점 = 셀 좌상단
    exist_logit: torch.Tensor  # (B, L, L, R)
    conn_dir: torch.Tensor  # (B, L, L, R, 2) 단위벡터, 원점 = 셀 중심
    t_logit: torch.Tensor  # (B, L, L, R)

    def __getitem__(self, index: int) -> "ModelOutput":
        return ModelOutput(**{f.name: getattr(self, f.name)[index] for f in fields(self)})

    def detach_cpu(self) -> "ModelOutput":
        return ModelOutput(**{f.name: _detach(getattr(self, f.name)) for f in fields(self)})


class StellaModel(nn.Module):
    def __init__(
        self,
        *,
        backbone: Backbone,
        neck: Neck,
        selector: NodeSelector,
        d_model: int,
        num_heads: int,
        num_conn_slots: int,
        layers: tuple[str, ...],
        window_size: int,
        ffn_dim: int,
        dropout: float,
        num_classes: int,
        grid_size: int,
    ):
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.selector = selector
        self.num_conn_slots = num_conn_slots
        self.window_size = window_size
        self.num_classes = num_classes
        self.grid_size = grid_size
        self.heatmap_head = HeatmapHead(d_model=d_model)
        self.role_embed = nn.Parameter(torch.zeros(num_conn_slots + 1, d_model))
        nn.init.normal_(self.role_embed, std=0.02)
        self.blocks = nn.ModuleList(
            StellaBlock(
                kind=kind, d_model=d_model, num_heads=num_heads, ffn_dim=ffn_dim, dropout=dropout
            )
            for kind in layers
        )
        self.self_head = SelfHead(d_model=d_model, num_classes=num_classes)
        self.conn_head = ConnHead(d_model=d_model)

    @classmethod
    def from_cfg(cls, module_cfg, cfg, **kwargs) -> "StellaModel":
        backbone = build_instance(module_cfg.backbone, cfg, base=Backbone)
        neck = build_instance(
            module_cfg.neck,
            cfg,
            base=Neck,
            in_channels=backbone.out_channels,
            d_model=module_cfg.d_model,
            upsample_steps=_upsample_steps(backbone.strides[0], cfg.data.grid_stride),
        )
        selector = NodeSelector(
            node_sampling=module_cfg.node_sampling,
            heatmap_thresh=module_cfg.heatmap_thresh,
            dilate=module_cfg.dilate,
            n_max=module_cfg.n_max,
        )
        return cls(
            backbone=backbone,
            neck=neck,
            selector=selector,
            d_model=module_cfg.d_model,
            num_heads=module_cfg.num_heads,
            num_conn_slots=module_cfg.num_conn_slots,
            layers=tuple(module_cfg.layers),
            window_size=module_cfg.window_size,
            ffn_dim=module_cfg.ffn_dim,
            dropout=module_cfg.dropout,
            num_classes=cfg.data.num_classes,
            grid_size=cfg.data.grid_size,
        )

    def forward(self, image: torch.Tensor, gt_positive: torch.Tensor | None = None) -> ModelOutput:
        feature = self.neck(self.backbone(image))
        heatmap_logit = self.heatmap_head(feature).float()
        cells_per_sample = self.selector(heatmap_logit, gt_positive, self.training)
        output = self._empty_output(heatmap_logit)
        for index, cells in enumerate(cells_per_sample):
            tokens, embeddings = self._run_tokens(feature[index], cells)
            self._scatter(output, index, cells, tokens)
            del embeddings
        return output

    def _run_tokens(
        self, feature: torch.Tensor, cells: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """선택 셀의 임베딩을 gather해 쿼리를 만들고 어텐션 스택을 돌린다 (7.5절)."""
        embeddings = feature[:, cells[:, 0], cells[:, 1]].transpose(0, 1)  # (N, D)
        tokens = embeddings.unsqueeze(1) + self.role_embed.unsqueeze(0)  # (N, K, D)
        positions = torch.stack([cells[:, 1], cells[:, 0]], dim=-1)  # (x, y) 순서
        neighbors = window_neighbors(cells, self.grid_size, self.window_size)
        for block in self.blocks:
            tokens = block(tokens, embeddings, positions, neighbors)
        return tokens, embeddings

    def _empty_output(self, heatmap_logit: torch.Tensor) -> ModelOutput:
        """dense 버퍼는 항상 fp32다 — 손실을 fp32로 승격해서 계산하기 때문(8.5절)."""
        batch, side = heatmap_logit.shape[0], self.grid_size
        slots, classes = self.num_conn_slots, self.num_classes
        zeros = heatmap_logit.new_zeros
        return ModelOutput(
            heatmap_logit=heatmap_logit,
            node_mask=torch.zeros(
                (batch, side, side), dtype=torch.bool, device=heatmap_logit.device
            ),
            class_logit=zeros((batch, side, side, classes)),
            self_coord=zeros((batch, side, side, 2)),
            exist_logit=zeros((batch, side, side, slots)),
            conn_dir=zeros((batch, side, side, slots, 2)),
            t_logit=zeros((batch, side, side, slots)),
        )

    def _scatter(
        self, output: ModelOutput, index: int, cells: torch.Tensor, tokens: torch.Tensor
    ) -> None:
        rows, cols = cells[:, 0], cells[:, 1]
        class_logit, self_coord = self.self_head(tokens[:, 0])
        exist_logit, conn_dir, t_logit = self.conn_head(tokens[:, 1:])
        output.node_mask[index, rows, cols] = True
        output.class_logit[index, rows, cols] = class_logit.float()
        output.self_coord[index, rows, cols] = self_coord.float()
        output.exist_logit[index, rows, cols] = exist_logit.float()
        output.conn_dir[index, rows, cols] = conn_dir.float()
        output.t_logit[index, rows, cols] = t_logit.float()


def _detach(tensor: torch.Tensor) -> torch.Tensor:
    """node_mask는 bool이라 float 변환하면 안 된다 (마스크 연산이 깨진다)."""
    moved = tensor.detach().cpu()
    return moved if moved.dtype == torch.bool else moved.float()


def _upsample_steps(backbone_stride: int, grid_stride: int) -> int:
    ratio = backbone_stride // grid_stride
    steps = max(ratio.bit_length() - 1, 0)
    if backbone_stride != grid_stride * (2**steps):
        raise ValueError(
            f"백본 stride {backbone_stride} 를 격자 stride {grid_stride} 로 2배씩 올릴 수 없다"
        )
    return steps

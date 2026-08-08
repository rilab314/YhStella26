"""Neck — 백본이 무엇이든 공통으로 (B, d_model, L, L)를 낸다 (design 7.3절).

이 격자가 이후 전부(히트맵·노드 선택·토큰 임베딩)의 좌표계다.
정규화는 LayerNorm/GroupNorm만 쓴다 — batch_size = 1로 시작하므로 BatchNorm은 통계가 무의미하다.
"""

import inspect

import torch
import torch.nn.functional as F
from torch import nn

from stella.builder import Buildable

GROUP_NORM_GROUPS = 32


class LayerNorm2d(nn.LayerNorm):
    """채널 축 LayerNorm을 (B, C, H, W)에 적용한다."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()


class Neck(nn.Module, Buildable):
    """계약: forward(list[Tensor]) -> (B, d_model, L, L)."""

    @classmethod
    def from_cfg(cls, module_cfg, cfg, **kwargs):
        """호출부(StellaModel)는 모든 neck이 쓸 수 있는 런타임 값을 한꺼번에 넘긴다.

        계열마다 필요한 값이 다르므로(SFP만 `upsample_steps`를 쓴다) 시그니처로 거른다.
        """
        accepted = inspect.signature(cls).parameters
        return super().from_cfg(
            module_cfg, cfg, **{k: v for k, v in kwargs.items() if k in accepted}
        )

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def check_levels(in_channels: tuple[int, ...], expected: int, name: str) -> None:
        if len(in_channels) != expected:
            raise ValueError(
                f"{name} 은 백본 출력 {expected}레벨을 기대하는데 {len(in_channels)}레벨을 받았다. "
                f"model.neck.name 을 확인하라 (in_channels={in_channels})"
            )


class SFP(Neck):
    """단일 스케일 ViT용 — ViTDet의 Simple Feature Pyramid를 1레벨로 축소.

    전치합성곱을 2단으로 나눠 올린다. 한 번에 4배 올리면 격자 무늬가 심해지기 때문이다.
    """

    def __init__(self, *, in_channels: tuple[int, ...], d_model: int, upsample_steps: int):
        super().__init__()
        self.check_levels(in_channels, 1, "SFP")
        self.upsample = nn.Sequential(
            *self._build_upsample(in_channels[0], d_model, upsample_steps)
        )
        self.smooth = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1), LayerNorm2d(d_model)
        )

    @staticmethod
    def _build_upsample(in_channels: int, d_model: int, steps: int) -> list[nn.Module]:
        layers: list[nn.Module] = []
        channels = in_channels
        for step in range(steps):
            out = d_model * 2 if step < steps - 1 else d_model
            layers += [
                nn.ConvTranspose2d(channels, out, kernel_size=2, stride=2),
                LayerNorm2d(out),
                nn.GELU(),
            ]
            channels = out
        if steps == 0:
            layers += [nn.Conv2d(channels, d_model, kernel_size=1), LayerNorm2d(d_model)]
        return layers

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        return self.smooth(self.upsample(features[0]))


class FPNLite(Neck):
    """멀티스케일 백본용 — 레벨별 출력을 만들지 않고 stride-4 한 레벨만 내보낸다.

    `out_blocks`는 출력단 3x3 블록 수다. 격자 해상도(192x192)에서 이웃 셀 문맥을 얼마나
    섞을지를 정하며, 연결 방향 예측이 국소 문맥에 의존한다는 가설을 시험할 손잡이다.
    """

    def __init__(self, *, in_channels: tuple[int, ...], d_model: int, out_blocks: int = 1):
        super().__init__()
        self.check_levels(in_channels, 4, "FPNLite")
        self.lateral = nn.ModuleList(
            nn.Sequential(nn.Conv2d(c, d_model, kernel_size=1), _group_norm(d_model))
            for c in in_channels
        )
        self.output_conv = nn.Sequential(*_output_blocks(d_model, out_blocks))

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        merged = self.lateral[-1](features[-1])
        for level in range(len(features) - 2, -1, -1):
            lateral = self.lateral[level](features[level])
            upsampled = F.interpolate(merged, size=lateral.shape[-2:], mode="nearest")
            merged = lateral + upsampled
        return self.output_conv(merged)


def _output_blocks(d_model: int, count: int) -> list[nn.Module]:
    """3x3 + GroupNorm 블록을 count개. 2개 이상이면 사이에 GELU를 넣는다."""
    layers: list[nn.Module] = []
    for index in range(max(count, 1)):
        if index:
            layers.append(nn.GELU())
        layers += [nn.Conv2d(d_model, d_model, kernel_size=3, padding=1), _group_norm(d_model)]
    return layers


def _group_norm(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(min(GROUP_NORM_GROUPS, channels), channels)

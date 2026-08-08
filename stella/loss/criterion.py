"""손실 조립 (impl_plan 8.0절).

`StellaCriterion`은 **곱하지 않고 더하기만 한다.** 가중치를 아는 것은 그 항목을 계산하는
모듈뿐이라, "어느 가중치가 어디에 걸리는지"를 한 곳만 보면 안다.
반환 dict가 그대로 로그가 된다(9.4절).
"""

import torch
from torch import nn

from stella.builder import Buildable, build_instance


class LossModule(nn.Module, Buildable):
    """공통 인터페이스: forward(output, targets) -> 항목별 원시 손실 dict + 'total'."""

    def forward(self, output, targets: dict) -> dict[str, torch.Tensor]:
        raise NotImplementedError


class StellaCriterion(nn.Module):
    @classmethod
    def from_cfg(cls, module_cfg, cfg, **kwargs) -> "StellaCriterion":
        return cls(
            heatmap=build_instance(module_cfg.heatmap, cfg, base=LossModule),
            self_slot=build_instance(module_cfg.self_slot, cfg, base=LossModule),
            conn=build_instance(module_cfg.conn, cfg, base=LossModule),
        )

    def __init__(self, *, heatmap: LossModule, self_slot: LossModule, conn: LossModule):
        super().__init__()
        self.losses = nn.ModuleDict({"heatmap": heatmap, "self_slot": self_slot, "conn": conn})

    def forward(self, output, targets: dict) -> dict[str, torch.Tensor]:
        collected: dict[str, torch.Tensor] = {}
        total = None
        for name, module in self.losses.items():
            values = module(output, targets)
            collected |= {f"{name}/{key}": value for key, value in values.items()}
            total = values["total"] if total is None else total + values["total"]
        collected["total"] = total
        return collected

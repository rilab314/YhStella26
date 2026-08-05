"""Optimizer / 스케줄 (impl_plan 9.2절).

AdamW. param group 3종: 백본(lr x lr_mult) / bias·norm(weight decay 0) / 나머지.
스케줄은 linear warmup + cosine decay (step 단위).
"""

import math

import torch
from torch.optim.lr_scheduler import LambdaLR

BACKBONE_PREFIX = "backbone."
MIN_LR_RATIO = 0.01


def build_optimizer(
    model: torch.nn.Module, *, lr: float, weight_decay: float, backbone_lr_mult: float
) -> torch.optim.Optimizer:
    groups = _param_groups(model, lr=lr, weight_decay=weight_decay, mult=backbone_lr_mult)
    return torch.optim.AdamW([g for g in groups if g["params"]], lr=lr, weight_decay=weight_decay)


def _param_groups(model: torch.nn.Module, *, lr: float, weight_decay: float, mult: float) -> list:
    buckets: dict[tuple[bool, bool], list] = {key: [] for key in _bucket_keys()}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_backbone = name.startswith(BACKBONE_PREFIX)
        no_decay = param.ndim <= 1  # bias 와 norm 파라미터
        buckets[(is_backbone, no_decay)].append(param)
    return [
        {
            "params": params,
            "lr": lr * mult if is_backbone else lr,
            "weight_decay": 0.0 if no_decay else weight_decay,
            "name": _group_name(is_backbone, no_decay),
        }
        for (is_backbone, no_decay), params in buckets.items()
    ]


def _bucket_keys() -> list[tuple[bool, bool]]:
    return [(False, False), (False, True), (True, False), (True, True)]


def _group_name(is_backbone: bool, no_decay: bool) -> str:
    return ("backbone" if is_backbone else "main") + ("_nodecay" if no_decay else "")


def build_scheduler(
    optimizer: torch.optim.Optimizer, *, warmup_steps: int, total_steps: int
) -> LambdaLR:
    horizon = max(total_steps, warmup_steps + 1)

    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(horizon - warmup_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return MIN_LR_RATIO + (1.0 - MIN_LR_RATIO) * cosine

    return LambdaLR(optimizer, factor)

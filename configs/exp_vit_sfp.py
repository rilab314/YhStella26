"""SFP 경로 검증용 — 게이트 없는 timm ViT 단일 스케일 백본."""

from configs.base import get_config as get_base
from configs.schema import ExperimentConfig


def get_config() -> ExperimentConfig:
    cfg = get_base()
    cfg.model.backbone.name = "TimmVitBackbone"
    cfg.model.backbone.pretrained = "vit_base_patch16_224.augreg_in21k"
    cfg.model.neck.name = "SFP"
    return cfg

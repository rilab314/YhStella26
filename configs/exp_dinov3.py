"""DINOv3 ViT-L/16 위성판 백본 + SFP (design 13절 결정 3).

HF 게이트 모델이라 계정 승인이 필요하다. 승인 전에는 `check_all`은 통과하지만
백본 생성 시 `GatedRepoError`가 난다.
"""

from configs.base import get_config as get_base
from configs.schema import ExperimentConfig


def get_config() -> ExperimentConfig:
    cfg = get_base()
    cfg.model.backbone.name = "Dinov3Backbone"
    cfg.model.backbone.pretrained = "facebook/dinov3-vitl16-pretrain-sat493m"
    cfg.model.neck.name = "SFP"
    return cfg

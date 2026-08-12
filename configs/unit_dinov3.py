"""단위 실험(U) 규격 + DINOv3 ViT-L/16 위성판 백본 (design 13절 결정 3).

지금까지의 모든 실측은 대역 백본(ConvNeXtV2-base) 기준이라는 단서를 달고 있다.
설계가 목표한 백본으로 한 번 재서 그 단서를 떼는 것이 이 config의 목적이다.

비교 대상은 같은 U 규격의 대조군뿐이다 — F 규격 절대값과 비교하지 않는다.
"""

from configs.schema import ExperimentConfig
from configs.unit import get_config as get_unit


def get_config() -> ExperimentConfig:
    cfg = get_unit()
    cfg.model.backbone.name = "Dinov3Backbone"
    cfg.model.backbone.pretrained = "facebook/dinov3-vitl16-pretrain-sat493m"
    cfg.model.neck.name = "SFP"  # ViT는 단일 스케일 → SFP가 피라미드를 만든다
    return cfg

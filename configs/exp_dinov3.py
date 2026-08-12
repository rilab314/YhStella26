"""DINOv3 ViT-L/16 위성판 백본 + SFP (design 13절 결정 3).

가중치는 로컬 캐시에 있고 조립·전방 통과가 확인됐다(303M). 단위 실험 규격은
`configs/unit_dinov3.py`를 쓴다.
"""

from configs.base import get_config as get_base
from configs.schema import ExperimentConfig


def get_config() -> ExperimentConfig:
    cfg = get_base()
    cfg.model.backbone.name = "Dinov3Backbone"
    cfg.model.backbone.pretrained = "facebook/dinov3-vitl16-pretrain-sat493m"
    cfg.model.neck.name = "SFP"
    return cfg

"""Table VI — **노드당 토큰 1개** 판 (논지 C2: 슬롯마다 토큰이 필요한가).

최종 모델(F06d)과 **어텐션을 지나는 토큰 구조 하나만** 다르다. 슬롯 수(2)·짝 맞춤 배정·
헤드 가중치 공유·파라미터 수가 전부 같다. 딸려 바뀌는 것은 **계산량(약 1/3)** 하나다.
"""

from configs.base import get_config as get_base
from configs.schema import ExperimentConfig


def get_config() -> ExperimentConfig:
    cfg = get_base()
    cfg.model.backbone.pretrained = "convnextv2_large.fcmae_ft_in22k_in1k_384"
    cfg.model.token_mode = "single"
    return cfg

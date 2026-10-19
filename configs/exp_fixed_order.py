"""Table VI — 슬롯 배정을 매칭 대신 **고정 순서(앞/뒤)** 로 한 판.

최종 모델(F06d)과 **배정 방식 하나만** 다르다. 슬롯 수(2)·가중치 공유·손실 형태가 전부 같다 —
3-C(각도 섹터)와 달리 슬롯 수가 따라 바뀌지 않으므로 **배정 방식의 효과만** 깨끗하게 잰다.
"""

from configs.base import get_config as get_base
from configs.schema import ExperimentConfig


def get_config() -> ExperimentConfig:
    cfg = get_base()
    cfg.model.backbone.pretrained = "convnextv2_large.fcmae_ft_in22k_in1k_384"
    cfg.loss.conn.path = "stella.loss.conn_fixed"
    cfg.loss.conn.name = "ConnLossFixed"
    return cfg

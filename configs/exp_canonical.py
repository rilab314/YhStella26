"""Table VI — 슬롯 배정을 매칭 대신 **각도 순서**로 고정한 판.

최종 모델(F06d)과 **배정 방식 하나만** 다르다 — 슬롯 수(2)·가중치 공유·손실 형태가 전부 같다.
3-C(각도 섹터)는 슬롯이 6개라 두 가지가 함께 달랐다. 이 판은 **단일 변수 비교**다.
"""

from configs.base import get_config as get_base
from configs.schema import ExperimentConfig


def get_config() -> ExperimentConfig:
    cfg = get_base()
    cfg.model.backbone.pretrained = "convnextv2_large.fcmae_ft_in22k_in1k_384"
    cfg.loss.conn.path = "stella.loss.conn_canonical"
    cfg.loss.conn.name = "ConnLossCanonical"
    return cfg

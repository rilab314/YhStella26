"""슬롯 수 ablation — 기본 R = 2에 여분 슬롯 1개를 줘 본다 (impl_plan 4.2절, 결정 1).

R > D 에서만 무매칭 슬롯(존재 0 감독)과 매칭의 존재 확률 항이 작동한다 (8.3~8.4절).
"""

from configs.base import get_config as get_base
from configs.schema import ExperimentConfig


def get_config() -> ExperimentConfig:
    cfg = get_base()
    cfg.model.num_conn_slots = 3  # K = 4
    return cfg

"""합성 데이터 실험 — 전 파이프라인 검증용 (design 6.6절)."""

from configs.base import get_config as get_base
from configs.schema import ExperimentConfig


def get_config() -> ExperimentConfig:
    cfg = get_base()
    cfg.data.path = "stella.data.synthetic"
    cfg.data.name = "SyntheticDataset"
    cfg.data.root = ""
    cfg.data.num_workers = 4
    cfg.data.cache_gt = "none"
    cfg.data.limit = 64
    cfg.train.epochs = 20
    cfg.train.accumulate = 4
    cfg.train.warmup_steps = 50
    return cfg

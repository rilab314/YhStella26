"""기본 실험 config — SEED-MAP 실데이터 + ConvNeXtV2 백본 + FPNLite."""

from configs.schema import ExperimentConfig


def get_config() -> ExperimentConfig:
    return ExperimentConfig()

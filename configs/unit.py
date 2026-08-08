"""단위 실험(U) 규격 — 가설 비교용 (improve-loop 스킬 · U 규격).

한 arm = 1 GPU. 4개를 동시에 띄워 한 라운드를 2.5시간 안에 끝내는 것이 목표다.
**절대값을 base(F 규격)와 비교하지 않는다** — 데이터량·유효 배치가 달라 의미가 없다.
같은 규격끼리(REF-U 대비)만 비교한다.
"""

from configs.base import get_config as get_base
from configs.schema import ExperimentConfig


def get_config() -> ExperimentConfig:
    cfg = get_base()
    cfg.data.limit = 3000  # 전체 8,979장에서 균등 간격 추출 (seedmap._subsample)
    cfg.data.num_workers = 6  # arm 4개 x 6 = 24 워커 (32 코어)
    cfg.train.epochs = 10
    cfg.train.devices = "1"
    cfg.train.accumulate = 4  # 유효 배치 4 — 짧은 실행에서는 step 수가 곧 학습량
    cfg.train.warmup_steps = 300  # 기본 1000은 이 길이의 실행에서 절반을 먹는다
    cfg.train.limit_val_batches = 320.0  # bs=1이므로 검증 320장
    cfg.log.max_batches = 4
    return cfg

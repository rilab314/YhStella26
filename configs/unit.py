"""단위 실험(U) 규격 — 가설 비교용 (research 스킬 · U 규격).

한 arm = 1 GPU. 4개를 동시에 띄워 한 라운드를 2.5시간 안에 끝내는 것이 목표다.
**절대값을 base(F 규격)와 비교하지 않는다** — 데이터량·유효 배치가 달라 의미가 없다.
같은 규격끼리(REF-U 대비)만 비교한다.
"""

from configs.base import get_config as get_base
from configs.schema import ExperimentConfig


def get_config() -> ExperimentConfig:
    cfg = get_base()
    cfg.data.limit = 3000  # 전체 8,979장에서 균등 간격 추출 (seedmap._subsample)
    # train/val 데이터로더가 **각각** 워커 풀을 만든다 → arm당 실제 워커는 이 값의 2배다.
    # 4로 두면 4 arm에서 32워커가 되어 부하 18.6(상한 16 초과) 실측. 2 = arm당 4, 4 arm = 16.
    cfg.data.num_workers = 2
    cfg.train.epochs = 10
    cfg.train.devices = "1"
    cfg.train.accumulate = 4  # 유효 배치 4 — 짧은 실행에서는 step 수가 곧 학습량
    cfg.train.warmup_steps = 300  # 기본 1000은 이 길이의 실행에서 절반을 먹는다
    # bs=1이라 이 값이 곧 검증 장수. **val 1,282장의 절반 이상을 쓴다** (사용자 지시 08-18).
    # 320장은 표본이 작아 판정이 흔들렸다 — 판정 밴드를 2%로 좁힌 이상 표본도 키워야 한다.
    cfg.train.limit_val_batches = 700.0
    cfg.log.max_batches = 4
    return cfg

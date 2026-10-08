"""3-C 비교 대상 — 연결 표현만 Sat2Graph 식 GTE(각도 섹터 고정 배정)로 바꾼 판.

**백본·넥·격자·클래스 헤드·학습 예산은 우리 최종 모델(F06d)과 같게 둔다.** 바뀌는 것은
"어느 슬롯이 어느 분기를 맡는가" 하나뿐이다 — 매칭으로 정하던 것을 각도로 고정 배정한다.
그래야 성능 차이를 표현 탓으로 돌릴 수 있다.

함께 바뀔 수밖에 없는 둘도 적어 둔다(표에 밝힌다):
  - 슬롯 수 R: 2 -> 6. 섹터가 2개면 한 칸이 180도라 방향을 못 가른다.
  - 슬롯 가중치 공유: 끈다. 섹터마다 뜻이 다르므로 같은 MLP 를 쓸 수 없다.
"""

from configs.base import get_config as get_base
from configs.schema import ExperimentConfig

NUM_SECTORS = 6  # Sat2Graph 의 방향 버킷 수 — 한 칸 60도


def get_config() -> ExperimentConfig:
    cfg = get_base()
    cfg.model.backbone.pretrained = "convnextv2_large.fcmae_ft_in22k_in1k_384"
    cfg.model.num_conn_slots = NUM_SECTORS
    cfg.model.share_slot_weights = False
    cfg.loss.conn.path = "stella.loss.conn_gte"
    cfg.loss.conn.name = "ConnLossGTE"
    return cfg

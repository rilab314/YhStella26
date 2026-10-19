"""비교 대상 — Sat2Graph 식 GTE 출력 표현 **전체**(각도 섹터 배정 + `(dx,dy)` 회귀).

`configs.exp_gte`(섹터 배정만) 위에 **거리 회귀 하나만** 얹는다. 백본·넥·격자·클래스 헤드·
학습 예산은 우리 최종 모델(F06d)과 같고, 섹터 수(6)와 슬롯 가중치 비공유도 `exp_gte` 그대로다.

그래서 이 판과 `exp_gte` 의 차이는 **"거리까지 배우는가" 하나**이고 (논지 C1-(2) 의 직접 근거),
이 판과 `configs.base` 의 차이는 **출력 표현 전체**다 (Table III 비교 대상 행).
"""

from configs.exp_gte import get_config as get_gte
from configs.schema import ExperimentConfig


def get_config() -> ExperimentConfig:
    cfg = get_gte()
    cfg.data.conn_store_length = True  # GT 에 분기 길이를 함께 싣는다 (캐시 폴더가 갈린다)
    cfg.model.conn_target = "delta"  # 헤드가 단위 방향 대신 변위를 낸다
    cfg.loss.conn.path = "stella.loss.conn_gte_delta"
    cfg.loss.conn.name = "ConnLossGTEDelta"
    cfg.decode.path = "stella.decode.graph_delta"
    cfg.decode.name = "DeltaChainDecoder"
    # 예측 지점에서 몇 셀까지 빗나간 후보를 받아줄지. 차선 간격(11.8 px = 2.95셀)에서 출발해
    # val 스윕(0.7~12, 8점)에서도 3.0 이 최고였다 (`experiment/result_0924.md` 2절).
    cfg.decode.perp_thresh = 3.0
    # 나머지 문턱은 우리 디코더와 **같은 절차**(좌표 하강 400장 → 전량 재채점)로 조율했다.
    # 손으로 찍은 값 0.3682 → 조율 후 0.3736 (G06_delta · val 전량).
    cfg.decode.end_thresh = 0.3
    cfg.decode.opp_thresh = 0.85
    cfg.decode.min_class_prob = 0.3
    cfg.decode.end_extend = 0.5
    return cfg

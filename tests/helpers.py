"""테스트 공용 헬퍼.

GT 주입 함수는 스크립트(디코더 전용 평가)도 쓰므로 패키지 안(`stella/model/inject.py`)에 있다.
여기서는 이름만 다시 내보낸다.
"""

from stella.model.inject import HIGH_LOGIT, gt_model_output

__all__ = ["HIGH_LOGIT", "gt_model_output"]

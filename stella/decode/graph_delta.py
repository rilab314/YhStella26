"""객체 생성(디코딩) — **예측 변위를 그대로 따라가는** 판 (Sat2Graph 식 GTE 대조군).

기본 판(`stella.decode.graph`)은 슬롯이 **방향만** 주므로 반경 안을 훑어 "예측한 선 위에
가장 잘 놓인" 정점을 고른다. 이 판은 슬롯이 **이웃 정점까지의 변위**를 주므로, 그 변위가
가리키는 **한 지점에 가장 가까운 정점**으로 간다 — 방향과 거리를 함께 쓴다.

그 외(정점 추출·마주봄 확인·꺾임각·종류 게이트·후처리)는 **전부 기본 판 그대로**다.
바뀌는 것은 후보 비용 하나이고, 그래야 이 판과 기본 판의 차이를 "출력 표현" 하나로 읽을
수 있다 (Table III 비교 대상 행).

`perp_thresh` 를 **빗나간 거리의 상한**으로 다시 쓴다 — 두 판 모두 "예측한 자리에서 몇 셀
어긋났는가"를 재는 값이라 자가 같다. 설정 필드를 새로 만들지 않는 이유다.
**이 값은 학습 뒤 val 에서 스윕해 정한다** — 우리 디코더의 문턱들도 그렇게 정해졌으므로,
같은 절차를 밟아야 비교 대상이 불리하게 시작하지 않는다 (D 트랙, GPU 없이 돈다).
"""

import numpy as np

from stella.decode.graph import ChainDecoder

UNIT_EPS = 1e-9


class DeltaChainDecoder(ChainDecoder):
    """슬롯의 변위가 찍은 지점에 가장 가까운 정점으로 사슬을 잇는다."""

    def _vertices(self, output) -> dict:
        """슬롯 출력을 방향과 거리로 쪼갠다 — 상속한 나머지 규칙이 단위 방향을 기대한다."""
        vertices = super()._vertices(output)
        delta = vertices["dir"]
        reach = np.linalg.norm(delta, axis=-1)
        vertices["reach"] = reach
        vertices["dir"] = np.divide(
            delta, reach[..., None], out=np.zeros_like(delta), where=reach[..., None] > UNIT_EPS
        )
        return vertices

    def _best_candidate(
        self, vertices, used, slot_used, vertex, k, label, previous
    ) -> tuple | None:
        """예측 지점에 가장 가까운 정점. 마주봄·꺾임각 게이트는 기본 판과 같다."""
        nearby = self._nearby_vertices(vertices, used, vertex, label)
        if nearby.size == 0:
            return None
        heading = vertices["dir"][vertex, k]
        delta = vertices["point"][nearby] - vertices["point"][vertex]
        distance = np.linalg.norm(delta, axis=-1)
        direction = delta / np.maximum(distance, UNIT_EPS)[:, None]
        opposite, _back = self._facing_slots(vertices, slot_used, nearby, heading)
        predicted = vertices["point"][vertex] + heading * vertices["reach"][vertex, k]
        missed = np.linalg.norm(vertices["point"][nearby] - predicted, axis=-1)
        cost = self._delta_cost(missed, opposite)
        return self._pick_candidate(cost, direction, previous, nearby, _back)

    def _delta_cost(self, missed: np.ndarray, opposite: np.ndarray) -> np.ndarray:
        """빗나간 거리 + 마주봄 벌점. 게이트를 못 넘은 후보는 inf."""
        facing = -opposite >= self.opp_thresh
        allowed = facing & (missed <= self.perp_thresh)
        return np.where(allowed, missed + self.w_opp * (1.0 + opposite), np.inf)

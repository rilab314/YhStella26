"""디코더 진단 카운터 (research 스킬 · 디코더 진단).

**정지 사유 분포가 곧 개선 지시서다.** 사슬이 왜 멈췄는지를 세어 두면
"조각남이 모델 탓인지 게이트 탓인지"를 지표 하나로 가른다.

- `end`      — 끝 확률이 임계를 넘어 멈췄다 (정상 종료)
- `nocand`   — 게이트를 통과한 후보가 없었다 (게이트가 엄하거나 방향이 틀렸다)
- `exist`    — 확장 슬롯의 존재 확률이 임계 미만이었다
- `slotused` — 이어갈 슬롯이 이미 사용됐다
"""

from collections import Counter

STOP_REASONS = ("end", "nocand", "exist", "slotused")


class ChainStats:
    """한 번의 디코딩마다 누적하고, 에폭 끝에 `summary()`로 비율을 낸다."""

    def __init__(self):
        self.counter: Counter = Counter()

    def reset(self) -> None:
        self.counter.clear()

    def add_stop(self, reason: str) -> None:
        self.counter[f"stop_{reason}"] += 1

    def add_chain(self, cells: int) -> None:
        self.counter["chains"] += 1
        self.counter["chain_cells"] += cells

    def add_reject(self) -> None:
        self.counter["purity_reject"] += 1

    def add_vertices(self, total: int, used: int) -> None:
        self.counter["vertices"] += total
        self.counter["vertex_used"] += used
        self.counter["images"] += 1

    def add_merge(self, count: int) -> None:
        self.counter["merged"] += count

    def summary(self) -> dict[str, float]:
        """비율로 환산한 진단 dict. 로그 키가 그대로 `val/dec/*`가 된다."""
        stops = sum(self.counter[f"stop_{r}"] for r in STOP_REASONS)
        chains = self.counter["chains"]
        result = {f"stop_{r}": _ratio(self.counter[f"stop_{r}"], stops) for r in STOP_REASONS}
        result["chains_per_img"] = _ratio(chains, self.counter["images"])
        result["chain_len"] = _ratio(self.counter["chain_cells"], chains)
        result["purity_reject"] = _ratio(self.counter["purity_reject"], self.counter["images"])
        # 영상당 검출 정점 수. `vertex_used`(쓰인 비율)만으로는 **정점이 모자란 것**과
        # **엮기에 실패한 것**을 못 가른다 — 분모가 여기 있어야 한다.
        result["vertices_per_img"] = _ratio(self.counter["vertices"], self.counter["images"])
        result["vertex_used"] = _ratio(self.counter["vertex_used"], self.counter["vertices"])
        result["merged_per_img"] = _ratio(self.counter["merged"], self.counter["images"])
        return result


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0

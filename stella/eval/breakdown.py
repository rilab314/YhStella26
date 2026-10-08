"""종류별 분해 · 중복 선 비율 — 논문 표가 요구하는데 채점기가 내지 않는 값 (6단계).

채점기(`ccq.InstanceCCQ`)는 종류별로는 F1·정밀도·재현율만 내고, 나머지(coverage·조각 수·
거짓 양성 두 갈래)는 전 종류를 합친 값만 낸다. 그리고 "같은 차선 위에 선을 두 번 그렸나"를
세는 함수가 없다. Table V(중복 선 비율)와 Table VII 하단(종류별 표)이 이 둘을 요구한다.

## 종류별 분해 — 한 종류만 남기고 같은 채점기를 돌린다

매칭은 원래 같은 종류끼리만 하므로(`ccq._greedy_match`) **F1·정밀도·재현율은 전체 채점과 같다.**
coverage·조각 수는 원래 "아무 종류의 예측이라도 덮으면" 세는데, 여기서는 **같은 종류의 예측만**
본다 — 종류별 표에서는 그쪽이 뜻이 맞다. 표 각주에 그렇게 적는다.

## 중복 선 비율 — `docs/paper_outline.md` Table V 의 정의

같은 종류의 **다른 예측선 하나**와 자기 길이의 80% 이상이 8 px 안에서 나란히 가는
(평가와 같은 각도 게이트) 예측선의 비율. 선 두 개의 합집합이 아니라 **한 선**과 재는 것이
요점이다 — 양옆 이웃 둘에 반씩 걸친 선은 중복이 아니다.
"""

import numpy as np

from stella.builder import build_instance
from stella.data.types import CLASS_NAMES
from stella.eval import geometry

REDUNDANT_BUFFER_PX = 8.0
REDUNDANT_SHARE = 0.8
# 종류마다 내는 값. 앞의 셋은 채점기가 종류별로 이미 내는 이름과 같다.
CLASS_METRICS = ("f1", "precision", "recall", "coverage", "frag", "fp_redundant", "fp_spurious")


class ClassBreakdown:
    """종류마다 채점기를 하나씩 두고, 예측·정답을 그 종류만 남겨 넣는다."""

    def __init__(self, cfg):
        excluded = {int(label) for label in cfg.eval.exclude_classes}
        self.labels = tuple(
            label for label in range(1, cfg.data.num_classes) if label not in excluded
        )
        self.metrics = {label: build_instance(cfg.eval, cfg) for label in self.labels}
        self.counts = {label: {"n_gt": 0, "n_pred": 0} for label in self.labels}

    def update(self, prediction: list[dict], target: list[dict]) -> None:
        for label in self.labels:
            mine = [item for item in prediction if int(item["class"]) == label]
            truth = [item for item in target if int(item["class"]) == label]
            self.counts[label]["n_gt"] += len(truth)
            self.counts[label]["n_pred"] += len(mine)
            if mine or truth:
                self.metrics[label].update(mine, truth)

    def merge(self, other: "ClassBreakdown") -> None:
        """다른 프로세스에서 누적한 것을 더한다 — 타일을 나눠 채점하기 위해서다."""
        for label in self.labels:
            self.metrics[label].merge_state(other.metrics[label])
            for key in ("n_gt", "n_pred"):
                self.counts[label][key] += other.counts[label][key]

    def summary(self) -> dict:
        return {CLASS_NAMES[label]: self._class_summary(label) for label in self.labels}

    def _class_summary(self, label: int) -> dict:
        metric = self.metrics[label]
        if not metric.update_called:  # 그 종류가 한 번도 안 나온 경우 — 채점기가 경고를 낸다
            return {**self.counts[label], **dict.fromkeys(CLASS_METRICS, 0.0)}
        computed = metric.compute()
        scores = {key: float(computed[key]) for key in CLASS_METRICS}
        return {**self.counts[label], **scores}


class RedundancyCounter:
    """예측선마다 "같은 종류의 다른 선 하나를 거의 그대로 따라가나"를 센다."""

    def __init__(self, *, angle_gate: float, sample_step: float, exclude_classes: tuple):
        self.angle_cos = float(np.cos(np.deg2rad(angle_gate)))
        self.sample_step = sample_step
        self.excluded = {int(label) for label in exclude_classes}
        self.lines = 0
        self.redundant = 0

    @classmethod
    def from_eval_cfg(cls, eval_cfg) -> "RedundancyCounter":
        return cls(
            angle_gate=eval_cfg.angle_gate,
            sample_step=eval_cfg.sample_step,
            exclude_classes=tuple(eval_cfg.exclude_classes),
        )

    def update(self, prediction: list[dict]) -> None:
        prepared = [self._prepare(item) for item in prediction]
        prepared = [item for item in prepared if item is not None]
        self.lines += len(prepared)
        self.redundant += sum(self._is_redundant(item, prepared) for item in prepared)

    def _prepare(self, item: dict) -> dict | None:
        points = np.asarray(item["points"], dtype=np.float64)
        if points.shape[0] < 2 or int(item["class"]) in self.excluded:
            return None
        sampled, tangent = geometry.resample(points, self.sample_step)
        box = geometry.bounding_box(points)
        return {"class": int(item["class"]), "points": sampled, "tangent": tangent, "box": box}

    def _is_redundant(self, item: dict, others: list[dict]) -> bool:
        for other in others:
            if other is item or other["class"] != item["class"]:
                continue
            if not geometry.boxes_overlap(item["box"], other["box"], REDUNDANT_BUFFER_PX):
                continue
            if self._shared_share(item, other) >= REDUNDANT_SHARE:
                return True
        return False

    def _shared_share(self, item: dict, other: dict) -> float:
        distance = geometry.gated_distance(
            item["points"], item["tangent"], other["points"], self.angle_cos
        )
        return float((distance <= REDUNDANT_BUFFER_PX).mean()) if distance.size else 0.0

    def merge(self, other: "RedundancyCounter") -> None:
        self.lines += other.lines
        self.redundant += other.redundant

    def summary(self) -> dict:
        ratio = self.redundant / self.lines if self.lines else 0.0
        return {"lines": self.lines, "redundant": self.redundant, "redundant_ratio": ratio}

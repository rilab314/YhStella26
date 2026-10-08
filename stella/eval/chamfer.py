"""Chamfer 거리 — 예측 선과 정답 선이 평균 몇 픽셀 떨어져 있나 (Table III · 교수 지시 09-30).

타일 한 장의 선을 일정 간격으로 촘촘히 찍어 점 집합 둘(예측 · 정답)을 만들고, 각 점에서 상대 집합의
가장 가까운 점까지 거리를 잰다. 두 방향을 따로 남긴다.

    pred_to_gt   예측 점 → 가장 가까운 정답 점   예측이 정답 밖으로 얼마나 벗어났나
    gt_to_pred   정답 점 → 가장 가까운 예측 점   정답을 얼마나 덮지 못했나
    chamfer      두 값의 평균                    타일마다 내고, 타일 평균을 표에 싣는다
    chamfer_median  타일별 chamfer 의 중앙값     통째로 놓친 선이 평균을 끌어올리는지 본다

**종류를 보지 않는다** — 선을 어디에 그렸는지만 재는 기하 지표다. 종류·인스턴스 구분은 F1 이 맡는다.
**선이 몇 조각인지도 보지 않는다** — 한 선을 스무 조각으로 그려도 점 집합은 같다. 설계 V-B 가
"점집합 거리는 덜 그린 것과 잘못 그린 것이 상쇄된다"고 쓰는 그 성질이라, 두 방향 값을 함께 둔다.

한쪽 집합이 비면 거리가 정의되지 않는다 — 그 타일은 평균에서 빼고 `skipped` 로 센다.
"""

import numpy as np
from scipy.spatial import cKDTree

from stella.eval import geometry


class ChamferDistance:
    """타일마다 `update` 하고 마지막에 `compute` 로 타일 평균을 낸다."""

    def __init__(self, *, step: float, exclude_classes: tuple):
        self.step = step
        self.excluded = {int(label) for label in exclude_classes}
        self.distances: list[tuple[float, float]] = []  # 타일별 (pred_to_gt, gt_to_pred)
        self.skipped = 0

    def update(self, prediction: list[dict], target: list[dict]) -> None:
        predicted, truth = self._points(prediction), self._points(target)
        if not len(predicted) or not len(truth):
            self.skipped += 1
            return
        pred_to_gt = cKDTree(truth).query(predicted)[0].mean()
        gt_to_pred = cKDTree(predicted).query(truth)[0].mean()
        self.distances.append((float(pred_to_gt), float(gt_to_pred)))

    def _points(self, instances: list[dict]) -> np.ndarray:
        """채점하는 종류의 선만 `step` 간격으로 찍어 한 점 집합으로 모은다."""
        sampled = [
            geometry.resample(item["points"], self.step)[0]
            for item in instances
            if int(item["class"]) not in self.excluded and len(item["points"]) >= 2
        ]
        return np.concatenate(sampled) if sampled else np.empty((0, 2))

    def compute(self) -> dict:
        values = np.asarray(self.distances, dtype=np.float64).reshape(-1, 2)
        pred_to_gt, gt_to_pred = values.mean(axis=0) if len(values) else (np.nan, np.nan)
        per_tile = values.mean(axis=1)
        return {
            "chamfer": float((pred_to_gt + gt_to_pred) / 2),
            "chamfer_median": float(np.median(per_tile)) if len(per_tile) else np.nan,
            "pred_to_gt": float(pred_to_gt),
            "gt_to_pred": float(gt_to_pred),
            "tiles": len(values),
            "skipped": self.skipped,
        }

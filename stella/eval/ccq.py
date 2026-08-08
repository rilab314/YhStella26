"""커버리지 중심 인스턴스 F1 — 비대칭 버퍼 CCQ (impl_plan 11절).

    C1(G, P) = |{x in G : d(x, P) <= rho}| / |G|      커버리지(완전성) — 쌍의 성질
    C2(P)    = |{y in P : d(y, 모든 GT) <= rho}| / |P| 정확성 — 예측 하나의 성질

    TP  <=>  C1 >= theta_cov (0.5, 관대)  and  C2 >= theta_cor (0.9, 엄격)

차선은 폭이 거의 없는 1차원 구조라 마스크 IoU 매칭이 맞지 않는다(11.3절).
"""

import numpy as np
import torch
from torchmetrics import Metric

from stella.data.types import CLASS_NAMES
from stella.eval import geometry


class InstanceCCQ(Metric):
    is_differentiable = False
    higher_is_better = True
    full_state_update = False

    @classmethod
    def from_cfg(cls, module_cfg, cfg, **kwargs) -> "InstanceCCQ":
        return cls(
            num_classes=cfg.data.num_classes,
            buffer_rho=module_cfg.buffer_rho,
            cov_thresh=module_cfg.cov_thresh,
            cor_thresh=module_cfg.cor_thresh,
            angle_gate=module_cfg.angle_gate,
            sample_step=module_cfg.sample_step,
            max_instances=module_cfg.max_instances,
            frag_min_cov=module_cfg.frag_min_cov,
        )

    def __init__(
        self,
        *,
        num_classes: int,
        buffer_rho: float,
        cov_thresh: float,
        cor_thresh: float,
        angle_gate: float,
        sample_step: float,
        max_instances: int,
        frag_min_cov: float,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.buffer_rho = buffer_rho
        self.cov_thresh = cov_thresh
        self.cor_thresh = cor_thresh
        self.angle_cos = float(np.cos(np.deg2rad(angle_gate)))
        self.sample_step = sample_step
        self.max_instances = max_instances
        self.frag_min_cov = frag_min_cov
        for name in ("tp", "fp", "fn", "fp_redundant", "fp_spurious"):
            self.add_state(name, torch.zeros(num_classes), dist_reduce_fx="sum")
        for name in ("gt_covered", "gt_total", "pred_covered", "pred_total"):
            self.add_state(name, torch.zeros(()), dist_reduce_fx="sum")
        for name in ("sq_error", "covered_points", "frag_sum", "frag_count", "frag_strict_sum"):
            self.add_state(name, torch.zeros(()), dist_reduce_fx="sum")

    def update(self, predictions: list[dict], targets: list[dict]) -> None:
        pred = _prepare(predictions[: self.max_instances], self.sample_step)
        gt = _prepare(targets[: self.max_instances], self.sample_step)
        correctness = self._correctness(pred, gt)
        coverage = self._coverage(pred, gt)
        self._accumulate_aggregate(pred, gt, correctness, coverage)
        self._accumulate_instances(pred, gt, correctness, coverage)

    def _correctness(self, pred: list[dict], gt: list[dict]) -> np.ndarray:
        """C2(P) — 예측이 어떤 GT 위에든 올라가 있는 비율. 모든 GT 대상이다."""
        values = np.zeros(len(pred))
        for index, item in enumerate(pred):
            distance = self._distance_to_union(item, gt)
            values[index] = float((distance <= self.buffer_rho).mean()) if distance.size else 0.0
        return values

    def _coverage(self, pred: list[dict], gt: list[dict]) -> np.ndarray:
        """C1(G, P) — GT가 각 예측에 덮인 비율 (쌍마다)."""
        values = np.zeros((len(gt), len(pred)))
        for gt_index, target in enumerate(gt):
            for pred_index, item in enumerate(pred):
                if not geometry.boxes_overlap(target["box"], item["box"], self.buffer_rho):
                    continue
                distance = geometry.gated_distance(
                    target["points"], target["tangent"], item["points"], self.angle_cos
                )
                values[gt_index, pred_index] = float((distance <= self.buffer_rho).mean())
        return values

    def _distance_to_union(self, item: dict, others: list[dict]) -> np.ndarray:
        distance = np.full(item["points"].shape[0], np.inf)
        for other in others:
            if not geometry.boxes_overlap(item["box"], other["box"], self.buffer_rho):
                continue
            candidate = geometry.gated_distance(
                item["points"], item["tangent"], other["points"], self.angle_cos
            )
            distance = np.minimum(distance, candidate)
        return distance

    def _accumulate_aggregate(self, pred, gt, correctness, coverage) -> None:
        """집계 커버리지·정확성·RMS 횡오차 (11.2절)."""
        for target in gt:
            distance = self._distance_to_union(target, pred)
            inside = distance <= self.buffer_rho
            self.gt_total += float(target["length"])
            self.gt_covered += (
                float(inside.mean()) * float(target["length"]) if inside.size else 0.0
            )
            self.sq_error += float((distance[inside] ** 2).sum())
            self.covered_points += float(inside.sum())
        for index, item in enumerate(pred):
            self.pred_total += float(item["length"])
            self.pred_covered += float(correctness[index]) * float(item["length"])

    def _accumulate_instances(self, pred, gt, correctness, coverage) -> None:
        matched_gt, matched_pred = _greedy_match(
            gt, pred, coverage, correctness, self.cov_thresh, self.cor_thresh
        )
        for gt_index, target in enumerate(gt):
            if gt_index in matched_gt:
                self.tp[target["class"]] += 1
                self._accumulate_fragments(coverage[gt_index], correctness)
            else:
                self.fn[target["class"]] += 1
        for pred_index, item in enumerate(pred):
            if pred_index in matched_pred:
                continue
            self.fp[item["class"]] += 1
            bucket = (
                self.fp_redundant
                if correctness[pred_index] >= self.cor_thresh
                else self.fp_spurious
            )
            bucket[item["class"]] += 1

    def _accumulate_fragments(self, coverage: np.ndarray, correctness: np.ndarray) -> None:
        """조각 수 두 가지. `frag`는 스치기만 해도 세고(정확성이 높을수록 부풀려진다),
        `frag_strict`는 그 GT를 `frag_min_cov` 이상 덮은 조각만 센다 (E00 관찰)."""
        accurate = correctness >= self.cor_thresh
        self.frag_sum += float(((coverage > 0) & accurate).sum())
        self.frag_strict_sum += float(((coverage >= self.frag_min_cov) & accurate).sum())
        self.frag_count += 1

    def compute(self) -> dict[str, torch.Tensor]:
        tp, fp, fn = self.tp, self.fp, self.fn
        precision = tp / (tp + fp).clamp(min=1e-9)
        recall = tp / (tp + fn).clamp(min=1e-9)
        f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-9)
        present = (tp + fn) > 0
        result = _micro_scores(tp.sum(), fp.sum(), fn.sum())
        result["f1_macro"] = f1[present].mean() if bool(present.any()) else torch.zeros(())
        result |= self._aggregate_scores()
        result |= _per_class(f1, precision, recall, present)
        return result

    def _aggregate_scores(self) -> dict[str, torch.Tensor]:
        return {
            "fp_redundant": self.fp_redundant.sum(),
            "fp_spurious": self.fp_spurious.sum(),
            "coverage": self.gt_covered / self.gt_total.clamp(min=1e-9),
            "correctness": self.pred_covered / self.pred_total.clamp(min=1e-9),
            "rms": (self.sq_error / self.covered_points.clamp(min=1e-9)).sqrt(),
            "frag": self.frag_sum / self.frag_count.clamp(min=1e-9),
            "frag_strict": self.frag_strict_sum / self.frag_count.clamp(min=1e-9),
        }


def _prepare(instances: list[dict], step: float) -> list[dict]:
    prepared = []
    for item in instances:
        points = np.asarray(item["points"], dtype=np.float64)
        if points.shape[0] < 2:
            continue
        sampled, tangent = geometry.resample(points, step)
        prepared.append(
            {
                "class": int(item["class"]),
                "points": sampled,
                "tangent": tangent,
                "length": geometry.polyline_length(points),
                "box": geometry.bounding_box(points),
            }
        )
    return prepared


def _greedy_match(
    gt: list[dict],
    pred: list[dict],
    coverage: np.ndarray,
    correctness: np.ndarray,
    cov_thresh: float,
    cor_thresh: float,
) -> tuple[set, set]:
    """두 조건을 통과한 쌍만 후보로 두고 C1 + C2 가중치로 일대일 최대가중 매칭한다."""
    candidates = []
    for gt_index in range(len(gt)):
        for pred_index in range(len(pred)):
            if gt[gt_index]["class"] != pred[pred_index]["class"]:
                continue
            c1, c2 = coverage[gt_index, pred_index], correctness[pred_index]
            if c1 >= cov_thresh and c2 >= cor_thresh:
                candidates.append((c1 + c2, gt_index, pred_index))
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    for _, gt_index, pred_index in sorted(candidates, reverse=True):
        if gt_index not in matched_gt and pred_index not in matched_pred:
            matched_gt.add(gt_index)
            matched_pred.add(pred_index)
    return matched_gt, matched_pred


def _micro_scores(tp: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor) -> dict[str, torch.Tensor]:
    precision = tp / (tp + fp).clamp(min=1e-9)
    recall = tp / (tp + fn).clamp(min=1e-9)
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall).clamp(min=1e-9),
    }


def _per_class(
    f1: torch.Tensor, precision: torch.Tensor, recall: torch.Tensor, present: torch.Tensor
) -> dict[str, torch.Tensor]:
    result = {}
    for label in range(1, f1.shape[0]):
        if not bool(present[label]):
            continue
        name = CLASS_NAMES[label] if label < len(CLASS_NAMES) else str(label)
        result[f"f1/{name}"] = f1[label]
        result[f"precision/{name}"] = precision[label]
        result[f"recall/{name}"] = recall[label]
    return result

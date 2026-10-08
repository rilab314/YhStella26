"""커버리지 중심 인스턴스 F1 — 비대칭 버퍼 CCQ (design 11절).

    C1(G, P) = |{x in G : d(x, P) <= rho}| / |G|      커버리지(완전성) — 쌍의 성질
    C2(G, P) = |{y in P : d(y, G) <= rho}| / |P|      정확성 — **같은 쌍**의 성질

    TP  <=>  C1 >= theta_cov (0.5, 관대)  and  C2 >= theta_cor (0.9, 엄격)

**C2 는 짝지어진 GT 기준으로 잰다** (paper_outline §7-⓪, 08-31 수정). 예측마다 "가장 잘 맞는
GT 하나"로 재면, 실제로는 GT-B 위에 올라간 예측이 GT-A 와 짝지어지면서 B 기준 값으로 매칭
문턱을 통과한다 — **옆 차선으로 갈아탄 예측**이 그 전형이다. 짝 기준 값은 TP 판정에 쓰고,
"이 예측이 *어떤* GT 위에라도 있나"를 묻는 FP 분류·집계 통계에는 GT 축 최대값을 쓴다.

차선은 폭이 거의 없는 1차원 구조라 마스크 IoU 매칭이 맞지 않는다(11.3절).
"""

import numpy as np
import torch
from torchmetrics import Metric

from stella.data.types import CLASS_NAMES
from stella.eval import geometry

# `label_instances` 가 선마다 돌려주는 판정 — 확인용 그림(`check_sheet.py`)이 읽는다.
TP, FN, FP_REDUNDANT, FP_SPURIOUS, EXCLUDED = "tp", "fn", "fp_redundant", "fp_spurious", "excluded"


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
            exclude_classes=module_cfg.exclude_classes,
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
        exclude_classes: tuple,
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
        self.exclude_classes = tuple(int(label) for label in exclude_classes)
        for name in ("tp", "fp", "fn", "fp_redundant", "fp_spurious"):
            self.add_state(name, torch.zeros(num_classes), dist_reduce_fx="sum")
        for name in ("gt_covered", "gt_total", "pred_covered", "pred_total"):
            self.add_state(name, torch.zeros(()), dist_reduce_fx="sum")
        for name in ("sq_error", "covered_points", "frag_sum", "frag_count", "frag_strict_sum"):
            self.add_state(name, torch.zeros(()), dist_reduce_fx="sum")

    def update(self, predictions: list[dict], targets: list[dict]) -> None:
        pred = _prepare(predictions[: self.max_instances], self.sample_step)
        gt = _prepare(targets[: self.max_instances], self.sample_step)
        coverage, correctness = self._score_matrices(pred, gt)
        best_correctness = _best_over_gt(correctness, len(pred))
        self._accumulate_aggregate(pred, gt, best_correctness)
        self._accumulate_instances(pred, gt, coverage, correctness, best_correctness)

    def label_instances(self, predictions: list[dict], targets: list[dict]) -> tuple[list, list]:
        """선마다 채점 판정을 돌려준다 — 확인용 그림이 **채점과 같은 짝**을 쓰게 한다.

        정답은 `TP`·`FN`, 예측은 `TP`·`FP_REDUNDANT`·`FP_SPURIOUS`. 채점에 들지 않는 선
        (제외 종류 · 점 2개 미만 · 상한 초과)은 `EXCLUDED` 다. 누적 상태는 건드리지 않는다.
        """
        pred = _prepare(predictions[: self.max_instances], self.sample_step)
        gt = _prepare(targets[: self.max_instances], self.sample_step)
        coverage, correctness = self._score_matrices(pred, gt)
        best_correctness = _best_over_gt(correctness, len(pred))
        matched_gt, matched_pred = _greedy_match(
            gt, pred, coverage, correctness, self.cov_thresh, self.cor_thresh
        )
        gt_labels = [TP if index in matched_gt else FN for index in range(len(gt))]
        pred_labels = [
            TP if index in matched_pred else self._fp_kind(best_correctness[index])
            for index in range(len(pred))
        ]
        excluded = self.exclude_classes
        return (
            _restore_labels(gt_labels, gt, len(targets), excluded),
            _restore_labels(pred_labels, pred, len(predictions), excluded),
        )

    def _fp_kind(self, best_correctness: float) -> str:
        """어떤 정답 위에는 올라가 있으면 중복, 어디에도 없으면 헛검출 (`_accumulate_instances`)."""
        return FP_REDUNDANT if best_correctness >= self.cor_thresh else FP_SPURIOUS

    def _score_matrices(self, pred: list[dict], gt: list[dict]) -> tuple[np.ndarray, np.ndarray]:
        """C1(G, P) 와 C2(G, P) 를 같은 쌍 루프에서 함께 낸다 — 둘 다 (GT x 예측) 행렬이다.

        두 값은 버퍼를 두르는 쪽만 반대다. C1 은 예측에 버퍼를 두르고 GT 점을 세고(조각남을
        처벌), C2 는 GT 에 버퍼를 두르고 예측 점을 센다(GT 를 벗어나는 것을 처벌).
        """
        coverage = np.zeros((len(gt), len(pred)))
        correctness = np.zeros((len(gt), len(pred)))
        for gt_index, target in enumerate(gt):
            for pred_index, item in enumerate(pred):
                if not geometry.boxes_overlap(target["box"], item["box"], self.buffer_rho):
                    continue
                coverage[gt_index, pred_index] = self._inside_ratio(target, item)
                correctness[gt_index, pred_index] = self._inside_ratio(item, target)
        return coverage, correctness

    def _inside_ratio(self, item: dict, other: dict) -> float:
        """`item` 의 샘플점 중 `other` 의 버퍼 안에 든 비율."""
        distance = geometry.gated_distance(
            item["points"], item["tangent"], other["points"], self.angle_cos
        )
        return float((distance <= self.buffer_rho).mean()) if distance.size else 0.0

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

    def _accumulate_aggregate(self, pred, gt, best_correctness) -> None:
        """집계 커버리지·정확성·RMS 횡오차 (11.2절).

        여기서 묻는 것은 "이 예측이 **어떤** GT 위에라도 있나"이므로 GT 축 최대값을 쓴다.
        """
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
            self.pred_covered += float(best_correctness[index]) * float(item["length"])

    def _accumulate_instances(self, pred, gt, coverage, correctness, best_correctness) -> None:
        matched_gt, matched_pred = _greedy_match(
            gt, pred, coverage, correctness, self.cov_thresh, self.cor_thresh
        )
        for gt_index, target in enumerate(gt):
            if gt_index in matched_gt:
                self.tp[target["class"]] += 1
                self._accumulate_fragments(coverage[gt_index], best_correctness)
            else:
                self.fn[target["class"]] += 1
        for pred_index, item in enumerate(pred):
            if pred_index in matched_pred:
                continue
            self.fp[item["class"]] += 1
            redundant = self._fp_kind(best_correctness[pred_index]) == FP_REDUNDANT
            bucket = self.fp_redundant if redundant else self.fp_spurious
            bucket[item["class"]] += 1

    def _accumulate_fragments(self, coverage: np.ndarray, best_correctness: np.ndarray) -> None:
        """조각 수 두 가지. `frag`는 스치기만 해도 세고(정확성이 높을수록 부풀려진다),
        `frag_strict`는 그 GT를 `frag_min_cov` 이상 덮은 조각만 센다 (E00 관찰)."""
        accurate = best_correctness >= self.cor_thresh
        self.frag_sum += float(((coverage > 0) & accurate).sum())
        self.frag_strict_sum += float(((coverage >= self.frag_min_cov) & accurate).sum())
        self.frag_count += 1

    def compute(self) -> dict[str, torch.Tensor]:
        tp, fp, fn = (self._counted(state) for state in (self.tp, self.fp, self.fn))
        precision = tp / (tp + fp).clamp(min=1e-9)
        recall = tp / (tp + fn).clamp(min=1e-9)
        f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-9)
        present = (tp + fn) > 0
        result = _micro_scores(tp.sum(), fp.sum(), fn.sum())
        result["f1_macro"] = f1[present].mean() if bool(present.any()) else torch.zeros(())
        result |= self._aggregate_scores()
        result |= _per_class(f1, precision, recall, present)
        return result

    def _counted(self, state: torch.Tensor) -> torch.Tensor:
        """제외 종류의 칸을 0 으로 만든 사본. 전체 점수·종류별 평균·종류별 행에서 함께 빠진다.

        `guiding_line` 처럼 **애초에 선이 아닌 것**(면으로 칠해진 영역)을 성능에서 빼기 위한
        손잡이다. 종류별 점수를 평균에서 빼는 것만으로는 안 된다 — 전체 점수는 종류를 합쳐
        세므로 **정답 쪽과 예측 쪽을 둘 다** 빼야 같은 판이 된다.
        """
        if not self.exclude_classes:
            return state
        kept = state.clone()
        kept[list(self.exclude_classes)] = 0.0
        return kept

    def _aggregate_scores(self) -> dict[str, torch.Tensor]:
        return {
            # 제외 종류(안전지대 등)는 TP·FP·FN 처럼 여기서도 뺀다 — 빼지 않아 Table VII 하단
            # 전체 행이 종류별 합과 97개 어긋났다(사용자 발견 10-05).
            "fp_redundant": self._counted(self.fp_redundant).sum(),
            "fp_spurious": self._counted(self.fp_spurious).sum(),
            "coverage": self.gt_covered / self.gt_total.clamp(min=1e-9),
            "correctness": self.pred_covered / self.pred_total.clamp(min=1e-9),
            "rms": (self.sq_error / self.covered_points.clamp(min=1e-9)).sqrt(),
            "frag": self.frag_sum / self.frag_count.clamp(min=1e-9),
            "frag_strict": self.frag_strict_sum / self.frag_count.clamp(min=1e-9),
        }


def _prepare(instances: list[dict], step: float) -> list[dict]:
    prepared = []
    for index, item in enumerate(instances):
        points = np.asarray(item["points"], dtype=np.float64)
        if points.shape[0] < 2:
            continue
        sampled, tangent = geometry.resample(points, step)
        prepared.append(
            {
                "index": index,  # 원래 목록에서의 자리 — `label_instances` 가 되돌릴 때 쓴다
                "class": int(item["class"]),
                "points": sampled,
                "tangent": tangent,
                "length": geometry.polyline_length(points),
                "box": geometry.bounding_box(points),
            }
        )
    return prepared


def _restore_labels(labels: list, prepared: list[dict], count: int, excluded: tuple) -> list:
    """준비 단계에서 빠진 선과 제외 종류를 원래 순서 자리에 `EXCLUDED` 로 되돌려 놓는다."""
    restored = [EXCLUDED] * count
    for label, item in zip(labels, prepared, strict=True):
        if item["class"] not in excluded:
            restored[item["index"]] = label
    return restored


def _greedy_match(
    gt: list[dict],
    pred: list[dict],
    coverage: np.ndarray,
    correctness: np.ndarray,
    cov_thresh: float,
    cor_thresh: float,
) -> tuple[set, set]:
    """두 조건을 통과한 쌍만 후보로 두고 C1 + C2 가중치로 일대일 최대가중 매칭한다.

    **두 값 모두 그 쌍 (G, P) 에서 잰 것이다** — C2 를 예측별 최대값으로 재면 옆 차선 위에
    올라간 예측이 다른 GT 와 짝지어지면서 문턱을 통과한다 (paper_outline §7-⓪).
    """
    candidates = []
    for gt_index in range(len(gt)):
        for pred_index in range(len(pred)):
            if gt[gt_index]["class"] != pred[pred_index]["class"]:
                continue
            c1, c2 = coverage[gt_index, pred_index], correctness[gt_index, pred_index]
            if c1 >= cov_thresh and c2 >= cor_thresh:
                candidates.append((c1 + c2, gt_index, pred_index))
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    for _, gt_index, pred_index in sorted(candidates, reverse=True):
        if gt_index not in matched_gt and pred_index not in matched_pred:
            matched_gt.add(gt_index)
            matched_pred.add(pred_index)
    return matched_gt, matched_pred


def _best_over_gt(matrix: np.ndarray, pred_count: int) -> np.ndarray:
    """예측마다 GT 축 최대값 — "어떤 GT 위에라도 있나"를 묻는 자리에서 쓴다."""
    if matrix.size == 0:
        return np.zeros(pred_count)
    return matrix.max(axis=0)


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

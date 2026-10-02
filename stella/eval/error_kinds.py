"""틀린 예측(FP)과 놓친 정답(FN)을 **원인별로** 가른다 — 교수 미팅 09-30 6·7절 (E2 · E4).

채점기(`ccq.InstanceCCQ`)는 FP 를 두 갈래(중복 · 헛검출)로만 센다. 여기서는 **채점기와 같은 짝**
(`_score_matrices` · `_greedy_match`)을 그대로 쓰고, 짝을 못 얻은 선마다 이유를 하나 붙인다.
기호는 채점기와 같다 — C1(G, P) = 예측이 정답을 덮은 비율, C2(G, P) = 예측이 정답 버퍼 안에 든 비율.

## FP — 짝을 못 얻은 예측 (위에서부터 처음 맞는 것 하나)

    duplicate       같은 종류 정답 위(C2 ≥ 0.9)인데 그 정답은 이미 짝이 있다 — 두 번 그림
    fragment        같은 종류 정답 위인데 그 정답은 짝이 없다(덮은 길이 부족) — 조각만 그림
    wrong_class     같은 종류 정답 위는 아니고 다른 종류 정답 위다          — 종류 혼동
    drift           어떤 정답 위에 절반 이상 있다가 벗어난다(0.5 ≤ C2 < 0.9) — 옆 차선·어긋남
    spurious_short  어떤 정답 위에도 거의 없고(C2 < 0.5) 짧다               — 짧은 가짜 선
    spurious_long   어떤 정답 위에도 거의 없고 길다                         — 긴 가짜 선

## FN — 짝을 못 얻은 정답

    wrong_class  다른 종류 예측 하나가 짝 조건을 만족한다            — 찾았는데 종류를 틀림
    drifted      같은 종류 예측이 절반 이상 덮지만 버퍼를 벗어난다  — 찾았는데 어긋남
    split        같은 종류 예측 여럿을 합치면 절반 이상 덮는다      — 조각나서 놓침
    partial      같은 종류 예측이 조금(10%~50%) 덮는다             — 일부만 그림
    missed       같은 종류 예측이 거의 안 덮는다(10% 미만)          — 통째로 놓침

채점에서 빼는 종류(배경 · 안전지대)는 처음부터 뺀다. 원인별 개수는 **종류별로도** 남기고(E3),
정답 선의 길이를 종류별로 모아 "어떤 모양의 선인가"를 함께 적는다.
"""

from collections import Counter, defaultdict

import numpy as np

from stella.builder import build_instance
from stella.data.types import CLASS_NAMES
from stella.eval import ccq

FP_KINDS = ("duplicate", "fragment", "wrong_class", "drift", "spurious_short", "spurious_long")
FN_KINDS = ("wrong_class", "drifted", "split", "partial", "missed")
DRIFT_FLOOR = 0.5  # C2 가 이 이상이면 "정답 위에 있다가 벗어났다"
TOUCH_FLOOR = 0.1  # 정답을 이만큼도 못 덮으면 "통째로 놓쳤다"
SHORT_PX = 40.0  # 가짜 선을 짧다/길다로 가르는 길이 — 10칸


class ErrorKinds:
    """타일마다 `update` 하고 `compute` 로 원인별 개수를 낸다."""

    def __init__(self, cfg):
        self.scorer = build_instance(cfg.eval, cfg)  # 짝 규칙·문턱·버퍼를 채점기에서 빌린다
        self.excluded = set(self.scorer.exclude_classes)
        self.counts = {"fp": Counter(), "fn": Counter(), "tp": 0}
        self.by_class = Counter()  # (갈래, 원인, 종류) — 갈래는 "tp" · "fp" · "fn"
        self.gt_lengths: dict[int, list[float]] = defaultdict(list)

    def update(self, prediction: list[dict], target: list[dict]) -> None:
        pred, gt = self._prepare(prediction), self._prepare(target)
        matched_gt, fp_kinds, fn_kinds = self._judge(pred, gt)
        self.counts["tp"] += len(matched_gt)
        for index in matched_gt:
            self.by_class["tp", "", gt[index]["class"]] += 1
        for item in gt:
            self.gt_lengths[item["class"]].append(float(item["length"]))
        for index, kind in fp_kinds.items():
            self._count("fp", kind, pred[index]["class"])
        for index, kind in fn_kinds.items():
            self._count("fn", kind, gt[index]["class"])

    def label_predictions(self, prediction: list[dict], target: list[dict]) -> list[str]:
        """예측 선마다 판정 — "tp" · FP 원인 이름 · 채점 제외는 "excluded". 누적하지 않는다.

        원인별 표본을 그려 눈으로 확인할 때 쓴다(E5). 순서는 `prediction` 그대로다.
        """
        kept = [i for i, item in enumerate(prediction) if int(item["class"]) not in self.excluded]
        pred, gt = self._prepare(prediction), self._prepare(target)
        _, fp_kinds, _ = self._judge(pred, gt)
        labels = ["excluded"] * len(prediction)
        for position, item in enumerate(pred):
            labels[kept[item["index"]]] = fp_kinds.get(position, "tp")
        return labels

    def _judge(self, pred: list[dict], gt: list[dict]) -> tuple[set, dict, dict]:
        """채점기와 같은 짝을 맺고, 짝 없는 예측·정답마다 원인을 붙인다."""
        coverage, correctness = self.scorer._score_matrices(pred, gt)
        matched_gt, matched_pred = ccq._greedy_match(
            gt, pred, coverage, correctness, self.scorer.cov_thresh, self.scorer.cor_thresh
        )
        same = _same_class(gt, pred)
        fp_kinds = {
            index: self._fp_kind(index, pred, correctness, same, matched_gt)
            for index in set(range(len(pred))) - matched_pred
        }
        fn_kinds = {
            index: self._fn_kind(index, gt, pred, coverage, correctness, same)
            for index in set(range(len(gt))) - matched_gt
        }
        return matched_gt, fp_kinds, fn_kinds

    def _count(self, side: str, kind: str, label: int) -> None:
        self.counts[side][kind] += 1
        self.by_class[side, kind, label] += 1

    def _prepare(self, instances: list[dict]) -> list[dict]:
        kept = [item for item in instances if int(item["class"]) not in self.excluded]
        return ccq._prepare(kept[: self.scorer.max_instances], self.scorer.sample_step)

    def _fp_kind(self, index, pred, correctness, same, matched_gt) -> str:
        on_gt = correctness[:, index] >= self.scorer.cor_thresh
        same_on = np.flatnonzero(on_gt & same[:, index])
        if len(same_on):
            return "duplicate" if any(g in matched_gt for g in same_on) else "fragment"
        if on_gt.any():
            return "wrong_class"
        best = correctness[:, index].max() if correctness.shape[0] else 0.0
        if best >= DRIFT_FLOOR:
            return "drift"
        return "spurious_short" if pred[index]["length"] < SHORT_PX else "spurious_long"

    def _fn_kind(self, index, gt, pred, coverage, correctness, same) -> str:
        cov, cor = coverage[index], correctness[index]
        passes = (cov >= self.scorer.cov_thresh) & (cor >= self.scorer.cor_thresh)
        if (passes & ~same[index]).any():
            return "wrong_class"
        if ((cov >= self.scorer.cov_thresh) & same[index]).any():
            return "drifted"
        peers = [item for item, flag in zip(pred, same[index], strict=True) if flag]
        union = self._union_coverage(gt[index], peers)
        if union >= self.scorer.cov_thresh:
            return "split"
        return "partial" if union >= TOUCH_FLOOR else "missed"

    def _union_coverage(self, target: dict, peers: list[dict]) -> float:
        """같은 종류 예측 **전부를 합쳐** 이 정답의 몇 %를 덮나."""
        distance = self.scorer._distance_to_union(target, peers)
        return float((distance <= self.scorer.buffer_rho).mean()) if distance.size else 0.0

    def compute(self) -> dict:
        fp, fn, tp = self.counts["fp"], self.counts["fn"], self.counts["tp"]
        return {
            "tp": tp,
            "fp": {kind: fp[kind] for kind in FP_KINDS},
            "fn": {kind: fn[kind] for kind in FN_KINDS},
            "fp_total": sum(fp.values()),
            "fn_total": sum(fn.values()),
            "per_class": {
                CLASS_NAMES[label]: self._class_summary(label) for label in sorted(self.gt_lengths)
            },
        }

    def _class_summary(self, label: int) -> dict:
        """한 종류의 원인별 개수 + 정답 선의 모양(개수 · 길이)."""
        lengths = np.asarray(self.gt_lengths[label])
        return {
            "tp": self.by_class["tp", "", label],
            "fp": {kind: self.by_class["fp", kind, label] for kind in FP_KINDS},
            "fn": {kind: self.by_class["fn", kind, label] for kind in FN_KINDS},
            "n_gt": int(lengths.size),
            "gt_length_mean": float(lengths.mean()),
            "gt_length_median": float(np.median(lengths)),
        }


def _same_class(gt: list[dict], pred: list[dict]) -> np.ndarray:
    """(정답 x 예측) 같은 종류인가."""
    gt_class = np.array([item["class"] for item in gt], dtype=int).reshape(-1, 1)
    pred_class = np.array([item["class"] for item in pred], dtype=int).reshape(1, -1)
    return gt_class == pred_class

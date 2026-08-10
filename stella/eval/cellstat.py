"""셀 단위 진단 지표 (improve-loop 스킬 · 셀 단위 진단).

인스턴스 지표(`val/inst/*`)는 "결과가 나쁘다"까지만 알려준다. 이 모듈은 **어느 출력이
나빠서** 그렇게 됐는지를 셀 단위로 분해한다. 전부 GPU 텐서 연산이라 비용이 사실상 없다.

가장 중요한 두 값:
- `link_ok`  — GT 분기 중 예측 방향이 디코더 정렬 게이트 안에 들어온 비율. 사슬이 이어질 확률.
- `chain_expect` = 1 / (1 - link_ok) — 링크 실패까지의 기대 사슬 길이.
  실측 `val/inst/frag` 와 비교하면 병목이 **모델(방향)** 인지 **디코더** 인지가 갈린다.
"""

import numpy as np
import torch
from torchmetrics import Metric

from stella.loss.matching import assign_slots, pad_branches

ANGLE_BINS = 181  # 0..180도, 1도 폭
VALID_NORM_THRESH = 0.5  # conn_dirs 는 단위벡터, 빈 분기는 0
STRICT_DEG = 20.0  # 여유 있는 링크 판정 기준
MAX_CHAIN_EXPECT = 999.0  # link_ok = 1 일 때의 발산 방지
COUNT_STATES = (
    "heat_gt",
    "heat_hit",
    "heat_pos_sum",
    "heat_neg_sum",
    "heat_neg_count",
    "node_count",
    "node_on_gt",
    "class_hit",
    "fg_hit",
    "class_count",
    "class_fg_hit",
    "bg_hit",
    "bg_count",
    "coord_err",
    "coord_count",
    "end_tp",
    "end_fp",
    "end_fn",
    "end_pos_sum",
    "end_pos_count",
    "end_neg_sum",
    "end_neg_count",
    "exist_pos_sum",
    "exist_pos_count",
    "exist_neg_sum",
    "exist_neg_count",
    "samples",
)


class CellDiagnostics(Metric):
    is_differentiable = False
    higher_is_better = True
    full_state_update = False

    @classmethod
    def from_cfg(cls, module_cfg, cfg, **kwargs) -> "CellDiagnostics":
        return cls(
            grid_stride=cfg.data.grid_stride,
            num_conn_slots=cfg.model.num_conn_slots,
            align_thresh=cfg.decode.align_thresh,
            end_thresh=cfg.decode.end_thresh,
        )

    def __init__(
        self,
        *,
        grid_stride: int,
        num_conn_slots: int,
        align_thresh: float,
        end_thresh: float,
    ):
        super().__init__()
        self.grid_stride = grid_stride
        self.num_conn_slots = num_conn_slots
        self.align_deg = float(np.rad2deg(np.arccos(np.clip(align_thresh, -1.0, 1.0))))
        self.end_thresh = end_thresh
        for name in COUNT_STATES:
            self.add_state(name, torch.zeros(()), dist_reduce_fx="sum")
        self.add_state("angle_hist", torch.zeros(ANGLE_BINS), dist_reduce_fx="sum")

    @torch.no_grad()
    def update(self, output, targets: dict) -> None:
        positive = targets["class_map"] > 0
        selected = output.node_mask
        both = positive & selected
        self.samples += float(positive.shape[0])
        self._update_heat(output, positive, selected)
        self._update_class(output, targets, both, selected & ~positive)
        self._update_coord(output, targets, both)
        self._update_end(output, targets, both)
        self._update_conn(output, targets, both, selected & ~positive)

    def _update_heat(self, output, positive: torch.Tensor, selected: torch.Tensor) -> None:
        """`heat_recall`은 선택 파이프라인(임계+팽창+n_max) 전체의 결과다. 그것만 보면
        "순위가 나쁜 것"과 "임계가 틀린 것"을 구분할 수 없어 확률 평균을 함께 남긴다."""
        probability = output.heatmap_logit.float().sigmoid()
        self.heat_gt += positive.sum()
        self.heat_hit += (positive & selected).sum()
        self.heat_pos_sum += probability[positive].sum()
        self.heat_neg_sum += probability[~positive].sum()
        self.heat_neg_count += (~positive).sum()
        self.node_count += selected.sum()
        self.node_on_gt += (positive & selected).sum()

    def _update_class(self, output, targets: dict, both, false_positive) -> None:
        """`class_acc`(정확한 클래스)와 `class_fg`(배경이 아니라고 본 비율)를 함께 본다.

        디코더는 `argmax != 0`인 셀만 정점으로 쓴다. 둘이 같이 낮으면 "GT 셀을 배경이라 부른다"
        (불균형 문제)이고, `class_fg`만 높으면 "찾긴 했는데 차선 종류를 틀린다"(분류 문제)다.
        원인이 전혀 다르므로 반드시 갈라 봐야 한다.

        **주의 — `class_acc`·`class_fg`는 분모가 "선택된 GT 셀"이라 실행 간 비교가 위험하다.**
        선택이 늘면 어려운 셀이 섞여 정확도가 내려간다(REF-F 실측: 붕괴 에폭을 뺀 구간에서
        corr(heat_recall, class_acc) = -0.54). 그래서 분모를 **전체 GT 셀**로 바꾼
        `class_recall`·`vertex_recall`을 함께 낸다 — 선택 수가 다른 arm끼리도 비교할 수 있고,
        `vertex_recall`은 곧 **디코더가 실제로 받는 정점 재현율**이다.
        """
        predicted = output.class_logit.argmax(dim=-1)
        self.class_count += both.sum()
        self.class_hit += (predicted[both] == targets["class_map"][both]).sum()
        self.class_fg_hit += (predicted[both] > 0).sum()
        self.fg_hit += (output.fg_logit[both] > 0.0).sum()  # 이진 헤드 기준 (E12)
        self.bg_count += false_positive.sum()
        self.bg_hit += (predicted[false_positive] == 0).sum()

    def _update_coord(self, output, targets: dict, both) -> None:
        if not bool(both.any()):
            return
        error = output.self_coord[both].float() - targets["coord_map"][both].float()
        self.coord_err += error.norm(dim=-1).sum()
        self.coord_count += both.sum()

    def _update_end(self, output, targets: dict, both) -> None:
        """끝 셀 양성이 ~2.5%뿐이라 임계 0.5 기준 재현율은 0으로 눌리기 쉽다.
        확률 평균을 함께 남겨 "순위가 없는 것"과 "임계가 높은 것"을 구분한다."""
        if not bool(both.any()):
            return
        probability = output.end_logit[both].float().sigmoid()
        target = targets["end_map"][both] > 0.5
        predicted = probability > self.end_thresh
        self.end_tp += (predicted & target).sum()
        self.end_fp += (predicted & ~target).sum()
        self.end_fn += (~predicted & target).sum()
        self.end_pos_sum += probability[target].sum()
        self.end_pos_count += target.sum()
        self.end_neg_sum += probability[~target].sum()
        self.end_neg_count += (~target).sum()

    def _update_conn(self, output, targets: dict, both, false_positive) -> None:
        self._update_exist(output, both, false_positive)
        if not bool(both.any()):
            return
        gt_dir = targets["conn_dirs"][both].float()
        valid = gt_dir.norm(dim=-1) > VALID_NORM_THRESH
        gt_dir, valid = pad_branches(gt_dir, valid, self.num_conn_slots)
        pred_dir = output.conn_dir[both].float()
        assignment, matched, _ = assign_slots(
            pred_dir, output.exist_logit[both].float(), gt_dir, valid, 1.0, 1.0
        )
        self._accumulate_angles(pred_dir, gt_dir, assignment, matched)

    def _update_exist(self, output, both, false_positive) -> None:
        probability = output.exist_logit.float().sigmoid()
        self.exist_pos_sum += probability[both].sum()
        self.exist_pos_count += float(both.sum()) * self.num_conn_slots
        self.exist_neg_sum += probability[false_positive].sum()
        self.exist_neg_count += float(false_positive.sum()) * self.num_conn_slots

    def _accumulate_angles(self, pred_dir, gt_dir, assignment, matched) -> None:
        if not bool(matched.any()):
            return
        index = assignment.unsqueeze(-1).expand(-1, -1, 2)
        assigned = torch.gather(gt_dir, 1, index)
        cosine = (pred_dir * assigned).sum(dim=-1)[matched].clamp(-1.0, 1.0)
        degrees = torch.rad2deg(torch.arccos(cosine))
        bins = degrees.long().clamp(0, ANGLE_BINS - 1)
        self.angle_hist += torch.bincount(bins, minlength=ANGLE_BINS).to(self.angle_hist)

    def compute(self) -> dict[str, torch.Tensor]:
        result = {
            "heat_recall": _ratio(self.heat_hit, self.heat_gt),
            "heat_precision": _ratio(self.node_on_gt, self.node_count),
            "heat_pos": _ratio(self.heat_pos_sum, self.heat_gt),
            "heat_neg": _ratio(self.heat_neg_sum, self.heat_neg_count),
            "node_per_img": _ratio(self.node_count, self.samples),
            "class_acc": _ratio(self.class_hit, self.class_count),
            "class_fg": _ratio(self.class_fg_hit, self.class_count),
            # 이진 전경 헤드가 같은 셀을 전경이라 부르는 비율 (E12). 헤드가 꺼져 있으면 0이다.
            "fg_acc": _ratio(self.fg_hit, self.class_count),
            # 분모가 **전체 GT 셀**이라 선택 수가 달라도 비교할 수 있다 (아래 주석).
            "class_recall": _ratio(self.class_hit, self.heat_gt),
            "vertex_recall": _ratio(self.class_fg_hit, self.heat_gt),
            "class_bg_recall": _ratio(self.bg_hit, self.bg_count),
            "coord_err_px": _ratio(self.coord_err, self.coord_count) * self.grid_stride,
            "end_recall": _ratio(self.end_tp, self.end_tp + self.end_fn),
            "end_precision": _ratio(self.end_tp, self.end_tp + self.end_fp),
            "end_pos": _ratio(self.end_pos_sum, self.end_pos_count),
            "end_neg": _ratio(self.end_neg_sum, self.end_neg_count),
            "exist_pos": _ratio(self.exist_pos_sum, self.exist_pos_count),
            "exist_neg": _ratio(self.exist_neg_sum, self.exist_neg_count),
        }
        return result | self._angle_scores()

    def _angle_scores(self) -> dict[str, torch.Tensor]:
        hist = self.angle_hist
        total = hist.sum()
        centers = torch.arange(ANGLE_BINS, device=hist.device, dtype=hist.dtype) + 0.5
        link_ok = _ratio(hist[: int(self.align_deg) + 1].sum(), total)
        return {
            "dir_err_deg": _ratio((hist * centers).sum(), total),
            "dir_err_p90": _percentile(hist, 0.9),
            "link_ok": link_ok,
            "link_ok_20deg": _ratio(hist[: int(STRICT_DEG) + 1].sum(), total),
            "chain_expect": (1.0 / (1.0 - link_ok)).clamp(max=MAX_CHAIN_EXPECT),
        }


def _ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    denominator = torch.as_tensor(denominator, dtype=torch.float32)
    return torch.as_tensor(numerator, dtype=torch.float32) / denominator.clamp(min=1e-9)


def _percentile(hist: torch.Tensor, fraction: float) -> torch.Tensor:
    total = float(hist.sum())
    if total <= 0:
        return torch.zeros((), device=hist.device)
    cumulative = torch.cumsum(hist, dim=0)
    index = int(torch.searchsorted(cumulative, torch.tensor(total * fraction, device=hist.device)))
    return torch.tensor(float(min(index, ANGLE_BINS - 1)) + 0.5, device=hist.device)

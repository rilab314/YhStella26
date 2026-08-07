"""self 슬롯 손실 — 클래스·좌표·끝 (impl_plan 8.2절, 9차 개정).

클래스는 선택된 전 셀에 준다 — 거짓 양성 셀은 배경(0)으로 감독한다(결정 24: 클래스 0을
학습하지 않으면 디코더의 배경 필터가 무력해진다). 구 설계의 "종점 셀 제외" 특례는
폐기했다 — 끝칸 미채움 규약(6.2절)이 라벨이 모호한 셀 자체를 없앤다.
좌표·끝(end)은 GT 양성 셀에만 준다.
"""

import torch
import torch.nn.functional as F

from stella.loss.criterion import LossModule


class SelfSlotLoss(LossModule):
    def __init__(self, *, w_class: float, w_coord: float, w_end: float):
        super().__init__()
        self.w_class = w_class
        self.w_coord = w_coord
        self.w_end = w_end

    def forward(self, output, targets: dict) -> dict[str, torch.Tensor]:
        positive = (targets["class_map"] > 0) & output.node_mask
        class_loss = self._class_loss(output, targets)
        coord_loss = self._coord_loss(output, targets, positive)
        end_loss = self._end_loss(output, targets, positive)
        total = self.w_class * class_loss + self.w_coord * coord_loss + self.w_end * end_loss
        return {"class": class_loss, "coord": coord_loss, "end": end_loss, "total": total}

    @staticmethod
    def _class_loss(output, targets: dict) -> torch.Tensor:
        """선택된 전 셀 — class_map이 배경 0이라 거짓 양성 셀의 라벨이 그대로 0이 된다."""
        selected = output.node_mask
        if not bool(selected.any()):
            return output.class_logit.sum() * 0.0
        return F.cross_entropy(output.class_logit[selected].float(), targets["class_map"][selected])

    @staticmethod
    def _coord_loss(output, targets: dict, positive: torch.Tensor) -> torch.Tensor:
        if not bool(positive.any()):
            return output.self_coord.sum() * 0.0
        predicted = output.self_coord[positive].float()
        return F.smooth_l1_loss(predicted, targets["coord_map"][positive].float())

    @staticmethod
    def _end_loss(output, targets: dict, positive: torch.Tensor) -> torch.Tensor:
        """end_map 직접 감독 — 양성의 ~2.5%라 우선 pos_weight 없이 시작한다 (8.2절)."""
        if not bool(positive.any()):
            return output.end_logit.sum() * 0.0
        predicted = output.end_logit[positive].float()
        return F.binary_cross_entropy_with_logits(predicted, targets["end_map"][positive].float())

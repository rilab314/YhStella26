"""self 슬롯 손실 — 클래스·좌표·끝 (design 8.2절, 9차 개정).

클래스는 선택된 전 셀에 준다 — 거짓 양성 셀은 배경(0)으로 감독한다(결정 24: 클래스 0을
학습하지 않으면 디코더의 배경 필터가 무력해진다). 구 설계의 "종점 셀 제외" 특례는
폐기했다 — 끝칸 미채움 규약(6.2절)이 라벨이 모호한 셀 자체를 없앤다.
좌표·끝(end)은 GT 양성 셀에만 준다.
"""

import torch
import torch.nn.functional as F

from stella.loss.criterion import LossModule


class SelfSlotLoss(LossModule):
    def __init__(
        self,
        *,
        w_class: float,
        w_coord: float,
        w_end: float,
        end_pos_weight: float,
        class_bg_weight: float,
    ):
        super().__init__()
        self.w_class = w_class
        self.w_coord = w_coord
        self.w_end = w_end
        self.end_pos_weight = end_pos_weight
        self.class_bg_weight = class_bg_weight

    def forward(self, output, targets: dict) -> dict[str, torch.Tensor]:
        positive = (targets["class_map"] > 0) & output.node_mask
        class_loss = self._class_loss(output, targets)
        coord_loss = self._coord_loss(output, targets, positive)
        end_loss = self._end_loss(output, targets, positive)
        total = self.w_class * class_loss + self.w_coord * coord_loss + self.w_end * end_loss
        return {"class": class_loss, "coord": coord_loss, "end": end_loss, "total": total}

    def _class_loss(self, output, targets: dict) -> torch.Tensor:
        """선택된 전 셀 — class_map이 배경 0이라 거짓 양성 셀의 라벨이 그대로 0이 된다.

        선택 셀의 70%가 배경이라 CE가 배경에 지배당한다(REF-F 에폭 1에서 class_acc 0.003 /
        bg_recall 0.997 관측). `class_bg_weight < 1`로 그 지배를 완화한다 (가설 백로그 B6).
        """
        selected = output.node_mask
        if not bool(selected.any()):
            return output.class_logit.sum() * 0.0
        label = targets["class_map"][selected]
        loss = F.cross_entropy(output.class_logit[selected].float(), label, reduction="none")
        if self.class_bg_weight == 1.0:
            return loss.mean()
        weight = torch.where(label > 0, 1.0, self.class_bg_weight)
        return (loss * weight).sum() / weight.sum().clamp(min=1e-9)

    @staticmethod
    def _coord_loss(output, targets: dict, positive: torch.Tensor) -> torch.Tensor:
        if not bool(positive.any()):
            return output.self_coord.sum() * 0.0
        predicted = output.self_coord[positive].float()
        return F.smooth_l1_loss(predicted, targets["coord_map"][positive].float())

    def _end_loss(self, output, targets: dict, positive: torch.Tensor) -> torch.Tensor:
        """end_map 직접 감독. 양성이 ~2.5%뿐이라 pos_weight를 열어 둔다 (8.2절, 가설 백로그 B2)."""
        if not bool(positive.any()):
            return output.end_logit.sum() * 0.0
        predicted = output.end_logit[positive].float()
        weight = predicted.new_tensor(self.end_pos_weight)
        return F.binary_cross_entropy_with_logits(
            predicted, targets["end_map"][positive].float(), pos_weight=weight
        )

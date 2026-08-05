"""self 슬롯 손실 — 클래스·좌표 (impl_plan 8.2절).

클래스와 좌표의 **감독 범위가 다르다.** 클래스는 종점 셀을 빼고(라벨이 모호하다),
좌표는 종점 셀에도 준다(끝점 좌표가 디코딩에 필요하다).
거짓 양성 셀(뽑혔지만 GT 양성이 아닌 셀)은 배경(0)으로 감독한다.
"""

import torch
import torch.nn.functional as F

from stella.loss.criterion import LossModule


class SelfSlotLoss(LossModule):
    def __init__(self, *, w_class: float, w_coord: float):
        super().__init__()
        self.w_class = w_class
        self.w_coord = w_coord

    def forward(self, output, targets: dict) -> dict[str, torch.Tensor]:
        class_loss = self._class_loss(output, targets)
        coord_loss = self._coord_loss(output, targets)
        total = self.w_class * class_loss + self.w_coord * coord_loss
        return {"class": class_loss, "coord": coord_loss, "total": total}

    @staticmethod
    def _class_loss(output, targets: dict) -> torch.Tensor:
        class_map = targets["class_map"]
        positive = class_map > 0
        supervised = (positive & (targets["end_map"] == 0)) | (output.node_mask & ~positive)
        if not bool(supervised.any()):
            return output.class_logit.sum() * 0.0
        logit = output.class_logit[supervised].float()
        label = torch.where(positive, class_map, torch.zeros_like(class_map))[supervised]
        return F.cross_entropy(logit, label)

    @staticmethod
    def _coord_loss(output, targets: dict) -> torch.Tensor:
        positive = targets["class_map"] > 0
        if not bool(positive.any()):
            return output.self_coord.sum() * 0.0
        predicted = output.self_coord[positive].float()
        return F.smooth_l1_loss(predicted, targets["coord_map"][positive].float())

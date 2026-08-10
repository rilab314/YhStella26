"""self 슬롯 손실 — 클래스·좌표·끝 (design 8.2절, 9차 개정).

클래스는 선택된 전 셀에 준다 — 거짓 양성 셀은 배경(0)으로 감독한다(결정 24: 클래스 0을
학습하지 않으면 디코더의 배경 필터가 무력해진다). 구 설계의 "종점 셀 제외" 특례는
폐기했다 — 끝칸 미채움 규약(6.2절)이 라벨이 모호한 셀 자체를 없앤다.
좌표·끝(end)은 GT 양성 셀에만 준다.
"""

import torch
import torch.nn.functional as F

from stella.data.types import CLASS_INSTANCE_COUNT
from stella.loss.criterion import LossModule


class SelfSlotLoss(LossModule):
    @classmethod
    def from_cfg(cls, module_cfg, cfg, **kwargs) -> "SelfSlotLoss":
        """클래스 수는 손실 config가 아니라 데이터 config가 갖는다 — 거기서 끌어온다."""
        return super().from_cfg(module_cfg, cfg, num_classes=cfg.data.num_classes, **kwargs)

    def __init__(
        self,
        *,
        w_class: float,
        w_coord: float,
        w_end: float,
        end_pos_weight: float,
        class_bg_weight: float,
        class_freq_power: float,
        num_classes: int,
        w_fg: float,
        fg_pos_weight: float,
    ):
        super().__init__()
        self.w_class = w_class
        self.w_coord = w_coord
        self.w_end = w_end
        self.end_pos_weight = end_pos_weight
        self.class_bg_weight = class_bg_weight
        self.w_fg = w_fg
        self.fg_pos_weight = fg_pos_weight
        self.register_buffer(
            "class_weight", self._build_class_weight(num_classes, class_freq_power, class_bg_weight)
        )

    @staticmethod
    def _build_class_weight(num_classes: int, freq_power: float, bg_weight: float) -> torch.Tensor:
        """전경은 인스턴스 빈도의 `-power` 승, 배경은 `bg_weight`.

        전경 가중은 **평균 1로 정규화**한다 — 그래야 `power`를 올려도 클래스 손실의 스케일이
        그대로라 손실 균형(SKILL 8절)이 흔들리지 않고 `power` 하나만 비교된다.
        빈도는 셀 수가 아니라 인스턴스 수(`CLASS_INSTANCE_COUNT`)라 선 길이만큼 근사가 섞인다.
        """
        weight = torch.ones(num_classes)
        if freq_power > 0.0:
            counts = torch.tensor(CLASS_INSTANCE_COUNT[1:num_classes], dtype=torch.float32)
            foreground = (counts.median() / counts) ** freq_power
            weight[1:] = foreground / foreground.mean()
        weight[0] = bg_weight
        return weight

    def forward(self, output, targets: dict) -> dict[str, torch.Tensor]:
        positive = (targets["class_map"] > 0) & output.node_mask
        class_loss = self._class_loss(output, targets)
        coord_loss = self._coord_loss(output, targets, positive)
        end_loss = self._end_loss(output, targets, positive)
        fg_loss = self._fg_loss(output, targets)
        total = (
            self.w_class * class_loss
            + self.w_coord * coord_loss
            + self.w_end * end_loss
            + self.w_fg * fg_loss
        )
        return {
            "class": class_loss,
            "coord": coord_loss,
            "end": end_loss,
            "fg": fg_loss,
            "total": total,
        }

    def _fg_loss(self, output, targets: dict) -> torch.Tensor:
        """전경/배경 이진 감독 (E12).

        디코더가 정점을 만들 때 실제로 묻는 것은 **"전경이냐"** 하나뿐인데(10.2절 배경 필터),
        12지 CE는 그 판정과 종류 분류를 겸한다. 손실의 대부분은 흔한 클래스끼리의 혼동이
        먹고, 그 오류는 정점을 살리므로 디코더에 덜 치명적이다. 즉 **치명적인 오류와 덜
        치명적인 오류가 한 항에서 경쟁**한다. 이 항이 앞의 판정만 따로 감독한다.

        `w_fg = 0`이면 계산 자체를 건너뛴다 — 기존 실행과 값이 그대로여야 비교가 선다.
        """
        selected = output.node_mask
        if self.w_fg == 0.0 or not bool(selected.any()):
            return output.fg_logit.sum() * 0.0
        predicted = output.fg_logit[selected].float()
        target = (targets["class_map"][selected] > 0).float()
        weight = predicted.new_tensor(self.fg_pos_weight)
        return F.binary_cross_entropy_with_logits(predicted, target, pos_weight=weight)

    def _class_loss(self, output, targets: dict) -> torch.Tensor:
        """선택된 전 셀 — class_map이 배경 0이라 거짓 양성 셀의 라벨이 그대로 0이 된다.

        선택 셀의 70%가 배경이라 CE가 배경에 지배당한다(REF-F 에폭 1에서 class_acc 0.003 /
        bg_recall 0.997 관측). `class_bg_weight < 1`로 그 지배를 완화하고, 전경끼리의
        불균형은 `class_freq_power`가 맡는다 (가설 백로그 B6, E09).
        가중 CE의 감소는 `sum(w_i * l_i) / sum(w_i)` — 전부 1이면 단순 평균과 같다.
        """
        selected = output.node_mask
        if not bool(selected.any()):
            return output.class_logit.sum() * 0.0
        label = targets["class_map"][selected]
        logit = output.class_logit[selected].float()
        return F.cross_entropy(logit, label, weight=self.class_weight.to(logit.dtype))

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

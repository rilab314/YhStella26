"""GT를 모델 출력 계약 형태로 주입한다 (design 7.1절 계약, M12 판정).

GT와 모델 출력이 같은 격자·같은 형태라(설계 방침 1) 키만 바꿔 채우면 **"완벽한 예측"** 이 된다.
이것으로 재는 점수가 파이프라인의 **천장**이다 — 디코더 검증(M12), 손실 0 수렴(M11),
그리고 개선 루프의 첫 실험 E00(상한 재측정)이 전부 이 함수를 쓴다.
"""

import torch

from stella.model.stella import ModelOutput

HIGH_LOGIT = 10.0


def gt_model_output(targets: dict, num_classes: int, num_slots: int) -> ModelOutput:
    """targets: 배치 차원이 있는 텐서 dict (collate_fn 출력과 같은 형태)."""
    class_map = targets["class_map"]
    positive = class_map > 0
    shape = class_map.shape
    output = ModelOutput(
        heatmap_logit=torch.where(positive, HIGH_LOGIT, -HIGH_LOGIT),
        node_mask=positive.clone(),
        class_logit=torch.zeros((*shape, num_classes)),
        self_coord=torch.zeros((*shape, 2)),
        end_logit=torch.where(targets["end_map"] > 0, HIGH_LOGIT, -HIGH_LOGIT),
        exist_logit=torch.full((*shape, num_slots), -HIGH_LOGIT),
        conn_dir=torch.zeros((*shape, num_slots, 2)),
    )
    output.self_coord[positive] = targets["coord_map"][positive].float()
    one_hot = torch.nn.functional.one_hot(class_map[positive], num_classes).float()
    output.class_logit[positive] = one_hot * 2 * HIGH_LOGIT
    _fill_connections(output, targets, positive, num_slots)
    return output


def _fill_connections(output: ModelOutput, targets: dict, positive, num_slots: int) -> None:
    gt_dirs = targets["conn_dirs"][positive].float()  # (P, D, 2)
    slots = min(num_slots, gt_dirs.shape[1])
    conn_dir = torch.zeros((gt_dirs.shape[0], num_slots, 2))
    conn_dir[:, :slots] = gt_dirs[:, :slots]
    exist = torch.full((gt_dirs.shape[0], num_slots), -HIGH_LOGIT)
    exist[:, :slots] = torch.where(gt_dirs[:, :slots].norm(dim=-1) > 0.5, HIGH_LOGIT, -HIGH_LOGIT)
    output.conn_dir[positive] = conn_dir
    output.exist_logit[positive] = exist

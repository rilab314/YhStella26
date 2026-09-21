"""GT를 모델 출력 계약 형태로 주입한다 (design 7.1절 계약, M12 판정).

GT와 모델 출력이 같은 격자·같은 형태라(설계 방침 1) 키만 바꿔 채우면 **"완벽한 예측"** 이 된다.
이것으로 재는 점수가 파이프라인의 **천장**이다 — 디코더 검증(M12), 손실 0 수렴(M11),
그리고 개선 루프의 첫 실험 E00(상한 재측정)이 전부 이 함수를 쓴다.
"""

from dataclasses import fields

import torch

from stella.model.stella import ModelOutput

HIGH_LOGIT = 10.0
# 부분 주입에서 바꿔 끼울 수 있는 묶음 (Table IV — 천장 분해).
INJECTABLE_PARTS = ("node", "class", "coord", "conn", "end")


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
        fg_logit=torch.where(positive, HIGH_LOGIT, -HIGH_LOGIT),
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


def inject_parts(
    output: ModelOutput, targets: dict, parts: tuple, num_classes: int, num_slots: int
) -> ModelOutput:
    """모델 예측에 GT 를 **묶음 단위로만** 얹는다 — 천장을 헤드별로 쪼개려는 것이다 (4-3절).

    `output` 은 **정답 칸에서 계산된 예측**이어야 한다(`model(image, gt_positive=...)`).
    그래야 정답 칸마다 예측값이 실제로 들어 있어 묶음을 바꿔 끼울 수 있다.

    parts 에 넣을 수 있는 이름은 `INJECTABLE_PARTS` 다:
      node  — 어느 칸이 정점인지 (히트맵·노드 마스크)
      class — 그 칸의 차선 종류
      coord — 셀 안 정확한 위치
      conn  — 연결 방향과 존재 여부
      end   — 사슬의 끝인지
    """
    unknown = set(parts) - set(INJECTABLE_PARTS)
    if unknown:
        raise ValueError(f"모르는 묶음이다: {sorted(unknown)} (가능: {INJECTABLE_PARTS})")
    truth = gt_model_output(targets, num_classes, num_slots)
    positive = targets["class_map"] > 0
    mixed = ModelOutput(**{f.name: getattr(output, f.name).clone() for f in fields(output)})
    _apply_parts(mixed, truth, positive, parts)
    return mixed


def _apply_parts(mixed: ModelOutput, truth: ModelOutput, positive, parts: tuple) -> None:
    """묶음마다 해당 필드만 GT 값으로 바꾼다. 정점 묶음만 배경 칸까지 손댄다."""
    if "node" in parts:
        mixed.heatmap_logit = truth.heatmap_logit.clone()
        mixed.node_mask = truth.node_mask.clone()
        mixed.fg_logit = truth.fg_logit.clone()
    for name, fields_of in _PART_FIELDS.items():
        if name in parts:
            for field in fields_of:
                getattr(mixed, field)[positive] = getattr(truth, field)[positive]


_PART_FIELDS = {
    "class": ("class_logit",),
    "coord": ("self_coord",),
    "conn": ("conn_dir", "exist_logit"),
    "end": ("end_logit",),
}

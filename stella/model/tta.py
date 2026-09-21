"""추론 시각 증강(TTA) — 격자 대칭 8가지로 추론해 예측을 평균한다 (2-5축, plan_0908).

학습은 하지 않고 **추론만 8배** 든다. 사진을 90도씩 돌리고 좌우로 뒤집어 넣은 뒤,
나온 예측을 **원본 방향으로 되돌려** 합친다.

되돌리기는 출력마다 다르다 — 지도는 칸 위치만 옮기면 되지만 `self_coord`(셀 안 위치)와
`conn_dir`(연결 방향)은 **벡터 자체도 돌려야** 한다. 둘은 원점이 다르다: 셀 안 위치는
셀 좌상단이 원점이라 0.5 를 중심으로 돌리고, 연결 방향은 노드 점이 원점이라 보정이 없다.

**★ 연결 슬롯은 평균하지 않는다.** 슬롯 R개는 순서가 무의미해서(매칭이 그때그때 배정한다)
뷰끼리 같은 번호를 평균하면 앞뒤가 섞인다. 슬롯을 가진 출력은 **그 칸을 고른 뷰 중 하나**
것을 통째로 쓴다 — 원본 뷰를 먼저 쓰고, 원본이 그 칸을 안 골랐으면 다음 뷰 것을 쓴다.

좌표 규약: 셀 인덱스만 `(행, 열)`, 그 외 2차원 벡터는 전부 `(x, y)`다 — 지도의 축 1이 x다.
"""

import itertools

import torch

from stella.model.stella import ModelOutput

# (회전 횟수, 좌우 뒤집기) 8가지 = 격자 대칭 전부. 첫 항목이 원본이다(슬롯 우선순위가 된다).
GRID_SYMMETRIES = tuple((turns, flip) for flip in (False, True) for turns in range(4))
# 칸마다 평균하는 출력 — "찾기" 쪽만. 슬롯을 가진 둘은 여기 없다.
AVERAGED_KEYS = ("class_logit", "self_coord", "end_logit", "fg_logit")
# 슬롯 순서가 무의미한 출력 — 합치는 법을 `slot_mode` 가 고른다.
SLOT_KEYS = ("exist_logit", "conn_dir")
SLOT_SINGLE = "single"  # 방안 A — 그 칸을 고른 첫 뷰 것을 통째로 쓴다
SLOT_MATCHED = "matched"  # 방안 B — 방향이 가장 잘 맞는 짝을 찾아 맞춘 뒤 평균한다
UNIT_EPS = 1e-6


class GridSymmetryTTA:
    """격자 대칭 8가지로 추론하고 원본 방향으로 되돌려 합친다."""

    def __init__(self, *, symmetries: tuple, min_views: int, slot_mode: str):
        """min_views: 이 수 이상의 뷰가 고른 칸만 노드로 남긴다 (1이면 합집합).

        slot_mode: 슬롯을 가진 출력을 합치는 법 — `SLOT_SINGLE` 또는 `SLOT_MATCHED`.
        """
        if slot_mode not in (SLOT_SINGLE, SLOT_MATCHED):
            raise ValueError(
                f"slot_mode 는 {SLOT_SINGLE} 또는 {SLOT_MATCHED} 여야 한다: {slot_mode}"
            )
        self.symmetries = symmetries
        self.min_views = min_views
        self.slot_mode = slot_mode

    def predict(self, model, image: torch.Tensor) -> ModelOutput:
        """image: (B, 3, H, W). 반환은 원본 방향의 ModelOutput 하나다."""
        views = [self._predict_view(model, image, turns, flip) for turns, flip in self.symmetries]
        return self._merge(views)

    def _predict_view(self, model, image: torch.Tensor, turns: int, flip: bool) -> ModelOutput:
        rotated = transform_image(image, turns, flip)
        return invert_output(model(rotated), turns, flip)

    def _merge(self, views: list[ModelOutput]) -> ModelOutput:
        masks = torch.stack([view.node_mask for view in views])  # (V, B, L, L)
        coverage = masks.sum(dim=0)
        merged = ModelOutput(
            heatmap_logit=torch.stack([v.heatmap_logit for v in views]).mean(dim=0),
            node_mask=coverage >= self.min_views,
            **self._average(views, masks, coverage),
            **self._merge_slots(views, masks, coverage),
        )
        _clear_non_nodes(merged)
        return merged

    def _average(self, views: list[ModelOutput], masks: torch.Tensor, coverage) -> dict:
        """그 칸을 고른 뷰들에 대해서만 평균한다 — 안 고른 뷰의 값은 0이라 섞으면 안 된다."""
        divisor = coverage.clamp(min=1)
        averaged = {}
        for key in AVERAGED_KEYS:
            stacked = torch.stack([getattr(view, key) for view in views])
            weight = _expand_mask(masks, stacked.dim())
            averaged[key] = (stacked * weight).sum(dim=0) / _expand_mask(divisor, stacked.dim() - 1)
        return averaged

    def _merge_slots(self, views: list[ModelOutput], masks: torch.Tensor, coverage) -> dict:
        if self.slot_mode == SLOT_SINGLE:
            return self._pick_slots(views, masks)
        return self._match_slots(views, masks, coverage)

    def _match_slots(self, views: list[ModelOutput], masks: torch.Tensor, coverage) -> dict:
        """뷰마다 슬롯 번호가 다르므로 **방향이 가장 잘 맞는 짝**으로 맞춘 뒤 평균한다."""
        dirs = torch.stack([view.conn_dir for view in views])
        exists = torch.stack([view.exist_logit for view in views])
        reference = self._reference_slots(dirs, masks)
        dirs, exists = _align_slots(dirs, exists, reference)
        mean_dir = _masked_mean(dirs, masks, coverage)
        mean_exist = _masked_mean(exists, masks, coverage)
        return {"conn_dir": _unit(mean_dir), "exist_logit": mean_exist}

    def _reference_slots(self, dirs: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """짝을 맞출 기준 — 그 칸을 고른 첫 뷰의 방향이다."""
        first = _first_selecting_view(masks)
        return dirs.gather(0, _expand_index(first, dirs)).squeeze(0)

    def _pick_slots(self, views: list[ModelOutput], masks: torch.Tensor) -> dict:
        """칸마다 **한 뷰**의 값을 통째로 쓴다 — 앞 뷰가 우선이다(첫 뷰가 원본)."""
        first = _first_selecting_view(masks)  # (B, L, L) — 뷰 번호, 아무도 없으면 0
        picked = {}
        for key in SLOT_KEYS:
            stacked = torch.stack([getattr(view, key) for view in views])
            index = _expand_index(first, stacked)
            picked[key] = stacked.gather(0, index).squeeze(0)
        return picked


def transform_image(image: torch.Tensor, turns: int, flip: bool) -> torch.Tensor:
    """사진을 반시계로 turns 번 돌리고(축 = 세로·가로) 필요하면 좌우로 뒤집는다."""
    moved = torch.rot90(image, turns, dims=(-2, -1)) if turns else image
    return moved.flip(-1) if flip else moved


def invert_output(output: ModelOutput, turns: int, flip: bool) -> ModelOutput:
    """돌린 사진에서 나온 예측을 원본 방향으로 되돌린다."""
    return ModelOutput(
        heatmap_logit=_invert_map(output.heatmap_logit, turns, flip),
        node_mask=_invert_map(output.node_mask, turns, flip),
        class_logit=_invert_map(output.class_logit, turns, flip),
        end_logit=_invert_map(output.end_logit, turns, flip),
        fg_logit=_invert_map(output.fg_logit, turns, flip),
        exist_logit=_invert_map(output.exist_logit, turns, flip),
        self_coord=_invert_cell_point(_invert_map(output.self_coord, turns, flip), turns, flip),
        conn_dir=_invert_vector(_invert_map(output.conn_dir, turns, flip), turns, flip),
    )


def _invert_map(tensor: torch.Tensor, turns: int, flip: bool) -> torch.Tensor:
    """칸 위치만 되돌린다. 지도 축은 (..., 행, 열, 남는 축들) 이라 행·열을 짚어 준다."""
    rows, cols = _grid_dims(tensor)
    moved = tensor.flip(cols) if flip else tensor
    return torch.rot90(moved, -turns, dims=(rows, cols)) if turns else moved


def _grid_dims(tensor: torch.Tensor) -> tuple[int, int]:
    """(B, L, L, ...) 에서 행·열 축 번호. 배치가 있으므로 앞에서 센다."""
    return 1, 2


def _invert_vector(vectors: torch.Tensor, turns: int, flip: bool) -> torch.Tensor:
    """방향 벡터 (x, y) 를 되돌린다 — 원점이 노드 점이라 보정이 없다."""
    x, y = vectors[..., 0], vectors[..., 1]
    if flip:
        x = -x
    for _ in range(turns):  # 반시계 회전의 역 = 시계 회전: (x, y) -> (-y, x)
        x, y = -y, x
    return torch.stack([x, y], dim=-1)


def _invert_cell_point(points: torch.Tensor, turns: int, flip: bool) -> torch.Tensor:
    """셀 안 위치 [0, 1) 을 되돌린다 — 원점이 셀 좌상단이라 0.5 를 중심으로 돌린다."""
    centered = points - 0.5
    return _invert_vector(centered, turns, flip) + 0.5


def _first_selecting_view(masks: torch.Tensor) -> torch.Tensor:
    """칸마다 그 칸을 고른 **가장 앞선 뷰**의 번호. 아무도 안 골랐으면 0."""
    views = masks.shape[0]
    order = torch.arange(views, device=masks.device).view(-1, *([1] * (masks.dim() - 1)))
    ranked = torch.where(masks, order, torch.full_like(order, views))
    first = ranked.min(dim=0).values
    return torch.where(first < views, first, torch.zeros_like(first))


def _align_slots(dirs: torch.Tensor, exists: torch.Tensor, reference: torch.Tensor) -> tuple:
    """뷰마다 슬롯 순서를 바꿔 보고 기준과 방향이 가장 잘 맞는 순서를 고른다."""
    orders = list(itertools.permutations(range(dirs.shape[-2])))
    permuted = torch.stack([dirs[..., list(order), :] for order in orders])
    score = (permuted * reference).sum(dim=(-1, -2))  # (P, V, B, L, L)
    best = score.argmax(dim=0)
    aligned_dir = permuted.gather(0, _expand_index(best, permuted)).squeeze(0)
    stacked_exist = torch.stack([exists[..., list(order)] for order in orders])
    aligned_exist = stacked_exist.gather(0, _expand_index(best, stacked_exist)).squeeze(0)
    return aligned_dir, aligned_exist


def _masked_mean(values: torch.Tensor, masks: torch.Tensor, coverage: torch.Tensor) -> torch.Tensor:
    """그 칸을 고른 뷰들에 대해서만 평균한다."""
    weight = _expand_mask(masks, values.dim())
    divisor = _expand_mask(coverage.clamp(min=1), values.dim() - 1)
    return (values * weight).sum(dim=0) / divisor


def _unit(vectors: torch.Tensor) -> torch.Tensor:
    """평균낸 방향은 길이가 1이 아니다 — 디코더 계약대로 단위벡터로 되돌린다."""
    norm = vectors.norm(dim=-1, keepdim=True)
    return torch.where(norm > UNIT_EPS, vectors / norm.clamp(min=UNIT_EPS), vectors)


def _expand_mask(mask: torch.Tensor, target_dim: int) -> torch.Tensor:
    """(V, B, L, L) 마스크를 (V, B, L, L, 1, ...) 로 늘려 값 텐서와 곱할 수 있게 한다."""
    view = mask.reshape(*mask.shape, *([1] * (target_dim - mask.dim())))
    return view.to(torch.float32)


def _expand_index(first: torch.Tensor, stacked: torch.Tensor) -> torch.Tensor:
    """gather 용 지수 — 뷰 축 하나만 남기고 값 텐서와 같은 모양으로 늘린다."""
    shape = (1, *stacked.shape[1:])
    index = first.reshape(1, *first.shape, *([1] * (stacked.dim() - 1 - first.dim())))
    return index.expand(shape)


def _clear_non_nodes(output: ModelOutput) -> None:
    """노드가 아닌 칸의 값은 의미가 없다 — 계약대로 0 으로 만든다."""
    keep = output.node_mask
    for key in (*AVERAGED_KEYS, *SLOT_KEYS):
        value = getattr(output, key)
        setattr(output, key, value * _expand_mask(keep, value.dim()))

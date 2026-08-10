"""예측 캐시 입출력 — 디코더 전용 트랙의 토대 (improve-loop 스킬 · D 트랙).

모델 예측을 한 번 디스크에 떨궈 두면 그 뒤로는 **GPU 없이** 디코더 파라미터를 스윕할 수 있다.
학습이 GPU를 다 쓰는 동안에도 디코더 실험이 계속 돌아가는 것이 이 모듈의 존재 이유다.

저장은 **노드 셀만 fp16** 으로 한다 — dense는 샘플당 3.3 MB지만 희소는 0.4 MB다.
`node_mask`가 거짓인 셀은 어차피 디코더가 보지 않으므로 정보 손실이 없다.
"""

from pathlib import Path

import numpy as np
import torch

from stella.model.stella import ModelOutput

SPARSE_KEYS = ("heat", "class_logit", "coord", "end", "exist", "dir")
# 나중에 생긴 키. **옛 캐시에는 없다** — 없으면 0으로 채운다(그 모델엔 전경 헤드가 없었다).
OPTIONAL_KEYS = ("fg",)
BACKGROUND_LOGIT = -30.0  # 비노드 셀의 히트맵 로짓 — sigmoid ~ 0


def save_prediction(path: Path, output: ModelOutput, instances: list[dict]) -> None:
    """output은 배치 차원이 없는 한 샘플(`output[i]`)이어야 한다."""
    arrays = {key: _to_numpy(getattr(output, key)) for key in vars(output)}
    cells = np.argwhere(arrays["node_mask"])
    rows, cols = cells[:, 0], cells[:, 1]
    np.savez_compressed(
        path,
        cells=cells.astype(np.int16),
        heat=arrays["heatmap_logit"][rows, cols].astype(np.float16),
        class_logit=arrays["class_logit"][rows, cols].astype(np.float16),
        coord=arrays["self_coord"][rows, cols].astype(np.float16),
        end=arrays["end_logit"][rows, cols].astype(np.float16),
        exist=arrays["exist_logit"][rows, cols].astype(np.float16),
        dir=arrays["conn_dir"][rows, cols].astype(np.float16),
        fg=arrays["fg_logit"][rows, cols].astype(np.float16),
        **_pack_instances(instances),
    )


def load_prediction(path: Path, shape: dict) -> tuple[ModelOutput, list[dict]]:
    """shape: {"grid_size", "num_classes", "num_slots"} — dense 버퍼 크기를 정한다."""
    with np.load(path) as data:
        sparse = {key: data[key] for key in ("cells", *SPARSE_KEYS)}
        sparse |= {key: data[key] for key in OPTIONAL_KEYS if key in data}
        instances = _unpack_instances(data)
    return _to_dense(sparse, shape), instances


def _to_dense(sparse: dict, shape: dict) -> ModelOutput:
    side, classes, slots = shape["grid_size"], shape["num_classes"], shape["num_slots"]
    cells = sparse["cells"].astype(np.int64)
    rows, cols = cells[:, 0], cells[:, 1]
    output = ModelOutput(
        heatmap_logit=torch.full((side, side), BACKGROUND_LOGIT),
        node_mask=torch.zeros((side, side), dtype=torch.bool),
        class_logit=torch.zeros((side, side, classes)),
        self_coord=torch.zeros((side, side, 2)),
        end_logit=torch.zeros((side, side)),
        fg_logit=torch.zeros((side, side)),
        exist_logit=torch.zeros((side, side, slots)),
        conn_dir=torch.zeros((side, side, slots, 2)),
    )
    output.node_mask[rows, cols] = True
    for key, field in zip((*SPARSE_KEYS, *OPTIONAL_KEYS), _DENSE_FIELDS):
        if key not in sparse:  # 옛 캐시 — 그 필드는 0으로 남는다
            continue
        getattr(output, field)[rows, cols] = torch.from_numpy(sparse[key].astype(np.float32))
    return output


_DENSE_FIELDS = (
    "heatmap_logit",
    "class_logit",
    "self_coord",
    "end_logit",
    "exist_logit",
    "conn_dir",
    "fg_logit",  # OPTIONAL_KEYS 와 짝이다 — 순서를 맞춰 둔다
)


def _to_numpy(value: torch.Tensor) -> np.ndarray:
    tensor = value.detach().cpu()
    return tensor.numpy() if tensor.dtype == torch.bool else tensor.float().numpy()


def _pack_instances(instances: list[dict]) -> dict[str, np.ndarray]:
    lengths = [len(item["points"]) for item in instances]
    points = (
        np.concatenate([np.asarray(item["points"]) for item in instances]).astype(np.float32)
        if instances
        else np.zeros((0, 2), np.float32)
    )
    return {
        "inst_points": points,
        "inst_offset": np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64),
        "inst_class": np.array([item["class"] for item in instances], dtype=np.int64),
    }


def _unpack_instances(data) -> list[dict]:
    points, offset, labels = data["inst_points"], data["inst_offset"], data["inst_class"]
    return [
        {"class": int(labels[k]), "points": points[offset[k] : offset[k + 1]]}
        for k in range(len(labels))
    ]

"""조각 병합·예측 캐시 (improve_plan §7 A2, 2.1절).

병합은 **잘못 이으면 조각남보다 나쁘다**(환각 FP가 된다). 그래서 "이어야 할 것을 잇는다"와
"이으면 안 되는 것을 안 잇는다"를 같은 무게로 시험한다.
"""

import numpy as np
import torch
from helpers import gt_model_output

from configs.exp_synthetic import get_config
from stella.builder import build_instance
from stella.data.encode import ChainEncoder
from stella.decode.cache import load_prediction, save_prediction
from stella.decode.postprocess import ChainMerger

IMAGE = 256
STRIDE = 4


def piece(label: int, points: list) -> dict:
    return {"class": label, "points": np.array(points, dtype=np.float32), "score": 1.0}


def test_collinear_fragments_merge_into_one():
    merger = ChainMerger(gap=8.0, align_cos=0.8)
    merged, removed = merger(
        [
            piece(3, [[10.0, 50.0], [40.0, 50.0]]),
            piece(3, [[45.0, 50.0], [80.0, 50.0]]),
        ]
    )
    assert removed == 1
    assert len(merged) == 1
    assert merged[0]["points"][:, 0].min() == 10.0
    assert merged[0]["points"][:, 0].max() == 80.0


def test_different_classes_do_not_merge():
    merger = ChainMerger(gap=8.0, align_cos=0.8)
    merged, removed = merger(
        [
            piece(3, [[10.0, 50.0], [40.0, 50.0]]),
            piece(5, [[45.0, 50.0], [80.0, 50.0]]),
        ]
    )
    assert removed == 0 and len(merged) == 2


def test_perpendicular_fragments_do_not_merge():
    """끝점은 가깝지만 서로를 향하지 않는다 — T접합에서 본선과 곁가지를 붙이면 안 된다."""
    merger = ChainMerger(gap=8.0, align_cos=0.8)
    merged, _ = merger(
        [
            piece(3, [[10.0, 50.0], [40.0, 50.0]]),
            piece(3, [[44.0, 50.0], [44.0, 90.0]]),
        ]
    )
    assert len(merged) == 2


def test_far_fragments_do_not_merge():
    merger = ChainMerger(gap=4.0, align_cos=0.8)
    merged, _ = merger(
        [
            piece(3, [[10.0, 50.0], [40.0, 50.0]]),
            piece(3, [[60.0, 50.0], [80.0, 50.0]]),
        ]
    )
    assert len(merged) == 2


def test_three_fragments_merge_in_order():
    merger = ChainMerger(gap=8.0, align_cos=0.8)
    merged, removed = merger(
        [
            piece(3, [[45.0, 50.0], [80.0, 50.0]]),
            piece(3, [[10.0, 50.0], [40.0, 50.0]]),
            piece(3, [[85.0, 50.0], [120.0, 50.0]]),
        ]
    )
    assert removed == 2 and len(merged) == 1
    xs = merged[0]["points"][:, 0]
    assert np.all(np.diff(xs) > 0)  # 이어 붙인 뒤에도 한 방향으로 진행한다


def test_disabled_merger_is_identity():
    merger = ChainMerger(gap=0.0, align_cos=0.8)
    items = [piece(3, [[10.0, 50.0], [40.0, 50.0]]), piece(3, [[45.0, 50.0], [80.0, 50.0]])]
    merged, removed = merger(items)
    assert removed == 0 and merged is items


def test_prediction_cache_roundtrip(tmp_path):
    """희소 캐시는 노드 셀의 값을 fp16 정밀도로 그대로 복원해야 한다."""
    cfg = get_config()
    cfg.data.image_size = IMAGE
    encoder = ChainEncoder(
        image_size=IMAGE, grid_stride=STRIDE, num_classes=12, max_degree=2, supersample=1
    )
    target = encoder.encode([piece(3, [[30.0, 100.0], [220.0, 100.0]])])
    targets = {k: torch.from_numpy(v).unsqueeze(0) for k, v in target.items()}
    output = gt_model_output(targets, cfg.data.num_classes, cfg.model.num_conn_slots)
    instances = [{"class": 3, "points": np.array([[30.0, 100.0], [220.0, 100.0]], np.float32)}]
    save_prediction(tmp_path / "a.npz", output[0], instances)
    shape = {"grid_size": IMAGE // STRIDE, "num_classes": 12, "num_slots": 2}
    loaded, back = load_prediction(tmp_path / "a.npz", shape)
    assert bool((loaded.node_mask == output.node_mask[0]).all())
    mask = output.node_mask[0]
    assert torch.allclose(loaded.self_coord[mask], output.self_coord[0][mask], atol=1e-3)
    assert torch.allclose(loaded.conn_dir[mask], output.conn_dir[0][mask], atol=1e-3)
    assert back[0]["class"] == 3


def test_cached_prediction_decodes_same_as_direct(tmp_path):
    """캐시를 거쳐도 디코딩 결과가 같아야 한다 — D 트랙 전체가 여기에 의존한다."""
    cfg = get_config()
    cfg.data.image_size = IMAGE
    encoder = ChainEncoder(
        image_size=IMAGE, grid_stride=STRIDE, num_classes=12, max_degree=2, supersample=1
    )
    target = encoder.encode([piece(3, [[30.0, 100.0], [220.0, 100.0]])])
    targets = {k: torch.from_numpy(v).unsqueeze(0) for k, v in target.items()}
    output = gt_model_output(targets, cfg.data.num_classes, cfg.model.num_conn_slots)
    save_prediction(tmp_path / "a.npz", output[0], [])
    shape = {"grid_size": IMAGE // STRIDE, "num_classes": 12, "num_slots": 2}
    loaded, _ = load_prediction(tmp_path / "a.npz", shape)
    decoder = build_instance(cfg.decode, cfg)
    direct = decoder(output[0])
    cached = decoder(loaded)
    assert len(direct) == len(cached) == 1
    assert np.allclose(direct[0]["points"], cached[0]["points"], atol=0.05)

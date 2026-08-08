"""셀 단위 진단 지표 (improve_plan 3절 층 2).

GT를 그대로 주입하면 모든 진단이 만점이어야 한다 — 그렇지 않으면 지표가 아니라
지표 계산이 틀린 것이다. 만점 확인이 이 파일의 존재 이유다.
"""

import torch
from helpers import gt_model_output

from configs.exp_synthetic import get_config
from stella.builder import build_instance
from stella.data.encode import ChainEncoder
from stella.eval.cellstat import CellDiagnostics

IMAGE = 256
STRIDE = 4


def make_cfg():
    cfg = get_config()
    cfg.data.image_size = IMAGE
    return cfg


def encode_batch(instances) -> dict:
    encoder = ChainEncoder(
        image_size=IMAGE, grid_stride=STRIDE, num_classes=12, max_degree=2, supersample=1
    )
    target = encoder.encode(instances)
    return {key: torch.from_numpy(value).unsqueeze(0) for key, value in target.items()}


def scores_for(instances) -> dict:
    cfg = make_cfg()
    targets = encode_batch(instances)
    output = gt_model_output(targets, cfg.data.num_classes, cfg.model.num_conn_slots)
    diagnostics = build_instance(cfg.cell_diag, cfg)
    diagnostics.update(output, targets)
    return {key: float(value) for key, value in diagnostics.compute().items()}


def test_gt_injection_scores_perfect():
    import numpy as np

    lines = [
        {"class": 3, "points": np.array([[30.0, 100.0], [220.0, 100.0]], dtype=np.float32)},
        {"class": 5, "points": np.array([[60.0, 20.0], [60.0, 200.0]], dtype=np.float32)},
    ]
    scores = scores_for(lines)
    assert scores["heat_recall"] == 1.0
    assert scores["heat_precision"] == 1.0
    assert scores["class_acc"] == 1.0
    assert scores["end_recall"] == 1.0
    assert scores["end_precision"] == 1.0
    assert scores["coord_err_px"] < 1e-3
    assert scores["dir_err_deg"] < 1.0
    assert scores["link_ok"] == 1.0
    assert scores["exist_pos"] > 0.99


def test_diagnostics_build_from_config():
    cfg = make_cfg()
    diagnostics = build_instance(cfg.cell_diag, cfg)
    assert isinstance(diagnostics, CellDiagnostics)
    assert 44.0 < diagnostics.align_deg < 47.0  # acos(0.7) = 45.57도


def test_rotated_prediction_lowers_link_ok():
    """방향을 90도 돌리면 link_ok가 무너져야 한다 — 지표가 실제로 방향을 본다는 확인."""
    import numpy as np

    cfg = make_cfg()
    lines = [{"class": 3, "points": np.array([[30.0, 100.0], [220.0, 100.0]], dtype=np.float32)}]
    targets = encode_batch(lines)
    output = gt_model_output(targets, cfg.data.num_classes, cfg.model.num_conn_slots)
    turned = output.conn_dir.clone()
    output.conn_dir[..., 0], output.conn_dir[..., 1] = -turned[..., 1], turned[..., 0]
    diagnostics = build_instance(cfg.cell_diag, cfg)
    diagnostics.update(output, targets)
    scores = diagnostics.compute()
    assert float(scores["link_ok"]) < 0.05
    assert 85.0 < float(scores["dir_err_deg"]) < 95.0

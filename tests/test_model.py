"""Neck·모델·손실의 shape과 계약 검증 (impl_plan M2·M5)."""

import torch

from configs.exp_synthetic import get_config
from stella.builder import build_instance
from stella.data.synthetic import SyntheticDataset
from stella.data.types import collate_fn
from stella.model.neck import SFP, FPNLite

GRID = 64
IMAGE = 256


def make_targets(batch: int = 2) -> dict:
    dataset = SyntheticDataset(
        split="val",
        image_size=IMAGE,
        grid_stride=4,
        num_classes=12,
        max_degree=3,
        encode_supersample=1,
        augment=False,
        limit=batch,
    )
    return collate_fn([dataset[i] for i in range(batch)])


def test_sfp_outputs_grid_resolution():
    neck = SFP(in_channels=(384,), d_model=256, upsample_steps=2)
    out = neck([torch.randn(1, 384, GRID // 4, GRID // 4)])
    assert out.shape == (1, 256, GRID, GRID)


def test_fpnlite_outputs_grid_resolution():
    channels = (96, 192, 384, 768)
    neck = FPNLite(in_channels=channels, d_model=256)
    levels = [torch.randn(1, c, GRID // (2**k), GRID // (2**k)) for k, c in enumerate(channels)]
    assert neck(levels).shape == (1, 256, GRID, GRID)


def test_neck_rejects_wrong_level_count():
    try:
        FPNLite(in_channels=(768,), d_model=256)
    except ValueError as error:
        assert "4레벨" in str(error)
    else:
        raise AssertionError("레벨 수가 맞지 않으면 에러를 내야 한다")


def test_criterion_returns_all_logged_keys():
    cfg = get_config()
    cfg.data.image_size = IMAGE
    criterion = build_instance(cfg.loss, cfg)
    targets = make_targets()
    output = _fake_output(targets, cfg)
    losses = criterion(output, targets)
    expected = {
        "heatmap/focal",
        "heatmap/total",
        "self_slot/class",
        "self_slot/coord",
        "conn/exist",
        "conn/dir",
        "conn/t",
        "conn/switch_rate",
        "conn/total",
        "self_slot/total",
        "total",
    }
    assert expected <= set(losses)
    assert torch.isfinite(losses["total"])


def test_perfect_prediction_drives_losses_to_zero():
    """GT를 모델 출력 형식으로 주입하면 좌표·방향·종점 손실이 0에 수렴한다."""
    cfg = get_config()
    cfg.data.image_size = IMAGE
    criterion = build_instance(cfg.loss, cfg)
    targets = make_targets(1)
    output = _fake_output(targets, cfg, perfect=True)
    losses = criterion(output, targets)
    assert float(losses["self_slot/coord"]) < 1e-6
    assert float(losses["conn/dir"]) < 1e-4
    assert float(losses["self_slot/class"]) < 1e-2


def _fake_output(targets: dict, cfg, perfect: bool = False):
    from stella.loss.conn import derive_branches
    from stella.model.stella import ModelOutput

    shape = targets["class_map"].shape
    slots, classes = cfg.model.num_conn_slots, cfg.data.num_classes
    positive = targets["class_map"] > 0
    output = ModelOutput(
        heatmap_logit=torch.zeros(shape),
        node_mask=positive.clone(),
        class_logit=torch.zeros((*shape, classes)),
        self_coord=torch.zeros((*shape, 2)),
        exist_logit=torch.zeros((*shape, slots)),
        conn_dir=torch.zeros((*shape, slots, 2)),
        t_logit=torch.zeros((*shape, slots)),
    )
    if not perfect:
        output.conn_dir[..., 0] = 1.0
        return output
    cells = positive.nonzero(as_tuple=False)
    gt_dir, gt_t, valid = derive_branches(targets, cells)
    output.self_coord[positive] = targets["coord_map"][positive]
    output.class_logit[positive] = (
        torch.nn.functional.one_hot(targets["class_map"][positive], classes).float() * 20.0
    )
    output.conn_dir[positive] = gt_dir[:, :slots]
    output.exist_logit[positive] = torch.where(valid[:, :slots], 20.0, -20.0)
    output.t_logit[positive] = torch.where(gt_t[:, :slots] > 0, 20.0, -20.0)
    return output

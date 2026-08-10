"""Neck·모델·손실의 shape과 계약 검증 (design M2·M11)."""

import torch
from helpers import gt_model_output

from configs.exp_synthetic import get_config
from stella.builder import build_instance
from stella.data.synthetic import SyntheticDataset
from stella.data.types import CLASS_NAMES, collate_fn
from stella.model.neck import SFP, FPNLite

GRID = 64
IMAGE = 256


def make_targets(batch: int = 2) -> dict:
    dataset = SyntheticDataset(
        split="val",
        image_size=IMAGE,
        grid_stride=4,
        num_classes=12,
        max_degree=2,
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


def test_heads_survive_module_apply():
    """`.to()`·`.float()`은 `nn.Module._apply`를 재귀 호출한다 — 헤드가 그 이름을 가리면
    모델을 장치로 옮기는 순간 죽는다. 조립 테스트로는 안 잡혀서 여기서 직접 시험한다."""
    from stella.model.heads import ConnHead, SelfHead

    for head in (SelfHead(d_model=32, num_classes=12), ConnHead(d_model=32, num_slots=2)):
        head.to(torch.float64).float()  # 예약 메서드를 가리면 여기서 TypeError가 난다
    shared = ConnHead(d_model=32, num_slots=2, share_slots=True).float()
    split = ConnHead(d_model=32, num_slots=2, share_slots=False).float()
    tokens = torch.randn(5, 2, 32)
    for head in (shared, split):
        exist, direction = head(tokens)
        assert exist.shape == (5, 2)
        assert direction.shape == (5, 2, 2)
        assert torch.allclose(direction.norm(dim=-1), torch.ones(5, 2), atol=1e-4)


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
    output = _imperfect_output(targets, cfg)
    losses = criterion(output, targets)
    expected = {
        "heatmap/focal",
        "heatmap/total",
        "self_slot/class",
        "self_slot/coord",
        "self_slot/end",
        "self_slot/total",
        "conn/exist",
        "conn/dir",
        "conn/match_ambiguity",
        "conn/total",
        "total",
    }
    assert expected <= set(losses)
    assert torch.isfinite(losses["total"])


def test_class_weight_is_neutral_without_frequency_power():
    """가중을 끄면 예전의 단순 평균 CE와 같다 — E09 리팩터가 기본 동작을 바꾸지 않았다."""
    cfg = get_config()
    cfg.data.image_size = IMAGE
    loss_module = build_instance(cfg.loss.self_slot, cfg)
    assert torch.allclose(loss_module.class_weight, torch.ones(cfg.data.num_classes))
    targets = make_targets()
    output = _imperfect_output(targets, cfg)
    selected = output.node_mask
    plain = torch.nn.functional.cross_entropy(
        output.class_logit[selected].float(), targets["class_map"][selected]
    )
    assert torch.allclose(loss_module(output, targets)["class"], plain)


def test_class_freq_power_lifts_rare_classes():
    """빈도 가중은 희소 클래스를 올리되 전경 평균은 1로 둔다 — 손실 스케일이 안 흔들린다 (E09)."""
    cfg = get_config()
    cfg.loss.self_slot.class_freq_power = 0.5
    weight = build_instance(cfg.loss.self_slot, cfg).class_weight
    rare = CLASS_NAMES.index("bus_only_lane")
    common = CLASS_NAMES.index("no_parking_stopping_line")
    assert weight[rare] > 1.0 > weight[common]
    assert torch.allclose(weight[1:].mean(), torch.tensor(1.0))
    assert float(weight[0]) == cfg.loss.self_slot.class_bg_weight


def test_perfect_prediction_drives_losses_to_zero():
    """GT를 모델 출력 형식으로 주입하면 좌표·방향·끝 손실이 0에 수렴한다 (M11 판정)."""
    cfg = get_config()
    cfg.data.image_size = IMAGE
    criterion = build_instance(cfg.loss, cfg)
    targets = make_targets(1)
    output = gt_model_output(targets, cfg.data.num_classes, cfg.model.num_conn_slots)
    losses = criterion(output, targets)
    assert float(losses["self_slot/coord"]) < 1e-6
    assert float(losses["self_slot/class"]) < 1e-2
    assert float(losses["self_slot/end"]) < 1e-3
    assert float(losses["conn/dir"]) < 1e-4
    assert float(losses["conn/exist"]) < 1e-3
    assert float(losses["conn/match_ambiguity"]) < 0.05  # 반평행 분기 2개는 배정이 명확하다


def test_grad_checkpoint_leaves_gradients_unchanged():
    """윈도우 층 재계산은 메모리만 바꾼다 — 기울기가 달라지면 안 된다."""
    cfg = get_config()
    cfg.data.image_size = IMAGE
    cfg.model.n_max = 200
    targets = make_targets(1)
    model = build_instance(cfg.model, cfg).train()
    gradients = []
    for flag in (False, True):
        model.grad_checkpoint = flag
        model.zero_grad(set_to_none=True)
        output = model(targets["image"], gt_positive=targets["class_map"] > 0)
        (output.conn_dir.square().sum() + output.class_logit.square().sum()).backward()
        gradients.append(
            {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
        )
    assert gradients[0] and gradients[0].keys() == gradients[1].keys()
    for name, grad in gradients[0].items():
        # 텐서 스케일 대비로 본다 — CUDA에서는 완전히 같고, CPU는 누적 순서가 달라
        # fp32 잡음(~3e-6)이 남는다. 재계산이 깨지면 오차는 이보다 몇 자릿수 커진다.
        scale = grad.abs().max().clamp(min=1e-9)
        assert float((grad - gradients[1][name]).abs().max() / scale) < 1e-4, name


def _imperfect_output(targets: dict, cfg):
    """전부 0에 가까운 미숙한 예측 — 손실 키·유한성 확인용."""
    from stella.model.stella import ModelOutput

    shape = targets["class_map"].shape
    slots, classes = cfg.model.num_conn_slots, cfg.data.num_classes
    positive = targets["class_map"] > 0
    output = ModelOutput(
        heatmap_logit=torch.zeros(shape),
        node_mask=positive.clone(),
        class_logit=torch.zeros((*shape, classes)),
        self_coord=torch.zeros((*shape, 2)),
        end_logit=torch.zeros(shape),
        exist_logit=torch.zeros((*shape, slots)),
        conn_dir=torch.zeros((*shape, slots, 2)),
    )
    output.conn_dir[..., 0] = 1.0
    return output

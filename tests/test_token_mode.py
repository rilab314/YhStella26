"""노드당 토큰 구조 전환 검증 (Table VI · 논지 C2).

`single` 은 어텐션을 토큰 하나로 통과시키고 역할 표시를 헤드 입구에서 더한다.
**파라미터 수가 같아야** 비교가 공정하고, **두 연결 슬롯이 여전히 서로 다른 값**을 낼 수
있어야 표현이 성립한다. 기본값(`per_slot`)은 현행과 같아야 한다.
"""

import pytest
import torch

from configs.exp_synthetic import get_config
from stella.builder import build_instance


def build(token_mode: str):
    cfg = get_config()
    cfg.model.token_mode = token_mode
    torch.manual_seed(0)
    return cfg, build_instance(cfg.model, cfg).train()


def forward(cfg, model):
    torch.manual_seed(1)
    size, side = cfg.data.image_size, cfg.data.grid_size
    image = torch.rand(1, 3, size, size)
    gt = torch.zeros(1, side, side, dtype=torch.bool)
    gt[0, side // 4 : side // 2, side // 3] = True
    with torch.no_grad():
        return model(image, gt_positive=gt)


def test_default_is_per_slot():
    """기본값이 현행 구조여야 기존 설정·체크포인트가 그대로 돈다."""
    assert get_config().model.token_mode == "per_slot"


def test_parameter_count_is_identical():
    """두 모드의 파라미터 수가 같다 — 차이가 모델 크기 탓이 아니다."""
    _, per_slot = build("per_slot")
    _, single = build("single")
    count = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
    assert count(per_slot) == count(single)


def test_single_mode_keeps_output_contract():
    """single 에서도 출력 모양이 계약대로다."""
    cfg, model = build("single")
    out = forward(cfg, model)
    slots = cfg.model.num_conn_slots
    assert out.conn_dir.shape[-2:] == (slots, 2)
    assert out.exist_logit.shape[-1] == slots
    assert int(out.node_mask.sum()) > 1


def test_single_mode_slots_can_differ():
    """토큰이 하나여도 역할 표시가 두 연결 슬롯을 갈라 **서로 다른 방향**을 낼 수 있다."""
    cfg, model = build("single")
    out = forward(cfg, model)
    nodes = out.node_mask
    first, second = out.conn_dir[nodes][:, 0], out.conn_dir[nodes][:, 1]
    assert not torch.allclose(first, second)


def test_modes_differ_in_output():
    """두 모드는 실제로 다른 계산을 한다 — 스위치가 먹혔는지 확인한다."""
    cfg, per_slot = build("per_slot")
    _, single = build("single")
    a, b = forward(cfg, per_slot), forward(cfg, single)
    assert not torch.allclose(a.conn_dir, b.conn_dir)


def test_unknown_mode_is_rejected():
    """오타를 조용히 넘기면 엉뚱한 구조로 학습된다 — 크게 실패해야 한다."""
    with pytest.raises(ValueError):
        build("one_token")

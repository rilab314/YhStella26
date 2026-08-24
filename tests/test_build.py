"""config 해석과 전체 조립 스모크 (design 5.5절).

`test_config_resolves`는 파일 이동·클래스 rename으로 문자열이 깨지는 것을 잡는 안전망이다.
"""

import dataclasses
import importlib
import types
from pathlib import Path

import pytest

from configs.schema import BackboneConfig, ExperimentConfig
from stella.builder import build_instance, check_all, resolve

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def all_configs() -> list[str]:
    names = sorted(p.stem for p in CONFIG_DIR.glob("*.py") if p.stem not in ("__init__", "schema"))
    return [f"configs.{name}" for name in names]


@pytest.mark.parametrize("module_name", all_configs())
def test_config_resolves(module_name: str):
    """모든 config의 path/name이 실재하는지. GPU·가중치 불필요, 수 초 이내."""
    check_all(importlib.import_module(module_name).get_config())


@pytest.mark.parametrize("module_name", all_configs())
def test_config_is_serializable(module_name: str):
    cfg = importlib.import_module(module_name).get_config()
    assert isinstance(cfg, ExperimentConfig)
    assert dataclasses.asdict(cfg)["data"]["grid_stride"] > 0


def test_unknown_class_name_suggests_candidates():
    bad = BackboneConfig(name="Dinov")
    with pytest.raises(AttributeError) as error:
        resolve(bad)
    assert "Dinov3Backbone" in str(error.value)


def test_class_imported_elsewhere_is_rejected():
    """neck.py 가 Backbone 을 import 하지 않더라도, 이름이 남의 모듈 것이면 막아야 한다."""
    bad = BackboneConfig(path="stella.model.stella", name="Backbone")
    with pytest.raises(TypeError) as error:
        resolve(bad)
    assert "stella.model.backbone" in str(error.value)


def test_unknown_module_path_names_the_config():
    bad = BackboneConfig(path="stella.model.nowhere")
    with pytest.raises(ModuleNotFoundError) as error:
        resolve(bad)
    assert "BackboneConfig.path" in str(error.value)


def test_leaf_modules_build_without_weights():
    """백본 다운로드가 필요 없는 부품들은 CI에서 항상 조립된다."""
    cfg = importlib.import_module("configs.exp_synthetic").get_config()
    for section in (cfg.loss, cfg.decode, cfg.eval, cfg.cell_diag):
        assert build_instance(section, cfg) is not None


@pytest.mark.slow
def test_full_build_smoke():
    """model·criterion·dataset·module 전체 조립 (백본 가중치 다운로드 필요)."""
    from stella.data.types import GridDatasetBase

    cfg = importlib.import_module("configs.exp_synthetic").get_config()
    parts = {
        "model": build_instance(cfg.model, cfg),
        "criterion": build_instance(cfg.loss, cfg),
        "decoder": build_instance(cfg.decode, cfg),
        "metric": build_instance(cfg.eval, cfg),
        "cell_diag": build_instance(cfg.cell_diag, cfg),
    }
    dataset = build_instance(cfg.data, cfg, base=GridDatasetBase, split="val")
    module = build_instance(cfg.train, cfg, **parts)
    assert len(dataset) > 0
    assert module.model is parts["model"]


def test_tuple_override_keeps_element_type():
    """튜플 손잡이를 덮어써도 원소 타입이 유지된다 — 문자열이 되면 손잡이가 조용히 꺼진다.

    `decode.short_classes` 를 덮어썼더니 원소가 `'9'` 가 되어 `label in short_classes`(int)가
    영원히 거짓이 됐고, 짧은 차선 보호가 사라진 채 "효과 없음"으로 판정할 뻔했다 (08-19).
    """
    from stella.config_io import cast_like

    assert cast_like((9, 10, 6), "(9,10,6,11,4)") == (9, 10, 6, 11, 4)
    assert cast_like((9, 10, 6), "9, 10, 4") == (9, 10, 4)
    assert cast_like((1.0, 2.0), "[1.5, 2.5]") == (1.5, 2.5)


def test_short_class_protection_is_reachable():
    """디코더가 실제로 그 명단을 int 로 비교하는지 — 계약이 깨지면 보호가 무음으로 사라진다."""
    cfg = importlib.import_module("configs.exp_synthetic").get_config()
    decoder = build_instance(cfg.decode, cfg)
    assert all(isinstance(label, int) for label in decoder.short_classes)
    assert decoder._length_floor(next(iter(decoder.short_classes))) == cfg.decode.min_points_short


def test_class_freq_floor_never_suppresses_common_classes():
    """`floor` 정규화는 희소 클래스만 올린다 — 흔한 클래스를 1 아래로 누르지 않는다.

    E09 는 `mean` 정규화(재분배)로 `lane_line` 가중을 0.42 까지 눌렀고, 셀의 대다수가 그
    흔한 클래스라 전경 인식 자체가 무너졌다(class_fg -42%). 기각의 원인은 축이 아니라
    정규화였다 — 이 계약이 그 사고를 막는다.
    """
    from stella.loss.self_slot import SelfSlotLoss

    mean = SelfSlotLoss._build_class_weight(12, 0.5, 1.0, "mean")
    floor = SelfSlotLoss._build_class_weight(12, 0.5, 1.0, "floor")
    assert mean[1:].min() < 1.0  # 기존 방식은 실제로 누른다
    assert floor[1:].min() >= 1.0
    assert floor[1:].max() > 1.0  # 그러면서 희소는 올라간다
    assert float(SelfSlotLoss._build_class_weight(12, 0.0, 1.0, "floor")[1:].max()) == 1.0


def test_gt_cache_key_separates_grid_stride(tmp_path):
    """격자 간격이 다르면 GT 캐시 폴더가 갈려야 한다 — 기본값(4)일 때는 이름이 그대로다.

    GT 맵의 크기가 stride 로 정해지는데 폴더 이름이 같으면 stride 8 실행이 stride 4 캐시를
    조용히 읽는다. 그러면 "stride 8 이 나쁘다"는 **틀린 결론**이 나온다.
    """
    from stella.data.seedmap import SeedMapDataset

    def suffix(stride: int, lookahead: int) -> str:
        dataset = object.__new__(SeedMapDataset)
        dataset.encoder = types.SimpleNamespace(grid_stride=stride, conn_lookahead=lookahead)
        return dataset._cache_suffix()

    assert suffix(4, 1) == ""
    assert suffix(8, 1) == "_s8"
    assert suffix(8, 2) == "_look2_s8"

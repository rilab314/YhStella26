"""config 해석과 전체 조립 스모크 (design 5.5절).

`test_config_resolves`는 파일 이동·클래스 rename으로 문자열이 깨지는 것을 잡는 안전망이다.
"""

import dataclasses
import importlib
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

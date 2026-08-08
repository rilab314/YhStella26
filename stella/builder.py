"""클래스 선택 단일 관문 (design 5.1절).

config의 `path`(모듈 경로)와 `name`(클래스 이름) 두 문자열만 보고 클래스를 찾는다.
다른 어디서도 `importlib`을 직접 부르지 않는다.
"""

import difflib
import importlib
import inspect
from dataclasses import fields, is_dataclass
from typing import Any

from configs.schema import ModuleConfig


def resolve(module_cfg: ModuleConfig, base: type | None = None) -> type:
    """(path, name) -> 클래스. 인스턴스는 만들지 않는다."""
    where = type(module_cfg).__name__
    module = _import_module(module_cfg, where)
    cls = _get_class(module, module_cfg, where)
    _check_defined_here(cls, module, module_cfg, where)
    if base is not None and not issubclass(cls, base):
        raise TypeError(f"{where}: {cls.__name__} 은 {base.__name__} 의 하위 클래스가 아니다")
    if not hasattr(cls, "from_cfg"):
        raise TypeError(f"{where}: {cls.__name__} 에 from_cfg 가 없다")
    return cls


def _import_module(module_cfg: ModuleConfig, where: str):
    try:
        return importlib.import_module(module_cfg.path)
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(f"{where}.path='{module_cfg.path}' 를 import 할 수 없다") from e


def _get_class(module, module_cfg: ModuleConfig, where: str) -> type:
    cls = getattr(module, module_cfg.name, None)
    if cls is not None:
        return cls
    here = sorted(
        n
        for n, o in vars(module).items()
        if isinstance(o, type) and o.__module__ == module.__name__
    )
    hint = difflib.get_close_matches(module_cfg.name, here, n=3)
    tail = f"혹시 이것? {hint}" if hint else f"이 모듈의 클래스: {here}"
    raise AttributeError(f"{where}.name='{module_cfg.name}' 이 {module_cfg.path} 에 없다. " + tail)


def _check_defined_here(cls: type, module, module_cfg: ModuleConfig, where: str) -> None:
    if cls.__module__ == module.__name__:
        return
    raise TypeError(
        f"{where}: '{module_cfg.name}' 은 {module_cfg.path} 에서 정의된 것이 아니라 "
        f"{cls.__module__} 에서 import 된 이름이다. path를 '{cls.__module__}' 로 고쳐라"
    )


def build_instance(module_cfg: ModuleConfig, cfg, base: type | None = None, **kwargs) -> Any:
    """부품 하나를 만든다. 클래스 선택은 resolve가, 조립은 클래스의 from_cfg가 한다."""
    return resolve(module_cfg, base).from_cfg(module_cfg, cfg, **kwargs)


def check_all(cfg) -> None:
    """cfg 트리의 모든 ModuleConfig를 미리 찾아본다. 무거운 초기화 전에 부른다."""
    for module_cfg in walk_module_configs(cfg):
        resolve(module_cfg)


def walk_module_configs(node):
    if isinstance(node, ModuleConfig):
        yield node
    if is_dataclass(node):
        for f in fields(node):
            yield from walk_module_configs(getattr(node, f.name))


class Buildable:
    """config 필드가 __init__ 인자와 이름까지 대응하는 클래스용 from_cfg 기본 구현 (5.4절).

    시그니처로 거르므로 `lr_mult` 같은 비생성자 필드는 섞여 들어가지 않는다.
    config 필드명과 __init__ 인자명은 반드시 같게 쓴다.
    """

    @classmethod
    def from_cfg(cls, module_cfg: ModuleConfig, cfg, **kwargs):
        params = inspect.signature(cls).parameters
        auto = {
            f.name: getattr(module_cfg, f.name)
            for f in fields(module_cfg)
            if f.name in params and not is_dataclass(getattr(module_cfg, f.name))
        }
        return cls(**{**auto, **kwargs})

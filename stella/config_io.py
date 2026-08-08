"""config 로드·덮어쓰기 (design 4.3절).

학습 진입점과 스크립트(디코더 평가·튜닝)가 같은 규칙으로 config를 읽어야 하는데,
`stella/train/train.py`는 Lightning·DataLoader를 끌고 온다. 그래서 순수한 config 조작만
여기로 분리했다 — 이 모듈은 표준 라이브러리와 `configs`만 의존한다.
"""

import dataclasses
import importlib


def load_config(module_name: str, overrides: list[str]):
    cfg = importlib.import_module(module_name).get_config()
    for item in overrides:
        path, _, raw = item.partition("=")
        apply_override(cfg, path.split("."), raw)
    return cfg


def apply_override(node, path: list[str], raw: str) -> None:
    for key in path[:-1]:
        node = getattr(node, key)
    current = getattr(node, path[-1])
    setattr(node, path[-1], cast_like(current, raw))


def cast_like(current, raw: str):
    """덮어쓸 값의 타입은 **현재 값**이 정한다 — config가 타입의 단일 출처다."""
    if isinstance(current, bool):
        return raw.lower() in ("1", "true", "yes")
    if isinstance(current, int):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    if isinstance(current, tuple):
        return tuple(raw.split(","))
    return raw


def apply_saved_config(cfg, saved: dict) -> None:
    """실행 폴더의 `config.json`을 현재 스키마 위에 덮어쓴다.

    스크립트가 "그 실행이 무슨 설정이었나"를 그대로 재현할 때 쓴다. 스키마에 없는 키는
    건너뛰므로, 스키마가 바뀌어도 옛 실행을 읽을 수 있다.
    """
    for field in dataclasses.fields(cfg):
        if field.name not in saved:
            continue
        current, value = getattr(cfg, field.name), saved[field.name]
        if dataclasses.is_dataclass(current):
            apply_saved_config(current, value)
        else:
            setattr(cfg, field.name, tuple(value) if isinstance(current, tuple) else value)

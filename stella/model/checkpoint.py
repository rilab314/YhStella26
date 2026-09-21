"""체크포인트에서 모델을 되살린다 — 예측을 뽑는 스크립트들이 공유한다.

`scripts/` 는 설치되는 패키지가 아니라 서로 import 할 수 없다. 두 스크립트가 같은 방식으로
가중치를 올려야 하므로 여기에 둔다(조립 규칙 — 절대 import 만 쓴다).
"""

from pathlib import Path

import torch

from stella.builder import build_instance
from stella.eval import runlog

MODEL_PREFIX = "model."


def load_model(cfg, ckpt: str, device: str, allow_missing: bool = False):
    """가중치가 하나라도 안 붙으면 **크게 실패한다.**

    조용히 넘어가면 그 부품만 무작위 초기값인 채로 평가가 돌아 결과가 통째로 거짓이 된다
    (실제로 헤드 이름을 바꾼 뒤 `conn_head` 가 무작위로 평가될 뻔했다).
    """
    if not ckpt:
        raise SystemExit("체크포인트가 없다 — --ckpt 또는 --run 을 주거나 --source gt 를 쓴다")
    model = build_instance(cfg.model, cfg)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
    weights = {k[len(MODEL_PREFIX) :]: v for k, v in state.items() if k.startswith(MODEL_PREFIX)}
    weights = _rename_legacy(weights, set(model.state_dict()))
    missing, unexpected = model.load_state_dict(weights, strict=False)
    print(f"[ckpt] {ckpt}  (missing {len(missing)}, unexpected {len(unexpected)})")
    if (missing or unexpected) and not allow_missing:
        raise SystemExit(
            f"가중치 불일치 — missing={list(missing)[:6]} unexpected={list(unexpected)[:6]}"
        )
    return model.to(device).eval()


def _rename_legacy(weights: dict, expected: set) -> dict:
    """옛 체크포인트의 키를 현재 구조로 옮긴다.

    `ConnHead` 가 슬롯별 가중치를 지원하면서 `mlp` -> `mlps.0` 이 됐다 (가설 백로그 C5).
    """
    renamed = {}
    for key, value in weights.items():
        new_key = key.replace("conn_head.mlp.", "conn_head.mlps.0.")
        renamed[new_key if new_key in expected else key] = value
    return renamed


def best_checkpoint(run: str) -> str:
    """실행 폴더에서 쓸 체크포인트 — **검증 점수가 가장 좋은 에폭**을 고른다.

    예전에는 `last.ckpt` 를 먼저 집었는데 그 파일은 최종 에폭이 아니다(`runlog` 참고).
    점수를 못 읽으면 그때만 예전 방식(가장 최근에 쓰인 파일)으로 물러선다.
    """
    if not run:
        return ""
    best = runlog.best_checkpoint(Path(run))
    if best is not None:
        return str(best)
    files = sorted((Path(run) / "checkpoints").glob("*.ckpt"), key=lambda p: p.stat().st_mtime)
    return str(files[-1]) if files else ""

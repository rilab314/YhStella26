"""고정 배정의 **이음매** 손해를 잰다 — 4단계 Table VI 각주용 (GPU 불필요).

각도 순서로 슬롯을 고정하면 **원 위 어딘가에 이음매**가 생긴다. 각도 0도 부근에서는 방향이
조금만 흔들려도 두 분기의 순서가 뒤집히므로, 그 구간의 칸은 배정이 불안정하다.

**묻는 것**: 각도 고정 판이 짝 맞춤 판보다 못한 몫이 **그 이음매 구간에 몰려 있는가?**
몰려 있으면 "고정 배정이 원리적으로 치르는 대가"라는 설명이 수치로 선다.

방향 오차는 평가와 **같은 방식**(짝을 맞춘 뒤)으로 잰다 — 배정 순서 탓으로 부풀리지 않는다.

사용:
    python scripts/seam_analysis.py --caches F06d_ctrl G04_canon --names 짝맞춤 각도고정
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from stella.builder import build_instance
from stella.config_io import apply_override, load_config
from stella.data.types import GridDatasetBase, collate_fn
from stella.decode.cache import load_prediction
from stella.loss.matching import assign_slots, pad_branches

CACHE_ROOT = Path("/data/ongoing/2026_stella/pred_cache")
SEAM_DEGREES = 10.0  # 0도에서 이 안쪽이면 '이음매 근처'
VALID_NORM = 0.5


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, [])
    for path, raw in (("data.root", args.root), ("data.cache_dir", args.gt_cache)):
        apply_override(cfg, path.split("."), raw)
    loader = build_loader(cfg, args.split)
    shape = {
        "grid_size": cfg.data.grid_size,
        "num_classes": cfg.data.num_classes,
        "num_slots": cfg.model.num_conn_slots,
    }
    print(f"{'모델':12} {'이음매 근처':>12} {'먼 곳':>10} {'몇 배 나쁜가':>12}")
    for cache, name in zip(args.caches, args.names):
        near, far = measure(loader, CACHE_ROOT / cache, shape, args.count)
        print(f"{name:12} {near:9.2f}도 {far:9.2f}도 {near / far:11.2f}배")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs.base")
    parser.add_argument("--caches", nargs="+", required=True)
    parser.add_argument("--names", nargs="+", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--count", type=int, default=400)
    parser.add_argument("--root", default="/data/ongoing/2026_stella/SEED_MAP_v1.2_splits")
    parser.add_argument("--gt-cache", default="/data/ongoing/2026_stella/gt_cache")
    return parser.parse_args()


def build_loader(cfg, split: str) -> DataLoader:
    dataset = build_instance(cfg.data, cfg, base=GridDatasetBase, split=split)
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=6, collate_fn=collate_fn)


def measure(loader, cache: Path, shape: dict, count: int) -> tuple[float, float]:
    """(이음매 근처 평균 오차, 먼 곳 평균 오차) — 도 단위."""
    near, far = [], []
    for index, batch in enumerate(loader):
        if index >= count:
            break
        stem = str(batch["meta"][0].get("filename", index)).replace("/", "_")
        path = cache / f"{stem}.npz"
        if not path.exists():
            continue
        output, _ = load_prediction(path, shape)
        errors, seam = cell_errors(output, batch, shape["num_slots"])
        if errors.numel():
            near.append(errors[seam])
            far.append(errors[~seam])
    return _mean(near), _mean(far)


def cell_errors(output, batch: dict, num_slots: int) -> tuple[torch.Tensor, torch.Tensor]:
    """칸마다 (짝을 맞춘 뒤의 방향 오차, 이음매 근처인가).

    캐시에서 읽은 예측은 **배치 차원이 없다** — 정답 쪽의 배치 차원을 벗겨 맞춘다.
    """
    supervised = (batch["class_map"][0] > 0) & output.node_mask
    if not bool(supervised.any()):
        return torch.zeros(0), torch.zeros(0, dtype=torch.bool)
    gt_dir = batch["conn_dirs"][0][supervised].float()
    valid = gt_dir.norm(dim=-1) > VALID_NORM
    gt_dir, valid = pad_branches(gt_dir, valid, num_slots)
    pred_dir = output.conn_dir[supervised].float()
    pred_exist = output.exist_logit[supervised].float()
    assignment, matched, _ = assign_slots(pred_dir, pred_exist, gt_dir, valid, 1.0, 1.0)
    index = assignment.unsqueeze(-1).expand(-1, -1, 2)
    cosine = (pred_dir * torch.gather(gt_dir, 1, index)).sum(dim=-1).clamp(-1.0, 1.0)
    degrees = torch.rad2deg(torch.arccos(cosine))
    return degrees[matched], _seam_flags(gt_dir, valid).unsqueeze(1).expand_as(matched)[matched]


def _seam_flags(gt_dir: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """분기 중 하나라도 각도 0도 부근이면 그 칸은 이음매 근처다."""
    angle = torch.rad2deg(torch.atan2(gt_dir[..., 1], gt_dir[..., 0]) % (2 * np.pi))
    near = ((angle < SEAM_DEGREES) | (angle > 360.0 - SEAM_DEGREES)) & valid
    return near.any(dim=1)


def _mean(chunks: list) -> float:
    if not chunks:
        return float("nan")
    joined = torch.cat([c for c in chunks if c.numel()])
    return float(joined.mean()) if joined.numel() else float("nan")


if __name__ == "__main__":
    main()

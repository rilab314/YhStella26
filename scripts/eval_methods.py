"""여러 방법을 **같은 자로** 채점하고, 논문 그림이 읽을 파일로 남긴다 — 5단계 진입점.

**GPU 를 쓰지 않는다.** 이미 떨궈 둔 예측(희소 캐시 · 분할 예측 PNG)만 읽는다.

    python scripts/eval_methods.py --method stella --split val
    python scripts/eval_methods.py --method seg_dp --split val --limit 50   # 빠른 점검

남기는 것 두 가지 — 둘 다 방법·split 별 폴더 하나다.

    paper/pred/{방법}/{split}/{타일}.npz   예측 폴리라인 (그림이 그린다)
    paper/score/{방법}_{split}/            summary.json(전체+종류별) · per_tile.csv(타일별)

**타일별 표가 이 진입점의 존재 이유다.** 논문 그림의 선별 조건은 전부 타일 단위인데
(`docs/paper_outline.md` §3), 지금까지의 채점은 전 타일 합계만 냈다.

폴더는 **통째로 지우고 다시 만든다**(`docs/6_paper_assets.md` 규약) — 지난 실행의 파일이
이번 결과에 섞이는 사고를 막는다.
"""

import argparse
import json
import shutil
import time
from pathlib import Path

from stella.baseline.segment_vectorize import CLASS_RGB, SegmentVectorizer
from stella.config_io import load_config
from stella.decode.sweep import build_cfg, read_meta, shape_of
from stella.eval.instance_io import save_instances
from stella.eval.methods import CacheMethod, SegmentMethod
from stella.eval.per_tile import TileScorer, write_rows

DATA_ROOT = Path("/data/ongoing/2026_stella")
# `scripts/figures/figure_base.py` · `scripts/tables/table_base.py` 와 같은 값이다.
# scripts/ 는 패키지가 아니라 서로 import 하지 못한다 — 경로를 바꾸면 세 곳을 함께 고친다.
PAPER_ROOT = DATA_ROOT / "paper"
CACHE_ROOT = DATA_ROOT / "pred_cache"
SEG_ROOT = DATA_ROOT / "baseline/mask2former/pred_large"

# Table VII 상단의 U 규격 실행. 이름 = 표의 행, 값 = 캐시 폴더(val 700장).
UNIT_CACHES = {
    "u_cnxb": "B_ctrl",
    "u_cnxl": "B_cnxL",
    "u_dinov3": "B_dinov3",
    "u_dinov3_lr02": "B_dinov3lr02",
    "u_swinb": "B_swin",
    "u_swinl": "B_swinL",
    "u_freeze": "B_freeze",
    "u_ref": "E30_len03_val700",
    "u_win9": "V7_A1_win9",
    "u_deep4": "V7_A1_deep4",
    "u_deep3": "V7_A2_deep3",
    "u_slots3": "V7_A1_slots3",
    "u_hrnet": "V7_A1_hrnet",
}

# 방법 목록. 캐시 갈래는 **디코더 설정만** 다르다 — 모델도 캐시도 같은 것을 쓴다.
# `config` 를 적은 방법만 그 config 의 디코더를 쓴다 (없으면 `--config`, 기본 `configs.base`).
METHODS = {
    "stella": {"cache": "F06d_ctrl", "decode": ()},
    # 후처리 없음 = 능선만 남기기(vertex_local_max) 끄고 + 중복 정리(dedup_high=0) 끈다.
    "stella_nopost": {"cache": "F06d_ctrl", "decode": ("vertex_local_max=False", "dedup_high=0")},
    # 각도 섹터만 가져온 판 — Table VI 의 한 행. Table III 의 비교 대상은 아래 `gte_delta` 다.
    "gte": {"cache": "G01_gte", "decode": ()},
    # Sat2Graph GTE 전체(각도 섹터 + 변위 회귀) — Table III 비교 대상. 조율한 문턱이 config 에 있다.
    "gte_delta": {"cache": "G06_delta", "config": "configs.exp_gte_delta", "decode": ()},
    # 정답 주입 천장은 **후처리 조건마다 다르다** (Table V 마지막 열) — 두 조건을 모두 둔다.
    "gt": {"cache": "gt_val_v12", "decode": ()},
    "gt_nopost": {"cache": "gt_val_v12", "decode": ("vertex_local_max=False", "dedup_high=0")},
    "seg_dp": {"segment": "pred_val", "epsilon": 2.0, "min_pixels": 50},
    # Table V — 후처리 두 단계를 하나씩만 켠다. 천장 열도 같은 조건으로 잰다.
    "stella_ridge": {"cache": "F06d_ctrl", "decode": ("dedup_high=0",)},
    "stella_dedup": {"cache": "F06d_ctrl", "decode": ("vertex_local_max=False",)},
    "gt_ridge": {"cache": "gt_val_v12", "decode": ("dedup_high=0",)},
    "gt_dedup": {"cache": "gt_val_v12", "decode": ("vertex_local_max=False",)},
    # Table VII 상단 — U 규격(train 3,000 · val 700 · 10에폭) 실행을 **현행 디코더로** 다시 잰다.
    **{name: {"cache": cache, "decode": ()} for name, cache in UNIT_CACHES.items()},
}
# 미리 정한 통과 조건 (검증셋 전량 · 현행 자). 어긋나면 원인을 찾기 전에 6단계로 넘어가지 않는다.
# 천장 두 값의 출처는 Table V 다 — 후처리를 켜면 0.9404, 끄면 0.9434(중복 정리가 진짜 선을 지운 몫).
EXPECTED_F1 = {
    "stella": 0.4489,
    "stella_nopost": 0.4031,
    "gte": 0.4148,
    "gte_delta": 0.3736,  # `result_0924.md` 2절 (조율 후)
    "seg_dp": 0.3088,
    "gt": 0.9404,
    "gt_nopost": 0.9434,
    # Table V (`result_0910b.md` C절 · val 전량)
    "stella_ridge": 0.4250,
    "stella_dedup": 0.4398,
    "gt_ridge": 0.9434,
    "gt_dedup": 0.9404,
    # Table VII 상단 (`result_0910b.md` · U 규격 val 700장)
    "u_cnxb": 0.3549,
    "u_cnxl": 0.3820,
    "u_dinov3": 0.3333,
    "u_dinov3_lr02": 0.2924,
    "u_swinb": 0.3056,
    "u_swinl": 0.3009,
    "u_freeze": 0.2573,
    "u_ref": 0.3571,
    "u_win9": 0.3494,
    "u_deep4": 0.3464,
    "u_deep3": 0.3349,
    "u_slots3": 0.3428,
    "u_hrnet": 0.2967,
}
TOLERANCE = 0.001


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    method, cfg = build_method(args)
    rows, summary = collect(method, cfg, args)
    elapsed = time.perf_counter() - started
    save(args, rows, summary, elapsed)
    report(args.method, summary, len(rows), elapsed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=sorted(METHODS))
    parser.add_argument("--split", default="val")
    parser.add_argument("--config", default="configs.base")
    parser.add_argument("--limit", type=int, default=0, help="0이면 전부")
    parser.add_argument("--cache", default="", help="캐시 폴더 직접 지정")
    parser.add_argument("--pred", default="", help="분할 예측 PNG 폴더 직접 지정")
    return parser.parse_args()


def build_method(args: argparse.Namespace):
    spec = METHODS[args.method]
    if "cache" in spec:
        return build_cache_method(args, spec)
    return build_segment_method(args, spec)


def build_cache_method(args: argparse.Namespace, spec: dict):
    """캐시를 만든 격자·슬롯 규격을 config 에 되돌린 뒤 디코더를 세운다."""
    cache = Path(args.cache) if args.cache else CACHE_ROOT / spec["cache"]
    meta = read_meta(cache)
    cfg = build_cfg(spec.get("config", args.config), list(spec["decode"]), meta)
    method = CacheMethod(
        cache=cache, cfg=cfg, decode_cfg=cfg.decode, shape=shape_of(meta), limit=args.limit
    )
    print(f"[{args.method}] 캐시 {cache.name} · 타일 {len(method)}장 · 슬롯 {meta['num_slots']}개")
    return method, cfg


def build_segment_method(args: argparse.Namespace, spec: dict):
    """색칠된 예측 PNG 를 폴리라인으로 바꾼다. 채점에서 빼는 종류는 뽑지도 않는다."""
    cfg = load_config(args.config, [])
    pred_dir = Path(args.pred) if args.pred else SEG_ROOT / spec["segment"]
    excluded = {int(label) for label in cfg.eval.exclude_classes}
    classes = tuple(label for label in sorted(CLASS_RGB) if label not in excluded)
    vectorizer = SegmentVectorizer(
        classes=classes, epsilon=spec["epsilon"], min_pixels=spec["min_pixels"]
    )
    method = SegmentMethod(
        pred_dir=pred_dir, cfg=cfg, vectorizer=vectorizer, split=args.split, limit=args.limit
    )
    print(f"[{args.method}] 예측 PNG {pred_dir.name} · 세는 종류 {len(classes)}개")
    return method, cfg


def collect(method, cfg, args: argparse.Namespace) -> tuple[list[dict], dict]:
    """타일마다 예측을 파일로 남기고 한 줄씩 채점한다."""
    pred_dir = fresh(PAPER_ROOT / "pred" / args.method / args.split)
    scorer = TileScorer(cfg)
    rows = []
    for tile, prediction, target in method.tiles():
        save_instances(pred_dir / f"{tile}.npz", prediction)
        rows.append(scorer.row(tile, prediction, target))
        if len(rows) % 200 == 0:
            print(f"  ... {len(rows)}장")
    return rows, scorer.summary()


def fresh(path: Path) -> Path:
    """폴더 단위로 지우고 다시 만든다 (docs/6_paper_assets.md 규약)."""
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save(args: argparse.Namespace, rows: list[dict], summary: dict, elapsed: float) -> None:
    out_dir = fresh(PAPER_ROOT / "score" / f"{args.method}_{args.split}")
    write_rows(out_dir / "per_tile.csv", rows)
    payload = {
        "method": args.method,
        "split": args.split,
        "tiles": len(rows),
        "seconds_per_tile": elapsed / max(len(rows), 1),
        "pred_per_tile": sum(row["n_pred"] for row in rows) / max(len(rows), 1),
        "summary": summary,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[{args.method}] 저장 {out_dir}")


def report(name: str, summary: dict, tiles: int, elapsed: float) -> None:
    """미리 정한 기대값과 대조한다 — 여기서 갈라지면 그림과 표가 서로 다른 말을 하게 된다."""
    f1 = summary.get("f1", 0.0)
    expected = EXPECTED_F1.get(name)
    print(f"\n[{name}] 타일 {tiles}장 · {elapsed:.1f} s · f1 {f1:.4f}")
    if expected is None:
        return
    gap = abs(f1 - expected)
    verdict = "일치" if gap <= TOLERANCE else "★ 어긋남 — 원인을 찾기 전에 다음 단계로 가지 않는다"
    print(f"  기대값 {expected:.4f} · 차이 {gap:.4f} · {verdict}")


if __name__ == "__main__":
    main()

"""원인 하나의 표본을 뽑아 확대해 그린다 — 사람이 원인을 확인하고 장면을 고른다 (E5 · Fig. 10).

E2 는 원인을 기하 조건으로만 센다. 라벨 누락인지 그늘인지는 사진을 봐야 안다.

    python scripts/sample_error_kind.py --kind spurious_long --count 30       # 틀린 예측 · 무작위
    python scripts/sample_error_kind.py --side fn --kind missed --order dark  # 놓친 정답 · 어둠순

한 장 = 문제의 선 주변을 잘라 키운 원본 | 정답 | 예측. 문제의 선은 굵은 빨강(틀린 예측은 예측 칸,
놓친 정답은 정답 칸), 나머지 선은 얇은 회색이다.
`--order dark` 는 문제의 선이 지나는 화소의 평균 밝기가 낮은 순 — 그늘 장면이 위로 온다.
→ experiment/data/e5/{방법}_{원인}/NN_{타일}_{선 번호}.jpg
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

from stella.builder import build_instance
from stella.config_io import load_config
from stella.data.types import CLASS_NAMES, GridDatasetBase
from stella.eval.error_kinds import ErrorKinds
from stella.eval.instance_io import load_instances
from stella.eval.methods import target_tiles
from stella.train import viz

PAPER_ROOT = Path("/data/ongoing/2026_stella/paper")
OUT_ROOT = Path(__file__).resolve().parents[1] / "experiment" / "data" / "e5"
SEED = 0
HIGHLIGHT = (255, 40, 40)  # 문제의 선 (RGB) — 흰 테두리를 둘러 어떤 배경에서도 보이게
TRUTH = (60, 230, 90)  # 정답 칸의 나머지 선 — 밝은 초록
PRED = (0, 220, 255)  # 예측 칸의 나머지 선 — 밝은 하늘색
EDGE = (255, 255, 255)
LINE_PX, PROBLEM_PX = 3, 5  # 키운 화면 기준 굵기(px)
CROP_MIN = 220  # 잘라 낼 창의 최소 한 변(px)
CROP_MARGIN = 40
VIEW = 600  # 잘라 낸 칸을 키운 한 변(px)
GAP = 10
FOCUS_GAP = 3.0  # 이보다 멀면 실패 자리로 본다(px) — 채점 버퍼 ρ 와 같다
HALO_PX, DASH_ON, DASH_OFF = 15, 22, 12  # 키운 화면 기준 띠 굵기 · 점선 길이(px)
HIDDEN_CLASSES = (0, CLASS_NAMES.index("safety_zone"))  # figure_base 와 같다 — 그리지 않는다


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, [])
    dataset = build_instance(cfg.data, cfg, base=GridDatasetBase, split=args.split)
    found = collect(cfg, args)
    chosen = choose(found, args, dataset)
    print(f"[{args.method}] {args.side}/{args.kind} {len(found)}개 중 {len(chosen)}개를 그린다")
    render_all(args, dataset, chosen)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="stella")
    parser.add_argument("--side", default="fp", choices=("fp", "fn"), help="틀린 예측 · 놓친 정답")
    parser.add_argument("--kind", default="spurious_long")
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--order", default="random", choices=("random", "dark"))
    parser.add_argument(
        "--plain", action="store_true", help="칸 위 글자 없이(논문용) — 폴더 끝에 _plain"
    )
    parser.add_argument(
        "--no-mark", action="store_true", help="문제의 선도 보통 색으로 — 폴더 끝에 _nomark"
    )
    parser.add_argument(
        "--class-color", action="store_true", help="모든 선을 종류 색으로(Fig. 1 범례) — 끝에 _cls"
    )
    parser.add_argument(
        "--ring", action="store_true", help="실패 구간에 흰 점선 띠(네 행 같은 표시) — 끝에 _ring"
    )
    parser.add_argument("--split", default="val")
    parser.add_argument(
        "--tiles", nargs="*", default=[], help="이 타일들만 본다 — 고른 장면 다시 그리기"
    )
    parser.add_argument("--out-suffix", default="", help="출력 폴더 이름 끝에 붙인다")
    parser.add_argument("--config", default="configs.base")
    return parser.parse_args()


def collect(cfg, args) -> list[tuple[str, int]]:
    """(타일, 선 번호) — 그 원인으로 판정된 선 전부. 번호는 예측 또는 정답 목록 안의 자리다."""
    kinds = ErrorKinds(cfg)
    labeler = kinds.label_predictions if args.side == "fp" else kinds.label_targets
    found = []
    for tile, target in target_tiles(cfg, args.split):
        if args.tiles and tile not in args.tiles:
            continue
        labels = labeler(read_prediction(args, tile), target)
        found += [(tile, index) for index, label in enumerate(labels) if label == args.kind]
    return found


def read_prediction(args, tile: str) -> list[dict]:
    path = PAPER_ROOT / "pred" / args.method / args.split / f"{tile}.npz"
    return load_instances(path) if path.exists() else []


def choose(found: list[tuple[str, int]], args, dataset) -> list[tuple[str, int]]:
    if args.order == "random":
        # 정렬해 두면 번호가 다시 그려도 같다 — E5 결과가 번호로 장면을 가리킨다
        return sorted(random.Random(SEED).sample(found, min(args.count, len(found))))
    position = {stem: index for index, stem in enumerate(dataset.stems)}
    brightness = []
    for tile, index in found:
        item = dataset[position[tile]]
        line = problem_line(args, item["instances"], read_prediction(args, tile), index)
        brightness.append(mean_brightness(viz.to_uint8_image(item["image"].numpy()), line))
    chosen, seen = [], set()
    for i in np.argsort(brightness):  # 한 타일에서 한 선만 — 같은 그늘 도로가 줄줄이 뽑히지 않게
        if found[i][0] not in seen:
            chosen.append(found[i])
            seen.add(found[i][0])
    return chosen[: args.count]


def problem_line(args, truth: list[dict], prediction: list[dict], index: int) -> np.ndarray:
    return np.asarray((prediction if args.side == "fp" else truth)[index]["points"])


def mean_brightness(image: np.ndarray, points: np.ndarray) -> float:
    """선이 지나는 화소의 평균 밝기(0~255) — 굵기 3 px 로 래스터화해 잰다."""
    mask = np.zeros(image.shape[:2], np.uint8)
    cv2.polylines(mask, [np.round(points).astype(np.int32)], False, 1, 3)
    return float(image.mean(axis=2)[mask > 0].mean()) if mask.any() else 255.0


def render_all(args, dataset, chosen: list[tuple[str, int]]) -> None:
    position = {stem: index for index, stem in enumerate(dataset.stems)}
    suffix = ("_plain" if args.plain else "") + ("_nomark" if args.no_mark else "")
    suffix += ("_cls" if args.class_color else "") + ("_ring" if args.ring else "")
    out_dir = OUT_ROOT / f"{args.method}_{args.kind}{suffix}{args.out_suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for order, (tile, index) in enumerate(chosen, start=1):
        item = dataset[position[tile]]
        sheet = render(
            args, item["image"].numpy(), item["instances"], read_prediction(args, tile), index
        )
        cv2.imwrite(str(out_dir / f"{order:02d}_{tile}_{index}.jpg"), sheet[:, :, ::-1])
    print(f"[{args.method}] 저장 {out_dir}")


def render(args, image, truth: list[dict], prediction: list[dict], index: int) -> np.ndarray:
    """잘라 키운 뒤에 선을 그린다 — 먼저 그리면 흐리고 가늘다(사용자 지적 10-02)."""
    plain = viz.to_uint8_image(image)
    window = crop_window(problem_line(args, truth, prediction, index), plain.shape[0])
    base = _crop(plain, window)
    if args.class_color:  # 네 행이 같은 규칙 — 종류 색 · 같은 굵기 · 표시 없음(교수 지시 10-07)
        marked = base.copy()
        if args.ring:  # 실패 구간을 네 행 모두 같은 모양으로 짚는다(사용자 결정 10-07)
            _halo(marked, _to_view(failure_path(args, truth, prediction, index), window))
        panels = [
            base,
            _draw_classes(marked, window, truth),
            _draw_classes(marked, window, prediction),
        ]
    elif args.kind == "wrong_class":
        under = _nearest(truth, np.asarray(prediction[index]["points"]))
        panels = [base, _draw(base, window, [under], 0, None, _class_color(under["class"]))]
        panels.append(
            _draw(
                base, window, [prediction[index]], 0, None, _class_color(prediction[index]["class"])
            )
        )
        if not args.plain:
            _caption(panels[1], f"GT: {CLASS_NAMES[under['class']]}")
            _caption(panels[2], f"Pred: {CLASS_NAMES[prediction[index]['class']]}")
    else:
        mark_gt, mark_pred = (index, -1) if args.side == "fn" else (-1, index)
        if args.no_mark:  # 빨강이 "이 선만 문제"·"틀린 선"으로 오해된다(사용자 지적 10-02)
            mark_gt = mark_pred = -1
        panels = [
            base,
            _draw(base, window, truth, mark_gt, TRUTH),
            _draw(base, window, prediction, mark_pred, PRED),
        ]
    gap = np.full((VIEW, GAP, 3), 255, np.uint8)
    return np.concatenate([panels[0], gap, panels[1], gap, panels[2]], axis=1)


def _draw(
    base, window, instances: list[dict], mark: int, color, mark_color=HIGHLIGHT
) -> np.ndarray:
    """`mark` 번 선은 흰 테두리를 두른 굵은 선으로, 나머지는 `color` 로(None 이면 그리지 않는다)."""
    canvas = base.copy()
    for number, item in enumerate(instances):
        if number != mark and color is not None:
            _polyline(canvas, _to_view(item["points"], window), color, LINE_PX)
    if 0 <= mark < len(instances):
        points = _to_view(instances[mark]["points"], window)
        _polyline(canvas, points, EDGE, PROBLEM_PX + 4)
        _polyline(canvas, points, mark_color, PROBLEM_PX)
    return canvas


def _draw_classes(base, window, instances: list[dict]) -> np.ndarray:
    """모든 선을 종류 색으로 — 논문 그림과 같이 배경·안전지대는 그리지 않는다."""
    canvas = base.copy()
    for item in instances:
        if int(item["class"]) not in HIDDEN_CLASSES:
            points = _to_view(item["points"], window)
            _polyline(canvas, points, _class_color(item["class"]), LINE_PX)
    return canvas


def failure_path(args, truth, prediction, index) -> np.ndarray:
    """실패 구간 — 문제의 선에서 벗어난 점 · 놓친 점 · 종류가 바뀐 점만 남긴다(타일 좌표)."""
    line = _dense(problem_line(args, truth, prediction, index))
    if args.kind == "wrong_class":
        bad = _nearest_points(line, truth)[1] != int(prediction[index]["class"])
    else:
        gap = _nearest_points(line, truth if args.side == "fp" else prediction)[0]
        bad = gap > FOCUS_GAP
    if args.kind == "drift":  # 벗어남은 선 전체가 판정 대상 — 넘는 점만 짚으면 안 보인다
        return line
    return line[bad] if bad.sum() > 1 else line


def _dense(points) -> np.ndarray:
    """폴리라인을 1 px 간격 점열로."""
    points = np.asarray(points, np.float64)
    steps = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    at = np.arange(0, steps[-1] + 1)
    return np.stack([np.interp(at, steps, points[:, 0]), np.interp(at, steps, points[:, 1])], 1)


def _nearest_points(line: np.ndarray, instances: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """선의 각 점에서 가장 가까운 다른 선 점까지의 (거리, 그 선의 종류)."""
    if not instances:
        return np.full(len(line), np.inf), np.full(len(line), -1)
    dense = [_dense(item["points"]) for item in instances]
    others = np.concatenate(dense)
    labels = np.concatenate([np.full(len(d), int(i["class"])) for d, i in zip(dense, instances)])
    distance = np.linalg.norm(line[:, None] - others[None], axis=2)
    nearest = distance.argmin(axis=1)
    return distance[np.arange(len(line)), nearest], labels[nearest]


def _halo(canvas: np.ndarray, points: np.ndarray) -> None:
    """선 밑에 까는 흰 점선 띠 — 1 px 간격 점열을 일정 길이씩 끊어 그린다."""
    period = DASH_ON + DASH_OFF
    length = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    phase = (length % period) < DASH_ON
    jump = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 3 * VIEW / CROP_MIN]
    run = np.cumsum(jump | np.r_[True, phase[1:] != phase[:-1]])
    for label in np.unique(run[phase]):
        piece = points[(run == label) & phase]
        if len(piece) > 1:
            _polyline(canvas, piece, EDGE, HALO_PX)


def _to_view(points, window: tuple[int, int, int, int]) -> np.ndarray:
    """타일 좌표 → 잘라 키운 화면 좌표."""
    x0, y0, x1, _ = window
    return (np.asarray(points, np.float64) - (x0, y0)) * (VIEW / (x1 - x0))


def _nearest(truth: list[dict], points: np.ndarray) -> dict:
    """예측 선의 점들에 평균적으로 가장 가까운 정답 선."""

    def gap(item):
        other = np.asarray(item["points"])
        return np.linalg.norm(points[:, None] - other[None], axis=2).min(axis=1).mean()

    return min(truth, key=gap)


def _class_color(label: int) -> tuple:
    return tuple(int(v) for v in viz.CLASS_COLOR[label % len(viz.CLASS_COLOR)])


def _caption(canvas: np.ndarray, text: str) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, text, (12, 40), font, 1.1, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(canvas, text, (12, 40), font, 1.1, (255, 255, 255), 2, cv2.LINE_AA)


def crop_window(points: np.ndarray, size: int) -> tuple[int, int, int, int]:
    """문제의 선을 감싸는 정사각 창 (x0, y0, x1, y1)."""
    center = points.mean(axis=0)
    half = max(CROP_MIN // 2, int(np.ptp(points, axis=0).max() / 2) + CROP_MARGIN)
    x0, y0 = (np.clip(center - half, 0, size - 2 * half)).astype(int)
    return int(x0), int(y0), int(x0 + 2 * half), int(y0 + 2 * half)


def _crop(panel: np.ndarray, window: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = window
    return cv2.resize(panel[y0:y1, x0:x1], (VIEW, VIEW), interpolation=cv2.INTER_CUBIC)


def _polyline(canvas: np.ndarray, points, color: tuple, thickness: int) -> None:
    pixels = np.round(np.asarray(points, np.float32) * 4).astype(np.int32)
    cv2.polylines(canvas, [pixels], False, color, thickness, cv2.LINE_AA, shift=2)


if __name__ == "__main__":
    main()

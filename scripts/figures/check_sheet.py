"""확인용 시트 — 원본 / 정답 / 예측 / 판정 네 칸을 한 줄로 (09-30 교수 미팅).

**논문에 싣는 그림이 아니다.** 후보 그림(`figure_*/candidates/`)은 예측만 보여 줘서 잘 찾았는지
못 찾았는지 알 수 없었다. 이 시트는 같은 후보를 **채점 결과와 함께** 보여 준다.
파일 이름이 후보 번호와 타일 id 라, 보고 고른 id 를 그대로 `selected.txt` 에 옮기면 된다.

**장면 하나가 파일 하나, 원본 크기(칸당 768 px) 그대로다.** 한 쪽에 열 장면을 줄여 넣었더니
선이 너무 작아 판정을 읽을 수 없었다(사용자 지적 10-01). 비교 대상이 여럿인 그림은 같은
장면 안에서 **방법마다 한 줄씩** 쌓아 위아래로 바로 견준다.

판정 칸의 색 — 채점기와 **같은 짝**에서 나온다(`InstanceCCQ.label_instances`):

    초록  맞음(TP)       예측이 정답 하나와 짝지어졌다
    주황  중복(FP)       정답 위에 있지만 다른 예측에 짝을 뺏겼다 — 끊긴 조각·두 줄 그림
    빨강  헛검출(FP)     어떤 정답 위에도 없다
    파랑  놓침(FN)       짝을 못 얻은 정답 선
    회색  채점 제외      안전지대 등 채점에서 빼는 종류

사용:
    python scripts/figures/check_sheet.py                       # 후보 폴더 전부
    python scripts/figures/check_sheet.py --figure figure_08    # 하나만
출력: paper/figure/{그림}/check/{후보 번호}_{타일}.jpg
"""

import argparse
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from figure_base import FIGURE_ROOT, GAP, MethodLines, TileImages, draw, fresh, join_row, label

from stella.builder import build_instance
from stella.config_io import load_config
from stella.eval import ccq
from stella.train import viz

SPLIT = "val"
PANEL = 768  # 한 칸의 한 변(px) — 원본 타일 크기 그대로, 줄이지 않는다
HEADER = 70  # 머리(열 제목 + 범례) 높이(px)
# 원본 크기 그대로라 PNG 는 장당 수 MB · 1,300장이면 수 GB 다 — 눈으로 보는 용도라 JPEG 로 충분하다
JPEG_QUALITY = 92
DIM = 0.45  # 판정 칸의 영상 밝기 — 선 색이 묻히지 않게 어둡게 깐다
STATUS_WIDTH = 3
STATUS_STYLE = {  # 판정 → (RGB 색, 범례 이름)
    ccq.TP: ((40, 200, 70), "TP"),
    ccq.FP_REDUNDANT: ((255, 150, 0), "FP dup"),
    ccq.FP_SPURIOUS: ((235, 40, 40), "FP false"),
    ccq.FN: ((60, 130, 255), "FN miss"),
    ccq.EXCLUDED: ((150, 150, 150), "excluded"),
}
METHOD_TITLE = {"stella": "STELLA", "seg_dp": "Segmentation", "gte_delta": "Sat2Graph GTE"}
# 그림마다 판정을 볼 방법 — 그 그림의 후보 열에 나오는 방법과 같다
FIGURE_METHODS = {
    "figure_01": ("stella",),
    "figure_08": ("seg_dp", "gte_delta", "stella"),
    "figure_09": ("stella",),
    "figure_11": ("seg_dp", "stella"),
}


class CheckSheet:
    """후보 폴더 하나를 장면마다 확인용 시트 한 장으로 그린다."""

    def __init__(self, cfg):
        self.images = TileImages(cfg, SPLIT)
        self.lines = MethodLines(SPLIT)
        self.metric = build_instance(cfg.eval, cfg)

    def write(self, figure: str) -> None:
        candidates = read_candidates(FIGURE_ROOT / figure / "candidates")
        out_dir = fresh(FIGURE_ROOT / figure / "check")
        methods = methods_of(figure)
        for order, stem in candidates:
            sheet = self.compose_scene(stem, methods)
            path = str(out_dir / f"{order}_{stem}.jpg")
            cv2.imwrite(path, sheet[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        print(f"[check] {figure}: 후보 {len(candidates)}장 × {methods_of(figure)} → {out_dir}")

    def compose_scene(self, stem: str, methods: tuple[str, ...]) -> np.ndarray:
        """장면 하나 — 머리 아래에 방법마다 한 줄."""
        rows = [self.compose_row(stem, method) for method in methods]
        gap = np.zeros((GAP, rows[0].shape[1], 3), np.uint8)
        pieces = [page_header(rows[0].shape[1])]
        for row in rows:
            pieces += [gap, row]
        return np.concatenate(pieces, axis=0)

    def compose_row(self, stem: str, method: str) -> np.ndarray:
        image, truth = self.images(stem)
        prediction = self.lines(method, stem)
        gt_labels, pred_labels = self.metric.label_instances(prediction, truth)
        panels = [
            label(viz.to_uint8_image(image), stem),
            label(draw(image, truth), "GT"),
            label(draw(image, prediction), METHOD_TITLE.get(method, method)),
            label(
                draw_status(image, truth, gt_labels, prediction, pred_labels),
                count_text(gt_labels, pred_labels),
            ),
        ]
        return join_row(panels)


def read_candidates(folder: Path) -> list[tuple[str, str]]:
    """후보 파일 이름 `{번호}_{타일}_{점수}.png` 에서 (번호, 타일) 을 번호순으로 꺼낸다."""
    names = sorted(path.stem for path in folder.glob("*.png"))
    return [(name.split("_", 1)[0], name.split("_", 1)[1].rsplit("_", 1)[0]) for name in names]


def methods_of(figure: str) -> tuple[str, ...]:
    """`figure_09_hh` 처럼 묶음 꼬리가 붙은 폴더도 앞 이름으로 찾는다."""
    for prefix, methods in FIGURE_METHODS.items():
        if figure.startswith(prefix):
            return methods
    raise KeyError(f"판정을 볼 방법이 정해지지 않은 그림이다: {figure}")


def draw_status(image, truth, gt_labels, prediction, pred_labels) -> np.ndarray:
    """어둡게 깐 영상 위에 놓친 정답을 먼저, 예측을 그 위에 판정 색으로 그린다."""
    canvas = (viz.to_uint8_image(image).astype(np.float32) * DIM).astype(np.uint8)
    missed = [
        (item, status)
        for item, status in zip(truth, gt_labels, strict=True)
        if status in (ccq.FN, ccq.EXCLUDED)
    ]
    for item, status in missed + list(zip(prediction, pred_labels, strict=True)):
        points = np.round(np.asarray(item["points"], np.float32)).astype(np.int32)
        width = 1 if status == ccq.EXCLUDED else STATUS_WIDTH
        cv2.polylines(canvas, [points], False, STATUS_STYLE[status][0], width, cv2.LINE_AA)
    return canvas


def count_text(gt_labels: list[str], pred_labels: list[str]) -> str:
    """판정 칸 제목 — 이 타일의 TP·FP·FN 개수."""
    gt, pred = Counter(gt_labels), Counter(pred_labels)
    fp = pred[ccq.FP_REDUNDANT] + pred[ccq.FP_SPURIOUS]
    return f"TP {pred[ccq.TP]}  FP {fp}  FN {gt[ccq.FN]}"


def page_header(width: int) -> np.ndarray:
    """머리 — 네 칸의 제목과 판정 색 범례."""
    header = np.full((HEADER, width, 3), 255, np.uint8)
    titles = ("Image", "GT", "Prediction", "Judgement")
    for index, title in enumerate(titles):
        origin = (index * (PANEL + GAP) + 10, 28)
        cv2.putText(header, title, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    x = 10
    for color, name in STATUS_STYLE.values():
        cv2.line(header, (x, 55), (x + 40, 55), color, 5, cv2.LINE_AA)
        cv2.putText(
            header, name, (x + 48, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA
        )
        x += 200
    return header


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--figure", default="", help="후보 폴더 이름 (비우면 전부)")
    parser.add_argument("--config", default="configs.base")
    args = parser.parse_args()
    sheet = CheckSheet(load_config(args.config, []))
    folders = sorted(path.parent.name for path in FIGURE_ROOT.glob("*/candidates"))
    for figure in [args.figure] if args.figure else folders:
        sheet.write(figure)


if __name__ == "__main__":
    main()

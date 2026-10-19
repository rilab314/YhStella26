"""논문 그림 PDF 의 바깥 여백을 잘라낸다 — 그리기 도구가 남긴 빈 공간을 없앤다.

파워포인트·Illustrator 로 뽑은 PDF 는 **캔버스 크기 그대로** 저장돼 내용 주위에 빈 공간이
남는다. 그대로 실으면 그림이 차지하는 지면만 늘어난다. 이 도구는 내용이 실제로 그려진
범위를 찾아 그만큼만 남긴다. **벡터는 그대로 두고 자르는 범위만 바꾸므로 화질 손해가 없다.**

    python scripts/figures/trim_pdf.py docs/figures/fig03_architecture.pdf
    python scripts/figures/trim_pdf.py docs/figures/*.pdf --margin 6

원본은 같은 폴더의 `_src/` 로 옮겨 보관한다 — 다시 자르거나 되돌릴 수 있다.
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

RENDER_DPI = 150  # 여백을 찾을 때만 쓰는 해상도 — 결과 화질과 무관하다
WHITE_LEVEL = 250  # 이 값보다 밝으면 빈 공간으로 본다
SOURCE_DIR = "_src"


def main() -> None:
    args = parse_args()
    for path in [Path(p) for p in args.paths]:
        trim_one(path, args.margin)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="논문 그림 PDF 의 바깥 여백을 잘라낸다")
    parser.add_argument("paths", nargs="+", help="자를 PDF 파일들")
    parser.add_argument("--margin", type=float, default=4.0, help="남길 여백(pt)")
    return parser.parse_args()


def trim_one(path: Path, margin: float) -> None:
    page = page_size(path)
    box = content_box(path, page, margin)
    if box is None:
        print(f"[trim] {path.name}: 내용을 못 찾았다 — 건너뛴다")
        return
    keep_source(path)
    write_cropped(path, box)
    width, height = box[2] - box[0], box[3] - box[1]
    print(f"[trim] {path.name}: {page[0]:.0f}x{page[1]:.0f} -> {width:.0f}x{height:.0f} pt")


def page_size(path: Path) -> tuple[float, float]:
    """PDF 한 쪽의 크기(pt)."""
    out = subprocess.run(["pdfinfo", str(path)], capture_output=True, text=True).stdout
    line = next(line for line in out.splitlines() if line.startswith("Page size:"))
    parts = line.split()
    return float(parts[2]), float(parts[4])


def content_box(path: Path, page: tuple[float, float], margin: float):
    """내용이 그려진 범위를 PDF 좌표(왼쪽 아래가 원점)로 돌려준다."""
    image = render(path)
    if image is None:
        return None
    dark = np.argwhere(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) < WHITE_LEVEL)
    if dark.size == 0:
        return None
    (top, left), (bottom, right) = dark.min(axis=0), dark.max(axis=0)
    height_px, width_px = image.shape[:2]
    x_scale, y_scale = page[0] / width_px, page[1] / height_px
    return (
        max(0.0, left * x_scale - margin),
        max(0.0, page[1] - (bottom + 1) * y_scale - margin),
        min(page[0], (right + 1) * x_scale + margin),
        min(page[1], page[1] - top * y_scale + margin),
    )


def render(path: Path) -> np.ndarray | None:
    """여백을 찾기 위해 한 쪽을 그림으로 그린다 (임시 폴더에 떨군 뒤 읽는다)."""
    with tempfile.TemporaryDirectory() as folder:
        prefix = Path(folder) / "page"
        done = subprocess.run(
            ["pdftoppm", "-png", "-r", str(RENDER_DPI), "-singlefile", str(path), str(prefix)],
            capture_output=True,
        )
        rendered = prefix.with_suffix(".png")
        if done.returncode != 0 or not rendered.exists():
            return None
        return cv2.imread(str(rendered), cv2.IMREAD_COLOR)


def keep_source(path: Path) -> None:
    """자르기 전 원본을 `_src/` 에 남긴다 — 이미 있으면 덮지 않는다."""
    backup = path.parent / SOURCE_DIR / path.name
    backup.parent.mkdir(exist_ok=True)
    if not backup.exists():
        shutil.copy2(path, backup)


def write_cropped(path: Path, box) -> None:
    mark = f"[/CropBox [{box[0]:.1f} {box[1]:.1f} {box[2]:.1f} {box[3]:.1f}] /PAGES pdfmark"
    temporary = path.with_suffix(".trim.pdf")
    done = subprocess.run(
        ["gs", "-q", "-o", str(temporary), "-sDEVICE=pdfwrite", "-c", mark, "-f", str(path)],
        capture_output=True,
    )
    if done.returncode != 0:
        sys.exit(f"[trim] ghostscript 실패: {done.stderr.decode()[:200]}")
    temporary.replace(path)


if __name__ == "__main__":
    main()

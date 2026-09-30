"""그림 생성기의 공통 뼈대 (design 15.3절 · 6단계).

## 선별 규칙이 바뀌었다 (교수 미팅 결정 09-15)

옛 규약은 "정량 조건으로 자동으로 고르고 상한 30~40장"이었다. **이제 자동으로 고르지 않는다.**
스크립트는 **후보를 그려 두기만** 하고, 사람이 눈으로 골라 타일 id 를 적으면 그 타일만 조판한다.

    python scripts/figures/figure_07.py --candidates     # 후보 100장을 그린다
    (사람이 figure_07/selected.txt 에 타일 id 를 한 줄에 하나씩 적는다)
    python scripts/figures/figure_07.py                  # 고른 타일만 조판한다

재현성은 그대로다 — **고른 목록이 파일로 남아** 다시 돌려도 같은 그림이 나온다.

## 후보는 두 묶음이다 (사용자 요청 09-29)

**교차로 100장(번호 001~100) + 그 밖의 장면 100장(101~200).** 교차로는 정답 유도선이
`MIN_GUIDING` 개 이상인 타일이다. 한 묶음으로만 뽑으면 선이 적은 장면이 위로 몰려 교차로가
거의 빠지고, 교차로로만 좁히면 평범한 도로가 빠진다. 그림마다 두 묶음의 자격을 따로 둔다.
정량 조건은 **고르기가 아니라 후보를 좁히는 데만** 쓴다(예: "정답 선이 8~25개인 타일").

## 입력은 5단계 산출물이다 — 모델도 GPU도 쓰지 않는다

    paper/pred/{방법}/{split}/{타일}.npz     예측 폴리라인
    paper/score/{방법}_{split}/per_tile.csv  타일별 점수 (후보 순위를 매긴다)

영상과 정답 선만 데이터셋에서 타일 이름으로 다시 읽는다(`scripts/viz_cache.py` 와 같은 방식).

## 렌더링은 한 규칙을 공유한다

예측이든 정답이든, 우리 것이든 비교 대상이든 **같은 함수로 그린다**(`stella/train/viz.py`).
클래스 색 폴리라인 + **양 끝점 흰 테두리 원**. 끝점 원이 있어야
한 선이 몇 조각 났는지 눈으로 세진다.
"""

import csv
import shutil
from pathlib import Path

import cv2
import numpy as np

from stella.builder import build_instance
from stella.data.types import CLASS_NAMES, GridDatasetBase
from stella.eval.instance_io import load_instances
from stella.train import viz

DATA_ROOT = Path("/data/ongoing/2026_stella")
# `scripts/tables/table_base.py` 에도 같은 값이 있다 — scripts/ 는 패키지가 아니다.
PAPER_ROOT = DATA_ROOT / "paper"
FIGURE_ROOT = PAPER_ROOT / "figure"
PRED_ROOT = PAPER_ROOT / "pred"
SCORE_ROOT = PAPER_ROOT / "score"

CAP = 100  # 후보 상한 (사용자 지시) — 이보다 훨씬 많이 통과하면 조건이 헐거운 것이다
GAP = 20  # 패널 사이 여백(px)
GAP_COLOR = (0, 0, 0)
LINE_WIDTH = 2  # 폴리라인 굵기(px) — 지면에서 사라지지 않을 최소치
LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
SELECTED = "selected.txt"
INTERSECTION, OTHER, ANY = "intersection", "other", "any"  # 후보 묶음
GUIDING = CLASS_NAMES.index("guiding_line")
# 교차로로 보는 정답 유도선 수 (사용자 요청 09-29 — 후보에 교차로가 드물었다).
# val 1,218장 중 350장이 이 조건을 넘는다. 이런 타일은 정답 선이 많아(중앙값 56개)
# 그림마다 두던 "정답 선 25개 이하" 상한이 교차로를 거의 다 걸러 내고 있었다.
MIN_GUIDING = 3


class TileImages:
    """타일 이름으로 원본 영상과 정답 선을 읽는다. 캐시에는 영상이 없다."""

    def __init__(self, cfg, split: str):
        self.dataset = build_instance(cfg.data, cfg, base=GridDatasetBase, split=split)
        self.index = {stem: position for position, stem in enumerate(self.dataset.stems)}

    def stems(self) -> list[str]:
        return list(self.index)

    def __call__(self, stem: str) -> tuple[np.ndarray, list[dict]]:
        item = self.dataset[self.index[stem]]
        return item["image"].numpy(), item["instances"]


class MethodLines:
    """5단계가 떨군 예측 폴리라인을 방법별로 읽는다."""

    def __init__(self, split: str):
        self.split = split

    def __call__(self, method: str, stem: str) -> list[dict]:
        path = PRED_ROOT / method / self.split / f"{stem}.npz"
        return load_instances(path) if path.exists() else []


class TileScores:
    """타일별 점수표(`per_tile.csv`)를 방법별로 읽어 둔다 — 후보 순위의 근거다."""

    def __init__(self, split: str, methods: tuple[str, ...]):
        self.table = {method: self._read(method, split) for method in methods}

    def _read(self, method: str, split: str) -> dict[str, dict]:
        path = SCORE_ROOT / f"{method}_{split}" / "per_tile.csv"
        if not path.exists():
            print(f"[figure_base] 점수표 없음: {path}")
            return {}
        with path.open(encoding="utf-8") as stream:
            return {row["tile"]: row for row in csv.DictReader(stream)}

    def value(self, method: str, stem: str, column: str) -> float:
        row = self.table.get(method, {}).get(stem)
        return float(row[column]) if row else 0.0


class SampleFigure:
    """후보를 그리고(사람이 고른 뒤) 고른 것만 조판하는 공통 절차."""

    name = "figure_00"
    split = "val"
    columns: tuple[tuple[str, str], ...] = ()  # (열 제목, 방법 이름 · ""면 원본/GT)
    score_column = "f1"
    pools: tuple[str, ...] = (INTERSECTION, OTHER)  # 묶음마다 `CAP` 장씩, 이 순서로 번호를 잇는다

    def __init__(self, cfg):
        self.images = TileImages(cfg, self.split)
        self.lines = MethodLines(self.split)
        self.scores = TileScores(self.split, tuple(m for _, m in self.columns if m))
        self.guiding: dict[str, int] = {}  # 타일별 정답 유도선 수 — 영상 읽기가 비싸 한 번만 센다

    # --- 후보 만들기 -------------------------------------------------------
    def write_candidates(self) -> None:
        out_dir = fresh(FIGURE_ROOT / self.name / "candidates")
        for index, pool in enumerate(self.pools):
            ranked = self.rank(pool)
            for order, (stem, score) in enumerate(ranked[:CAP], start=index * CAP + 1):
                sheet = self.compose(stem)
                name = f"{order:03d}_{stem}_{score:.3f}.png"
                cv2.imwrite(str(out_dir / name), sheet[:, :, ::-1])
            print(
                f"[{self.name}] {pool}: 훑은 {len(self.images.stems())}장 → 후보 {len(ranked)}장"
                f" → {min(len(ranked), CAP)}장 저장 (번호 {index * CAP + 1}부터)"
            )
        listing = out_dir.parent / SELECTED
        print(f"[{self.name}] 이제 사람이 고른다 — 타일 id 를 {listing} 에 적는다")

    def rank(self, pool: str = ANY) -> list[tuple[str, float]]:
        scored = [(stem, self.score(stem, pool)) for stem in self.images.stems()]
        passed = [(stem, value) for stem, value in scored if value is not None]
        return sorted(passed, key=lambda pair: -pair[1])

    def score(self, stem: str, pool: str):
        """묶음 `pool` 의 후보 자격과 순위 점수. None 이면 뺀다 — 하위 클래스가 정한다."""
        raise NotImplementedError

    def in_pool(self, stem: str, pool: str) -> bool:
        """타일이 묶음에 드는가. 영상까지 읽으므로 싼 조건 뒤에 부른다."""
        if pool == ANY:
            return True
        return self.is_intersection(stem) == (pool == INTERSECTION)

    def is_intersection(self, stem: str) -> bool:
        """정답 유도선이 `MIN_GUIDING` 개 이상인 타일."""
        return self.guiding_count(stem) >= MIN_GUIDING

    def guiding_count(self, stem: str) -> int:
        if stem not in self.guiding:
            truth = self.images(stem)[1]
            self.guiding[stem] = sum(1 for item in truth if item["class"] == GUIDING)
        return self.guiding[stem]

    # --- 고른 것 조판하기 ---------------------------------------------------
    def write_selected(self) -> None:
        listed = self.read_selected()
        if not listed:
            print(f"[{self.name}] {SELECTED} 가 비어 있다 — 먼저 --candidates 로 후보를 만든다")
            return
        out_dir = fresh(FIGURE_ROOT / self.name / "selected")
        for order, stem in enumerate(listed, start=1):
            sheet = self.compose(stem)
            cv2.imwrite(str(out_dir / f"{order:02d}_{stem}.png"), sheet[:, :, ::-1])
        print(f"[{self.name}] 고른 {len(listed)}장 조판: {out_dir}")

    def read_selected(self) -> list[str]:
        path = FIGURE_ROOT / self.name / SELECTED
        if not path.exists():
            return []
        lines = path.read_text(encoding="utf-8").splitlines()
        return [line.strip() for line in lines if line.strip() and not line.startswith("#")]

    # --- 한 타일을 가로 한 줄로 ---------------------------------------------
    def compose(self, stem: str) -> np.ndarray:
        image, truth = self.images(stem)
        panels = [self._panel(image, title, method, truth, stem) for title, method in self.columns]
        return join_row(panels)

    def _panel(self, image, title: str, method: str, truth: list[dict], stem: str) -> np.ndarray:
        if not method:
            plain = title == "원본"
            drawn = image if plain else draw(image, truth)
            return label(viz.to_uint8_image(drawn) if plain else drawn, title)
        return label(draw(image, self.lines(method, stem)), title)


def draw(image: np.ndarray, instances: list[dict]) -> np.ndarray:
    """모든 패널이 같은 규칙으로 그린다 — 굵기 2 px · 끝점 원 반지름 4 px.

    1 px 은 화면에서는 보이지만 2단 지면(3.5 in)으로 줄이면 사라진다(09-16 육안 확인).
    """
    return viz.draw_instances(image, instances, thickness=LINE_WIDTH)


def join_row(panels: list[np.ndarray]) -> np.ndarray:
    """패널을 가로로 잇는다 — 사이는 검은 띠, 원본 해상도 그대로."""
    height = panels[0].shape[0]
    gap = np.full((height, GAP, 3), GAP_COLOR, np.uint8)
    pieces = []
    for index, panel in enumerate(panels):
        if index:
            pieces.append(gap)
        pieces.append(panel)
    return np.concatenate(pieces, axis=1)


def label(panel: np.ndarray, text: str) -> np.ndarray:
    canvas = panel.copy()
    cv2.putText(canvas, text, (12, 34), LABEL_FONT, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(canvas, text, (12, 34), LABEL_FONT, 1.0, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def fresh(path: Path) -> Path:
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    return path


def add_arguments(parser) -> None:
    parser.add_argument("--candidates", action="store_true", help="후보를 그린다(사람이 고를 것)")
    parser.add_argument("--config", default="configs.base")

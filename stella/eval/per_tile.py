"""타일 하나하나의 지표 — 논문 그림이 **어떤 장면을 실을지** 고를 때 쓴다 (5단계).

지금까지의 채점은 전 타일을 합쳐 값 하나를 냈다(`decode/sweep.py`). 그런데 그림 선별 조건은
전부 타일 단위다 — "우리가 분할 방식보다 0.20 이상 앞선 타일", "F1×coverage 사분면",
"GT 인스턴스가 8~25개인 타일". 그래서 **타일마다 채점기를 새로 만들어 한 장만 채점한다.**

채점기 상태가 타일마다 초기화되므로 전체 합계와 섞이지 않는다. 합계는 부르는 쪽이 따로
누적한다(`TileScorer.total`) — 같은 짝을 두 번 계산하지 않기 위해서다.
"""

import csv
from pathlib import Path

from stella.builder import build_instance

# 타일 한 장에서 뽑는 지표. 합계 전용 지표(정밀도·재현율)는 한 장에서는 뜻이 약해 뺐다.
TILE_METRICS = ("f1", "coverage", "correctness", "frag", "rms")
ROW_FIELDS = ("tile", "n_gt", "n_pred", "n_class", *TILE_METRICS)


class TileScorer:
    """타일마다 한 줄을 만들고, 같은 짝으로 전체 합계도 함께 누적한다."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.total = build_instance(cfg.eval, cfg)

    def row(self, tile: str, prediction: list[dict], target: list[dict]) -> dict:
        self.total.update(prediction, target)
        scored = self._score_once(prediction, target)
        return {
            "tile": tile,
            "n_gt": len(target),
            "n_pred": len(prediction),
            "n_class": len({int(item["class"]) for item in target}),
            **scored,
        }

    def _score_once(self, prediction: list[dict], target: list[dict]) -> dict:
        metric = build_instance(self.cfg.eval, self.cfg)
        metric.update(prediction, target)
        computed = metric.compute()
        return {key: float(computed[key]) for key in TILE_METRICS if key in computed}

    def summary(self) -> dict:
        """전체 합계 — 지금까지 쓰던 값과 같은 자다(종류별 지표를 포함한다)."""
        return {key: float(value) for key, value in self.total.compute().items()}


def write_rows(path: Path, rows: list[dict]) -> None:
    """타일별 표를 csv 로 남긴다 — 그림 스크립트가 읽는 기계용 파일이다."""
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(ROW_FIELDS), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

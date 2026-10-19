"""Chamfer 거리 — 손으로 셀 수 있는 장면에서 값이 맞는지."""

import numpy as np
import pytest

from stella.eval.chamfer import ChamferDistance

STEP = 0.5


def line(cls: int, start: tuple, end: tuple) -> dict:
    return {"class": cls, "points": np.array([start, end], dtype=np.float64)}


def build() -> ChamferDistance:
    return ChamferDistance(step=STEP, exclude_classes=(0, 10))


def test_identical_lines_are_zero():
    metric = build()
    metric.update([line(1, (0, 0), (40, 0))], [line(1, (0, 0), (40, 0))])
    assert metric.compute()["chamfer"] == pytest.approx(0.0)


def test_parallel_offset_equals_gap():
    metric = build()
    metric.update([line(1, (0, 3), (40, 3))], [line(1, (0, 0), (40, 0))])
    result = metric.compute()
    assert result["pred_to_gt"] == pytest.approx(3.0)
    assert result["gt_to_pred"] == pytest.approx(3.0)
    assert result["chamfer"] == pytest.approx(3.0)


def test_half_drawn_line_only_hurts_coverage_side():
    """정답의 절반만 그리면 예측→정답은 0, 정답→예측만 커진다 — 두 방향을 따로 두는 이유."""
    metric = build()
    metric.update([line(1, (0, 0), (20, 0))], [line(1, (0, 0), (40, 0))])
    result = metric.compute()
    assert result["pred_to_gt"] == pytest.approx(0.0)
    assert result["gt_to_pred"] == pytest.approx(5.0, abs=0.1)  # 20~40 구간의 평균 거리 10 의 절반


def test_class_is_ignored_but_excluded_classes_are_dropped():
    metric = build()
    target = [line(1, (0, 0), (40, 0)), line(10, (0, 50), (40, 50))]
    metric.update([line(2, (0, 0), (40, 0))], target)
    assert metric.compute()["chamfer"] == pytest.approx(0.0)


def test_empty_side_is_skipped():
    metric = build()
    metric.update([], [line(1, (0, 0), (40, 0))])
    metric.update([line(1, (0, 2), (40, 2))], [line(1, (0, 0), (40, 0))])
    result = metric.compute()
    assert result["skipped"] == 1
    assert result["tiles"] == 1
    assert result["chamfer"] == pytest.approx(2.0)

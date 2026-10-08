"""틀린 예측·놓친 정답의 원인 분류 — 손으로 셀 수 있는 장면에서 이름이 맞는지."""

import numpy as np
import pytest

from stella.config_io import load_config
from stella.eval.error_kinds import ErrorKinds

LANE, CENTER = 1, 2  # 채점하는 종류 둘 (배경 0 · 안전지대 10 이 아니면 된다)


@pytest.fixture(scope="module")
def cfg():
    return load_config("configs.base", [])


def line(cls: int, *points) -> dict:
    return {"class": cls, "points": np.array(points, dtype=np.float64)}


def run(cfg, prediction, target) -> dict:
    kinds = ErrorKinds(cfg)
    kinds.update(prediction, target)
    return kinds.compute()


def test_second_copy_is_duplicate(cfg):
    truth = [line(LANE, (0, 0), (100, 0))]
    result = run(cfg, [line(LANE, (0, 0), (100, 0)), line(LANE, (0, 1), (100, 1))], truth)
    assert result["tp"] == 1
    assert result["fp"]["duplicate"] == 1


def test_short_piece_of_unmatched_line(cfg):
    result = run(cfg, [line(LANE, (0, 0), (30, 0))], [line(LANE, (0, 0), (100, 0))])
    assert result["fp"]["fragment"] == 1
    assert result["fn"]["partial"] == 1


def test_right_place_wrong_class(cfg):
    result = run(cfg, [line(CENTER, (0, 0), (100, 0))], [line(LANE, (0, 0), (100, 0))])
    assert result["fp"]["wrong_class"] == 1
    assert result["fn"]["wrong_class"] == 1


def test_pieces_that_cover_together_are_split(cfg):
    pieces = [line(LANE, (0, 0), (40, 0)), line(LANE, (45, 0), (80, 0))]
    result = run(cfg, pieces, [line(LANE, (0, 0), (100, 0))])
    assert result["fn"]["split"] == 1
    assert result["fp"]["fragment"] == 2


def test_far_lines_are_spurious_by_length(cfg):
    far = [line(LANE, (0, 200), (20, 200)), line(LANE, (0, 300), (150, 300))]
    result = run(cfg, far, [line(LANE, (0, 0), (100, 0))])
    assert result["fp"]["spurious_short"] == 1
    assert result["fp"]["spurious_long"] == 1
    assert result["fn"]["missed"] == 1


def test_line_that_leaves_the_buffer_drifts(cfg):
    leaving = line(LANE, (0, 0), (60, 0), (100, 6))
    result = run(cfg, [leaving], [line(LANE, (0, 0), (100, 0))])
    assert result["fp"]["drift"] == 1
    assert result["fn"]["drifted"] == 1


def test_labels_follow_original_order_and_skip_excluded(cfg):
    prediction = [
        line(10, (0, 50), (40, 50)),  # 안전지대 — 채점 제외
        line(LANE, (0, 300), (150, 300)),
        line(LANE, (0, 0), (100, 0)),
    ]
    labels = ErrorKinds(cfg).label_predictions(prediction, [line(LANE, (0, 0), (100, 0))])
    assert labels == ["excluded", "spurious_long", "tp"]


def test_target_labels_name_missed_lines(cfg):
    target = [line(LANE, (0, 0), (100, 0)), line(LANE, (0, 300), (100, 300))]
    labels = ErrorKinds(cfg).label_targets([line(LANE, (0, 0), (100, 0))], target)
    assert labels == ["tp", "missed"]

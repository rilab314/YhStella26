"""종류별 분해 · 중복 선 비율 (6단계 — Table V · VII 하단).

**종류별로 나눠 채점한 TP·FP·FN 을 더하면 전체 채점과 같아야** 한다 — 매칭이 원래 같은 종류끼리만
일어나기 때문이다. 이것이 깨지면 종류별 표와 주 비교표가 서로 다른 숫자를 말한다.
"""

import numpy as np

from stella.config_io import load_config
from stella.eval.breakdown import ClassBreakdown, RedundancyCounter
from stella.eval.ccq import InstanceCCQ


def line(label: int, y: float, start: float = 0.0, end: float = 60.0) -> dict:
    points = np.array([[start, y], [end, y]], np.float32)
    return {"class": label, "points": points}


def scene() -> tuple[list[dict], list[dict]]:
    target = [line(1, 10.0), line(3, 40.0), line(3, 80.0)]
    prediction = [line(1, 10.5), line(3, 40.0), line(3, 120.0), line(5, 80.0)]
    return prediction, target


def test_class_counts_add_up_to_overall():
    cfg = load_config("configs.base", [])
    prediction, target = scene()
    overall = InstanceCCQ.from_cfg(cfg.eval, cfg)
    overall.update(prediction, target)
    breakdown = ClassBreakdown(cfg)
    breakdown.update(prediction, target)
    for name in ("tp", "fp", "fn"):
        split_sum = sum(float(getattr(m, name).sum()) for m in breakdown.metrics.values())
        assert split_sum == float(getattr(overall, name).sum())


def test_class_summary_counts_instances():
    cfg = load_config("configs.base", [])
    prediction, target = scene()
    breakdown = ClassBreakdown(cfg)
    breakdown.update(prediction, target)
    summary = breakdown.summary()
    assert summary["lane_line"]["n_gt"] == 2 and summary["lane_line"]["n_pred"] == 2
    assert summary["lane_line"]["f1"] == 0.5


def test_merge_equals_single_pass():
    """타일을 나눠 누적하고 더한 값이 한 번에 누적한 값과 같아야 한다."""
    cfg = load_config("configs.base", [])
    prediction, target = scene()
    single = ClassBreakdown(cfg)
    first, second = ClassBreakdown(cfg), ClassBreakdown(cfg)
    for tile in range(2):
        single.update(prediction, target)
        (first if tile == 0 else second).update(prediction, target)
    first.merge(second)
    assert first.summary() == single.summary()


def counter() -> RedundancyCounter:
    return RedundancyCounter(angle_gate=30.0, sample_step=2.0, exclude_classes=(0, 10))


def test_parallel_copy_is_redundant():
    """같은 종류의 선을 3 px 옆에 한 번 더 그리면 두 선 모두 중복이다."""
    redundancy = counter()
    redundancy.update([line(3, 10.0), line(3, 13.0)])
    assert redundancy.summary()["redundant"] == 2


def test_far_or_other_class_is_not_redundant():
    redundancy = counter()
    redundancy.update([line(3, 10.0), line(3, 30.0), line(5, 11.0)])
    assert redundancy.summary()["redundant"] == 0


def test_half_overlap_is_not_redundant():
    """짧은 선이 긴 선 위에 겹치면 짧은 쪽만 중복이다 — 긴 선은 길이의 절반만 겹친다."""
    redundancy = counter()
    redundancy.update([line(3, 10.0, 0.0, 100.0), line(3, 12.0, 0.0, 50.0)])
    assert redundancy.summary()["redundant"] == 1


def test_crossing_lines_are_not_redundant():
    """각도 게이트 — 직각으로 지나는 같은 종류의 선은 중복이 아니다."""
    vertical = {"class": 3, "points": np.array([[30.0, -30.0], [30.0, 30.0]], np.float32)}
    redundancy = counter()
    redundancy.update([line(3, 0.0), vertical])
    assert redundancy.summary()["redundant"] == 0

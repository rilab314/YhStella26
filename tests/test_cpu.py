"""CPU 예산 — 코어·스레드 상한이 config 하나로 정해지는지 검증한다.

부하가 예측 가능해야 학습·D 트랙을 겹쳐 돌릴 수 있다. 이 계약이 깨지면 프로세스마다
코어를 전부 잡아 부하가 튀고, 같은 기계를 쓰는 사람이 불편해진다.
"""

import os

from configs.exp_synthetic import get_config
from stella.builder import build_instance
from stella.runtime.cpu import CpuBudget, parse_cores


def test_parse_cores_expands_ranges_and_lists():
    assert parse_cores("0-3") == [0, 1, 2, 3]
    assert parse_cores("0,2,4") == [0, 2, 4]
    assert parse_cores("0-1,4-5") == [0, 1, 4, 5]
    assert parse_cores("2-3, 0-1") == [0, 1, 2, 3]  # 정렬·중복 제거


def test_reserved_cores_leaves_room_for_the_human():
    """`cores`가 비면 뒤쪽 `reserved_cores`개를 사람 몫으로 남긴다."""
    total = os.cpu_count() or 1
    budget = CpuBudget(torch_threads=2, interop_threads=1, cores="", reserved_cores=2)
    ids = budget.core_ids()
    assert len(ids) == total - 2
    assert max(ids) == total - 3  # 뒤 2개는 건드리지 않는다


def test_explicit_cores_wins_over_reserved():
    budget = CpuBudget(torch_threads=2, interop_threads=1, cores="0-1", reserved_cores=30)
    assert budget.core_ids() == [0, 1]


def test_reserving_everything_falls_back_to_no_pinning():
    """예약이 코어 수 이상이면 고정하지 않는다 — 0코어에 묶어 프로세스를 굶기지 않는다."""
    total = os.cpu_count() or 1
    assert (
        CpuBudget(torch_threads=1, interop_threads=1, cores="", reserved_cores=total).core_ids()
        == []
    )


def test_expected_threads_scales_with_processes():
    """부하를 미리 가늠하는 용도 — arm 수를 곱하면 예상 러너블 스레드가 나온다."""
    budget = CpuBudget(torch_threads=2, interop_threads=1, cores="", reserved_cores=10)
    assert budget.expected_threads(processes=4) == 8


def test_budget_builds_from_config():
    cfg = get_config()
    budget = build_instance(cfg.cpu, cfg)
    assert isinstance(budget, CpuBudget)
    assert budget.torch_threads == cfg.cpu.torch_threads

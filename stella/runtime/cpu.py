"""CPU 예산 — 이 프로세스가 몇 코어·몇 스레드를 쓸지 한 곳에서 정한다.

**왜 필요한가.** torch는 기본적으로 프로세스마다 코어를 **전부** 잡는다(32코어 기계에서
프로세스당 32스레드). 그래서 학습 arm 두셋만 겹쳐도 러너블 스레드가 폭증해 부하가 튀고,
같은 기계를 쓰는 사람의 에디터·터미널이 끊긴다. 실측으로 부하가 5.7과 18.6 사이를 오갔고,
그 진폭 때문에 "지금 여유가 있나"를 한 번 찍어 판단하는 것이 불가능했다.

이 모듈은 그 숫자를 **config 한 곳**으로 모아 부하를 예측 가능하게 만든다.

- `torch_threads` · `interop_threads` — 프로세스당 스레드 상한.
- `cores` / `reserved_cores` — 이 프로세스가 붙을 코어. 나머지는 사람 몫으로 남는다.
  코어 친화도는 **자식 프로세스가 물려받는다**(데이터로더 워커 포함).

부하 상한을 쫓아다니는 대신 **쓸 코어를 미리 떼어 두는 방식**이다. 사람이 쓸 코어가
구조적으로 보장되므로, 학습이 아무리 튀어도 대화형 작업은 영향을 받지 않는다.
"""

import os

import torch

from stella.builder import Buildable


class CpuBudget(Buildable):
    """스레드 수와 코어 친화도를 실제로 거는 객체. `apply()`가 유일한 부작용 지점이다."""

    def __init__(
        self,
        *,
        torch_threads: int,
        interop_threads: int,
        cores: str,
        reserved_cores: int,
    ):
        self.torch_threads = torch_threads
        self.interop_threads = interop_threads
        self.cores = cores
        self.reserved_cores = reserved_cores

    def apply(self) -> dict:
        """예산을 이 프로세스에 건다. 반환값은 로그·기록용 요약이다."""
        core_ids = self.core_ids()
        if core_ids:
            os.sched_setaffinity(0, core_ids)
        self._apply_threads()
        return {
            "cores": len(core_ids) if core_ids else os.cpu_count(),
            "torch_threads": self.torch_threads,
            "interop_threads": self.interop_threads,
            "expected_threads": self.expected_threads(),
        }

    def _apply_threads(self) -> None:
        """torch 스레드 상한. 환경변수는 자식 프로세스(데이터로더 워커)에 물려주려고 함께 쓴다."""
        if self.torch_threads > 0:
            torch.set_num_threads(self.torch_threads)
            os.environ["OMP_NUM_THREADS"] = str(self.torch_threads)
            os.environ["MKL_NUM_THREADS"] = str(self.torch_threads)
        if self.interop_threads > 0:
            _set_interop_threads(self.interop_threads)

    def core_ids(self) -> list[int]:
        """붙을 코어 목록. `cores`가 있으면 그것을, 없으면 뒤쪽을 사람 몫으로 남긴다."""
        if self.cores:
            return parse_cores(self.cores)
        total = os.cpu_count() or 1
        usable = total - self.reserved_cores
        return list(range(usable)) if 0 < usable < total else []

    def expected_threads(self, processes: int = 1) -> int:
        """프로세스 수를 주면 예상 러너블 스레드 수를 낸다 — 부하를 미리 가늠하는 용도."""
        return processes * max(self.torch_threads, 1)


def _set_interop_threads(count: int) -> None:
    """inter-op 스레드는 병렬 작업 시작 후엔 못 바꾼다 — 이미 정해졌으면 조용히 넘어간다."""
    try:
        torch.set_num_interop_threads(count)
    except RuntimeError:
        pass


def parse_cores(spec: str) -> list[int]:
    """`"0-21"` · `"0-3,8-11"` · `"0,1,2"` 를 코어 번호 목록으로 편다."""
    ids: list[int] = []
    for chunk in spec.split(","):
        piece = chunk.strip()
        if not piece:
            continue
        if "-" in piece:
            start, _, end = piece.partition("-")
            ids.extend(range(int(start), int(end) + 1))
        else:
            ids.append(int(piece))
    return sorted(set(ids))

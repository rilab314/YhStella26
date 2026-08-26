"""논문 그림·표 생성의 공통 뼈대 (design 15절).

여기에는 **재사용되는 것만** 둔다 — 경로 규약(`paths`), 그림 베이스(`figure_base`),
표 베이스(`table_base`). 그림·표 하나하나는 진입점이므로 `scripts/paper/`에 있다.

`stella` 안에 두는 이유는 하나다. **저장소는 editable 설치이고 `sys.path` 조작을 하지
않는다**(CLAUDE.md). `scripts/`는 패키지가 아니라 import 할 수 없으므로, 여러 스크립트가
공유하는 코드는 반드시 `stella` 안에 있어야 한다 — `stella/eval/runlog.py`를
`summarize_runs.py`와 `judge_round.py`가 함께 쓰는 것과 같은 구조다.
"""

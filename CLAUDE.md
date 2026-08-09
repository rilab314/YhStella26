# CLAUDE.md

## 프로젝트

위성영상에서 **차선(도로 노면 선형 객체)을 검출하는 딥러닝 모델**을 개발하는 프로젝트다.
입력은 항공/위성 타일 영상(768×768), 출력은 차선 종류별 폴리라인(line string) 인스턴스다.

설계의 핵심은 **토큰 기반 연결성 출력 헤드**다. 셀마다 self 토큰 1개 + 연결 슬롯 R개를 두고,
연결 슬롯이 "이 셀에서 선이 어느 방향으로 이어지는가"를 예측한다. 디코더가 그 방향들을
그래프로 엮어 폴리라인을 만든다. 설계 근거는 `docs/`에 있다 — `docs/design.md`가 색인과
문서 작성 원칙을 담고, 본문은 `structure`·`data`·`model`·`pipeline`·`decisions`로 나뉘며
변경 내역은 `docs/history.md`에만 쌓는다.

## 환경

- `uv` + `pyproject.toml`. Python 3.11, torch 2.6/cu124, PyTorch Lightning.
- `uv sync --extra dev` 로 `.venv` 구성. 실행은 `.venv/bin/python`.
- 저장소는 editable 설치라 `sys.path` 조작이 없다. **절대 import만** 쓴다.

## 구조

```
configs/            schema.py(모든 config dataclass) + base.py + exp_*.py
stella/
  builder.py        resolve / build_instance / check_all — 클래스 선택 단일 관문
  data/             types(출력 계약)·encode(GT 인코더)·synthetic·augment·seedmap
  model/            backbone·neck·heatmap·rope·blocks·heads·stella
  loss/             criterion(조립) + heatmap·self_slot·conn + matching
  decode/graph.py   GraphDecoder — 정점→간선→그래프→폴리라인
  eval/             ccq(인스턴스 F1 지표)·geometry
  train/            module·optim·viz·callbacks·train(진입점)
scripts/            viz_gt.py(GT 육안 확인)·stat_labels.py(라벨 통계)
tests/              pytest — 인코더 불변식·매칭·RoPE·디코더·지표·config 해석
```

## 조립 규칙 (엄격히 지킨다)

1. **`__init__`은 config를 모른다.** keyword-only, 기본값 없음 — 기본값의 단일 출처는 config다.
2. **클래스 선택은 `build_instance` 한 곳에서만.** config의 `path`+`name` 문자열을 보고 찾는다.
   다른 어디서도 `importlib`을 직접 부르지 않는다.
3. `from_cfg(module_cfg, cfg, **kwargs)` — 시그니처가 전부 같다. `kwargs`는 config에 없는
   런타임 값만 나른다.
4. **최상위 배선은 `stella/train/train.py` 한 곳.** 진입 직후 `check_all(cfg)`로 오타를 먼저 잡는다.

## 좌표 규약

- 셀 인덱스만 `(i, j) = (행, 열)`, **그 외 모든 2차원 벡터는 `(x, y)` 순서**.
- 셀 내 좌표(`coord_map`)의 원점 = 셀 좌상단, 연결 방향의 원점 = 셀 중심.
- 인코더는 픽셀의 **면적 중심(+0.5)** 을 쓴다. 디코더는 출력 직전에 `-0.5` 해서 라벨 좌표계로 돌린다.

## 실행

```bash
.venv/bin/python -m pytest tests/ -q                      # 전체 테스트 (slow 제외하려면 -m "not slow")
.venv/bin/python -m ruff check . && .venv/bin/python -m ruff format .
.venv/bin/python scripts/viz_gt.py --config configs.base --split val --count 4 --out /tmp/gt
.venv/bin/python scripts/stat_labels.py --split train --workers 10
.venv/bin/python -m stella.train.train --config configs.base --tag myrun \
    --override train.epochs=25 data.batch_size=1
```

- 데이터셋: `/media/humpback/.../Ongoing/2026_stella/SEED_MAP_v1.1`
  (평평한 `image/`·`label/` + `dataset.json`으로 split을 표현한다. split 폴더 구조가 아니다.)
- 학습 결과: 데이터셋 옆 `.../2026_stella/log/{YYMMDD_HHMMSS}_{config}_{tag}/`
  — `config.json`, 소스 전체 복사본 `src/`, `git_sha.txt`, `checkpoints/`, `metrics.csv`, `viz/`.

## 개선 실험 (진행 중인 작업)

지금 이 저장소의 주 작업은 **성능 개선 루프**다. 실험·성능 관련 요청을 받으면
**먼저 `experiment/STATE.md`를 읽는다** — 현재 무엇이 돌고 있고 다음 행동이 무엇인지가
거기 한 파일에 있다. 방법·규약은 `.claude/skills/improve-loop/SKILL.md`.

- **`/stella`** 한 마디로 루프가 한 바퀴 돈다. 반복은 **`/loop /stella`** (간격 생략).
- 판정은 `scripts/judge_round.py`, 무인 배정은 `scripts/dispatch.py`가 한다.
- **병합 전에 `scripts/gate.py`를 통과시킨다.** 게이트가 못 잡았을 실패를 발견하면
  그 자리에서 `gate_baseline.json`에 검사를 추가한다 (SKILL 16절).
- 학습이 도는 중에는 게이트·D 트랙을 얹지 않는다 — CPU 부하 상한 16.

## 작성 규칙

- **코드를 새로 쓰거나 고칠 때는 `coding` 스킬(`.claude/skills/coding/SKILL.md`)을 먼저 읽고 그 규칙을 따른다.**
  OOP·단일 책임, 길이 제한(모듈 400 / 클래스 200 / 함수 30줄, 한 줄 100자), 명사·동사 네이밍,
  호출 트리 깊이우선 순서 배치, 상수 분리, numpy 벡터화가 핵심이다.
- 대화·문서·주석은 한국어로 쓴다.
- **설계 문서를 고칠 때는 `docs/design.md`의 작성 원칙을 먼저 읽는다** — 본문에는 수정 이력을
  남기지 않고 현재 상태만 쓰며, 변경 내역은 `docs/history.md`에 날짜순으로 요약한다.
- **설계 문서의 값과 실측이 다르면 실측을 따르고, config에 근거를 주석으로 남긴다.**
  `n_max`·`max_conn_dist`·`w_dist`가 그렇게 정해졌다.

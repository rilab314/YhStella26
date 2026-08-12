# CLAUDE.md

## 프로젝트

위성영상에서 **차선(도로 노면 선형 객체)을 검출하는 딥러닝 모델**을 개발하는 프로젝트다.
입력은 항공/위성 타일 영상(768×768), 출력은 차선 종류별 폴리라인(line string) 인스턴스다.

설계의 핵심은 **토큰 기반 연결성 출력 헤드**다. 셀마다 self 토큰 1개 + 연결 슬롯 R개를 두고,
연결 슬롯이 "이 셀에서 선이 어느 방향으로 이어지는가"를 예측한다. 디코더가 그 방향들을
**사슬로 확장**해 폴리라인을 만든다 (선 하나 = 사슬 하나 — 인코딩과 디코딩이 같은 모양이다).
설계 근거는 `docs/`에 있다 — `docs/design.md`가 색인과 문서 작성 원칙을 담고, 본문은
`structure`·`data`·`model`·`pipeline`·`decisions`로 나뉘며 변경 내역은 `docs/history.md`에만 쌓는다.

**파이프라인 구현은 끝났고 지금은 개선(실험) 단계다.** 아래 "개선 실험" 절을 먼저 본다.

## 환경

- `uv` + `pyproject.toml`. Python 3.11, torch 2.6/cu124, PyTorch Lightning.
- `uv sync --extra dev` 로 `.venv` 구성. 실행은 `.venv/bin/python`.
- 저장소는 editable 설치라 `sys.path` 조작이 없다. **절대 import만** 쓴다.
  (editable 설치가 worktree를 가리키면 `python scripts/*.py`가 남의 코드로 돈다 —
  `scripts/gate.py`의 `install` 검사가 그 사고를 막는다.)
- GPU는 RTX 4090 4장, CPU 32코어. **CPU 예산은 `cfg.cpu`가 단일 출처**다 —
  학습은 코어 `0-21`만 쓰고 `22-31`은 사람 몫으로 비워 둔다 (`stella/runtime/cpu.py`).

## 구조

```
configs/            schema.py(모든 config dataclass) + base.py(F 규격) + unit.py(U 규격)
                    + exp_*.py(변형) + unit_dinov3.py
stella/
  builder.py        resolve / build_instance / check_all / Buildable — 클래스 선택 단일 관문
  config_io.py      config 로드·점 경로 덮어쓰기 — 학습 진입점과 스크립트가 공유
  runtime/cpu.py    CpuBudget — 스레드 수·코어 친화도를 실제로 거는 유일한 지점
  data/             types(출력 계약)·encode(GT 인코더)·seedmap(실데이터)·synthetic·augment
  model/            backbone·neck·heatmap·rope·blocks·heads·stella
                    inject.py — GT를 모델 출력 형태로 주입("완벽한 예측") = 파이프라인 천장
  loss/             criterion(조립) + heatmap·self_slot·conn + matching
  decode/           graph.py(ChainDecoder — 사슬 확장 본체)·vertices(정점 추출·시드)
                    postprocess(조각 병합·RDP)·cache(희소 예측 캐시)·sweep(스윕 코어)
                    stats(정지 사유 카운터 — end/nocand/exist/slotused)
  eval/             ccq(인스턴스 F1)·geometry·cellstat(셀 단위 진단)·runlog(metrics.csv 판독)
  train/            module·optim·viz·callbacks·train(진입점)
scripts/            아래 "실행" 참고 — 데이터 확인 · 학습 운영 · 디코더(D) 트랙 · 진단
tests/              pytest — 인코더 불변식·매칭·RoPE·디코더·후처리·지표·셀진단·CPU예산·config 해석
experiment/         STATE.md(냉시동 진입점) · plan_MMDD.md · result_MMDD.md · queue.json · data/
gate_baseline.json  PR 전 게이트의 검사 목록과 하한 — 코드가 아니라 데이터다
```

## 조립 규칙 (엄격히 지킨다)

1. **`__init__`은 config를 모른다.** keyword-only, 기본값 없음 — 기본값의 단일 출처는 config다.
2. **클래스 선택은 `build_instance` 한 곳에서만.** config의 `path`+`name` 문자열을 보고 찾는다.
   다른 어디서도 `importlib`을 직접 부르지 않는다.
3. `from_cfg(module_cfg, cfg, **kwargs)` — 시그니처가 전부 같다. `kwargs`는 config에 없는
   런타임 값만 나른다. 이름이 그대로 대응하면 `Buildable`을 상속해 기본 구현을 쓴다.
4. **최상위 배선은 `stella/train/train.py` 한 곳.** 진입 직후 `check_all(cfg)`로 오타를 먼저 잡고,
   그 다음 `cfg.cpu`로 코어·스레드 상한을 건다.
5. `configs/schema.py`는 `stella`를 import 하지 않는다 — config는 순수한 데이터다.

## 좌표 규약

- 셀 인덱스만 `(i, j) = (행, 열)`, **그 외 모든 2차원 벡터는 `(x, y)` 순서**.
- 격자는 stride 4 — 768 / 4 = **L = 192**.
- 셀 내 좌표(`coord_map`)의 원점 = 셀 좌상단, 연결 방향의 원점 = 셀 중심.
- 인코더는 픽셀의 **면적 중심(+0.5)** 을 쓴다. 디코더는 출력 직전에 `-0.5` 해서 라벨 좌표계로 돌린다.

## 실행

```bash
.venv/bin/python -m pytest tests/ -q                      # 전체 테스트 (가벼운 것만: -m "not slow")
.venv/bin/python -m ruff check . && .venv/bin/python -m ruff format .
.venv/bin/python -m stella.train.train --config configs.base --tag myrun \
    --override train.epochs=25 data.batch_size=1
```

- 데이터셋: `/media/humpback/435806fd-.../Ongoing/2026_stella/SEED_MAP_v1.1_splits`
  — **`{train,val,test}/{image,label}` 구조**다. `label/*.json`을 glob 해서 인덱스를 만들므로
  `dataset.json` 파싱이 없다. train 8,979 / val 1,282 / test 2,567장.
  (원본 평평한 `SEED_MAP_v1.1`은 그 옆에 남아 있지만 코드가 읽지 않는다.)
- GT 캐시 `.../2026_stella/gt_cache/{split}/` · 예측 캐시 `.../pred_cache/`.
- 학습 결과: `.../2026_stella/log/{YYMMDD_HHMMSS}_{config}_{tag}/`
  — `config.json`, 소스 전체 복사본 `src/`, `git_sha.txt`, `checkpoints/`, `metrics.csv`, `viz/`.

### scripts — 무엇이 어디에 있나

| 갈래 | 스크립트 |
| --- | --- |
| 데이터 확인 | `viz_gt.py`(GT 육안) · `stat_labels.py`(라벨 통계) |
| 실험 운영 | `run_experiments.py`(arm을 GPU에 배치) · `dispatch.py`(무인 배정) · `judge_round.py`(판정) · `gate.py`(PR 전 관문) |
| 결과 판독 | `show_run.py`(실행 하나) · `summarize_runs.py`(여러 실행 비교표) |
| 디코더(D) 트랙 | `dump_predictions.py`(예측→희소 캐시) · `eval_decode.py`(CPU만으로 스윕) · `tune_decoder.py`(좌표 하강) |
| 진단 | `loss_balance.py`(손실 균형·가중치 제안) · `class_confusion.py`("배경이라 부름" vs "종류 혼동") |

```bash
.venv/bin/python scripts/viz_gt.py --config configs.base --split val --count 4 --out /tmp/gt
.venv/bin/python scripts/summarize_runs.py --last 8
.venv/bin/python scripts/gate.py                      # 학습 중이면 --workers 2
.venv/bin/python scripts/judge_round.py --round E08 --control U1_ref
```

**D 트랙이 요점이다** — 예측을 한 번 캐시로 떨궈 두면 **GPU 없이** 디코더 파라미터를 스윕할 수
있다. 학습이 GPU를 다 쓰는 동안에도 디코더 실험은 계속 돈다.

## 개선 실험 (진행 중인 작업)

지금 이 저장소의 주 작업은 **성능 개선 루프**다. 실험·성능 관련 요청을 받으면
**먼저 `experiment/STATE.md`를 읽는다** — 현재 무엇이 돌고 있고 다음 행동이 무엇인지가
거기 한 파일에 있다. 방법·규약은 `.claude/skills/improve-loop/SKILL.md`.

- **`/stella`** 한 마디로 루프가 한 바퀴 돈다. 반복은 **`/loop /stella`** (간격 생략).
- 실험 규격은 둘 — **U**(`configs/unit.py`: 3,000장·10에폭·arm당 1 GPU, 가설 비교용)와
  **F**(`configs/base.py`: 전체 데이터). **규격이 다른 실행의 절대값을 비교하지 않는다.**
- 판정은 `scripts/judge_round.py`, 무인 배정은 `scripts/dispatch.py`(대기열 `experiment/queue.json`).
- **병합 전에 `scripts/gate.py`를 통과시킨다.** 게이트가 못 잡았을 실패를 발견하면
  그 자리에서 `gate_baseline.json`에 검사를 추가한다 (SKILL 16절).
- 학습이 도는 중에는 게이트·D 트랙을 얹지 않는다 — CPU 부하 상한 16.
- **학습을 띄우기 전에 설계안(`experiment/plan_MMDD.md`)을 내고 승인받는다** (사용자 지시).
- **PR을 올릴 때는 `report` 스킬(`.claude/skills/report/SKILL.md`)을 먼저 읽는다.**
  사용자는 대화가 아니라 **PR로 경과를 읽으므로**, PR 본문은 그 자체로 완결된 보고서여야
  한다 — 용어를 전부 풀어 쓰고(`glossary.md`), 조건이 무엇을 바꿨는지 밝히고, 표에 열
  설명을 달고, 다음 실험 계획까지 넣는다.

### 반복해서 당한 함정 (상세는 `experiment/STATE.md`)

- **기본값을 바꾸면 그 시점을 기준으로 실행이 두 집단으로 갈린다.** 실행 폴더가 시작 시 config를
  고정하므로 이후 실행만 새 설정이다. 비교 전에 **캐시를 떠서 같은 조건에서 다시 잰다.**
- **새 필드를 만들면 그 필드가 지나는 모든 경계를 훑는다** — 캐시·config·시각화·판정.
- **파라미터를 고치면 그것이 가리던 축이 되살아난다** — 옛 진단 수치를 근거로 축을 배제하지 않는다.

## 작성 규칙

- **코드를 새로 쓰거나 고칠 때는 `coding` 스킬(`.claude/skills/coding/SKILL.md`)을 먼저 읽고 그 규칙을 따른다.**
  OOP·단일 책임, 길이 제한(모듈 400 / 클래스 200 / 함수 30줄, 한 줄 100자), 명사·동사 네이밍,
  호출 트리 깊이우선 순서 배치, 상수 분리, numpy 벡터화가 핵심이다.
- 대화·문서·주석은 한국어로 쓴다.
- **설계 문서를 고칠 때는 `docs/design.md`의 작성 원칙을 먼저 읽는다** — 본문에는 수정 이력을
  남기지 않고 현재 상태만 쓰며, 변경 내역은 `docs/history.md`에 날짜순으로 요약한다.
- **설계 문서의 값과 실측이 다르면 실측을 따르고, config에 근거를 주석으로 남긴다.**
  `n_max`(실측 최대 8,909 → 9,500) · `decode.radius`(2 → 24, f1 +65%) ·
  `focal_alpha`(0.25 → 0.75, f1 +22%) · **`decode.w_dist`(0.001 → 0.03, f1 +33%)** 가
  그렇게 정해졌다. 마지막 것은 문서가 "폐기"로 적어 둔 값이었다 — **설계 문서의 기각 판정도
  그때의 설정에 한한다.**

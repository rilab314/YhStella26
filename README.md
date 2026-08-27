# STELLA — 위성영상 차선 검출

위성/항공 타일 영상에서 차선(도로 노면 선형 객체)을 **폴리라인 인스턴스**로 검출한다.

셀마다 self 토큰 1개와 연결 슬롯 R개를 두고, 연결 슬롯이 "이 셀에서 선이 어느 방향으로
이어지는가"를 단위 방향 벡터로 예측한다. GT는 선(인스턴스)마다 독립된 사슬로 인코딩되고
(셀당 분기 2개), 디코더는 클래스 확률 국소 피크에서 마주봄 확인으로 사슬을 한 노드씩
확장하며 폴리라인을 복원한다.

## 설치

```bash
uv sync --extra dev          # 새로 만들 때
uv sync --extra dev --frozen # 잠금 파일 그대로 복원할 때 (재해석 금지)
```

Python 3.11 / torch 2.6 (cu124) / PyTorch Lightning. 저장소는 editable 설치된다.

**환경은 세 파일이 함께 고정한다** — `.python-version`(인터프리터 3.11) · `pyproject.toml`
(직접 의존성과 cu124 인덱스) · `uv.lock`(전 패키지 정확한 버전). 셋 다 커밋돼 있으므로
`--frozen`으로 동기하면 같은 환경이 그대로 복원된다. `uv.lock`이 pyproject와 어긋났는지는
`uv lock --check`로 확인한다.

> editable 설치가 **다른 worktree**를 가리키면 `python scripts/*.py`가 남의 코드로 돈다.
> `scripts/gate.py`의 `install` 검사가 그 사고를 막는다 — 환경을 새로 만든 뒤 한 번 돌려 본다.

## 빠른 확인

```bash
.venv/bin/python -m pytest tests/ -q                # 인코더 불변식·매칭·RoPE·디코더·지표
.venv/bin/python scripts/viz_gt.py --config configs.exp_synthetic --count 4 --out /tmp/gt
```

## 학습

```bash
.venv/bin/python -m stella.train.train --config configs.base --tag myrun
```

- `--config` : `configs/` 아래 모듈 이름 (`configs.base`, `configs.exp_synthetic`, ...)
- `--override data.batch_size=2 train.epochs=40` : 점 경로로 config 필드를 덮어쓴다
- `--resume <ckpt>` : 체크포인트에서 이어서 학습

출력은 `{train.output_root}/{YYMMDD_HHMMSS}_{config}_{tag}/`에 실행마다 새로 만들어진다.
`config.json` · 소스 전체 복사본 `src/` · `git_sha.txt` · `checkpoints/` · `metrics.csv` · `viz/`.

## 데이터

SEED-MAP v1.2 (768×768, GSD 0.2550 m/px). `SEED_MAP_v1.2_splits/{train,val,test}/{image,label}`
재정리 사본(`scripts/build_split_dataset.py`)을 읽고, `label/{id}.json`이 `LINE_STRING`
폴리라인을 담는다. train 8,979 / val 1,218 / test 2,457장.
학습 클래스는 `category_id` 11종 + 배경.

라벨은 **폴리라인 그대로** 두고 `__getitem__`에서 격자 GT로 온라인 인코딩한다.
증강은 인코딩 전 벡터 단계에서 하므로 좌표 뒤집기 해킹이 필요 없다.

합성 데이터셋(`configs.exp_synthetic`)으로 전 파이프라인을 실데이터 없이 돌려볼 수 있다.

## 문서

설계 근거와 결정 사항은 `docs/`에 있다 — `docs/0_design.md`가 색인이다. 코드 규약은 `CLAUDE.md`에 있다.

# STELLA 재구현 계획 (impl_plan.md)

[architecture.md](architecture.md)의 **토큰 기반 연결성 출력 헤드** 설계를 코드로 옮기기 위한 구현 계획이다.
기존 저장소(STELLA2026)를 고치는 것이 아니라 **새 저장소를 만들어 처음부터 다시 구현**한다.

- 범위: **학습 루프부터 디코딩(10절)·인스턴스 평가(11절)까지.** 로깅(9.4)·시각 로그(9.5)도 포함한다. (초기 범위는 "학습이 도는 것까지"였고 6·8차 개정에서 확장했다.)
- 데이터셋: **SEED-MAP v1.1**(6.7절). 샘플 300장으로 구조·통계를 확정했고 전체 데이터는 수령 대기 중이다. 개발 초반에는 출력 계약(6절)을 따르는 **합성 데이터셋**으로 전체 파이프라인을 먼저 검증한다.
- **이번 개정(10차, 2026-08-07) — target을 모델 출력 형태에 맞춘다 + 디코더 시드 전략 변경.**
  ① 구 설계 방침 "GT는 모델 출력 형식을 흉내 내지 않는다"를 **뒤집었다** — 분기가 항상 2로 고정되면서
  "사실 저장 + criterion 유도"의 간접층이 더는 값을 못 한다. GT가 **연결 방향 2개를 직접 저장**한다
  (`conn_cells`·`end_point` 폐지 → `conn_dirs`, 자기 점 → 이웃 점 단위벡터, 6.2절). criterion은
  유도 없이 저장된 방향과 매칭만 한다(8.3절). ② 디코더 시드를 끝 셀에서 **클래스 확률 국소 피크**로
  바꾸고 거기서 **양방향 확장**한다. 확장 게이트에 **사슬 클래스 확률 하한**(`min_class_prob`)을
  추가하고, 완성된 사슬은 **순도 검사**(시드 클래스 일치 60% 초과)를 통과해야 남는다(10.3절).
  ③ **연결 슬롯 $R = 2$로 고정** — GT 분기 수와 일치시킨다. Y자 분기도 "직진성이 강한 주도선 +
  옆으로 빠지는 보조선" 두 인스턴스로 찾으면 되므로 셀 하나가 세 방향을 낼 필요가 없다.
  $R = D = 2$가 되면서 매칭 순열이 2개로 줄고, 존재 항이 배정에서 무효가 되고, 양성 셀의 모든
  슬롯이 매칭된다(8.3~8.4절).
- 개정(9차, 2026-08-07) — 인코딩·디코딩 재설계: "선 하나 = 사슬 하나".** 재구현(M0~M9 완료, `feat/reimplement`) 후 확인된 구조적 조각남(GT 주입에도 F1 상한 0.63~0.69)의 뿌리가 **인코딩(클래스 단위 그래프)과 디코딩(그래프 절단)의 비대칭**임을 실측으로 확인하고 둘을 같은 모양으로 다시 설계했다. ① GT를 **선(인스턴스) 단위 사슬**로 인코딩 — 셀당 분기 항상 2, 끝칸 미채움, 3×3 순위 규칙으로 지그재그·고립 노드 제거(6.4절). ② 연결 방향의 원점을 셀 중심에서 **자기 노드 점**으로 변경(6.1·6.2절). ③ $t$ 슬롯 폐기, **`end_map` 직접 감독**(7.7·8.2절). ④ 디코더를 그래프 절단에서 **단방향 사슬 확장**(마주봄 확인 $\mathbf{c}\cdot\mathbf{n} \to -1$, 반경 2)으로 교체(10절). ⑤ 재구현 실측 반영 — `n_max` 9500, `window_size` 7 + 윈도우 층 gradient checkpointing(OOM 해소), 전체 train 8,979장 통계, `category_id` 표 확정, split 폴더 재정리, `switch_rate` → `match_ambiguity`. 남은 의문은 **14절**.
- 개정(8차) — **평가 지표 신설(11절).** 얇은 선에 맞는 **커버리지 중심 인스턴스 F1**을 3축으로 설계. 주 지표는 **비대칭 버퍼 CCQ**(커버리지 $\ge 0.5$ 관대 / 정확성 $\ge 0.9$ 엄격, 정확성은 특정 GT가 아니라 **모든 GT** 대상), 보조로 집계 커버리지·RMS 횡오차·GT당 조각 수. 주 지표는 클래스별 + micro/macro로 산출. FP를 **중복/환각**으로 분해. Chamfer 매칭·confidence AP·마스크 IoU·양쪽 버퍼 IoU를 쓰지 않는 이유 정리(11.3). `MetricConfig` 추가·`val/inst/*` 로깅(9.4)·평가 지표 설계 완료. **마일스톤 재정렬** — 디코더·평가 지표를 학습 루프보다 앞으로(M6 디코더 / M7 지표 / M8 학습 루프 / M9 실데이터).
- 개정(5차) — **실데이터 파이프라인 추가(6.7절).** SEED-MAP 라벨 JSON 구조 분석, 학습 클래스 확정(`category_id` 11종 + 배경, 논문 Table V 일치), `LINE_STRING`만 사용, **타일 경계 자르기**(전체 점의 49.8%가 이미지 밖), 실측 통계로 설계값 검증 — 차수 분포가 $R = 3$을 뒷받침하고 `n_max`는 3000 → 8000으로 상향.
- 개정(4차) — **조립 방식 확정.** ① 클래스 선택을 config의 `path` + `name` 문자열과 **단일 관문 `build_instance`**로 확정(5절). 하위클래스 자동 registry 방식은 폐기 — config가 코드를 import 하지 않아 순환참조가 구조적으로 불가능하고, 실제로 쓰는 클래스만 지연 import된다. ② `from_cfg(module_cfg, cfg, **kwargs)` — **자기 섹션 config와 전체 cfg를 함께** 받는다. ③ **`check_all(cfg)`** 로 무거운 초기화 전에 모든 클래스 참조를 미리 검증. ④ 부품 config를 `ModuleConfig` 상속 + `kw_only=True`로 통일(4.1절). ⑤ backbone을 함수에서 **클래스**로 전환(7.2절). ⑥ 실행 재현은 **소스 전체 복사**로 확정(4.3절).
- 개정(6차) — **손실 모듈화·로깅·객체 생성.** ① shape 기호 정리: `B`는 **배치 전용**, 연결 슬롯 수는 **`R`**(6.1절 기호표). ② 백본을 **3층 구조**로 — 베이스 / 라이브러리별 중간 인터페이스 / **모델 계열별 클래스**, 계열 안의 스케일은 한 클래스가 처리(7.2절). ③ `SFP`·`FPNLite` 내부 구조를 층 단위로 명시(7.3절). ④ 어텐션의 key/value가 **선택된 노드 임베딩**임을 근거와 함께 확정(7.6절). ⑤ 손실을 **`HeatmapLoss`·`SelfSlotLoss`·`ConnLoss` 3모듈 + `StellaCriterion` 조립**으로 분해. 가중치는 **최하위 손실 항목마다 하나씩 단일 층**(6개)으로 두고 모듈 단위 상위 가중치는 두지 않는다(8.0절). ⑥ **로깅(9.4)·시각 로그(9.5)** 신설 — 에폭 단위, 손실 dict 전체 기록, 검증에서 heat/class/slot 3종 PNG. ⑦ **10절 객체 생성(디코딩) 신설** — 4단계 알고리즘, 매 에폭 검증에서 실행되어 인스턴스 평가로 이어진다. 디코더가 필수 마일스톤으로 승격, 평가 지표 항목 추가.
- 개정(7차) — **GT 인코더 재설계 + 남은 확인 정리.** ① GT 노드를 **인스턴스 단위 대표점에서 픽셀 무게중심으로** 바꿈(6.4절) — 클래스별로 선을 그리고 셀 안 픽셀의 무게중심을 좌표로, 픽셀이 많은 클래스를 셀 소유자로 쓴다. **이중선을 두 인스턴스로 라벨링한 실데이터**에서 노드가 갈라지지 않는다. ② 연결성을 **간선 합집합**으로 유도하고 `end_map`을 **차수 1**로 유도 — 이중선·Y자 분기가 별도 규칙 없이 처리된다. ③ **GT 캐시 정책**(6.4.1절) — 온라인 기본, val/test는 무조건 캐시, 학습 캐시는 실측 후 판단. ④ 디코딩 간선 비용에 **클래스 불일치 항** 추가(종점 슬롯은 면제, 10.3절). ⑤ **종점 셀 클래스 손실 제외 / 거짓 양성 셀 배경 CE 추가**(8.2절)와 그에 따른 디코더 다수결 클래스(10.5절). ⑥ 남은 확인 9건 → 5건으로 정리.
- 3차 개정: ① GT를 **self 정보 맵(class/coord/end) + 연결 이웃 셀 좌표**로 단순화 — 연결 슬롯 형태의 GT(방향·t 텐서)는 제거하고 criterion이 유도. ② 종점 셀은 연결 예측을 내지 않는 것으로 규약화. ③ 학습 노드 선택 = **GT 양성 ∪ 예측 히트맵 마스크**. ④ 매칭 비용을 방향 정렬 + 존재 확률로 단순화. ⑤ 결정사항 8건 확정 반영(13절).

---

## 1. 설계 원칙 — 기존 단점을 이렇게 극복한다

| #   | 기존 문제 (STELLA2026)                                              | 새 저장소의 해법                                                                                                                                                                                                    |
| --- | --------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1   | `build_from_cfg`가 **cfg 전체**를 받아 실제 의존성이 시그니처에 안 보임             | 문제의 뿌리는 `__init__`이 cfg를 받은 것이었다. **`__init__`은 config를 모른다** — keyword-only·기본값 없는 typed 인자만 받으므로 의존성이 시그니처에 전부 드러난다. `from_cfg`는 그 인자를 채우는 얇은 어댑터일 뿐이라 cfg 전체를 봐도 무해하다 (5절)                                |
| 2   | `importlib` 문자열 기반 동적 import — IDE 추적·타입 검사 불가                  | `importlib`은 유지하되(config가 코드에 의존하지 않아 순환참조가 불가능하고, 쓰는 클래스만 지연 import된다) 과거 문제였던 **"흩어진 호출 + 검증 없음"** 을 없앤다: `builder.py` **단일 관문**, 해석 시 4중 검증, 학습 전 `check_all(cfg)` 사전 검사, CI의 `test_config_resolves` (5절) |
| 3   | `LightningModel.build_from_cfg`의 postprocessors 미전달 같은 조립 누락 버그 | 최상위 배선이 `train.py` 한 화면에 모이고, **"모든 config로 전부 build되는지" 스모크 테스트**를 CI에 둔다                                                                                                                                   |
| 4   | dict 기반 `CfgNode` — 오타가 실행 중간에야 `AttributeError`로 터짐            | **typed dataclass config.** 오타는 로드 즉시(또는 타입 체커에서) 잡힌다                                                                                                                                                        |
| 5   | `settings.py`의 `sys.path` 해킹                                    | 정식 패키지(`pyproject.toml` + editable install, 2절). `sys.path` 조작을 코드가 아니라 **설치가** 대신한다. 어느 cwd·실행 방식(pytest·DDP 자식 프로세스·IDE)에서도 동일하게 해석되고, 같은 파일이 두 모듈로 중복 로드되는 사고가 구조적으로 없다. **절대 import만** 사용                |
| 6   | `model/ops`의 커스텀 CUDA 빌드(ms_deform_attn)                        | 새 모델은 deformable attention을 쓰지 않는다. **순수 PyTorch(SDPA)** 만으로 구현 — 빌드 단계 자체가 없다                                                                                                                               |
| 7   | 백본마다 래퍼 클래스 + forward hook                                      | **모델마다** 클래스를 만들지 않는다. **라이브러리별 어댑터 두 개**(`Dinov3`·`TimmBackbone`)가 전부다. forward hook 없이 각 라이브러리의 공식 특징 추출 경로만 쓴다 (7.2절)                                                                                     |
| 8   | 9채널 dense 라벨을 npy로 사전 생성 — 증강이 부호 뒤집기 해킹으로 제한됨                  | 라벨을 **벡터(폴리라인) 그대로 저장**하고 `__getitem__`에서 **온라인 인코딩.** 증강은 벡터 단계에서 수행                                                                                                                                        |
| 9   | left/right 용어 혼란, 단위벡터 vs 오프셋 불일치(TODO)                         | 슬롯은 무순서(매칭으로 배정)라 prev/next가 없다. 연결은 GT도 예측도 **단위 방향 벡터 하나**로 같은 형태다(6.2절, 10차 개정) — 표현이 하나뿐이라 불일치가 생길 자리가 없다                                                                                               |
| 10  | 학습·손실에서 GT 노드와 예측을 인덱스로 대응시키는 복잡성                               | GT와 모델 출력을 **같은 $(L, L)$ 격자·같은 형태**(방향 벡터)에 둔다. 같은 셀 = 같은 인덱스라 정렬 문제가 없고, 남는 일은 셀 안 슬롯 배정(매칭, 8.3절)뿐이다                                                                                                       |

**조립 규칙 (엄격히 지킨다):**

1. **`from_cfg(cls, module_cfg, cfg, **kwargs)`.** 시그니처가 전부 같다. `module_cfg`는 자기 섹션 config,
   `cfg`는 전체 `ExperimentConfig`. 다른 섹션 값(`cfg.data.num_classes` 등)은 `cfg`에서 직접 읽는다 —
   공유 값은 여전히 `DataConfig` 한 곳에만 있고, 중간 계층이 named parameter로 릴레이하지 않는다.
   `kwargs`는 **config에 없는 값만** 나른다: 런타임 산출물(`in_channels=backbone.out_channels`)이나
   호출 지점의 선택(`split="train"`).
2. **`__init__`은 config를 모른다.** keyword-only 인자만 받고 **기본값을 두지 않는다** — 기본값의 단일 출처는
   config dataclass다. config 없이도 생성·테스트할 수 있고, 인자가 하나라도 빠지면 즉시 `TypeError`가 난다.
3. **클래스 선택은 `build_instance` 한 곳에서만** 한다(5절). config에는 `path`(모듈 경로)와 `name`(클래스 이름)
   두 문자열이 있고, 다른 어디서도 `importlib`을 직접 부르지 않는다.
4. **최상위 배선은 `train.py` 한 곳**에 둔다. 진입 직후 `check_all(cfg)`를 불러 **무거운 초기화 전에**
   모든 클래스 참조를 검증한다.

---

## 2. 기술 스택

| 항목       | 선택                                      | 비고                                                                                                                                                                                                                   |
| -------- | --------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Python   | 3.11+                                   |                                                                                                                                                                                                                      |
| 패키지 관리   | `uv` + `pyproject.toml`                 | `uv.lock`으로 재현성 확보(`uv sync --frozen`). 저장소는 **editable 설치**(원칙 #5) — `[tool.hatch.build.targets.wheel] packages = ["stella", "configs"]`. CUDA torch는 PyPI에 없으므로 `[[tool.uv.index]]` + `[tool.uv.sources]`로 인덱스를 명시 |
| 프레임워크    | PyTorch ≥ 2.5                           | SDPA(`scaled_dot_product_attention`) 사용                                                                                                                                                                              |
| 학습 루프    | PyTorch Lightning ≥ 2.4                 | DDP·bf16·체크포인트 위임                                                                                                                                                                                                    |
| 백본       | `transformers` ≥ 4.55 (`timm` ≥ 1.0 보조) | 7.2절                                                                                                                                                                                                                 |
| 수치       | `numpy`, `einops`, (`scipy`)            | scipy는 매칭 단위테스트의 대조 구현(LSA)용                                                                                                                                                                                         |
| 시각화(개발용) | `opencv-python`                         | GT 인코딩 확인 스크립트                                                                                                                                                                                                       |
| 품질       | `ruff`, `pytest`                        | 포매팅+린트+테스트                                                                                                                                                                                                           |

---

## 3. 폴더/파일 구조

```
stella2/                        # 새 저장소 루트
├── pyproject.toml              # 패키지·의존성·ruff·pytest 설정
├── README.md
├── configs/
│   ├── schema.py               # ★ 모든 config dataclass 정의 (단일 파일)
│   ├── base.py                 # get_config() -> ExperimentConfig (기본 실험)
│   └── exp_*.py                # 변형 실험: base를 불러와 필드만 수정
├── stella/
│   ├── __init__.py             # 비워 둔다 (import 목록을 관리하지 않는다)
│   ├── builder.py              # resolve / build_instance / check_all — 클래스 선택 단일 관문 (5절)
│   ├── data/
│   │   ├── types.py            # GridDatasetBase(출력 계약 docstring 포함) + collate_fn
│   │   ├── encode.py           # 폴리라인 → 격자 GT 인코더 (6.4절)
│   │   ├── synthetic.py        # SyntheticDataset — 개발용 합성 데이터 (6.6절)
│   │   ├── augment.py          # 벡터 단계 증강 (flip/rot90) + 색상 증강
│   │   └── seedmap.py          # SeedMapDataset — SEED-MAP 로더·경계 자르기 (6.7절, M9)
│   ├── model/
│   │   ├── backbone.py         # Backbone 베이스 + Dinov3 / TimmBackbone (7.2절)
│   │   ├── neck.py             # Neck 베이스 + SFP / FPNLite → (256,L,L) (7.3절)
│   │   ├── heatmap.py          # 보조 히트맵 헤드 + 노드 선택 (7.4절)
│   │   ├── rope.py             # 2D axial RoPE (7.6절)
│   │   ├── blocks.py           # slot self-attn / cross-attn(전역·윈도우) / FFN
│   │   ├── heads.py            # self 헤드·연결 슬롯 헤드 (7.7절)
│   │   └── stella.py           # StellaModel(from_cfg 포함) + ModelOutput 정의 (7.1절)
│   ├── loss/
│   │   ├── matching.py         # 셀별 슬롯 배정 — R! 순열 완전탐색 벡터화 (8.3절)
│   │   ├── heatmap.py          # HeatmapLoss (8.1절)
│   │   ├── self_slot.py        # SelfSlotLoss — 클래스·좌표·끝(end) (8.2절)
│   │   ├── conn.py             # ConnLoss — 매칭 + 존재·방향 (8.3~8.4절)
│   │   └── criterion.py        # StellaCriterion — 위 셋을 조립·가중합 (8.0절)
│   ├── decode/
│   │   └── graph.py            # ChainDecoder — 정점 추출 → 사슬 확장 (10절, M6)
│   ├── eval/
│   │   └── ccq.py              # InstanceCCQ — 커버리지 중심 인스턴스 F1 (11절, M7)
│   └── train/
│       ├── module.py           # StellaTrainModule (LightningModule, 얇게)
│       ├── optim.py            # param group 분리·워밍업 스케줄
│       ├── viz.py              # 시각 로그 그리기 — 순수 함수, Lightning 무관 (9.5절)
│       ├── callbacks.py        # VizCallback — 검증 배치마다 첫 샘플 저장 (9.5절)
│       └── train.py            # 진입점 + ★최상위 조립 배선 (5절)
├── scripts/
│   ├── viz_gt.py               # GT 인코딩·합성 데이터 육안 확인
│   └── stat_labels.py          # SEED-MAP 라벨 통계 — 6.7.5절 표를 재생성 (전체 데이터 수령 시 재실행)
└── tests/
    ├── test_build.py           # ① 전 config의 path/name 해석 (빠름) ② 전체 조립 스모크 (느림, 5절)
    ├── test_encode.py          # GT 인코더 불변식 검증 (6.4절)
    ├── test_rope.py            # RoPE 상대위치 성질 검증
    ├── test_matching.py        # 순열 매칭을 scipy LSA와 대조 검증
    ├── test_decode.py          # GT를 모델 출력 형식으로 넣으면 원본 폴리라인이 복원되는지 (10절)
    ├── test_metric.py          # 인스턴스 CCQ: 완전복원=F1 1, 조각 예측의 TP/redundant FP 판정 (11절)
    ├── test_viz.py             # 시각 로그 함수의 shape·색상 규약 (9.5절)
    └── test_model.py           # shape 테스트 + 1-이미지 과적합 테스트
```

파일 수를 일부러 적게 유지한다. 한 파일 = 한 책임. `util/misc.py` 같은 잡동사니 파일은 만들지 않는다.
**계열 하나 = 파일 하나**로 둔다(베이스와 구현체를 같은 파일에). `backbone.py` 하나에 `Backbone`·`Dinov3`·`TimmBackbone`이 함께 있는 식이다.
`__init__.py`는 전부 비운다 — import 목록을 관리하지 않는다. 필요한 모듈은 `build_instance`가 `path`를 보고 그때 import 한다(5절).
중앙 factory 파일은 없다. 클래스를 찾는 일만 `builder.py`가 하고, 조립은 각 클래스의 `from_cfg`가, 최상위 배선은 `train.py`가 한다.

---

## 4. Config 시스템

### 4.1. 스키마 — `configs/schema.py`

전부 `@dataclass(kw_only=True)`로 정의한다. **공유 값은 `DataConfig`에만 존재한다** (`num_classes`, `image_size`, `grid_stride`).
`build_instance`로 만드는 부품의 config는 **`ModuleConfig`를 상속**해 `path`·`name` 두 문자열을 갖는다(5절).

**`schema.py`는 `stella` 패키지를 import 하지 않는다.** config는 순수한 데이터라서 코드 쪽으로 의존이 생기지 않고,
따라서 순환참조가 구조적으로 불가능하다.

```python
from dataclasses import dataclass, field


@dataclass(kw_only=True)
class ModuleConfig:
    """build_instance로 만드는 부품 config의 공통 베이스 (5절).
    하위 클래스가 두 필드에 기본값을 준다."""

    path: str  # 모듈 경로   예: "stella.model.backbone"
    name: str  # 클래스 이름 예: "Dinov3"
```

`kw_only=True`는 취향이 아니라 **필수**다. dataclass 상속은 하위 클래스 필드를 베이스 필드 **뒤에** 붙이는데,
일반 dataclass는 "기본값 있는 필드 뒤에 기본값 없는 필드"를 허용하지 않는다. `ModuleConfig`가 이미 기본값을 가지므로,
하위 config에 기본값 없는 필드를 하나라도 추가하는 순간 `TypeError: non-default argument follows default argument`가
**import 시점에** 터진다. `kw_only=True`면 생성자가 `def __init__(self, *, ...)`가 되어 이 제약 자체가 사라진다.
덤으로 위치 인자 생성이 막혀서, 필드 순서를 바꿔도 기존 코드가 조용히 어긋나지 않는다.

```python
@dataclass(kw_only=True)
class DataConfig(ModuleConfig):
    path: str = "stella.data.synthetic"
    name: str = "SyntheticDataset"  # 실데이터는 "stella.data.seedmap" / "SeedMapDataset"
    root: str = ""  # SEED_MAP_v1.1_splits 루트. 하위에 {train,val,test}/{image,label} (6.7.2절)
    image_size: int = 768  # SEED-MAP 원본 크기와 동일 — 리사이즈 없음
    grid_stride: int = 4  # 격자 배율 s. L = image_size // grid_stride = 192
    num_classes: int = 12  # 0 = background + 차선 11종 (6.7.1절 Table V, 전체 train 재집계로 확정)
    batch_size: int = 1  # 확정 — bs=2는 처리량 +16%뿐, accumulate가 유효 배치를 만든다 (9.3절)
    num_workers: int = 8
    max_degree: int = 2  # D: 셀당 GT 분기 수. 선 단위 사슬이라 **항상 정확히 2** (6.4절)
    encode_supersample: int = 1  # GT 래스터화 배율. 1 = 픽셀 해상도 (6.4절 A단계)
    cache_gt: str = "val_test"  # GT 캐시: "none" | "val_test"(기본) | "all" (6.4.1절)

    @property
    def grid_size(self) -> int:
        return self.image_size // self.grid_stride


@dataclass(kw_only=True)
class BackboneConfig(ModuleConfig):
    path: str = "stella.model.backbone"
    name: str = "Dinov3"  # Backbone 하위 클래스: "Dinov3" | "TimmBackbone"
    pretrained: str = "facebook/dinov3-vitl16-pretrain-sat493m"  # HF/timm 모델 ID (확정, 13절)
    lr_mult: float = 0.1  # optim.py가 읽는다 (__init__ 인자 아님)
    freeze: bool = False


@dataclass(kw_only=True)
class NeckConfig(ModuleConfig):
    path: str = "stella.model.neck"
    name: str = "SFP"  # "SFP"(ViT 단일 스케일) | "FPNLite"(멀티스케일)


@dataclass(kw_only=True)
class ModelConfig(ModuleConfig):
    path: str = "stella.model.stella"
    name: str = "StellaModel"
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    neck: NeckConfig = field(default_factory=NeckConfig)
    d_model: int = 256  # neck·블록·헤드가 공유 — 여기 한 곳에만 둔다
    num_heads: int = 8
    num_conn_slots: int = 2  # R = 2 확정 (K = 3) — GT 분기 수와 일치 (결정 1, 10차 개정)
    layers: tuple[str, ...] = ("global", "window", "window", "window", "window", "window")
    window_size: int = 7  # w. 9 → 7 (실측: peak 12.09 → 9.72 GiB, step 455 → 291 ms, 7.6절)
    ffn_dim: int = 1024
    dropout: float = 0.0
    grad_checkpoint: bool = True  # 윈도우 층만 재계산 — 활성의 대부분이 거기 있다 (7.6절)
    node_sampling: str = "gt+pred"  # 학습 노드 선택: "gt+pred"(기본) | "gt" (7.4절)
    n_max: int = 9500  # 노드 수 상한. 전체 train 실측 최대 8,909 (p99 5,893, 6.7.5절)
    heatmap_thresh: float = 0.3  # τ_h
    dilate: int = 3  # 예측 마스크 팽창: 0 | 3 | 5


# --- 손실: 종류별 모듈 config + 조립 config (8절) ---
@dataclass(kw_only=True)
class HeatmapLossConfig(ModuleConfig):
    path: str = "stella.loss.heatmap"
    name: str = "HeatmapLoss"
    w_heatmap: float = 1.0  # 총합에 그대로 곱해지는 유일한 가중치
    focal_alpha: float = 0.25  # 가중치가 아니라 focal 형태 파라미터
    focal_gamma: float = 2.0


@dataclass(kw_only=True)
class SelfSlotLossConfig(ModuleConfig):
    path: str = "stella.loss.self_slot"
    name: str = "SelfSlotLoss"
    w_class: float = 1.0
    w_coord: float = 1.0
    w_end: float = 1.0  # 끝 셀 BCE — end_map 직접 감독 (8.2절, 9차 개정)


@dataclass(kw_only=True)
class ConnLossConfig(ModuleConfig):
    path: str = "stella.loss.conn"
    name: str = "ConnLoss"
    w_exist: float = 1.0
    w_dir: float = 1.0  # 연결 방향 손실 (1 - 내적)
    match_w_dir: float = 1.0  # λ_dir — 손실 가중치가 아니라 **매칭 비용** 계수 (8.3절)
    match_w_exist: float = 1.0  # λ_e   — 〃


@dataclass(kw_only=True)
class LossConfig(ModuleConfig):
    path: str = "stella.loss.criterion"
    name: str = "StellaCriterion"
    heatmap: HeatmapLossConfig = field(default_factory=HeatmapLossConfig)
    self_slot: SelfSlotLossConfig = field(default_factory=SelfSlotLossConfig)
    conn: ConnLossConfig = field(default_factory=ConnLossConfig)


# --- 디코딩(객체 생성) + 로깅 ---
@dataclass(kw_only=True)
class DecodeConfig(ModuleConfig):  # 9차 개정 — 사슬 확장 디코더 (10절)
    path: str = "stella.decode.graph"
    name: str = "ChainDecoder"
    heatmap_thresh: float = 0.3  # τ_h — 노드 후보 (추론 경로, 7.4절)
    exist_thresh: float = 0.5  # τ_e — 연결 슬롯 존재
    end_thresh: float = 0.5  # τ_end — 끝 셀 판정 (σ(end_logit)), 사슬 정지 조건
    radius: int = 2  # 탐색 반경(셀). 5×5 — 교차점에서 잃은 한 칸을 건너뛴다 (6.4절)
    align_thresh: float = 0.7  # 내 슬롯 방향과 실제 상대 방향의 코사인 하한 (c·u_ab)
    opp_thresh: float = 0.7  # 마주봄 하한 — -(c·n) ≥ 이 값 (10.3절)
    w_opp: float = 1.0  # 후보 비용에서 마주봄 항의 비중
    min_class_prob: float = 0.1  # 확장 게이트 — 후보의 사슬 클래스 softmax 확률 하한 (10.3절)
    purity_thresh: float = (
        0.6  # 사슬 순도 하한 — argmax 클래스 일치 비율. 이하면 사슬 폐기 (10.3절)
    )
    end_extend: float = 1.0  # 끝 셀에서 끝방향 슬롯으로 연장하는 길이(셀) — 10.3절 끝 연장
    min_points: int = 2  # 이보다 짧은 폴리라인은 버린다 (연장점 포함)
    simplify_tol: float = 0.0  # >0이면 RDP 단순화 (픽셀)
    # 임계값들은 학습된 체크포인트로 검증 셋에서 스윕해 확정한다 (13절 남은 확인).
    # 구 GraphDecoder의 mutual·w_dist·max_conn_dist·t_thresh는 폐기 — 10절 참고.


@dataclass(kw_only=True)
class MetricConfig(ModuleConfig):  # 인스턴스 평가 지표 (11절)
    path: str = "stella.eval.ccq"
    name: str = "InstanceCCQ"
    buffer_rho: float = 12.0  # ρ (픽셀). 차선 간격의 절반 이하. GSD 확정 후 재설정 (11.1)
    cov_thresh: float = 0.5  # θ_cov — 커버리지(완전성) 하한, 관대
    cor_thresh: float = 0.9  # θ_cor — 정확성 하한, 엄격 (모든 GT 대상)
    angle_gate: float = 30.0  # 매칭 시 접선 방향 차 상한(도) — 수직 교차 배제
    sample_step: float = 2.0  # 길이 계산용 폴리라인 샘플 간격(픽셀)


@dataclass(kw_only=True)
class LogConfig(ModuleConfig):
    path: str = "stella.train.callbacks"
    name: str = "VizCallback"
    every_n_epochs: int = 1  # 시각 로그를 남길 에폭 간격
    max_batches: int = 20  # 에폭당 최대 배치 수 (배치당 1장, 9.5절)
    heat_alpha: float = 0.5  # 히트맵 블렌딩 비율
    slot_line_len: float = 6.0  # 슬롯 방향선 길이 (픽셀)
    exist_thresh: float = 0.5  # 이보다 낮은 슬롯은 그리지 않는다
    class_thresh: float = 0.5  # class map에 칠할 최소 히트맵 확률


@dataclass(kw_only=True)
class TrainConfig(ModuleConfig):
    path: str = "stella.train.module"
    name: str = "StellaTrainModule"
    lr: float = 1e-4
    weight_decay: float = 0.05
    warmup_steps: int = 1000
    epochs: int = 100  # 아래 넷은 pl.Trainer가 읽는다 (__init__ 인자 아님)
    accumulate: int = 16  # 유효 배치 = batch_size × accumulate × GPU 수
    grad_clip: float = 0.1
    precision: str = "bf16-mixed"
    seed: int = 42  # train.py가 읽는다
    output_root: str = "results"  # 〃


@dataclass(kw_only=True)
class ExperimentConfig:  # 이것 자체는 build 대상이 아니다
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    decode: DecodeConfig = field(default_factory=DecodeConfig)
    eval: MetricConfig = field(default_factory=MetricConfig)
    log: LogConfig = field(default_factory=LogConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
```

**config 필드가 전부 `__init__` 인자인 것은 아니다.** 상당수는 다른 곳이 읽는다.

| 필드                                                           | 읽는 곳                          |
| ------------------------------------------------------------ | ----------------------------- |
| `BackboneConfig.lr_mult`                                     | `optim.py` param group (9.2절) |
| `DataConfig.batch_size`, `num_workers`                       | `DataLoader`                  |
| `TrainConfig.epochs`, `accumulate`, `grad_clip`, `precision` | `pl.Trainer` (9.3절)           |
| `TrainConfig.seed`, `output_root`                            | `train.py`                    |
| `ModuleConfig.path`, `name`                                  | `builder.py`                  |

**공유 값은 여전히 한 곳에만 둔다.** `ConnLoss`가 필요로 하는 슬롯 수는 `ModelConfig.num_conn_slots`,
`ChainDecoder`가 필요로 하는 격자 크기는 `DataConfig.grid_size`에서 읽는다 — `from_cfg`가 전체 cfg를 받으므로
복제할 이유가 없다(조립 규칙 1). `DecodeConfig.heatmap_thresh`처럼 이름이 겹치는 것은 **의도적으로 분리한 값**이다:
학습 중 노드 선택(`model.heatmap_thresh`)과 객체 생성 시 노드 채택(`decode.heatmap_thresh`)은 따로 튜닝한다.

### 4.2. 실험 변형

YAML·상속 병합을 쓰지 않는다. 변형 실험은 **base를 불러와 필드를 코드로 수정**한다.

```python
# configs/exp_r3.py — 슬롯 수 ablation 예시 (기본 R = 2에 여분 슬롯 1개를 줘 본다)
from configs.base import get_config as get_base


def get_config():
    cfg = get_base()
    cfg.model.num_conn_slots = 3  # K = 4
    return cfg


# configs/exp_timm.py — 백본 교체 예시. path 기본값이 있어 대개 name만 바꾸면 된다
def get_config():
    cfg = get_base()
    cfg.model.backbone.name = "TimmBackbone"
    cfg.model.backbone.pretrained = "swinv2_large_window12to16_192to256"
    cfg.model.neck.name = "FPNLite"
    return cfg
```

장점: IDE에서 "이 필드를 어디서 바꾸는지"가 참조 검색으로 다 나온다. 오타는 실행 중 `AttributeError`가 아니라 **정적 검사**에서 잡힌다.
`path`·`name` 문자열만은 정적 검사가 안 되므로, 4.3의 `check_all`과 5절의 `test_config_resolves`가 그 자리를 맡는다.

### 4.3. 로드 규칙

- 진입점은 `--config <모듈 이름>`을 받는다(예: `--config configs.exp_k5`). `importlib.import_module(...).get_config()`를 호출한다.
  `configs`가 설치된 패키지이므로(2절) 파일 경로 대신 모듈 이름을 받는 편이 오타 시 에러가 명확하다.

- **config를 로드한 직후 `check_all(cfg)`를 부른다**(5절). 백본 가중치 다운로드·CUDA 초기화 전에
  모든 `path`/`name` 오타가 여기서 걸린다.

- 학습 시작 시 출력 폴더에 다음을 남긴다.
  
  - `config.json` — `dataclasses.asdict(cfg)`.
  
  - **`src/` — 소스 전체 복사.** git commit·diff 방식을 쓰지 않는다. 실험마다 커밋을 강제하면
    "시작하자마자 끄고 숫자 하나 바꿔 다시 돌리는" 작업 방식에서 커밋 이력이 못 쓰게 되고,
    복원하려면 그 커밋이 살아 있는 저장소가 필요하다. 소스 전체는 수백 KB라 체크포인트에 비하면 무시할 크기이고,
    결과 폴더만으로 자기완결적이다.
    
    ```python
    shutil.copytree(
        repo_root,
        out_dir / "src",
        ignore=shutil.ignore_patterns("__pycache__", ".git", ".venv", "results", "data", "*.pyc"),
    )
    ```
    
    제외 패턴은 필수다 — 그냥 복사하면 `data/`·`results/`까지 끌려온다.
  
  - `git_sha.txt` — git 저장소일 때만, `git rev-parse HEAD`와 dirty 여부 한 줄. 커밋을 강제하지 않으며
    소스 복사를 대체하지도 않는다. "대략 어느 시점 코드인가"를 훑을 때만 쓴다.

- 체크포인트가 하나도 없는 실행 폴더를 지우는 정리 스크립트를 둔다(시작 직후 중단한 실행 정리용).

---

## 5. 조립 — `builder.py` 단일 관문 + 클래스별 `from_cfg` + `train.py` 배선

역할이 셋으로 나뉜다. **클래스를 찾는 일**은 `builder.py`가, **부품을 만드는 일**은 각 클래스의 `from_cfg`가,
**최상위 배선**은 `train.py`가 한다.

### 5.1. 클래스 선택 — `stella/builder.py`

config에 `path`(모듈 경로)와 `name`(클래스 이름)을 적고, `builder.py`가 그것만 보고 클래스를 찾는다.
이 방식을 고른 이유는 두 가지다. **config가 코드를 import 하지 않아 순환참조가 구조적으로 불가능**하고,
**실제로 쓰는 클래스만 지연 import**되어 `__init__.py`도 전역 import 스캔도 필요 없다.

대가는 하나 — `path`/`name`은 문자열이라 IDE가 rename 해 주지 못한다. 그래서 **해석 단계에서 최대한 많이 검증**하고,
그 검증을 **학습 시작 전에 몰아서** 돌린다.

```python
# stella/builder.py
import difflib, importlib
from dataclasses import fields, is_dataclass
from typing import Any


def resolve(module_cfg: ModuleConfig, base: type | None = None) -> type:
    """(path, name) → 클래스. 인스턴스는 만들지 않는다."""
    where = type(module_cfg).__name__  # 어느 config가 문제인지 알려주기 위함

    try:
        module = importlib.import_module(module_cfg.path)
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(f"{where}.path='{module_cfg.path}' 를 import 할 수 없다") from e

    cls = getattr(module, module_cfg.name, None)
    if cls is None:
        here = sorted(
            n
            for n, o in vars(module).items()
            if isinstance(o, type) and o.__module__ == module.__name__
        )
        hint = difflib.get_close_matches(module_cfg.name, here, n=3)
        raise AttributeError(
            f"{where}.name='{module_cfg.name}' 이 {module_cfg.path} 에 없다. "
            + (f"혹시 이것? {hint}" if hint else f"이 모듈의 클래스: {here}")
        )

    if cls.__module__ != module.__name__:
        raise TypeError(
            f"{where}: '{module_cfg.name}' 은 {module_cfg.path} 에서 정의된 것이 아니라 "
            f"{cls.__module__} 에서 import 된 이름이다. path를 '{cls.__module__}' 로 고쳐라"
        )

    if base is not None and not issubclass(cls, base):
        raise TypeError(f"{where}: {cls.__name__} 은 {base.__name__} 의 하위 클래스가 아니다")

    if not hasattr(cls, "from_cfg"):
        raise TypeError(f"{where}: {cls.__name__} 에 from_cfg 가 없다")

    return cls


def build_instance(module_cfg: ModuleConfig, cfg, base: type | None = None, **kwargs) -> Any:
    return resolve(module_cfg, base).from_cfg(module_cfg, cfg, **kwargs)


def check_all(cfg) -> None:
    """cfg 트리의 모든 ModuleConfig를 미리 찾아본다. 무거운 초기화 전에 부른다."""
    for mc in _walk(cfg):
        resolve(mc)


def _walk(node):
    if isinstance(node, ModuleConfig):
        yield node
    if is_dataclass(node):
        for f in fields(node):
            yield from _walk(getattr(node, f.name))
```

**네 가지 검사가 각각 막는 사고:**

| 검사                       | 없으면 생기는 일                                                                        |
| ------------------------ | -------------------------------------------------------------------------------- |
| 모듈 내 클래스 목록 + `difflib`  | `has no attribute 'Dinov'` 만 보고 후보를 직접 찾아야 함                                     |
| `cls.__module__ == path` | `neck.py`가 `Backbone`을 import 하므로 `path="…neck", name="Backbone"` 이 **조용히 성공**한다 |
| `issubclass(cls, base)`  | 계열을 잘못 적어도 통과 → forward 깊은 곳에서 shape 에러                                          |
| `hasattr(from_cfg)`      | 어느 config가 문제인지 안 나오는 `AttributeError`                                           |

`base`는 호출부에서 넘긴다. `Backbone`·`Neck`은 부모의 `__init__` 타입 힌트로 **이미 import 되어 있으므로** 추가 비용이 없다.

**`check_all`이 왜 따로 필요한가.** `build_instance`의 검사는 *그 부품을 만들 때* 돈다. 그런데 부품은 하나씩 순서대로
만들어지므로, `cfg.loss`의 오타는 `model`이 완성된 뒤에야 — 즉 DINOv3 가중치를 다 받고 GPU에 올린 뒤에야 — 드러난다.
`check_all`은 같은 검사를 **트리 전체에 대해 인스턴스 생성 없이** 먼저 돌려서 이 대기를 없앤다.
덤으로 대상 모듈을 실제 import 하므로 그 파일의 문법 오류나 최상단 import 실패도 이 시점에 드러난다.

`check_all`이 못 잡는 것도 적어 둔다. ① `base` 검사 — 어느 config가 어느 계열을 기대하는지는 호출부에만 있는 정보다.
② 실제로 만들어 봐야 아는 것(`__init__` 인자 불일치, shape 오류). 후자는 아래 `test_full_build`가 맡는다.

### 5.2. 부품 만들기 — 클래스별 `from_cfg`

```python
# stella/model/backbone.py
class Backbone(nn.Module):  # 인터페이스: forward(x) -> list[Tensor], out_channels
    ...


class Dinov3(Backbone):
    def __init__(self, *, pretrained: str, freeze: bool):  # keyword-only, 기본값 없음
        ...
    @classmethod
    def from_cfg(cls, module_cfg: BackboneConfig, cfg: ExperimentConfig, **kwargs):
        return cls(pretrained=module_cfg.pretrained, freeze=module_cfg.freeze)


# stella/model/stella.py — 부모가 자식을 조립한다
from stella.model.backbone import Backbone
from stella.model.neck import Neck


class StellaModel(nn.Module):
    @classmethod
    def from_cfg(cls, module_cfg: ModelConfig, cfg: ExperimentConfig, **kwargs) -> "StellaModel":
        backbone = build_instance(module_cfg.backbone, cfg, base=Backbone)
        neck = build_instance(
            module_cfg.neck,
            cfg,
            base=Neck,
            in_channels=backbone.out_channels,
            d_model=module_cfg.d_model,
        )
        return cls(
            backbone=backbone,
            neck=neck,
            d_model=module_cfg.d_model,
            num_heads=module_cfg.num_heads,
            num_conn_slots=module_cfg.num_conn_slots,
            layers=module_cfg.layers,
            window_size=module_cfg.window_size,
            num_classes=cfg.data.num_classes,  # 다른 섹션은 cfg에서 직접
            grid_size=cfg.data.grid_size,
        )
```

`num_classes`·`grid_size`가 중간 계층을 거쳐 릴레이되지 않는 것이 요점이다. 공유 값은 여전히 `DataConfig`에만 있고,
그것을 읽는 곳이 값을 직접 가져간다. `kwargs`(`in_channels`, `split` 등)는 **config에 없는 값만** 나른다.

### 5.3. 최상위 배선 — `stella/train/train.py`

```python
from stella.data.types import GridDatasetBase


def main(cfg: ExperimentConfig):
    check_all(cfg)  # ← 오타는 여기서 전부 걸린다

    model = build_instance(cfg.model, cfg)
    criterion = build_instance(cfg.loss, cfg)
    train_set = build_instance(cfg.data, cfg, base=GridDatasetBase, split="train")
    val_set = build_instance(cfg.data, cfg, base=GridDatasetBase, split="val")
    module = build_instance(cfg.train, cfg, model=model, criterion=criterion)
    ...
```

모든 줄이 같은 모양이라 "이 부품은 어떻게 만들더라"를 다시 확인할 일이 없다.

### 5.4. `from_cfg` 기본 구현 — `Buildable`

`Dinov3`·`SFP`처럼 config 필드가 `__init__` 인자와 이름까지 그대로 대응하는 클래스는 `from_cfg` 본문이
`x=c.x`의 반복이다. 작은 믹스인 하나로 없앤다.

```python
class Buildable:
    @classmethod
    def from_cfg(cls, module_cfg, cfg, **kwargs):
        params = inspect.signature(cls).parameters  # __init__이 받는 것만 고른다
        auto = {
            f.name: getattr(module_cfg, f.name)
            for f in fields(module_cfg)
            if f.name in params and not is_dataclass(getattr(module_cfg, f.name))
        }
        return cls(**{**auto, **kwargs})
```

**모든 계열 베이스는 `Buildable`을 상속한다** (`class Backbone(nn.Module, Buildable)`). 잎 클래스는 `from_cfg`를
쓰지 않고, 조립이 필요한 클래스(`StellaModel`처럼 자식을 만들어야 하는 것)만 override 한다.
시그니처로 거르므로 `lr_mult`처럼 **생성자 인자가 아닌 필드**(4.1절 표)가 섞여 들어가지 않는다.

이 방식의 유일한 위험은 config 필드명과 `__init__` 인자명이 어긋날 때의 **조용한 누락**인데,
조립 규칙 2("`__init__`에 기본값을 두지 않는다")가 그것을 `TypeError`로 바꾼다.
따라서 **config 필드명과 `__init__` 인자명은 반드시 같게 쓴다.**

### 5.5. 테스트 — `tests/test_build.py`

두 단계로 나눈다. 앞의 것이 **파일 이동·클래스 rename으로 문자열이 깨지는 것**을 잡는 안전망이다.

```python
ALL_CONFIGS = ["configs.base", *glob_exp_configs()]


@pytest.mark.parametrize("mod", ALL_CONFIGS)
def test_config_resolves(mod):
    """모든 config의 path/name이 실재하는지. GPU·가중치 불필요, 수 초 이내. 매 커밋."""
    check_all(importlib.import_module(mod).get_config())


@pytest.mark.slow
@pytest.mark.parametrize("mod", ALL_CONFIGS)
def test_full_build(mod):
    """model·criterion·dataset·module 전체 조립. 조립 누락(과거 postprocessors 버그)이 여기서 터진다."""
    ...
```

---

## 6. 데이터 인터페이스 명세 (핵심)

데이터셋 내부 구현은 자유지만, **`__getitem__`의 출력은 아래 계약을 반드시 따른다.**
합성 데이터셋(6.6)·SEED-MAP 실데이터셋(6.7)·GT 인코더·criterion(8절)이 모두 이 계약으로 맞물린다.

**설계 방침 1 (10차 개정): target은 가급적 모델 출력과 같은 형태로 만든다.**
구 방침은 반대였다 — "GT는 사실(이웃 셀 좌표)만 저장하고 방향은 criterion이 유도한다."
그 간접층의 근거는 가변 분기 수(그래프 차수)와 유도 규칙($t$ 등)의 복잡성이었는데, 9차 개정으로
**모든 셀의 분기가 정확히 2개**가 되면서 근거가 사라졌다. 이제 인코더가 **연결 방향 2개를 직접
계산해 저장**한다:

- 셀별 self 정보: 클래스, 셀 내 좌표, **자기가 사슬의 끝 셀인지**(end) — $(L, L)$ 격자 맵.
- 연결성: **자기 점에서 사슬 이웃의 점으로 향하는 단위 방향 2개** (`conn_dirs`).

criterion은 아무것도 유도하지 않는다 — 저장된 방향 2개와 예측 슬롯 2개를 매칭해 손실을 줄
뿐이다(8.3절). GT와 모델 출력이 같은 격자·같은 형태·**같은 개수**($R = D = 2$, 결정 1)라
셀 인덱싱만으로 정렬되고, 남는 차이는 슬롯 순서(매칭이 배정)뿐이다. 시각화·디버깅도 같은
그리기 함수를 쓴다(9.5절).

**설계 방침 2 (9차 개정): 선 하나 = 사슬 하나 — 인코딩과 디코딩이 같은 모양이다.**
구 설계(7차)는 클래스 단위로 간선을 합집합한 **그래프**를 GT로 만들고, 디코더가 그 그래프를
절단해 폴리라인을 만들었다. 그래프(합집합)와 라벨 인스턴스 목록(선 단위)이 어긋나 GT를 완벽히
주입해도 F1 상한이 0.63~0.69에 막혔고(조각 1.7배), 끝점 소유권·$t$ 해석 같은 규약 충돌이
연쇄로 생겼다. 새 설계는 **선(인스턴스)마다 독립된 사슬**을 인코딩한다 — 모든 셀의 분기가
정확히 2개(앞·뒤)이고, 디코더도 사슬을 한 노드씩 확장하며 복원한다(10절).
Y·T자 분기는 "끝칸 미채움" 규약이, X·+자 교차는 셀 소유권 + 탐색 반경 2가 처리한다(6.4절).

### 6.1. 좌표·축 규약 (전 코드 공통)

**shape 기호 (전 문서 공통).** `B`는 **언제나 배치**다. 연결 슬롯 수는 `R`로 쓴다.

| 기호  | 뜻                            | 기본값                        |
| --- | ---------------------------- | -------------------------- |
| $B$ | **배치 크기** (다른 뜻으로 쓰지 않는다)    | 1 (`data.batch_size`)      |
| $L$ | 격자 한 변                       | 192 (`= image_size / s`)   |
| $s$ | 격자 배율(stride)                | 4 (`data.grid_stride`)     |
| $C$ | 클래스 수 (0 = 배경 포함)            | 12 (`data.num_classes`)    |
| $R$ | **연결 슬롯 수** — GT 분기 수와 일치    | 2 (`model.num_conn_slots`) |
| $K$ | 노드당 토큰 수 $= R + 1$ (self 1개) | 3                          |
| $D$ | 셀당 GT 분기 수 — 사슬이라 **항상 2**   | 2 (`data.max_degree`)      |
| $N$ | 이번 forward에서 선택된 노드 수        | 가변 (≤ `model.n_max`)       |

- 이미지 픽셀: $x$ 오른쪽+, $y$ 아래쪽+. 크기 $H = W = 768$.
- 격자: 배율 $s = 4$, 크기 $L = 192$. 셀 인덱스는 $(i, j)$ = (행, 열) = $(y, x)$ 순서.
- 연속 격자 좌표: $u = x_{\mathrm{px}} / s$, $v = y_{\mathrm{px}} / s$. 점이 속한 셀은 $(i, j) = (\lfloor v \rfloor, \lfloor u \rfloor)$.
- **모든 2차원 벡터 텐서는 $(x, y)$ 순서**로 저장한다 (셀 인덱스만 $(i,j)$ 순서). 이 두 가지가 유일한 규약이고, `types.py` docstring에 박아 둔다.
- **원점 규약.**
  - 셀 내 좌표(`coord`)의 원점 = **셀 좌상단.** 노드 점의 절대 위치(격자 단위):

$$
\mathbf{p}^{\mathrm{full}} = (j + c^x,\; i + c^y), \qquad (c^x, c^y) \in [0, 1)^2
$$

  - 연결 방향의 원점 = **자기 노드 점 $\mathbf{p}^{\mathrm{full}}$** (9차 개정 — 구 설계는 셀 중심
    $\mathbf{o}_{ij} = (j + 0.5, i + 0.5)$였다). 방향은 점 사이를 잇는 **선 자체의 접선**을
    예측하는 것이므로, 좌표 예측(`coord`)과 원점을 공유하는 편이 자연스럽고 인코딩·디코딩에서
    같은 식을 쓴다.

### 6.2. 한 샘플의 출력 스펙

`__getitem__(idx)` → `(image, targets)`. 아래 shape은 배치 차원을 뺀 한 샘플 기준이다.
$C = 12$(0 = 배경), $D$ = `max_degree`(셀당 GT 분기 저장 칸 수 — 사슬이라 항상 2).

**image**

| 항목            | 값                                                  |
| ------------- | -------------------------------------------------- |
| dtype / shape | `float32`, `(3, 768, 768)`                         |
| 범위            | $[0, 1]$ (RGB, 정규화 전)                              |
| 비고            | mean/std 정규화는 **모델 내부**에서 한다(버퍼). 백본을 바꿔도 데이터셋은 불변 |

**targets** (dict)

| key          | dtype     | shape              | 값 범위               | 의미                                                                          |
| ------------ | --------- | ------------------ | ------------------ | --------------------------------------------------------------------------- |
| `class_map`  | `int64`   | `(192, 192)`       | $[0, 11]$          | 셀 소유 선의 클래스. 0 = 배경. **히트맵 GT는 `class_map > 0`으로 유도**(별도 키 없음)              |
| `coord_map`  | `float32` | `(192, 192, 2)`    | $[0, 1)$           | 셀 내 정밀 좌표 $(c^x, c^y)$ — 소유 선 픽셀의 무게중심, 원점 = 셀 좌상단. 양성 셀만 유효                |
| `end_map`    | `float32` | `(192, 192)`       | $\{0, 1\}$         | **이 셀이 사슬의 끝 셀인지.** 유도값이 아니라 **직접 감독 대상**이다(8.2절)                           |
| `conn_dirs`  | `float32` | `(192, 192, D, 2)` | 단위벡터               | **연결 방향 2개** ($D = 2$) — 자기 점에서 사슬 이웃의 점으로. 칸 순서는 무의미(매칭이 배정). 양성 셀만 유효     |
| `instances`  | list      | —                  | —                  | 평가용 원본 폴리라인 `{class: int, points: float32 (P,2) 픽셀좌표}`. **학습 미사용**, 그대로 통과만 |
| `meta`       | dict      | —                  | —                  | `filename` 등                                                                |

방향은 **인코더가 직접 계산해 저장한다** (10차 개정 — 구 설계의 `conn_cells`(이웃 셀 좌표) +
criterion 유도를 폐지). 셀 $a$의 사슬 이웃이 $b$일 때 (원점 = **자기 점**, 6.1절):

$$
\mathbf{d}^{gt}_{(a \to b)} = \frac{\mathbf{p}^{\mathrm{full}}_b - \mathbf{p}^{\mathrm{full}}_a}{\left\lVert \mathbf{p}^{\mathrm{full}}_b - \mathbf{p}^{\mathrm{full}}_a \right\rVert},
\qquad
\mathbf{d}^{gt}_{(a \to \mathrm{end})} = \frac{\mathbf{q} - \mathbf{p}^{\mathrm{full}}_a}{\left\lVert \mathbf{q} - \mathbf{p}^{\mathrm{full}}_a \right\rVert}
\;\;(\mathbf{q} = \text{선의 실제 끝점 — 인코더 내부 값, 저장하지 않는다})
$$

**모든 양성 셀의 GT 분기는 정확히 2개다** — 중간 셀은 (앞 이웃 방향, 뒤 이웃 방향), 끝 셀은
(안쪽 이웃 방향, **끝점 방향**). 구 설계의 $t^{gt}$(종점·다른 클래스 접합 겸용 슬롯 라벨)는 폐기했다 —
끝은 `end_map`이 셀 단위로 직접 감독하고, 다른 클래스 접합 간선은 새 인코딩에 존재하지 않는다.
끝점 좌표 자체는 방향 계산에만 쓰이고 target에 남지 않는다(9차의 `end_point` 키 폐지).

**끝 규약 (중요, 9차 개정).** 선이 지나는 셀 열의 **양 끝 칸은 채우지 않는다.**
선이 10칸(0~9번)을 지나면 0·9번은 미채움이고, **1번·8번이 끝 셀**이다(`end_map = 1`).
끝 셀의 둘째 분기 방향은 이웃 셀이 아니라 **실제 끝점**을 향한다 — "선이 이 방향으로
끝났다"는 방향 감독만 남는다.

- **왜 끝칸을 비우나.** Y·T자 접합에서 선 B의 끝칸은 본선 A가 지나가는 셀이다. 끝칸을 채우면
  A와 B가 그 셀의 소유권을 다투고, 구 설계의 규약 충돌(종점 셀 vs 다른 클래스 접합 간선,
  구 open_questions 3·4)이 전부 여기서 나왔다. 끝칸을 아예 비우면 **다툼 자체가 없다** —
  그 셀은 지나가는 선의 것이고, B는 한 칸 물러난 끝 셀에서 끝점 방향만 남긴다.
- 이 규약은 접합·자유 끝·타일 경계 절단에 **동일하게** 적용한다(확정 — 사용자 결정, 9차 개정).
  **선이 짧아지는 것이 아니다**: 끝방향 슬롯이 실제 끝점을 향하도록 감독되므로, 디코더가 끝 셀에서
  그 방향으로 마지막 점을 연장해 찍는다(10.4절 끝 연장). 3칸짜리 선도 소멸하지 않는다 —
  채워지는 셀은 1개지만 그 셀의 두 분기가 모두 끝방향이라, 디코더가 양쪽으로 연장해 3점
  폴리라인을 만든다.

### 6.3. 미니 예시

수평선 하나(클래스 3)가 셀 $(10,5) \sim (10,12)$ 여덟 칸의 중앙을 지날 때 ($D = 2$):

```python
class_map[10, 6:12] = 3  # 양 끝칸 (10,5)·(10,12)는 미채움
coord_map[10, 6:12] = [0.5, 0.5]
end_map[10, 6] = end_map[10, 11] = 1  # 끝 셀 = 끝에서 두 번째 칸

# 인코더가 직접 저장하는 방향 (모든 양성 셀에서 2개, 원점 = 자기 점):
conn_dirs[10, 7] = [(-1, 0), (+1, 0)]  # 중간 셀: 앞 이웃(10,6)·뒤 이웃(10,8) 방향
conn_dirs[10, 6] = [(+1, 0), (-1, 0)]  # 끝 셀: 안쪽 이웃(10,7) 방향 + 끝점 방향
conn_dirs[10, 11] = [(-1, 0), (+1, 0)]  # 끝 셀: 안쪽 이웃(10,10) 방향 + 끝점 방향

# 끝점 좌표(예: (10,5) 칸 안의 실제 선 끝)는 방향 계산에만 쓰고 저장하지 않는다.
# criterion은 유도 없이 conn_dirs 2개 vs 예측 슬롯 R개를 매칭한다(8.3절).
```

### 6.4. GT 인코더 — `data/encode.py` (9차 개정 — 선 단위 사슬)

입력: 폴리라인 인스턴스 목록(픽셀 좌표 + 클래스). 출력: 6.2절 targets.

**설계 방침: 선 하나 = 사슬 하나.** 구 설계(7차)는 클래스 단위로 그려 무게중심·간선 합집합
그래프를 만들었다 — 이중선이 갈라지는 것을 막으려는 선택이었지만, GT 그래프가 라벨 인스턴스
목록과 어긋나 **GT를 완벽히 주입해도 인스턴스 F1이 0.63~0.69에 막혔다**(조각 1.7배 실측).
새 설계는 **선(인스턴스)마다 따로** 인코딩한다. 셀의 소유권 다툼은 선 사이에서만 일어나고,
사슬 위상은 각 선이 독자적으로 갖는다. 이중선이 두 인스턴스면 **두 사슬**이다 — 셀에 찍히는
픽셀의 무게중심을 정확히 찾는 것이 요점이지, 이중선을 한 줄로 합치는 것이 목적이 아니다.

#### (A) 선별 래스터 → 셀 소유권, `class_map`, `coord_map`

1. **선마다 따로 그린다.** 선 $l$의 폴리라인을 768×768 캔버스에 두께 1로 그리고,
   셀별 픽셀 수 $n_l(i,j)$와 셀 내 픽셀 좌표 합(무게중심용)을 센다.
2. **끝칸 제거.** 선 $l$의 셀 열에서 **첫 칸과 마지막 칸을 지운다**(6.2절 끝 규약).
   지운 칸의 픽셀은 아래 소유권 경쟁에도 참여하지 않는다.
3. **지그재그·스침 제거 (3×3 순위 규칙).** 선 $l$이 지나는 각 셀에 대해 **자기 3×3 이웃 안에
   있는 $l$의 셀들** 중 픽셀 수 순위가 **4등 이하**면 그 셀을 지운다. 선이 3×3 창을 지날 때
   정상적으로 걸치는 셀은 3개 안팎이므로, 모서리를 1~2 px 스치는 잉여 셀만 제거된다.
   - 이 규칙이 구 설계의 두 문제를 함께 없앤다: **고립 노드**(전체 train의 6.15% — 래스터에는
     잡히나 위상 셀 열에 안 잡히던 스침 셀)와 **지그재그 사슬**(인접 셀이 번갈아 걸쳐
     폴리라인이 톱니로 이어지는 것).
   - 절대값 필터(`min_cell_pixels`)는 쓰지 않는다 — 실측에서 사슬 중간·끝 셀까지 걸러
     오히려 나빴다(GT 주입 F1 0.686 → 0.557). 순위 규칙은 그 셀 없이도 주변에 사슬을 이을
     셀이 3개 이상 있을 때만 지우므로 연결이 끊기지 않는다.
4. **셀 소유권 (선 사이).** 여러 선이 같은 셀에 픽셀을 남기면 **픽셀이 더 많은 선이 소유한다**
   (구 설계의 클래스 단위 규칙을 선 단위로 바꾼 것). 동점이면 희소 클래스 우선(6.7.1절),
   그래도 같으면 앞선 인스턴스. X·+자 교차에서 진 선은 그 칸을 잃는다 — 사슬은 (B)에서
   그 칸을 건너뛰고, 디코더의 탐색 반경 2가 이를 복원한다(10절).
5. **`class_map`·`coord_map`.** 셀의 클래스 = 소유 선의 클래스. `coord_map` = **소유 선의
   픽셀만**으로 계산한 셀 안 무게중심:

$$
\mathbf{c}^{gt}_{ij} = \frac{1}{n_{l^*}(i,j)}\sum_{(x,y)\,\in\,\text{cell}(i,j)\,\cap\,M_{l^*}}
\left(\frac{x + 0.5}{s} - j,\;\; \frac{y + 0.5}{s} - i\right) \;\in\; [0,1)^2
$$

   원점은 셀 좌상단이다(6.1절). 픽셀 중심 보정(+0.5)을 잊지 않는다.

> 두께는 1로 둔다. 두껍게 그려도 무게중심은 같은 자리로 수렴하고 계산량만 는다.
> 좌표 정밀도가 부족하면 ×2 supersampling을 옵션으로 연다(`encode_supersample`).

#### (B) 선별 위상 → `conn_dirs`, `end_map`

6. **소유 셀 사슬.** 선 $l$의 셀 열(끝칸·3×3 제거 후)에서 **$l$이 소유한 셀만** 순서대로 남긴다.
   잃은 칸은 **건너뛰어 양옆을 직접 잇는다** — 그래서 사슬 이웃이 $3\times3$ 밖(2칸 거리)일 수
   있다. 방향은 단위벡터라 거리와 무관하게 표현에 제약이 없다.
7. **`conn_dirs`.** 사슬의 앞·뒤 이웃에 대해 **자기 점 → 이웃 점 단위 방향을 직접 계산해
   저장**한다(6.2절 식). 중간 셀은 방향 2개, 사슬 양 끝 셀은 이웃 방향 1개 + **끝점 방향** 1개
   (끝점 = 원본 폴리라인의 실제 끝, 인코더 내부 값). 모든 양성 셀이 정확히 2개 —
   구 설계의 차수 초과 절단(0.307% 실측)이 사라진다.
8. **`end_map`.** 사슬의 첫·마지막 셀이 끝 셀이다: `end_map = 1`.
9. (구 설계의 간선 합집합·차수 1 유도·다른 클래스 접합 단방향 간선, 그리고 9차의
   `conn_cells`·`end_point` 저장은 **전부 폐기** — 방향 직접 저장으로 대체.)

**불변식 (`test_encode.py`가 검증):**

1. `conn_dirs`는 양성 셀에서 **단위벡터 2개**다 ($\lVert \mathbf{d} \rVert = 1$, 항상 2개 유효).
2. **사슬 반평행성:** 같은 선의 이웃 $a$·$b$에 대해 $a$의 "$b$ 방향"과 $b$의 "$a$ 방향"은
   정확히 반대다 ($\mathbf{d}_{(a \to b)} = -\mathbf{d}_{(b \to a)}$ — 같은 두 점을 잇는 방향이므로).
3. **방향의 표적 검증:** $a$의 점에서 $\mathbf{d}_{(a \to b)}$ 방향 반직선이 $b$의 점을 지난다
   (저장된 방향을 따라가면 실제 이웃 점이 나온다 — 인코딩·디코딩 대칭의 근거).
4. `coord_map`·`end_map`·`conn_dirs`는 양성 셀에서만 유효(그 외 0).
5. **무게중심은 셀 안에 있다:** $\mathbf{c}^{gt} \in [0,1)^2$.
6. **각 양성 셀은 정확히 한 선에 속한다** (소유권 규칙의 결과).
7. **T자 접합:** 본선 A 위에서 끝나는 선 B를 넣으면, B의 끝칸(A가 지나는 셀)은 A 소유로 남고
   B의 사슬은 한 칸 물러난 끝 셀에서 끝점 방향으로 접합점을 가리킨다. A의 사슬은 끊기지 않는다.
8. **X자 교차:** 두 선이 한 셀에서 교차하면 진 선의 사슬이 그 칸을 건너뛰고(2칸 거리 이웃의
   점을 향하는 방향), 이긴 선의 사슬은 연속이다.
9. **지그재그:** 대각선에 가까운 완만한 선을 넣어도 사슬이 톱니(지그재그)가 아니라
   단조로운 셀 열로 나온다 (3×3 순위 규칙 검증).

증강(`data/augment.py`)은 **인코딩 전 벡터 단계**에서 한다. 좌우/상하 반전과 90° 회전은 폴리라인 좌표 변환만으로 끝난다. 색상 증강(밝기·대비·감마·노이즈)은 이미지에만 적용한다.

#### 6.4.1. 온라인 인코딩 vs 사전 생성

**기본은 온라인 인코딩이다. 다만 검증/테스트 split은 캐시한다.**

사전 생성(npy)의 유혹은 명확하다 — 인코딩 비용이 0이 된다. 문제는 **증강**이다. 증강을 벡터 단계에서
하기로 한 이유가 기존 저장소의 단점 #8(사전 생성 npy 때문에 증강이 부호 뒤집기 해킹으로 제한됨)이었다.
인코딩된 맵을 flip/rot90 하려면 `coord_map`은 성분을 바꾸고 $x \to 1-x$를 해야 하고,
`conn_dirs`는 방향 성분의 부호·순서를 다시 매핑해야 한다. 정확히 그 해킹으로 되돌아간다.

**비용 추정.** (A) 래스터 단계는 클래스당 `cv2.polylines` 한 번 + reshape-sum 한 번이다. 한 이미지에
등장하는 클래스는 보통 5~8종이고, 768×768 uint8 연산이라 클래스당 1 ms 안팎이다. (B) 위상 단계는
폴리라인 36개(6.7.5절 실측) × 중앙 19점이라 훨씬 싸다. **합쳐서 샘플당 20~50 ms 정도**로 본다.

이 값이 병목인지는 **GPU 스텝 시간과 `num_workers`가 정한다.** 판정 기준은 하나다.

$$
\text{샘플 준비 시간} \;>\; \text{GPU 스텝 시간} \times \text{num\_workers} \quad\Rightarrow\quad \text{캐시가 필요하다}
$$

(워커 $W$개가 병렬로 준비하므로, 워커 하나는 샘플 하나를 $W$스텝 안에만 만들면 GPU를 굶기지 않는다.
초안의 "스텝 시간 / 워커 수" 판정식은 부등호 방향이 잘못된 것이라 9차 개정에서 바로잡았다.)

**실측 결과 (재구현, 학습 캐시 불필요 확정):** 샘플 전체 처리 70 ms(파싱 30.0 + 인코딩 15.7 ms,
p99 58 ms) vs 예산 328 ms × workers 8 = 2,624 ms — **여유 37배.**
학습 캐시는 넣지 않는다. val/test 캐시는 정책대로 유지한다.
(선 단위 인코딩(9차)은 선마다 캔버스를 그려 클래스 단위보다 그리기 횟수가 늘지만 — 장당 평균
38개 — 선별 bounding box만 그리면 비용 증가는 제한적이다. M10에서 재실측한다.)

**단계적 대응 (위에서부터 순서대로):**

1. `num_workers`를 늘린다. 가장 싸다.
2. **검증/테스트 split을 캐시한다.** 증강을 하지 않으므로 GT가 결정적이다. 매 에폭 같은 계산을 반복하는
   순수한 낭비이므로 **이건 조건 없이 한다** (`DataConfig.cache_gt`, 기본 `"val_test"`).
3. 학습도 캐시해야 하면 **희소 형식**으로 저장한다. 양성 셀만 저장하면 샘플당 약 46 KB —
   `(cell_idx int32, class uint8, coord 2×float16, end uint8, conn D×2 int16)` × 노드 2,101개.
   전체 12,828장이면 약 590 MB다. dense npy(샘플당 664 KB → 8.5 GB)보다 14배 작다.
   증강 8종(flip×rot90)을 전부 구우면 4.7 GB로 여전히 감당 가능하다.
4. 그래도 부족하면 인코더를 최적화한다(클래스 마스크를 uint8 한 장에 클래스 인덱스로 그리고 bincount).

**캐시를 쓰더라도 온라인 경로가 정답이다.** 캐시는 온라인 인코더의 출력을 그대로 저장한 것이어야 하고,
`test_encode.py`가 **캐시와 온라인 결과가 같은지** 확인한다. 두 경로가 갈라지면 디버깅이 불가능해진다.

### 6.5. collate — `data/types.py`

모든 GT 키가 고정 크기라서 **전부 그대로 stack**한다.

- `image` → `(B, 3, 768, 768)`, `class_map` → `(B, 192, 192)`, `conn_dirs` → `(B, 192, 192, D, 2)`, ...
- `instances`·`meta`만 list로 유지.

### 6.6. 합성 데이터셋 — `data/synthetic.py`

실데이터 준비 전에 전 파이프라인을 검증할 개발용 데이터셋. 랜덤 시드로 이미지를 즉석 생성한다.

- 랜덤 폴리라인(직선·베지어 곡선) 수 개를 배경 위에 그린 이미지 + 같은 폴리라인으로 GT 인코딩.
- Y자 분기·다른 클래스 T자 접합·교차·완만한 대각선을 **의도적으로 섞어** 생성한다
  (6.4절 끝칸 미채움·셀 소유권·건너뛰기·3×3 순위 규칙의 실전 검증).
- 학습 초기 과적합 테스트(M5)·DDP 스모크(M8)에 사용.

### 6.7. 실데이터 파이프라인 — SEED-MAP (`data/seedmap.py`)

실데이터는 **SEED-MAP v1.1**(Shin & Choi, IEEE JSTARS vol. 18, 2025)이다.
위성 영상 768×768, 지상 해상도 0.1278 m/px, 타일당 지상 범위 약 98 m.
아래 수치는 전부 **샘플 300장(train/val/test 각 100장, 차선 인스턴스 10,787개)** 을 직접 집계한 값이다.

#### 6.7.1. 클래스 정의 — `category_id` 11종 + 배경

**클래스는 `category_id`(기능 분류)로 정한다.** `type_id`(색·실선/점선 같은 시각 속성)는 클래스에 쓰지 않는다.
목록은 논문 Table V(Mask2Former classwise 성능)의 차선 클래스와 정확히 일치하며, 배경을 합쳐 $C = 12$다.

| 학습 라벨 | `category_id` | 이름                             | 논문 IoU | 논문 전체 인스턴스 | 샘플 300장 |
| ----- | ------------- | ------------------------------ | ------ | ---------- | ------- |
| 0     | —             | background                     | 96.11  | —          | —       |
| 1     | 501           | `center_line`                  | 42.42  | 136,182    | 1,125   |
| 2     | 502           | `u_turn_zone_line`             | 45.69  | 27,381     | 153     |
| 3     | 503           | `lane_line`                    | 34.83  | 196,614    | 2,582   |
| 4     | 504           | `bus_only_lane`                | 24.79  | 9,186      | 83      |
| 5     | 505           | `edge_line`                    | 25.01  | 75,276     | 1,108   |
| 6     | 506           | `path_change_restriction_line` | 29.24  | 95,188     | 1,594   |
| 7     | 515           | `no_parking_stopping_line`     | 29.61  | 212,132    | 2,162   |
| 8     | 525           | `guiding_line`                 | 34.61  | 31,455     | 593     |
| 9     | 530           | `stop_line`                    | 31.52  | 48,353     | 1,100   |
| 10    | 531           | `safety_zone`                  | 9.60   | 10,145     | 198     |
| 11    | 535           | `bicycle_lane`                 | 33.85  | 14,309     | 89      |

- **`599`(`others`)는 제외한다.** Table V에도 없고 클래스로 세울 근거가 없다.
- **표는 확정됐다 (전체 train 8,979장 재집계, 9차 개정).** 표에 없는 `category_id`는
  `599` 697개 / `5011` 17개 / `None` 3개가 전부다 — 전체 차선의 0.2%. 셋 다 버린다.
  **`5011`은 클래스로 세울 근거가 없어 제외 확정** (구 open_questions 7). 로더는 여전히
  미등록 id를 경고와 함께 집계한다(새 데이터 유입 대비).
- **셀 소유권에 별도의 클래스 priority 표를 두지 않는다.** 한 셀에 두 클래스가 걸리면 **그 셀에 더 많이
  그려진 클래스**가 이긴다(6.4절 래스터 단계 3). 통과하는 선은 셀을 길게 가로질러 픽셀이 많고, 스쳐 지나거나
  거기서 끝나는 선은 픽셀이 적으므로, "통과선이 소유한다"는 성질이 규칙 없이 따라 나온다.
  픽셀 수가 같을 때만 위 표의 **인스턴스 수 역순**(희소 클래스 우선)으로 가른다.

#### 6.7.2. 폴더 구조와 파일 규약 (9차 개정)

원본 `SEED_MAP_v1.1`은 평평한 `image/`·`label/` + `dataset.json`(`{train, validation, test}` 키)으로
split을 표현한다. 재구현 코드는 `dataset.json`을 직접 읽었지만, **split 폴더 구조로 재정리했다** —
원본 옆 `SEED_MAP_v1.1_splits/`에 **복사**(사용자 결정 — 링크·이동 아님. 원본 무손상, 완전 독립 사본).
**파일 정리는 완료됐고**(train 8,979 / val 1,282 / test 2,567, 이미지·라벨 쌍 일치 확인),
로더 수정이 M13으로 남아 있다.

```
{data.root} = .../2026_stella/SEED_MAP_v1.1_splits/
├── train/  image/{id}.png   label/{id}.json      # 8,979장
├── val/    image/{id}.png   label/{id}.json      # 1,282장 — 원본 "validation" → "val"
└── test/   image/{id}.png   label/{id}.json      # 2,567장
```

`{id}`는 `"126.6219,37.3851"` 형태의 **타일 좌상단 경위도**다. 확장자만 다르고 stem이 같아 짝을 이룬다.
`split` kwarg로 폴더를 고르고, `label/*.json`을 glob 해서 인덱스를 만든다 — `dataset.json` 파싱이
로더에서 사라진다.

#### 6.7.3. 라벨 JSON 구조

파일당 리스트 하나. 첫 원소가 `"class": "MetaData"`, 나머지가 `"class": "RoadObject"`다.

```python
[
    {
        "class": "MetaData",
        "image_x1y1x2y2": [lon_min, lat_min, lon_max, lat_max],
        "coordinate_format": "longitude, latitude",
        "format_code": "EPSG:4326",
        "region": "Incheon, Korea",
    },
    {
        "class": "RoadObject",
        "id": "B3238A281281",
        "category_id": "5321",
        "category": "crosswalk",  # 기능 분류  ← 학습 클래스
        "type_id": "5",
        "type": "crosswalk",  # 시각 속성  ← 미사용
        "geometry_type": "POLYGON",  # "POLYGON" | "LINE_STRING"
        "image_points": [[353, 22], ...],  # 픽셀 좌표  ← 사용
        "global_points": [[126.62, 37.38], ...],
    },  # 경위도      ← 미사용
    ...,
]
```

`category_id`·`type_id`는 **문자열**로 저장돼 있다(`"5321"`). 정수로 변환해 쓰되 비교는 문자열 키로 한다.

#### 6.7.4. 로딩 규칙 (`__getitem__` 전반부)

1. **`geometry_type == "LINE_STRING"` 만 사용한다.** 샘플 16,882개 중 `POLYGON` 6,069개(화살표·횡단보도 등
   노면 기호)는 전부 버린다. 남는 것이 `LINE_STRING` 10,813개다.
2. **`category_id`를 6.7.1 표로 라벨에 매핑**한다. 표에 없으면 버리고 카운트한다 (→ 10,787개).
3. **연속 중복 점 제거.** 샘플에 7,651개의 중복 인접 점이 있다(같은 좌표가 두 번 연속). 그대로 두면
   방향 계산에서 0 벡터가 나온다. 인코딩 전에 반드시 없앤다.
4. **이미지 경계로 자른다 (중요).** `image_points`는 타일 밖까지 이어진다 — 전체 좌표 범위가
   $x \in [-17009, 14762]$, $y \in [-4770, 6062]$이고 **전체 점의 49.8%가 $[0, 767]$ 밖**이다.
   자르지 않으면 격자 인덱싱이 통째로 깨진다.
   - 경계를 넘는 구간은 **테두리와의 교점을 새 정점으로 넣어** 자른다. 이러면 잘린 끝이 정확히
     이미지 가장자리에 놓인다.
   - 실제 영향은 작다: 완전히 밖이라 버려지는 인스턴스 0.5%, 두 조각 이상으로 쪼개지는 인스턴스 0.1%.
   - **잘린 끝은 그냥 선의 끝으로 취급한다.** 6.2절 끝 규약이 동일하게 적용된다(끝칸 미채움 +
     끝 셀의 끝점 방향 = 경계 교점 방향). "타일을 벗어났다"를 구분하는 별도 플래그는 두지 않는다 —
     타일 안에서 관측 가능한 사실이 같고, 타일 밖을 예측할 방법도 없다. 경계 끝도 끝칸 미채움을
     동일 적용한다(결정 31).
5. **좌표는 정수가 아니다.** 대부분 정수지만 실수 좌표가 섞여 있다. `float32`로 읽는다.
6. 남은 폴리라인을 6.4절 인코더에 넣어 `class_map`·`coord_map`·`end_map`·`conn_dirs`를 만든다.
   `instances`(평가용 원본 폴리라인)는 **자른 뒤의 픽셀 좌표**를 그대로 통과시킨다.

#### 6.7.5. 실측 통계 — 전체 train 8,979장 재집계 (9차 개정, `scripts/stat_labels.py`)

| 항목                          | 실측 (전체 train 8,979장)                                     | 설계에 주는 함의                                        |
| --------------------------- | -------------------------------------------------------- | ------------------------------------------------ |
| 이미지                         | 768×768 RGB, 전부 동일                                       | `image_size = 768` 리사이즈 불필요                      |
| 차선 인스턴스/장                   | 평균 38.3, 중앙 34, 최대 255                                   |                                                  |
| 점/폴리라인 (경계 자른 뒤)            | 중앙 14, p99 199, 최대 478                                    | 온라인 인코딩 비용 상한 — 실측 70 ms/샘플로 여유 (6.4.1절)        |
| **GT 노드 수/장**               | **평균 2,121, p90 3,681, p99 5,893, 최대 8,909**             | **`n_max` 8000으로도 최대치를 못 담아 → 9500 확정**          |
| 미등록 `category_id`           | `599` 697 / `5011` 17 / `None` 3 (차선의 0.2%)               | 6.7.1 표 확정 — 전부 제외                               |
| 노드 차수 분포 (구 인코딩)            | 0: 6.15% / 1: 2.52% / 2: 90.63% / 3: 0.40% / 4: 0.31%    | 차수 0(고립)·3+·초과 절단(0.307%)은 **구 인코딩의 산물** — 아래 참고 |
| 종점 셀 (구 인코딩)                | 노드의 2.51%                                                 |                                                  |
| 파싱·인코딩 시간                   | 30.0 + 15.7 ms (p99 58 ms)                                | 학습 캐시 불필요 확정 (6.4.1절)                            |

**차수 통계는 구(클래스 그래프) 인코딩 기준이다.** 새 선 단위 인코딩(6.4절)에서는 분기가 구조적으로
항상 2라 차수 분포·초과 절단·고립 노드가 **정의상 사라진다**(고립의 원인이던 스침 셀은 3×3 순위
규칙이 제거). M10에서 새 인코더 기준 통계 — 사슬 길이 분포, 소유권으로 잃는 셀 비율, 2셀 이하로
줄어드는 선 비율 — 를 다시 뽑는다. 차수 2가 90.6%였다는 사실은 "사슬 가정이 데이터에 맞는다"는
근거이기도 하다.

#### 6.7.6. 증강

6.4절대로 **인코딩 전 벡터 단계**에서 한다. SEED-MAP은 위성 정사영상이라 회전·반전이 물리적으로 타당하다.

- 기하: 좌우/상하 반전, 90° 회전 (폴리라인 좌표 변환만). 클래스 의미가 바뀌지 않는다.
- 색상: 밝기·대비·감마·가우시안 노이즈 — 이미지에만.
- **적용하지 않는 것:** 임의 각도 회전과 크롭. 전자는 빈 영역을 만들고 후자는 타일 경계를 다시 만든다.
  둘 다 4번의 경계 자르기를 매번 다시 해야 해서 v1에서는 넣지 않는다.

#### 6.7.7. 알려진 제약

- 샘플 300장은 전체 12,828장의 2.3%다. 위 통계는 **경향 확인용**이고, 전체 데이터를 받으면 M9에서
  같은 스크립트로 다시 집계해 `n_max`·`max_degree`·클래스 목록을 확정한다.
- 논문은 타일 정렬(ICP)이 실패한 이미지를 이미 제외했다고 밝혔지만, 급경사 지형의 잔여 오정렬은 남아 있다.
- `safety_zone`은 면(area)인데 `LINE_STRING`으로 저장된다(논문도 이 점을 IoU 9.6의 원인으로 지목).
  샘플에서 닫힌 폴리라인은 2개뿐이라 실제로는 열린 외곽선 조각으로 들어온다. 별도 처리는 하지 않는다.

---

## 7. 모델 설계 — `stella/model/`

### 7.1. 전체 forward 흐름과 출력 계약

내부 계산은 **선택된 셀만 희소하게**(토큰 단위) 하고, 반환 직전에 결과를 격자에 **scatter해서 dense로** 되돌린다.
어텐션 스택은 $N$개 노드 위에서만 돌고, scatter 비용은 무시할 수준이다. dense 반환의 목적은
GT의 self 맵(6.2)과 **같은 격자 좌표계**를 쓰기 위함이다 — criterion이 셀 인덱싱만으로 GT와 예측을 짝짓는다.

```
image (B,3,768,768), [0,1]
 → (버퍼 mean/std로 정규화)
 → backbone                       # SFP 경로: (B,C_b,48,48) | FPNLite 경로: 4레벨
 → neck                           # (B,256,192,192) = (B, d_model, L, L)
 → heatmap head                   # (B,192,192) logit
 → 노드 선택 (7.4)                 # 셀 목록 (N,2) — 학습: GT ∪ 예측 마스크, 추론: 예측 마스크
 → 임베딩 gather + 쿼리 구성 (7.5)  # q: (N, K, 256)
 → 어텐션 스택 ×6 (7.6)            # (N, K, 256)
 → 출력 헤드 (7.7)                 # 토큰별 예측
 → scatter to grid                # 아래 dense 텐서로 반환
```

`StellaModel.forward(image, gt_positive=None)` → `ModelOutput` (dataclass).
아래 표가 **모델 출력 계약**이다. shape은 배치 차원을 뺀 한 샘플 기준, $C = 12$, $R$ = 연결 슬롯 수(2, 6.1절 기호표).
`node_mask`가 거짓인 셀의 값은 전부 0이며 **의미가 없다** — criterion·디코더는 반드시 `node_mask`(또는 GT 양성 맵)로 걸러 쓴다.

| 필드              | dtype / shape              | 범위       | 의미                                                            |
| --------------- | -------------------------- | -------- | ------------------------------------------------------------- |
| `heatmap_logit` | `float32 (192, 192)`       | logit    | 보조 히트맵 (셀에 노면 표시가 있는가). $\sigma$를 씌우면 확률                      |
| `node_mask`     | `bool (192, 192)`          | —        | 이번 forward에서 토큰을 계산한 셀 (7.4의 선택 결과)                           |
| `class_logit`   | `float32 (192, 192, C)`    | logit    | self 슬롯의 클래스 예측 (0 = 배경 포함)                                   |
| `self_coord`    | `float32 (192, 192, 2)`    | $[0, 1]$ | self 슬롯의 셀 내 좌표. sigmoid. 원점 = **셀 좌상단**                      |
| `end_logit`     | `float32 (192, 192)`       | logit    | self 슬롯의 "**이 셀이 사슬의 끝**"일 확률 — `end_map` 직접 감독 (9차 개정)       |
| `exist_logit`   | `float32 (192, 192, R)`    | logit    | 슬롯별 연결 존재 확률                                                  |
| `conn_dir`      | `float32 (192, 192, R, 2)` | 단위벡터     | 슬롯별 연결 방향. `F.normalize`. 원점 = **자기 노드 점**(6.1절), 목표 = 상대 노드 점 |

구 설계의 슬롯별 `t_logit`(연결 대상이 종점/다른 클래스 접합)은 폐기했다 — 끝 판정은 셀 단위
`end_logit`이 맡고, 다른 클래스 접합 간선은 새 인코딩에 없다(6.4절). *(사용자 확정 — 결정 32.)*

**GT와 모델 출력은 같은 형태다 (10차 개정 — 설계 방침 1).** heatmap↔`class_map > 0`,
`class_logit`↔`class_map`, `self_coord`↔`coord_map`, `end_logit`↔`end_map`,
`conn_dir`↔`conn_dirs`가 짝을 이룬다. $R = D = 2$(결정 1)라 개수까지 같고, 남는 차이는
**슬롯 순서**(무순서)뿐이다. 그래서 criterion이 하는 일은 유도가 아니라 **매칭**(8.3)과
손실 계산뿐이다.

### 7.2. Backbone — `model/backbone.py`

여러 백본을 비교 실험할 것이므로 **3층 구조**로 둔다. 모델 계열마다 출력 형태가 다르므로 **계열별로 클래스 하나**를
만들고, 라이브러리 공통 작업(가중치 로드·전처리 상수·특징 추출 호출 규약)은 **중간 인터페이스 클래스**로 모은다.

```
Backbone(nn.Module, Buildable)          # 계약: forward(x) -> list[Tensor], out_channels, strides
├── HuggingFaceBackbone                 # transformers 공통: AutoModel/AutoImageProcessor 로드,
│   │                                   #   processor에서 mean/std 추출, 게이트 토큰 처리
│   ├── Dinov3Backbone                  #   ViT 패치 토큰 → 1레벨 맵
│   ├── PerceptionEncoderBackbone
│   └── ...
└── TimmBackbone                        # timm 공통: create_model(features_only=True),
    │                                   #   default_cfg에서 mean/std 추출
    ├── SwinBackbone                    #   4레벨 (stride 4/8/16/32)
    ├── ConvNeXtBackbone
    └── ...
```

**계열 안의 스케일 변화(large/base/small/tiny)는 한 클래스가 처리한다.** 클래스를 고르는 것은 `name`이고,
스케일은 `pretrained` 문자열이 정한다. 채널 수·레이어 수는 로드한 모델에서 읽어 `out_channels`에 채운다.

```python
cfg.model.backbone.name = "Dinov3Backbone"
cfg.model.backbone.pretrained = (
    "facebook/dinov3-vitl16-pretrain-sat493m"  # → vitb16으로 바꾸면 base
)
```

| 층                     | 책임                                                                                            |
| --------------------- | --------------------------------------------------------------------------------------------- |
| `Backbone`            | 계약만 정의. `out_channels: tuple[int,...]`, `strides: tuple[int,...]`, `pixel_mean/std` 버퍼        |
| `HuggingFaceBackbone` | `AutoModel.from_pretrained` 로드, `AutoImageProcessor`에서 정규화 상수 추출, `freeze` 처리                 |
| `TimmBackbone`        | `timm.create_model(pretrained=True, features_only=True)` 로드, `default_cfg`에서 정규화 상수 추출        |
| 계열 클래스                | **출력 형태를 `list[Tensor]`(stride 오름차순)로 맞추는 일.** ViT류는 패치 토큰 → `(B, C, h, w)` reshape, 계층형은 그대로 |

정규화 상수를 백본이 들고 있으므로 데이터셋은 `[0,1]` RGB만 내놓으면 되고(6.2절), 백본을 바꿔도
`StellaModel` 바깥은 불변이다.

| 우선순위       | 클래스 / 모델                                                                                 | 출력                          | 비고                                                 |
| ---------- | ---------------------------------------------------------------------------------------- | --------------------------- | -------------------------------------------------- |
| **1 (확정)** | `Dinov3Backbone` — **DINOv3 ViT-L/16 위성 사전학습** `facebook/dinov3-vitl16-pretrain-sat493m` | 패치 토큰 → `(B, 1024, 48, 48)` | 위성 영상 4.9억 장 사전학습 dense 특화. HF 게이트 라이선스 — 계정 동의 필요 |
| 2          | `SwinBackbone` — SwinV2-L 등                                                              | 4레벨 (stride 4/8/16/32)      | 게이트 없음. **FPNLite 경로 검증용**                         |
| 3          | `ConvNeXtBackbone` — ConvNeXtV2-L 등                                                      | 4레벨                         | CNN 대조군                                            |
| 4          | `PerceptionEncoderBackbone` 등                                                            | 계열마다 다름                     | 필요할 때 계열 클래스 하나 추가로 확장                             |

- 백본 추가 = **계열 클래스 하나 추가**. 라이브러리가 이미 있으면 중간 클래스를 상속해 `forward` 출력 정리만 하면 된다.
- InternImage는 **지원하지 않는다**(13절 결정 4 — DCNv3 커스텀 CUDA 커널이 원칙 #6 위반).
- **`timm`·`transformers` import는 중간 인터페이스 클래스 안에서 한다.** 모듈 최상단에 두면 한쪽 라이브러리만
  깔린 환경에서 `check_all`(5.1절)이 통째로 죽는다.
- 1레벨만 내는 백본(ViT)은 `SFP`와, 4레벨을 내는 백본은 `FPNLite`와 짝을 이룬다. `Neck.from_cfg`가
  `backbone.out_channels` 길이를 보고 맞지 않으면 즉시 에러를 낸다.

### 7.3. Neck — `model/neck.py`

`Neck` 베이스클래스의 하위로 구현하고(config `model.neck`으로 고른다, 5절), 백본이 무엇이든
**공통으로 `(B, 256, 192, 192)`** = `(B, d_model, L, L)`를 낸다. 이 격자가 이후 전부(히트맵·노드 선택·토큰 임베딩)의 좌표계다.

**정규화는 LayerNorm/GroupNorm만 쓴다.** `batch_size = 1`로 시작하므로(결정 8) BatchNorm은 통계가 무의미하다.

#### `SFP` — 단일 스케일 ViT용 (ViTDet의 Simple Feature Pyramid를 1레벨로 축소)

입력은 stride 16의 한 레벨 `(B, C_b, 48, 48)`(DINOv3 ViT-L이면 $C_b = 1024$). stride 16 → 4는 4배 확대다.
**전치합성곱 2단**으로 올린다. 한 번에 4배(`stride=4`) 올리지 않는 이유는 격자 무늬(checkerboard)가 심해지기 때문이다.

```
x: (B, C_b, 48, 48)                                   stride 16
├─ ConvTranspose2d(C_b, 512, kernel=2, stride=2)      → (B, 512, 96, 96)     stride 8
├─ LayerNorm2d(512) → GELU
├─ ConvTranspose2d(512, 256, kernel=2, stride=2)      → (B, 256, 192, 192)   stride 4
├─ LayerNorm2d(256) → GELU
├─ Conv2d(256, 256, kernel=3, padding=1)              ← 전치합성곱의 격자 무늬 완화 + 국소 평활
└─ LayerNorm2d(256)                                   → (B, 256, 192, 192)
```

- `kernel=2, stride=2`는 입력 픽셀당 출력 2×2가 겹치지 않게 대응해 정렬이 단순하다. 그래도 학습 초기에는
  블록 경계가 보이므로 **마지막 3×3 합성곱이 필수**다.
- `LayerNorm2d`는 채널 축 LayerNorm을 `(B,C,H,W)`에 적용한 것(`permute → LayerNorm → permute`).
- 파라미터가 첫 단(`1024×512×2×2 ≈ 2.1M`)에 몰린다. 메모리가 문제면 1×1 Conv로 채널을 먼저 512로 줄인다.

#### `FPNLite` — 멀티스케일 백본용

입력은 4레벨 `[(B,C_1,192,192), (B,C_2,96,96), (B,C_3,48,48), (B,C_4,24,24)]`(stride 4/8/16/32).
표준 FPN에서 **레벨별 출력을 만들지 않고 stride-4 한 레벨만 내보내는 것**이 "Lite"의 뜻이다.

```
c4 (s=32) ─ 1×1 Conv(C_4→256) + GN ─────────────── p4
                                                    │ nearest ×2
c3 (s=16) ─ 1×1 Conv(C_3→256) + GN ───── (+) ────── p3
                                                    │ nearest ×2
c2 (s=8)  ─ 1×1 Conv(C_2→256) + GN ───── (+) ────── p2
                                                    │ nearest ×2
c1 (s=4)  ─ 1×1 Conv(C_1→256) + GN ───── (+) ────── p1
                                                    │
                                   3×3 Conv(256→256) + GN
                                                    ↓
                                          (B, 256, 192, 192)
```

- lateral은 전부 `1×1 Conv + GroupNorm(32)`. 상위 레벨은 `F.interpolate(mode="nearest")`로 2배 올려 더한다.
  nearest를 쓰는 이유는 bilinear가 얇은 선을 흐리기 때문이다.
- 마지막 `3×3 Conv + GN`은 top-down 덧셈이 만드는 계단 현상(aliasing)을 없앤다. FPN의 output conv와 같은 역할이다.
- 출력이 한 레벨뿐이라 상위 레벨은 **문맥 주입용**으로만 쓰인다. 얇은 선의 위치 정밀도는 $c_1$이 결정한다.

> 참고: 메모리가 부족하면 `grid_stride=8`(L=96)로 낮출 수 있게 열어 둔다. `SFP`는 전치합성곱을 1단으로,
> `FPNLite`는 $p_2$에서 멈추면 된다. 6절이 전부 $s$로 매개변수화되어 있어 라벨 인코딩도 자동으로 따라간다.

### 7.4. 히트맵 헤드 + 노드 선택 — `model/heatmap.py`

- 헤드: `1×1 Conv(256→1)` — 셀이 노면 표시 위인지 이진 판단. focal 손실(8.1)로만 학습된다.
- **학습 시 선택 (기본, `node_sampling="gt+pred"`):** **GT 양성 맵 ∪ 예측 마스크.**
  예측 마스크는 추론과 같은 경로($\sigma(\text{logit}) > \tau_h$ → dilation)로 만든다.
  GT 쪽은 감독이 있는 셀을 보장하고, 예측 쪽은 추론에서 만날 거짓 양성 셀을 미리 노출시켜
  "존재 안 함"(exist=0)을 학습시킨다(8.4). $N > N_{\max}$면 GT 셀은 전부 유지하고 예측 셀만 확률순으로 자른다.
  (`"gt"` 옵션: GT 셀만 사용 — 초기 디버깅·과적합 테스트용.)
- **추론 시 선택:** $\sigma(\text{logit}) > \tau_h$ → `max_pool2d` dilation(3×3) → $N_{\max}$ 상한.
  (architecture.md 3.2절: 낮은 임계값 + dilation으로 재현율 우선.)
- **최소 1노드 보장:** 선택이 비면 히트맵 최대값 셀 1개를 강제로 넣는다. 뒤 모듈이 항상 실행되어야 DDP unused-parameter 문제가 없다(9.6).
- 선택은 하드 연산이라 그래디언트가 없다. 단, gather된 임베딩을 통해 **encoder까지는 그래디언트가 흐른다.**

### 7.5. 쿼리 구성 — `model/stella.py` 내부

- 역할 임베딩 `role_embed: nn.Parameter (K, 256)` — $\mathbf{u}_0$ = self, $\mathbf{u}_{1..R}$ = 연결 슬롯.
- 노드 임베딩: 선택 셀 $(i_n, j_n)$에서 $\mathbf{z}_n = F[:, :, i_n, j_n]$ gather → `(N, 256)`.

$$
\mathbf{q}_{n,k} = \mathbf{z}_n + \mathbf{u}_k \qquad \Rightarrow \quad \text{q: } (N, K, 256)
$$

- cross-attention의 key/value는 **선택된 노드 임베딩 $\mathbf{z}$ 그대로**(스택을 지나도 갱신하지 않는 고정 memory). 근거는 7.6절.
- 배치 처리: 노드 단계는 **이미지별 루프**로 처리한다(v1). 배치를 $2^n$으로 키우는 실험(9.3)에서 병목이 되면 패딩+마스크로 벡터화한다.

### 7.6. 어텐션 스택 — `model/blocks.py`, `model/rope.py`

블록 구성(×6, pre-LN, residual):

```
[슬롯 간 self-attn]  q(N,K,256) 내 같은 노드의 K개 토큰끼리   ← 슬롯 분화 유도
→ [cross-attn]       q(N·K, 256) → kv: z(N, 256). layer 0 = 전역, 1~5 = 윈도우 w=7
→ [FFN]              256 → 1024 → 256
```

**key/value는 전체 feature map이 아니라 선택된 $N$개 노드의 임베딩 $\mathbf{z}$다.** 세 가지 이유다.

1. **비용.** 전체 격자를 memory로 쓰면 키가 $L^2 = 36{,}864$개다. 노드만 쓰면 $N \approx 2{,}000$(6.7.5절 실측 평균)로
   **18배 작다.** 쿼리는 $NK \approx 6{,}000$개($K = 3$)이므로 전역 층의 어텐션 행렬이
   $6{,}000 \times 36{,}864$ 대신 $6{,}000 \times 2{,}000$이 된다.
2. **정보 손실이 거의 없다.** 노드 선택은 히트맵 기반이라(7.4) 노면 표시가 있을 만한 셀을 이미 다 포함한다.
   버려지는 것은 아스팔트·건물 같은 배경 셀이고, 연결성 추론에 필요한 것은 **다른 노드가 어디 있는가**다.
3. **윈도우 마스크가 셀 좌표로 정의된다.** 쿼리도 키도 셀 좌표를 가진 노드라야 $\max(|\Delta i|, |\Delta j|) \le (w-1)/2$가
   그대로 성립한다.

**memory는 스택을 지나도 갱신하지 않는다.** neck 출력에서 한 번 gather한 $\mathbf{z}$를 6개 층이 공유한다.
갱신형(self-attention 스택)으로 바꾸면 층마다 $N \times N$ 갱신이 추가되는데, 노드 임베딩 자체를 정제하는 것이
목적이 아니라 **슬롯 쿼리가 주변 노드를 훑는 것**이 목적이므로 고정 memory로 충분하다 (architecture.md 3.3절).

- **MHA는 직접 구현**한다(선형 qkv + `F.scaled_dot_product_attention` + 출력 프로젝션, 약 30줄). `nn.MultiheadAttention`은 RoPE를 끼워 넣을 수 없다.
- **윈도우 층은 $N \times N$ 마스크가 아니라 이웃 gather 방식이다 (재구현에서 채택 확정).**
  셀당 노드가 최대 하나이므로 $w \times w$ 격자 오프셋을 그대로 gather 하면 결과가 같으면서
  어텐션 행렬이 $(N, K, w^2)$로 줄어든다. 9.6절이 대비책으로 예고했던 교체안이 기본이 됐다.
- **`window_size = 7` (계획 9에서 하향, 실측 근거).** 활성 메모리가 $w^2$에 비례하는데
  ($k$·$v$·RoPE 사본이 노드당 $w^2 \times 256$짜리 텐서로 층마다 생긴다) 실제 연결은
  디코더 탐색 반경 2셀 안에서 일어난다 — $\pm 4$셀(w=9)을 볼 근거가 없다.
  실측(n_max 6000, bs 1): **peak 12.09 → 9.72 GiB, step 455 → 291 ms.**
- **윈도우 층만 gradient checkpointing** (`grad_checkpoint`, 기본 on). 활성의 대부분이 위
  gather에 있고 재계산 비용은 gather + 어텐션뿐이라 싸다. 전역 층은 kv가 $(N, 256)$ 하나라
  제외한다. 기울기 불변은 테스트로 고정(`test_grad_checkpoint_leaves_gradients_unchanged`).
  실측: w=7과 함께 **peak 6.21 GiB / step 334 ms** → `n_max = 9500`·`batch_size = 2`가
  모두 가능해졌다(9.3절).
- w=7의 **정확도 영향은 미검증**이다 — 수용 영역 $\pm 4 \to \pm 3$셀은 모델 변경이므로
  재학습에서 w=9와 비교한다(14절 의문). 손해가 보이면 w=9 + checkpointing만으로도
  bs=2가 들어간다(6.33 GiB).
- **2D axial RoPE** (`rope.py`, RoPE-ViT 방식): head 차원을 반으로 나눠 앞쪽은 $x$(=$j$), 뒤쪽은 $y$(=$i$) 좌표로 회전. 주파수 베이스 100. cross-attn의 q·k 양쪽에 적용한다. 위치는 **셀 정수 좌표**를 쓴다. 같은 노드의 K개 슬롯은 같은 위치를 공유한다. 슬롯 간 self-attn은 위치가 전부 같아 RoPE가 무의미하므로 생략한다.
- 단위 테스트(`test_rope.py`): 모든 노드 위치를 $(+\Delta i, +\Delta j)$ 평행이동해도 attention 로짓이 불변인지 확인(상대위치 성질).

### 7.7. 출력 헤드 — `model/heads.py`

최종 토큰 `(N, K, 256)`에서 토큰별 작은 MLP를 거친 뒤, 7.1의 dense 텐서로 scatter한다.

| 입력 토큰                       | 헤드     | 출력 (셀당)          | 활성화               | 원점/범위                |
| --------------------------- | ------ | ---------------- | ----------------- | -------------------- |
| self ($k=0$)                | 2층 MLP | `self_coord` 2   | sigmoid           | 셀 좌상단, $[0,1]$       |
| self ($k=0$)                | 2층 MLP | `class_logit` 12 | 없음                | logit                |
| self ($k=0$)                | 2층 MLP | `end_logit` 1    | 없음                | logit (9차 개정)        |
| 연결 ($k \ge 1$, 슬롯 간 가중치 공유) | 2층 MLP | `exist_logit` 1  | 없음                | logit                |
| 〃                           | 〃      | `conn_dir` 2     | **`F.normalize`** | **자기 점** 기준 **단위 방향** |

- 연결 슬롯은 **방향만 예측한다.** 상대 노드까지의 거리·좌표는 예측하지 않고, 감독도 방향 차이(1−내적)로만 준다(8.4).
  디코딩에서 상대 정점을 고를 때는 **양쪽 슬롯의 마주봄**($\mathbf{c}\cdot\mathbf{n} \to -1$)과
  실제 상대 방향 정렬로 비용을 만든다(10.3절).
- **끝 판정은 자기 셀의 $\hat{\mathrm{end}}$가 담당한다** — 구 설계처럼 이웃 셀의 슬롯($\hat{t}$)에
  맡기지 않는다. 끝 셀도 분기 2개(안쪽 + 끝방향)를 정상적으로 예측한다(6.2 끝 규약).

---

## 8. 손실 — `stella/loss/`

### 8.0. 구조 — 세 모듈 + 조립

모델과 같은 방식으로 조립한다. 손실 종류마다 클래스 하나를 두고, `StellaCriterion`이 그것들을 모아 총합을 낸다.
각 모듈은 **자기 config를 갖고, 그 안에 세부 항목별 가중치가 있다.**

```
StellaCriterion (LossConfig)                      # 조립 + 가중합
├── HeatmapLoss   (HeatmapLossConfig)             # 8.1  히트맵 focal BCE
├── SelfSlotLoss  (SelfSlotLossConfig)            # 8.2  self 슬롯: 클래스·좌표·끝(end)
└── ConnLoss      (ConnLossConfig)                # 8.3~8.4  매칭 + 존재·방향
                  └── loss/matching.py            # 8.3  셀별 슬롯 배정 (ConnLoss가 호출)
```

**가중치는 한 층뿐이다.** 실제로 계산되는 **최하위 손실 항목마다 가중치 하나**를 두고, 그것이 총합에
그대로 들어간다. 모듈 단위의 상위 가중치는 **두지 않는다** — 항목 하나의 실효 가중치가 두 값의 곱이 되면
"conn을 절반으로 줄이려면 어디를 건드려야 하나" 같은 혼란이 생기고, 파라미터만 늘어 실수하기 쉽다.

$$
\mathcal{L} = w_{hm}\mathcal{L}_{hm}
\;+\; w_{cls}\mathcal{L}_{cls} + w_{coord}\mathcal{L}_{coord} + w_{end}\mathcal{L}_{end}
\;+\; w_{e}\mathcal{L}_{e} + w_{dir}\mathcal{L}_{dir}
$$

가중치 6개가 전부이고 **각각이 정확히 하나의 손실 항목에 대응한다.**
(9차 개정: 슬롯 단위 $w_t$가 빠지고 셀 단위 $w_{end}$가 들어왔다.)

| 가중치         | 항목                   | 소유 config            |
| ----------- | -------------------- | -------------------- |
| $w_{hm}$    | 히트맵 focal BCE (8.1)  | `HeatmapLossConfig`  |
| $w_{cls}$   | self 슬롯 클래스 CE (8.2) | `SelfSlotLossConfig` |
| $w_{coord}$ | self 슬롯 좌표 SmoothL1  | 〃                    |
| $w_{end}$   | 끝 셀 BCE (8.2)        | 〃                    |
| $w_{e}$     | 연결 존재 BCE (8.4)      | `ConnLossConfig`     |
| $w_{dir}$   | 연결 방향 (1 − 내적)       | 〃                    |

`focal_alpha`·`focal_gamma`와 `match_w_dir`·`match_w_exist`는 **가중치가 아니다.** 앞의 둘은 focal 항의 형태를,
뒤의 둘은 **매칭 비용**(8.3절, 어느 슬롯이 어느 분기를 맡을지)을 정한다. 총합의 크기에 곱해지지 않는다.

**공통 인터페이스.** 각 모듈은 `forward(output: ModelOutput, targets: dict) -> dict[str, Tensor]`로
**항목별 원시 손실 dict**를 낸다. `"total"` 키에 자기 항목들의 가중합을 담는다.
원시 값을 같이 내는 이유는 가중치를 조정할 때 항목별 실제 크기를 봐야 하기 때문이다.

```python
class HeatmapLoss(nn.Module, Buildable):
    def __init__(self, *, w_heatmap: float, focal_alpha: float, focal_gamma: float): ...
    def forward(self, output, targets) -> dict[str, Tensor]:
        return {"focal": l_hm, "total": self.w_heatmap * l_hm}


class SelfSlotLoss(nn.Module, Buildable):
    def __init__(self, *, w_class: float, w_coord: float): ...
    def forward(self, output, targets) -> dict[str, Tensor]:
        return {
            "class": l_cls,
            "coord": l_coord,
            "end": l_end,
            "total": self.w_class * l_cls + self.w_coord * l_coord + self.w_end * l_end,
        }


class ConnLoss(nn.Module, Buildable):
    def __init__(
        self,
        *,
        num_conn_slots: int,
        w_exist: float,
        w_dir: float,
        match_w_dir: float,
        match_w_exist: float,
    ): ...
    def forward(self, output, targets) -> dict[str, Tensor]:
        return {
            "exist": l_e,
            "dir": l_dir,
            "match_ambiguity": amb,  # 손실 아님 — 감시 지표(8.3)
            "total": self.w_exist * l_e + self.w_dir * l_dir,
        }


class StellaCriterion(nn.Module):
    @classmethod
    def from_cfg(cls, module_cfg: LossConfig, cfg: ExperimentConfig, **kwargs):
        return cls(
            heatmap=build_instance(module_cfg.heatmap, cfg, base=LossModule),
            self_slot=build_instance(module_cfg.self_slot, cfg, base=LossModule),
            conn=build_instance(module_cfg.conn, cfg, base=LossModule),
        )

    def forward(self, output, targets) -> dict[str, Tensor]:
        out, total = {}, 0.0
        for name, mod in self.losses.items():  # "heatmap" | "self_slot" | "conn"
            d = mod(output, targets)
            out |= {f"{name}/{k}": v for k, v in d.items()}
            total = total + d["total"]  # ← 계수 없이 그대로 더한다
        out["total"] = total
        return out
```

**`StellaCriterion`은 곱하지 않고 더하기만 한다.** 가중치를 아는 것은 그 항목을 계산하는 모듈뿐이라,
"어느 가중치가 어디에 걸리는지"를 한 곳만 보면 안다.

`ConnLoss`가 `num_conn_slots`를 필요로 하는데 이 값은 `cfg.model`에 있다. `from_cfg`가 전체 cfg를 받으므로
(조립 규칙 1) `cfg.model.num_conn_slots`를 직접 읽으면 된다 — `LossConfig`에 중복 저장하지 않는다.

**반환 dict가 그대로 로그가 된다**(9.4절). `StellaTrainModule`은 키를 알 필요 없이 전부 로깅한다.
손실 항목을 추가해도 로깅 코드를 고치지 않아도 된다.

### 8.0.1. 표기와 감독 범위

배치에 대해 평균한다는 전제로, 아래는 모두 한 이미지 기준이다.

| 기호                                            | 뜻                                                                 | 출처                     |
| --------------------------------------------- | ----------------------------------------------------------------- | ---------------------- |
| $L, C, R, D$                                  | 격자 192 / 클래스 12 / 슬롯 2 / GT 분기 2 (6.1절 기호표)                        | config                 |
| $Y_{ij}$                                      | 셀의 GT 클래스 (0 = 배경)                                                | `class_map`            |
| $\mathcal{P}$                                 | GT 양성 셀 집합 $\{(i,j): Y_{ij} > 0\}$, 크기 $                          | \mathcal{P}            |
| $\mathcal{S}$                                 | 이번 스텝에 토큰이 계산된 셀 (7.4 선택 결과, $\mathcal{P} \subseteq \mathcal{S}$) | `node_mask`            |
| $\mathbf{c}^{gt}_{ij}$, $\mathrm{end}_{ij}$   | 셀 내 GT 좌표, 끝 셀 여부                                                 | `coord_map`, `end_map` |
| $\mathbf{d}^{gt}_m$                           | 분기 $m$의 GT 방향 ($m = 1, 2$ — 항상 2개). **인코더가 직접 저장** (6.2절, 유도 없음)  | `conn_dirs`            |
| $\hat{h}, \hat{\mathbf{s}}, \hat{\mathbf{c}}, \hat{\mathrm{end}}$ | 예측 히트맵 확률·클래스 로짓·셀 내 좌표·끝 로짓                      | 모델                     |
| $\hat{e}_k, \hat{\mathbf{d}}_k$               | 슬롯 $k$의 존재 로짓·단위 방향 (셀 첨자 생략)                                     | 모델                     |
| $\sigma$                                      | sigmoid                                                           | —                      |

**손실을 주는 범위.** 셀 집합마다 감독이 다르다. (9차 개정 — 끝 셀 특례가 사라졌다:
끝칸을 채우지 않으므로 모든 양성 셀의 감독이 균일하다.)

| 셀 집합                                | 뜻                      | 히트맵   | class       | coord | end | 연결 exist  | 연결 dir |
| ----------------------------------- | ---------------------- | ----- | ----------- | ----- | --- | --------- | ------ |
| $\mathcal{P}$                       | GT 노드 (분기 항상 2)        | 음성/양성 | ✅           | ✅     | ✅   | 전 슬롯 1 ($R = D$) | 매칭된 쌍만 |
| $\mathcal{S} \setminus \mathcal{P}$ | **거짓 양성** — 뽑혔지만 도로 아님 | 음성    | 배경 CE (8.2) | ❌     | ❌   | 전부 0      | —      |

**거짓 양성 셀($\mathcal{S} \setminus \mathcal{P}$)이란**: 7.4절의 노드 선택은 GT 양성 셀 $\mathcal{P}$에
**히트맵 예측 마스크**를 합집합한다. 히트맵이 아직 부정확하면 실제로는 도로 표시가 없는 셀도 뽑히는데, 그것이
$\mathcal{S} \setminus \mathcal{P}$다. 이 셀들을 일부러 넣는 이유는 **추론에서 반드시 만나기 때문**이다 —
추론에는 GT가 없어 히트맵 마스크만으로 노드를 고르므로, 거짓 양성이 섞인 상태로 뒤 모듈이 돌아간다.
학습 때 미리 노출시켜 "여긴 아무것도 없다"를 배우게 한다.

### 8.1. `HeatmapLoss` — focal BCE

히트맵 GT는 $\mathbf{1}[Y > 0]$이다. 배경 셀이 압도적으로 많아(양성 수천 / 전체 36,864) 그대로 BCE를 쓰면
배경이 학습을 지배하므로, focal 항으로 쉬운 배경의 기여를 줄인다. **모든 셀**에서 계산하고 양성 수로 정규화한다.

$$
\mathcal{L}_{hm} = -\frac{1}{|\mathcal{P}|} \sum_{i,j}
\begin{cases}
\alpha \, (1 - \hat{h}_{ij})^{\gamma} \log \hat{h}_{ij} & Y_{ij} > 0 \\
(1-\alpha) \, \hat{h}_{ij}^{\gamma} \log (1 - \hat{h}_{ij}) & Y_{ij} = 0
\end{cases}
$$

기본값 $\alpha = 0.25$, $\gamma = 2$ (`HeatmapLossConfig.focal_alpha/gamma`).

### 8.2. `SelfSlotLoss` — 클래스·좌표·끝

**클래스 손실** — $\mathcal{P} \cup (\mathcal{S} \setminus \mathcal{P})$, 즉 선택된 전 셀

$$
\mathcal{L}_{cls} = \frac{1}{|\mathcal{S}|} \sum_{(i,j) \in \mathcal{S}}
\mathrm{CE}\!\left(\hat{\mathbf{s}}_{ij},\; Y_{ij}\right), \qquad Y_{ij} = 0 \ \text{on}\ \mathcal{S} \setminus \mathcal{P}
$$

- **구 설계의 "종점 셀 제외" 특례는 폐기했다** (9차 개정). 특례의 근거는 "선의 끝 셀은 다른
  차선이 지나가는 자리일 수 있어 라벨이 모호하다"였는데, 새 끝 규약(6.2)이 그 다툼 셀 자체를
  채우지 않으므로 **모호한 셀이 애초에 존재하지 않는다.** 모든 양성 셀의 클래스는 소유 선이
  유일하게 정한다.
- **거짓 양성 셀은 배경(0)으로 감독한다.** 클래스 0을 한 번도 학습하지 않으면 `argmax`가 0을 낼 이유가
  없어져 **디코더의 배경 필터(10.2절 $\arg\max \neq 0$)가 무력해진다.** 히트맵 임계값 하나에만 의존하는
  대신 두 번째 걸름망을 만든다.

**좌표 손실** — $\mathcal{P}$ 전체

$$
\mathcal{L}_{coord} = \frac{1}{|\mathcal{P}|} \sum_{(i,j) \in \mathcal{P}} \mathrm{SmoothL1}\!\left(\hat{\mathbf{c}}_{ij} - \mathbf{c}^{gt}_{ij}\right)
$$

**끝 손실** (9차 개정) — $\mathcal{P}$ 전체. `end_map`을 직접 감독한다:

$$
\mathcal{L}_{end} = \frac{1}{|\mathcal{P}|} \sum_{(i,j) \in \mathcal{P}}
\mathrm{BCE}\!\left(\sigma(\hat{\mathrm{end}}_{ij}),\; \mathrm{end}_{ij}\right)
$$

끝 셀은 양성 셀의 약 2.5%(6.7.5절 — 새 인코딩에서는 끝칸 미채움으로 소폭 는다)라 불균형이
있지만, 히트맵과 달리 수백:1이 아니므로 일단 pos_weight 없이 시작하고 곡선을 보고 판단한다.

### 8.3. 연결 슬롯 매칭 — `loss/matching.py` (`ConnLoss`가 호출)

**왜 필요한가.** 슬롯에는 방향 역할이 미리 정해져 있지 않다(앞/뒤 순서가 없다). 셀의 슬롯 2개 중
**어느 슬롯이 어느 GT 분기를 맡는지**만 정하면 된다. GT 분기는 **모든 셀에서 정확히 2개**고
(중간 셀 = 앞·뒤 이웃, 끝 셀 = 안쪽 이웃 + 끝방향, 6.2절) $R = D = 2$(결정 1)라 배정은
"그대로 vs 교차" 두 순열 중 하나를 고르는, 셀마다 독립인 최소 크기 문제다. 무매칭 슬롯은 없다.

**매칭 비용 (방향 정렬 + 존재 확률, 13절 결정 5):**

$$
\mathcal{C}(k, m) = \lambda_{dir}\left(1 - \hat{\mathbf{d}}_k \cdot \mathbf{d}^{gt}_m\right) - \lambda_{e}\,\sigma(\hat{e}_k)
$$

1. **방향 정렬 (주 신호).** 두 단위벡터의 내적이 1이면 완전 일치(비용 0), 반대 방향이면 2.
2. **존재 확률.** 이 항은 $m$에 무관해서, 원래 역할은 분기 수 $< R$일 때 어느 슬롯이 매칭에
   뽑힐지를 정하는 것이었다. **$R = D = 2$에서는 모든 순열이 모든 슬롯을 쓰므로 순열 간
   상수가 되어 배정에 영향이 없다** — $R > D$ ablation(`exp_r3`)에서만 작동한다.
   식은 일반형으로 유지한다.

**알고리즘 (전 셀 동시, GPU 벡터 연산).** $P = |\mathcal{P}|$.

1. $\mathcal{P}$의 셀에서 예측 슬롯 `(P, 2, ·)`과 GT 분기 방향 `conn_dirs` `(P, 2, 2)`를 gather —
   유도 계산이 없다(10차 개정).
2. 비용 텐서 `C: (P, R, D)` 계산. $R = D = 2$라 패딩이 없다
   ($R > D$인 ablation에서만 분기 축을 패딩하고 그 칸의 비용을 상수 0으로 둔다).
3. 슬롯 순열을 전부 나열: `perms (R!, R)`. $R = 2$면 **2개**(그대로 / 교차).
4. 순열별 총비용 `(P, R!)`을 gather+sum으로 만들고 `argmin` → 셀별 최적 배정.
5. 결과: `matched (P, R)` bool 마스크(기본 설정에서는 전부 참)와, 슬롯이 맡은 분기 인덱스.

$R \le 4$ 전제의 완전탐색이다 ($R \ge 5$가 필요해지면 그때 LSA로 교체).
`test_matching.py`는 무작위 비용에서 이 결과를 `scipy.optimize.linear_sum_assignment`와 대조한다.

**감시 지표 `match_ambiguity`** (구 이름 `switch_rate` — 9차 개정에서 개명). 계획서 초안은
"배정이 스텝 간에 얼마나 바뀌는지"를 요구했지만, 스텝마다 배치 이미지가 달라 같은 셀을 추적할 수
없다. 같은 목적(매칭 불안정성)을 **한 스텝 안에서** 잰다 — **최적 순열과 차선 순열의 총비용 차가
0.05 미만인 셀의 비율**(= 다음 스텝에 뒤집히기 쉬운 배정). 재구현 실측에서 의도한 신호를 준다:
합성 과적합 1.000 → 0.098, 실데이터 0.503(ep0) → 0.099(ep22). 구 인코딩에서는 유효 분기
$\le 1$인 셀(고립·종점, 9.9%)이 구조적으로 항상 "모호"로 세어져 지표가 바닥에 붙는 문제가
있었는데, 새 인코딩은 모든 셀의 분기가 2라 이 바닥이 사라진다. 손실이 아니라 감시 지표이고,
로깅 경로가 손실과 같아서 따로 배선할 필요가 없다(9.4절).

### 8.4. 매칭 후 연결 손실 (`ConnLoss` 본체)

$N_{match} = 2\,|\mathcal{P}|$ 는 매칭된 쌍의 총수다 (모든 양성 셀의 분기가 2개).

**존재 손실** — 감독 범위가 셀 종류마다 다르다:

- $\mathcal{P}$의 셀: 매칭된 슬롯 1. $R = D = 2$에서는 **모든 슬롯이 매칭되므로 전 슬롯 1**이다
  (무매칭 슬롯 0 감독은 $R > D$ ablation에서만 나타난다).
- $\mathcal{S} \setminus \mathcal{P}$의 셀(거짓 양성): **전 슬롯 0** ("존재하지 않음만 학습").

즉 기본 설정에서 exist의 변별 신호는 거짓 양성 셀에서만 나온다 — 사실상 "이 셀이 진짜 노드인가"의
셀 단위 신호가 슬롯별로 복제된 것이다. 디코더의 슬롯 게이트($\sigma(\hat e) > \tau_e$, 10.3절)로는
여전히 쓰이므로 유지한다.

$$
\mathcal{L}_{e} = \frac{1}{|\mathcal{S}| \cdot R} \sum_{(i,j) \in \mathcal{S}} \sum_{k=1}^{R}
\mathrm{BCE}\!\left(\sigma(\hat{e}_k),\; \mathbf{1}[k \text{ matched}]\right)
$$

**방향 손실** — 매칭된 쌍에만. 크기·좌표 감독 없이 **방향 차이만** 학습한다. 값 범위 $[0, 2]$:

$$
\mathcal{L}_{dir} = \frac{1}{N_{match}} \sum_{\text{matched}\,(k,m)} \left(1 - \hat{\mathbf{d}}_k \cdot \mathbf{d}^{gt}_m\right)
$$

끝 셀의 끝방향 분기도 똑같이 존재 1 + 방향으로 감독된다 — "선이 이쪽으로 끝났다"를 슬롯이
말하게 하고, 셀이 끝이라는 사실은 $\mathcal{L}_{end}$(8.2)가 따로 말한다.
구 설계의 슬롯 종점 손실 $\mathcal{L}_t$는 폐기했다(6.2절). 매칭 안 된 슬롯의 방향에는 손실을 주지 않는다.

### 8.5. 총 손실

8.0절의 **단일 층 가중합**이다. 항목마다 가중치 하나, 곱은 없다.

$$
\mathcal{L} = w_{hm}\mathcal{L}_{hm}
\;+\; w_{cls}\mathcal{L}_{cls} + w_{coord}\mathcal{L}_{coord} + w_{end}\mathcal{L}_{end}
\;+\; w_{e}\mathcal{L}_{e} + w_{dir}\mathcal{L}_{dir}
$$

config dataclass는 4.1절에 있다(`HeatmapLossConfig`·`SelfSlotLossConfig`·`ConnLossConfig`·`LossConfig`).
기본값은 6개 가중치 전부 1이다.

- 필드 이름을 `self`가 아니라 `self_slot`으로 둔 것은 `self`가 파이썬 예약어라 dataclass 생성자에서 충돌하기 때문이다.
- 손실 모듈을 추가하려면 config dataclass 하나 + `LossConfig` 필드 하나 + `StellaCriterion.from_cfg` 한 줄이면 된다.
  새 모듈의 가중치도 **최하위 항목마다 하나씩**만 둔다.
- 손실 계산은 bf16 학습 중에도 **fp32로 승격**해서 한다(focal의 log 안정성).

---

## 9. 학습 파이프라인 — `stella/train/`

### 9.1. `StellaTrainModule` (LightningModule, 얇게 유지)

```python
class StellaTrainModule(pl.LightningModule):
    def __init__(self, *, model: StellaModel, criterion: StellaCriterion,
                 decoder: ChainDecoder, metric: InstanceCCQ,
                 lr: float, weight_decay: float, warmup_steps: int): ...

    def training_step(self, batch):        # forward → criterion → 손실 dict 로깅 → total 반환
    def validation_step(self, batch):      # forward → criterion 로깅 → 디코딩(10절) → 지표 누적
    def on_validation_epoch_end(self):     # 누적된 인스턴스 지표를 집계·로깅
    def configure_optimizers(self):        # optim.py 호출
```

- 받는 것은 `model`, `criterion`, `decoder`와 옵티마이저 값 몇 개뿐. **전역 cfg를 들고 다니지 않는다.**
- **`validation_step`은 디코딩까지 한다.** 매 에폭 검증마다 모델 출력을 폴리라인 객체로 만들고(10절),
  인스턴스 지표(11절)를 누적한다. 여기서는 **호출 지점과 누적 구조만** 다룬다.
- 시각 로그는 module이 아니라 **callback**이 맡는다(9.5절). 학습 로직과 그리기 로직을 섞지 않는다.

### 9.2. Optimizer / 스케줄 — `optim.py`

- AdamW. param group 3개: (a) 백본 — `lr × lr_mult(0.1)`, (b) bias·norm 파라미터 — weight decay 0, (c) 나머지 — 기본 lr·wd.
- 스케줄: **linear warmup(1000 step) + cosine decay** (step 단위).

### 9.3. Trainer 설정과 진입점 — `train.py`

5.3절의 배선 코드에 이어서:

```python
trainer = pl.Trainer(
    max_epochs=cfg.train.epochs,
    check_val_every_n_epoch=1,  # 학습 1에폭 ↔ 검증 1에폭 (9.4)
    precision=cfg.train.precision,  # bf16-mixed
    accelerator="gpu",
    devices="auto",
    strategy="ddp",
    gradient_clip_val=cfg.train.grad_clip,
    accumulate_grad_batches=cfg.train.accumulate,
    callbacks=[
        ModelCheckpoint(
            monitor="val/total", save_top_k=5, save_last=True, auto_insert_metric_name=False
        ),
        build_instance(cfg.log, cfg),  # VizCallback (9.5)
    ],
    logger=CSVLogger(out_dir),
)
trainer.fit(module, train_loader, val_loader, ckpt_path=args.resume)
```

- **배치 크기 확정 (13절 결정 8, 9차 개정에서 실측으로 종결):** 초기 실측에서 bs=2가 OOM이었으나
  원인은 가중치가 아니라 윈도우 어텐션의 활성이었고(7.6절 — 파라미터·옵티마이저는 1.43 GiB로 전체의
  15%뿐), `window_size = 7` + 윈도우 층 checkpointing으로 해소됐다.
  확정 설정(w=7, ckpt, `n_max = 9500`) 실측(RTX 4090):

  | bs | peak | step | img/s |
  | --- | --- | --- | --- |
  | 1 | 7.50 GiB | 423 ms | 2.37 |
  | 2 | 10.95 GiB | 724 ms | 2.76 |
  | 4 | 17.83 GiB | 1585 ms | 2.52 |

  bs=2는 가능하지만 이득이 처리량 +16%뿐이고 bs=4는 오히려 느리다. `accumulate = 16`이 유효
  배치를 만들고 정규화가 전부 LayerNorm이라 BN 통계 문제도 없다. **`batch_size = 1`로 확정**하고,
  아낀 메모리는 `n_max` 9500(GT 셀 절단 제거 — 실제 정확도에 걸리는 값)에 쓴다.
  NF4류 가중치 양자화는 이 병목(활성 85%)에 듣지 않는다 — 백본이 DINOv3 ViT-L(~300M)로
  바뀌어 가중치 몫이 ~4.8 GiB가 되면 QLoRA를 재검토한다.
- 출력 폴더: `results/{YYMMDD_HHMM}_{config이름}/` — `config.json` + `src/`(소스 전체 복사) + `git_sha.txt` + `checkpoints/` (4.3절).
- EarlyStopping은 `val/inst/f1`(11절)이 실데이터에서 검증된 뒤 붙인다. 일단 손실 기반 체크포인트만.

### 9.4. 로깅 — 에폭 단위

**학습 1에폭 → 검증 1에폭을 번갈아 돈다.** Lightning 기본 동작이므로 `check_val_every_n_epoch=1`만 명시하면 된다.
모든 스칼라는 **에폭 평균**으로 남긴다(`on_step=False, on_epoch=True`). step 단위 곡선은 남기지 않는다 —
`accumulate=16`이라 step 단위 값이 튀고, 비교해야 하는 것은 에폭 간 추세다.

**손실 dict를 그대로 로깅한다.** `StellaCriterion`이 8.0절의 이름 붙은 dict를 내므로, train module은
키를 알 필요가 없다.

```python
def training_step(self, batch):
    out = self.model(batch["image"], gt_positive=batch["class_map"] > 0)
    losses = self.criterion(out, batch)  # dict[str, Tensor]
    self.log_dict(
        {f"train/{k}": v for k, v in losses.items()}, on_step=False, on_epoch=True, sync_dist=True
    )
    return losses["total"]
```

기록되는 키 (`val/`도 같은 구조):

| 키                        | 뜻                                                                        |
| ------------------------ | ------------------------------------------------------------------------ |
| `train/total`            | 총 손실 (8.5)                                                               |
| `train/heatmap/focal`    | 히트맵 focal BCE — **가중치 적용 전 원시 값**                                        |
| `train/heatmap/total`    | $w_{hm}\mathcal{L}_{hm}$                                                 |
| `train/self_slot/class`  | self 슬롯 클래스 CE (원시)                                                      |
| `train/self_slot/coord`  | self 슬롯 좌표 SmoothL1 (원시)                                                 |
| `train/self_slot/end`    | 끝 셀 BCE (원시, 9차 개정)                                                      |
| `train/self_slot/total`  | $w_{cls}\mathcal{L}_{cls} + w_{coord}\mathcal{L}_{coord} + w_{end}\mathcal{L}_{end}$ |
| `train/conn/exist`       | 연결 존재 BCE (원시)                                                           |
| `train/conn/dir`         | 연결 방향 1 − 내적 (원시)                                                        |
| `train/conn/match_ambiguity` | **손실 아님.** 배정 모호 셀 비율 — 매칭 불안정성 감시(8.3, 구 `switch_rate`)             |
| `train/conn/total`       | $w_{e}\mathcal{L}_{e} + w_{dir}\mathcal{L}_{dir}$                        |
| `lr`, `lr_backbone`      | 스케줄 확인용 (9.2)                                                            |

`*/total`은 그 모듈의 가중합이고 `train/total`은 그것들의 **단순 합**이다(8.0절 — 상위 가중치가 없다).
개별 항목은 **원시 값**으로 남기므로, 가중치를 조정할 때 항목별 실제 크기를 그대로 비교할 수 있다.

**검증에서만 추가로 남기는 것:**

- 위와 같은 `val/*` 손실 전부.

- **인스턴스 단위 성능 지표 (`val/inst/*`)** — 10절 디코딩 결과와 GT `instances`를 **11절 지표**로
  비교해 산출한다. `on_validation_epoch_end`에서 집계한다.
  
  | 키                                      | 뜻                                               |
  | -------------------------------------- | ----------------------------------------------- |
  | `val/inst/f1` · `precision` · `recall` | 비대칭 CCQ 인스턴스 F1 — **주 지표**, 전체 micro (11.1)     |
  | `val/inst/f1_macro`                    | 클래스 단순평균 F1 (11.1)                              |
  | `val/inst/f1/{cls}` 등                  | **클래스별** F1·precision·recall — 차선 11종 각각 (11.1) |
  | `val/inst/fp_redundant`                | 매칭 안 됐지만 GT 위($C_2 \ge 0.9$)인 잉여 예측 수 — 병합으로 해결 |
  | `val/inst/fp_spurious`                 | GT 밖($C_2 < 0.9$)에 그린 예측 수 — 삭제 필요              |
  | `val/inst/coverage`                    | 집계 완전성 — 순수 커버리지, 연결성 무시 (11.2)                 |
  | `val/inst/correctness`                 | 집계 정확성 (11.2)                                   |
  | `val/inst/rms`                         | 매칭 구간 RMS 횡오차 (11.2)                            |
  | `val/inst/frag`                        | GT당 예측 조각 수 — 연결성·작업량 (11.2)                    |
  
  주 지표는 **클래스별로 따로** 낸 뒤 전체 micro와 클래스 평균 macro로 묶는다(11.1). **`f1`과
  `coverage`의 격차가 조각남의 크기**이므로 둘을 나란히 본다(11.2).

- **시각 로그** (9.5절) — 스칼라가 아니라 PNG 파일이다.

`logger=CSVLogger(out_dir)` → `metrics.csv` 한 장에 에폭별 행이 쌓인다. `ModelCheckpoint`의 `monitor`는
`"val/total"`로 두고, 파일 이름에 `/`가 들어가지 않도록 `auto_insert_metric_name=False`를 준다.

### 9.5. 시각 로그 — `train/viz.py` + `train/callbacks.py`

검증 중 예측을 눈으로 확인하기 위한 PNG를 남긴다. **배치마다 첫 번째 샘플 하나만** 그린다(전부 그리면 느리다).

```
results/{run}/viz/epoch{E:03d}/{sample_id}_heat.png
                              /{sample_id}_class.png
                              /{sample_id}_slot.png
```

`stella/train/viz.py`는 **Lightning을 모르는 순수 함수 모음**이다(`np.ndarray` in → `np.ndarray` out).
그래서 단위 테스트가 가능하고, 같은 함수로 **GT도 그릴 수 있다** — GT의
`class_map`·`coord_map`·`end_map`·`conn_dirs`는 모델 출력과 격자는 물론 **형태까지 같아서**
(설계 방침 1, 10차 개정) 인자만 바꿔 넣으면 되고, 방향 유도 코드가 필요 없다.
`stella/train/callbacks.py`의 `VizCallback(pl.Callback)`이 `on_validation_batch_end`에서 샘플 0을 꺼내 호출한다.

**세 가지 그림** (원본 이미지 768×768 위에 그린다):

| 파일            | 내용                                                                                                         |
| ------------- | ---------------------------------------------------------------------------------------------------------- |
| `*_heat.png`  | 히트맵 확률 $\sigma(\text{heatmap\_logit})$를 **파랑→빨강** 컬러맵으로 칠하고 원본과 **반씩 블렌딩**. 192×192를 nearest로 768×768까지 확대 |
| `*_class.png` | 원본 위에 **4×4 셀마다 중심 2×2 픽셀**을 클래스 색으로 칠하기                                                                   |
| `*_slot.png`  | 원본 위에 **self 좌표 = 검은 점**, **연결 슬롯 방향 = R/G 선** (자기 점에서 시작, 6.1절 원점 규약)                                    |

```python
# heat: 파랑(0) → 빨강(1). matplotlib 없이 직접 만든다 (의존성 최소화)
prob = sigmoid(heatmap_logit)  # (192,192)
prob = upsample_nearest(prob, 4)  # (768,768)
heat = np.stack([prob, np.zeros_like(prob), 1.0 - prob], -1)  # R=p, G=0, B=1-p
out = (1 - a) * image + a * heat * 255  # a = log.heat_alpha (0.5)

# class: 셀 (i,j) → 픽셀 블록 [4i:4i+4, 4j:4j+4] 중 중심 2×2 = [4i+1:4i+3, 4j+1:4j+3]
cls = class_logit.argmax(-1)  # (192,192)
draw = (cls > 0) & (sigmoid(heatmap_logit) > log.class_thresh)
for i, j in np.argwhere(draw):
    out[4 * i + 1 : 4 * i + 3, 4 * j + 1 : 4 * j + 3] = CLASS_COLOR[cls[i, j]]

# slot: self 점 + 슬롯 방향선 (둘 다 자기 점에서 — 6.1 원점 규약, 9차 개정)
p = (np.array([j, i]) + self_coord[i, j]) * s  # 셀 좌상단 원점 → 픽셀 (6.1)
cv2.circle(out, p.astype(int), radius=1, color=(0, 0, 0), thickness=-1)
for k in range(R):  # R = 2 → 슬롯 0,1 = R,G
    if sigmoid(exist_logit[i, j, k]) < log.exist_thresh:
        continue
    cv2.line(
        out,
        p.astype(int),
        (p + conn_dir[i, j, k] * log.slot_line_len).astype(int),
        color=SLOT_COLOR[k],
        thickness=1,
    )
```

**클래스 색상표** (`stella/data/types.py`에 `CLASS_COLOR`로 둔다 — 데이터 정의와 같은 곳). RGB.

```python
CLASS_COLOR = [
    (0, 0, 0),  # 0  background/ignore — 칠하지 않는다
    (77, 77, 255),  # 1  center_line
    (77, 178, 255),  # 2  u_turn_zone_line
    (77, 255, 77),  # 3  lane_line
    (255, 153, 77),  # 4  bus_only_lane
    (255, 77, 77),  # 5  edge_line
    (178, 77, 255),  # 6  path_change_restriction_line
    (77, 255, 178),  # 7  no_parking_stopping_line
    (255, 178, 77),  # 8  guiding_line
    (255, 215, 0),  # 9  stop_line          — 흰 배경과 구분되도록 금색
    (255, 77, 128),  # 10 safety_zone
    (0, 139, 139),  # 11 bicycle_lane       — 녹색·보라 선과 구분되도록 진한 청록
]
SLOT_COLOR = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # 슬롯 0,1 (셋째 색은 exp_r3 ablation용)
```

- 클래스 0은 배경이라 칠하지 않는다(색이 검정이라 칠해도 원본을 가리기만 한다).
- 슬롯 색은 **슬롯 번호 순서**일 뿐 의미가 없다. 슬롯 배정은 매칭이 정하므로(8.3), 같은 방향이 늘 같은 색은 아니다.
  학습이 진행되면 슬롯별로 방향이 분화되는지 보는 용도다.
- **`batch_size = 1`이면 "배치당 1장"이 곧 "전 이미지"다.** 검증 100장이면 에폭당 300개 PNG가 쌓인다.
  `log.max_batches`(기본 20)로 상한을 두고, `log.every_n_epochs`로 간격을 조절한다.
- 그리기는 CPU에서 하고, 텐서는 `.detach().cpu().float()`로 옮긴 뒤 넘긴다.

### 9.6. 알려진 함정과 대비책

| 함정                                                              | 대비                                                                                                                                                 |
| --------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| 선(라벨)이 하나도 없는 타일: 노드 0개 → 토큰 스택이 안 돌아 DDP가 unused parameter로 죽음 | 최소 1노드 강제(7.4). 실데이터에서 빈 타일이 많으면 학습 셋에서 제외도 검토                                                                                                     |
| 전역 cross-attn의 attention 행렬 $KN \times N$ (추론 최악 12000×3000)    | 마스크 없는 전역 층은 SDPA flash/mem-efficient 커널이 잡아 실체화 안 됨. 윈도우 층은 **이웃 gather 방식이 기본**(7.6절 — 대비책이었으나 재구현에서 채택). gather 활성이 커서 윈도우 층만 gradient checkpointing (7.6절 실측) |
| 가변 $N$ 때문에 `torch.compile`이 재컴파일 반복                             | v1은 compile 끔. 안정화 후 `dynamic=True`로 시도                                                                                                            |
| `F.normalize`의 0 벡터 (학습 초기 conn_dir 원시 출력이 0 근처)                | `eps` 지정 + 방향 손실이 매칭된 슬롯에만 걸리므로 발산하지 않음                                                                                                            |
| GT 연결 수 > $D$인 셀                                                | 인코더가 경고 + 절단 + 통계 출력(6.4). 실데이터에서 빈도 확인 후 $R$ 조정(ablation과 연결)                                                                                     |
| 매 에폭 검증마다 디코딩(10절)을 도는 비용                                       | 디코딩은 CPU·numpy 작업이라 GPU와 겹칠 수 있다. 느리면 검증 셋을 부분 샘플링                                                                    |

---

## 10. 객체 생성 (디코딩) — `stella/decode/graph.py`

모델 출력은 **셀 단위 예측**이다(7.1). 평가와 실사용에 필요한 것은 **폴리라인 객체 목록**이다.
이 절이 그 변환을 정의한다. `ChainDecoder`는 **매 에폭 검증 단계에서 실행되고**(9.1), 그 결과가
인스턴스 단위 성능 평가의 입력이 된다.

**입력** `ModelOutput` 한 샘플 + `DecodeConfig`.
**출력** `list[dict]` — `{"class": int, "points": float32 (P, 2) 픽셀 좌표, "score": float}`.
6.2절 `targets["instances"]`와 **같은 형식**이라 GT와 예측을 바로 비교할 수 있다.

**왜 재설계했나 (9차 개정).** 구 `GraphDecoder`(정점 → 간선 후보 → 양방향 확인 → 그래프 절단)는
GT 주입에서도 간선 재현율이 97.7%에 그쳤고, 성분당 간선이 평균 47개라 **2.3%의 간선 손실이
연결 성분 수 1.8배로 증폭**됐다(SEED-MAP val 12장 실측 — 소실 원인: 슬롯이 GT 아닌 셀을 지목
278, `mutual` 탈락 211). 긴 사슬에서는 간선 하나만 끊겨도 인스턴스가 쪼개진다. 새 디코더는
인코딩(6.4절)과 같은 모양으로 — **사슬을 한 노드씩 확장**하며, 전역 그래프·상호 최선 확인·경로
절단 단계가 없다. 검증 기준: **GT를 출력 형식으로 주입하면 인스턴스 F1 ≈ 1** (M12).

### 10.1. 3단계 개요

```
ModelOutput
 → ① 정점 추출        노드 셀 → (클래스·클래스 확률, 점, 점수, 끝 확률, 슬롯) 목록
 → ② 사슬 확장        클래스 확률 국소 피크에서 양방향으로, 마주봄 확인으로 한 노드씩 연장
                      + 완성된 사슬의 클래스 순도 검사 (탈락 시 정점 반환)
 → ③ 후처리           최소 길이·RDP 단순화·점수
```

각 단계가 독립 함수라 단위 테스트가 가능하다. 합성 데이터 과적합 모델에서 **GT 폴리라인이 그대로 복원되는지**가
전체 검증 기준이다(M6·M12).

### 10.2. ① 정점 추출

7.4절 **추론 경로와 같은 방식**으로 노드 셀을 고른다 — 학습 때와 같은 코드를 쓰되 임계값만 `decode.heatmap_thresh`다.

$$
\mathcal{V} = \{(i,j) : \sigma(\hat{h}_{ij}) > \tau_h \;\land\; \arg\max_c \hat{s}_{ij,c} \neq 0 \}
$$

정점 하나의 속성:

| 속성     | 값                                                                                                    |
| ------ | ---------------------------------------------------------------------------------------------------- |
| 클래스    | $y_{ij} = \arg\max_{c} \hat{s}_{ij,c}$ (배경이면 정점이 아니다)                                                |
| 클래스 확률 | $\pi_{ij} = \mathrm{softmax}(\hat{s}_{ij})$ — **확률 벡터를 보관.** 시드 선정·확장 게이트·순도 검사에 쓴다(10.3)          |
| 픽셀 좌표  | $\mathbf{p}_{ij} = \left((j + \hat{c}^x_{ij}),\ (i + \hat{c}^y_{ij})\right) \cdot s$ — 셀 좌상단 원점(6.1) |
| 점수     | $\sigma(\hat{h}_{ij}) \cdot \max_c \pi_{ij,c}$                                                       |
| 끝 확률   | $\sigma(\hat{\mathrm{end}}_{ij})$ — 사슬 확장의 정지 신호(10.3)                                               |
| 슬롯     | $(\sigma(\hat{e}_k),\ \hat{\mathbf{d}}_k)$, $k = 1..R$ — 방향의 원점은 자기 점 $\mathbf{p}_{ij}$(6.1)         |

**학습 때와 달리 dilation을 쓰지 않는다.** 학습에서 dilation은 거짓 양성 셀을 일부러 노출시키려는 것이지만(7.4),
디코딩에서는 중복 정점을 만들 뿐이다.

### 10.3. ② 사슬 확장 — 한 노드씩, 단방향으로

핵심 확인은 **"서로가 서로의 점을 향하는가"**다. 현재 정점 $a$의 확장 슬롯 방향을 $\mathbf{c}$,
후보 정점 $b$의 슬롯 방향을 $\mathbf{n}$이라 하면, 둘이 마주보면 $\mathbf{c} \cdot \mathbf{n} \to -1$이다.
구 설계의 "상호 최선(mutual best)" 검사는 하지 않는다 — 한 방향씩 확장하며 그때그때 최선 후보를
붙인다. mutual best는 슬롯이 한 칸 건너 셀을 지목하는 사소한 오차만으로 간선을 통째로 버려
긴 사슬을 끊었다(GT 주입에서 소실 간선 489개 중 211개가 mutual 탈락).

**시드 선정 — 클래스 확률 국소 피크에서 양방향으로 (10차 개정).** 자기 클래스 확률
$\max_c \pi_c$가 **정점 이웃($3\times3$) 중 최대인 정점**(국소 피크)을 확률 내림차순으로 시드로
삼는다. 모델이 가장 확신하는 지점에서 출발해야 사슬 클래스가 안정적으로 정해진다 — 끝 셀은
감독이 상대적으로 어렵고(양성의 ~2.5%) 예측이 흔들리면 시작점 자체가 틀어지므로, 끝에서
출발하는 구 방식을 버렸다. **사슬 클래스 $y^\ast$ = 시드의 argmax 클래스**로 고정하고, 시드의
활성 슬롯들을 따라 **양방향으로** 확장한 뒤 두 반쪽을 이어 붙인다. 국소 피크가 소진되면 남은
미사용 정점을 확률 내림차순으로 시드에 쓴다(안전망).

**확장 한 스텝.** 정점 $a$, 미사용 슬롯 $k$ ($\sigma(\hat{e}_{a,k}) > \tau_e$)에서:

1. **후보 집합.** $a$의 셀에서 체비셰프 반경 `radius`(= 2, $5\times5$) 안의 미사용 정점 $b$ 중
   **사슬 클래스 확률이 하한을 넘는 것**: $\pi_{b,y^\ast} \ge$ `min_class_prob`(기본 0.1).
   argmax 일치를 요구하지 않는다 — 중간에 잠깐 다른 클래스가 이길 수 있기 때문이다. 다만
   사슬 클래스 확률이 0에 가까운 정점을 붙이는 것은 위험하므로 하한으로 거른다(순도 검사가
   뒤에서 한 번 더 잡는다). 반경 2는 교차점에서 소유권으로 잃은 한 칸을 건너뛰기
   위함이다(6.4절). GT 주입 실측에서 간선의 98.7%가 1칸, 1.3%가 2칸이다.
2. **되가리킴 슬롯.** $b$의 활성 슬롯 중 $a$ 쪽을 가장 잘 향하는 것:
   $\mathbf{n}_b = \arg\min_{\mathbf{n}} \mathbf{c} \cdot \mathbf{n}$.
3. **게이트.** $\mathbf{u}_{ab} = (\mathbf{p}_b - \mathbf{p}_a)/\lVert\cdot\rVert$ 에 대해
   둘 다 통과해야 후보로 남는다:

$$
\hat{\mathbf{d}}_{a,k} \cdot \mathbf{u}_{ab} \ge \tau_{align}
\qquad\wedge\qquad
-\,(\hat{\mathbf{d}}_{a,k} \cdot \mathbf{n}_b) \ge \tau_{opp}
$$

   (내 슬롯이 실제로 $b$를 향하고, $b$의 슬롯이 마주 본다.)
4. **채택.** 남은 후보 중 비용 최소를 붙인다:

$$
\mathcal{C}(b) = \left(1 - \hat{\mathbf{d}}_{a,k} \cdot \mathbf{u}_{ab}\right)
+ w_{opp}\left(1 + \hat{\mathbf{d}}_{a,k} \cdot \mathbf{n}_b\right)
$$

   거리 항은 없다 — 구 설계 실측에서 거리 항은 정렬 나쁜 가까운 셀을 끌어들여 해로웠다
   (`w_dist = 0.3` → 간선 정확도 0.985 → 0.953). 대신 방향이 같은 원거리 후보와의 동률은
   **마주봄 항**이 가른다(건너뛴 셀은 되가리키는 슬롯이 없다).
5. **이동.** $b$를 사슬에 붙이고, $b$의 되가리킴 슬롯 $\mathbf{n}_b$를 사용 처리한 뒤
   $b$의 **반대쪽 활성 슬롯**으로 계속 확장한다.

**정지 조건.** ① $b$의 끝 확률 $> \tau_{end}$ (사슬 끝 도달) ② 게이트(방향 + **사슬 클래스 확률
하한**)를 통과한 후보 없음 ③ 사슬의 시작 정점으로 복귀(고리 폐쇄) ④ 정점 수 상한(이상 동작 안전망).

**순도 검사 (10차 개정).** 양방향 확장이 끝나 사슬이 완성되면, 구성 정점 중 **argmax 클래스가
사슬 클래스 $y^\ast$와 일치하는 비율**을 잰다. `purity_thresh`(기본 0.6) **이하면 사슬을 버린다** —
정점과 슬롯을 미사용으로 되돌리고(다른 시드가 다시 쓸 수 있게), 그 시드는 실패로 표시해 재시도
하지 않으며, 다음 시드로 넘어간다. `min_class_prob` 게이트가 느슨한 만큼(0.1 — 일시적 확률
하락 허용) 여기서 "사슬이 통째로 다른 클래스를 따라간" 경우를 걸러낸다.

**끝 연장 (사용자 결정 — "선이 짧아지지 않는다").** 사슬이 끝 셀에서 멈추면, 그 셀의 남은 활성
슬롯(끝방향 — GT가 실제 끝점을 향하도록 감독했다) 방향으로 **연장점을 하나 추가**한다:
$\mathbf{p}_{\text{ext}} = \mathbf{p} + \hat{\mathbf{d}} \cdot \texttt{end\_extend}$ (기본 1셀 — 끝점은
미채움 이웃 칸 안에 있으므로 평균 거리가 약 1셀이다. 정확한 값은 스윕). 끝칸 미채움으로 잘린
길이가 여기서 복원된다. **1셀 사슬**(3칸짜리 선 — 분기 2개가 모두 끝방향)은 양쪽으로 연장해
3점 폴리라인이 된다. `min_points`는 연장점을 포함한 점 수에 적용한다.

구 설계의 클래스 불일치 벌점·종점 슬롯 면제·다른 클래스 접합 예외(구 10.3~10.4)는 전부 필요
없다 — 클래스는 **사슬 클래스 확률 하한 + 순도 검사**가 부드럽게 관리한다. T자 접합에서 본선은
곁가지 끝 셀을 후보로 만나도 방향 게이트가 거르고, 곁가지는 자기 끝 셀에서 정지한다 —
**본선이 접합점에서 잘리던 구 문제(구 open_questions 4)가 구조적으로 사라진다.**

### 10.4. ③ 후처리

1. **폴리라인의 클래스 = 사슬 클래스 $y^\ast$(시드의 클래스).** 순도 검사(10.3)가 구성 정점의
   60% 초과 일치를 보증하므로 다수결과 항상 일치한다 — 별도 다수결 단계를 두지 않는다.
2. 정점 수 < `min_points`면 버린다(연장점 포함). `simplify_tol > 0`이면 RDP로 단순화한다(기본 0 = 안 함).
3. **점수.** 폴리라인의 점수 = 구성 정점 점수의 평균. 평가에서 confidence 임계값 스윕에 쓴다.

**이 규약은 모델이 아니라 디코더에만 있다.** 확장 방식을 바꾸고 싶으면 재학습 없이 여기만 고치면 된다.
`radius`·$\tau_{align}$·$\tau_{opp}$·$\tau_{end}$·$\tau_e$·`min_class_prob`·`purity_thresh`는
학습된 체크포인트로 검증 셋에서 스윕해 확정한다 — GT 주입만으로는 오탐 필터 강도를 정할 수 없다
(13절 남은 확인). GT 주입에서는 확률이 0/1이라 `min_class_prob`·`purity_thresh`가 자명하게
통과되므로, M12 수용 기준(F1 ≈ 1)에는 영향이 없다.

### 10.5. 학습 파이프라인과의 연결

```python
# StellaTrainModule.validation_step
out = self.model(batch["image"])  # 추론 경로 (GT 미사용)
losses = self.criterion(out, batch)
self.log_dict({f"val/{k}": v for k, v in losses.items()}, on_step=False, on_epoch=True)

for b in range(len(batch["image"])):  # 샘플별 디코딩
    pred = self.decoder(out[b])  # list[dict] — 10절 출력
    self.metric.update(pred, batch["instances"][b])  # 지표 누적 (11절 InstanceCCQ)

# on_validation_epoch_end
self.log_dict({f"val/inst/{k}": v for k, v in self.metric.compute().items()})
self.metric.reset()
```

- `ChainDecoder`도 `build_instance(cfg.decode, cfg, ...)`로 만든다 — 다른 부품과 같은 규약(5절).
  디코딩 방식을 통째로 바꾸려면 `decode.name`만 바꾸면 된다(구 `GraphDecoder`가 실제로 그렇게 교체됐다).
- **지표 객체(`self.metric`)는 `build_instance(cfg.eval, cfg)`로 만든다**(11.4절). 인터페이스는
  `update(pred, gt)` / `compute() -> dict` / `reset()`이고, 실제 지표는 11절의 커버리지 중심 인스턴스 F1이다.
- DDP에서는 지표를 `all_gather`로 모아야 한다. `torchmetrics.Metric`을 상속하면 자동으로 처리된다.

---

## 11. 평가 지표 — 커버리지 중심 인스턴스 F1

`ChainDecoder`(10절)가 만든 예측 폴리라인을 GT 폴리라인(`targets["instances"]`)과 비교해 성능을 수치화한다.
결과 객체는 10.5절 `self.metric` 자리에 꽂혀 매 에폭 검증에서 `val/inst/*`로 로깅된다(9.4절).
조사 근거는 `references/metric/metric_survey.md`에 있다.

**방침.** 차선은 폭이 거의 없는 1차원 구조라 **마스크 IoU 매칭이 맞지 않는다**(11.3절). 대신 **버퍼 기반
인스턴스 CCQ**를 쓴다. 하나의 지표로 다 담지 못하므로 **세 축**을 함께 본다. 주 지표는 검출과 연결성을,
보조 지표는 순수 커버리지와 기하 정밀도를 맡는다.

### 11.1. 주 지표 — 비대칭 CCQ 인스턴스 F1

예측 폴리라인 $P$와 GT 폴리라인 $G$에 대해, 버퍼 폭 $\rho$에서 두 겹침 비율을 정의한다. $d(\cdot, L)$은
점에서 폴리라인 $L$까지의 거리, $|\cdot|$은 호 길이, $\mathcal{G}$는 **모든 GT 폴리라인의 합집합**이다.

$$
C_1(G, P) = \frac{\big|\{x \in G : d(x, P) \le \rho\}\big|}{|G|}
$$

$$
C_2(P) = \frac{\big|\{y \in P : d(y, \mathcal{G}) \le \rho\}\big|}{|P|}
$$

- $C_1$ = 이 GT가 예측에 덮인 비율. **커버리지**(완전성, recall 쪽). 쌍의 성질이다.
- $C_2$ = 이 예측이 어떤 GT 위에든 올라가 있는 비율. **정확성**(precision 쪽). 예측 하나의 성질이며 **모든 GT 대상**이다.

$C_2$를 특정 GT가 아니라 모든 GT 대상으로 재는 것이 요점이다. 가림·타일 경계로 GT가 여러 조각으로
나뉘어 있어도(6.7.4절), 예측이 그 조각들을 이으면 그 부분이 전부 어떤 GT의 버퍼 안이라 정확성이 유지된다.
반대로 **GT가 실제로 없는 구간을 이어버린** 예측은 그 부분이 어떤 버퍼에도 안 들어가 $C_2$가 떨어진다.
즉 "라벨만 끊긴 것"과 "실제로 끊긴 것"의 차이를 그 구간 아래 GT의 유무로 가른다.

**TP 판정 — 비대칭 임계값.**

$$
\text{TP} \iff C_1(G, P) \ge \theta_{cov} = 0.5 \;\;\wedge\;\; C_2(P) \ge \theta_{cor} = 0.9
$$

- 커버리지는 **관대하게**($\theta_{cov} = 0.5$): GT를 절반 넘게 덮은 조각 하나는 살린다. 아깝게 끊긴 예측도 대표 조각이 인정된다.
- 정확성은 **엄격하게**($\theta_{cor} = 0.9$): GT 밖에 그린 선, 진짜 끊긴 곳을 이은 헛다리를 걸러낸다.

**매칭.** 두 조건을 통과한 $(G, P)$ 쌍만 후보로 두고 $C_1 + C_2$를 가중치로 **일대일 최대가중 매칭**한다.
매칭된 쌍 = TP, 남은 GT = FN, 남은 예측 = FP.

$\rho$가 인접 차선 간격의 절반보다 작으면 예측 위 한 점은 많아야 하나의 GT 버퍼에만 들 수 있어 $C_2$의
"모든 GT"가 중복 없이 잘 정의된다. **$\rho$는 반드시 차선 간격의 절반 이하로 잡는다.** 이 선을 넘으면
예측이 자기 GT보다 이웃 GT에 가까워져 매칭이 뒤바뀐다.

**FP를 두 종류로 나눠 보고한다.** HD맵 보정 관점에서 비용이 다르기 때문이다.

- **redundant FP**: 매칭 안 됐지만 $C_2 \ge 0.9$. 실제 GT 위의 잉여 조각이다. 병합만 하면 된다.
- **spurious FP**: 매칭 안 됐고 $C_2 < 0.9$. GT 밖에 그린 선이다. 삭제하고 확인해야 한다.

둘 다 precision에는 FP로 센다 — 조각남을 벌하는 것이 이 지표의 의도다. 분해는 진단용이며 $C_2$에서 공짜로 갈린다.

$\text{precision} = TP/(TP+FP)$, $\text{recall} = TP/(TP+FN)$, $F1$은 조화평균이다. **매칭과 집계는
클래스별로 따로 한다** — 차선 11종 각각에 대해 precision·recall·F1을 내고, 그 위에 전체를 합친
micro와 클래스 단순평균 macro를 함께 보고한다. redundant/spurious FP 분해도 클래스별로 유지한다.

### 11.2. 보조 지표

**집계 커버리지 (연속값).** 인스턴스 매칭 없이 전체 길이 기준으로 완전성·정확성을 잰다. $\mathcal{P}$는
모든 예측의 합집합이다.

$$
\text{Completeness} = \frac{\sum_{G} \big|\{x \in G : d(x, \mathcal{P}) \le \rho\}\big|}{\sum_{G} |G|}, \qquad
\text{Correctness} = \frac{\sum_{P} \big|\{y \in P : d(y, \mathcal{G}) \le \rho\}\big|}{\sum_{P} |P|}
$$

임계값 절벽이 없고, 조각이 몇 개든 합집합으로 세므로 심하게 쪼개진 GT도 덮인 만큼 반영된다. **주 지표
$F1$과 이 커버리지의 격차가 곧 조각남의 크기다.** 커버리지는 높은데 인스턴스 $F1$이 낮으면 "덮긴 덮었는데
못 이었다"이고, 그 격차의 감소가 연결성 개선의 정량 증거가 된다. 조각난 것까지 $F1$에 넣으면 이 지표가
그냥 커버리지가 되어 연결성 신호가 사라지므로, 커버리지는 **별도 축**으로 둔다.

**RMS 횡오차.** 버퍼 안에서 얼마나 정확히 붙었는지를 잰다. $C_1$·$C_2$는 $\rho$ 안이면 통과라 이 정보를 못 본다.
$M$은 예측에 덮인 GT 구간이다.

$$
\text{RMS} = \sqrt{\frac{1}{|M|} \sum_{x \in M} d(x, \mathcal{P})^2}, \qquad M = \{x \in \mathcal{G} : d(x, \mathcal{P}) \le \rho\}
$$

**GT당 조각 수 (연결성·작업량).** 매칭된 각 GT를 덮는 깨끗한 예측 조각 수다. 이상값은 1이고, 클수록
수작업 스티치가 많다. $\mathcal{M}$은 매칭된 GT 집합이다.

$$
\text{Frag} = \frac{1}{|\mathcal{M}|} \sum_{G \in \mathcal{M}} \#\{P : C_1(G, P) > 0 \;\wedge\; C_2(P) \ge \theta_{cor}\}
$$

### 11.3. 채택하지 않은 지표와 이유

상세 근거는 `metric_survey.md`. 요지만 적는다.

- **Chamfer distance 매칭 F1/AP** — 조각난 예측을 둘 다 탈락시켜 0점을 준다. 커버리지가 높은 2조각
  예측과 아무것도 없는 것을 구분하지 못한다. 우리가 강조할 커버리지·연결성이 안 보인다. 점을 성기게 찍으면
  최근접이 비스듬한 대각선 거리를 재는 문제도 있다. → 버퍼·길이 기반 CCQ가 이 둘을 다 피한다.
- **confidence 기반 AP** — 차선은 셀당 출력이 하나이고 배경만 이기면 노드가 나와서 confidence의 변별력이
  약하다. 셀별 값을 평균낼수록 분포가 좁아져 순위가 무의미하다. AP는 한 셀에서 출력이 여럿 나오는 박스
  검출에서 온 방식이라 맞지 않는다. → 운영점 하나의 $F1$로 보고한다. 도로망 추출 분야(TOPO·APLS)도 AP를 쓰지 않는다.
- **마스크 IoU F1 (CULane식, $AP^M$)** — 얇은 선은 몇 px만 어긋나도 IoU가 0.5를 못 넘어 지나치게
  엄격하다. 결과가 래스터화 두께에 좌우된다. → 기존 벤치마크 대조용으로만 참고 가능.
- **양쪽 버퍼 IoU 매칭** — 횡방향 거리와 매칭 길이가 한 숫자에 섞여 "가까워서 좋은지 많이 겹쳐서 좋은지"
  구분이 안 된다. → 종방향(커버리지)과 횡방향(RMS)을 별도 축으로 분리했다.
- **clDice·Boundary IoU** — 세그멘테이션 품질 지표라 인스턴스 단위 TP/FP/FN 집계가 안 된다. 닫힌
  영역용이라 열린 폴리라인에 부적합하다.
- **APLS·TOPO·junction F1 (도입 안 함)** — 도로망 전역 그래프의 연결성용이다. 차선은 평행한 인스턴스가
  많아 전역 경로·분기점 지표가 과하다. 연결성은 주 지표의 조각남 벌점과 `frag`(GT당 조각 수)로 충분히 잡는다.

### 11.4. 구현 연결

- 지표는 `build_instance(cfg.eval, cfg)`로 만들어(5절) `StellaTrainModule`에 넘기고, 10.5절 `self.metric`
  자리에 꽂는다. `torchmetrics.Metric`을 상속해 DDP `all_gather`를 위임한다. 인터페이스는
  `update(pred, gt)` / `compute() -> dict` / `reset()`이다.
- 파라미터는 `MetricConfig`(4.1절)에 둔다: $\rho$(버퍼), $\theta_{cov} = 0.5$, $\theta_{cor} = 0.9$,
  방향 게이트, 길이 샘플 간격. 계산은 폴리라인을 일정 간격으로 샘플해 점-폴리라인 거리로 하며, 그래서
  성긴 점의 대각선 문제가 없다.
- 출력 dict가 그대로 `val/inst/*`로 로깅된다(9.4절).
- **파선 GT 규약 — 종결 (9차 개정).** "GT 쪽 일대다 매칭 허용"이나 "평가 전 공선 GT 병합" 보정은
  **도입하지 않는다.** 재구현 실측에서 GT 주입 F1 상한(0.63~0.69)의 주범은 GT 라벨의 조각남이
  아니라(구 인코더 그래프 성분 451 < GT 인스턴스 504 — 오히려 병합하고 있었다) **디코더의 간선
  소실 증폭**이었다. 선 단위 인코딩(6.4절)은 라벨 인스턴스와 사슬이 1:1이라 지표와 정확히
  정합한다 — dash마다 별도 객체면 사슬도 별도라 매칭이 그대로 성립한다. 새 수용 기준:
  **GT 주입 인스턴스 F1 ≈ 1** (M12). 이 기준이 안 나오면 그때 지표가 아니라 인코더·디코더를 본다.

---

## 12. 구현 순서 (마일스톤별 완료 판정)

| 단계     | 내용                                                                                                                   | 완료 판정                                                                                                            |
| ------ | -------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| **M0** | 저장소 스캐폴딩: pyproject(editable 설치), 패키지 뼈대, `schema.py`, `base.py`, `builder.py`, 각 클래스 `from_cfg` 골격, CI(ruff+pytest) | `test_config_resolves` + `test_full_build` 통과 (모든 config로 해석·조립 성공). 오타 config를 일부러 넣어 에러 메시지가 후보를 보여주는지 확인      |
| **M1** | `types.py` + `encode.py`(래스터+위상 2단계) + `synthetic.py` + `viz_gt.py`                                                  | `test_encode.py`(6.4 불변식 7종, **이중선 검증 포함**) 통과. 합성 샘플 시각화 육안 확인 — 분기·T접합·교차·종점·이중선 케이스 포함. 인코딩 시간 측정치 기록(6.4.1절) |
| **M2** | `Backbone`(Dinov3/TimmBackbone) + `Neck`(SFP/FPNLite)                                                                | 두 경로 모두 `(B,256,192,192)` shape 테스트 통과. lr_mult 그룹 분리 확인                                                         |
| **M3** | 히트맵 헤드 + 노드 선택(gt+pred/추론 두 경로)                                                                                      | 합성 8장 과적합 → 히트맵 손실 수렴, 추론 선택의 GT 셀 재현율 ≈ 1                                                                       |
| **M4** | RoPE + 어텐션 블록                                                                                                        | `test_rope.py`(평행이동 불변), 윈도우 마스크 정확성 테스트(범위 밖 attention = 0)                                                     |
| **M5** | 출력 헤드 + scatter + GT 유도(방향·t) + matching + criterion                                                                 | `test_matching.py`(scipy LSA 대조) 통과. **합성 1장 과적합에서 total loss → ~0**, 배정 switch rate 0 수렴, 종점 셀 슬롯의 exist → 0 확인 |
| **M6** | **`GraphDecoder`**(10절 4단계 전부) + `test_decode` — 학습 루프와 무관한 잎 모듈 | M5 과적합 모델·GT 주입에서 **합성 GT 폴리라인 완전 복원**. 분기·T접합 케이스에서 경로가 의도대로 잘리는지 시각 확인. 단독 통과 |
| **M7** | 인스턴스 평가 지표(11절) — `stella/eval/ccq.py`(`InstanceCCQ`) + `test_metric` — 학습 루프와 무관한 잎 모듈 | 완전 복원 GT에서 F1 = 1·커버리지 = 1. 조각 예측의 TP/redundant FP·클래스별 F1 판정. 단독 통과 |
| **M8** | Lightning module + train.py + **로깅**(9.4) + **시각 로그**(9.5) + `validation_step`에 디코더(M6)·지표(M7) 연결 | 합성 데이터 2-GPU DDP 100 step 스모크. 체크포인트 저장→재개 시 손실 곡선 연속. **검증 에폭이 디코딩+평가 포함으로 완주**, `metrics.csv`에 손실·`val/inst/*` 전 항목 기록, `viz/epoch000/`에 3종 PNG 생성 |
| **M9** | `SeedMapDataset`(6.7절) + `stat_labels.py` 전체 데이터 재집계 (미등록 category, 차수>$D$ 빈도, 노드 수 분포) | 실데이터 1000 step 학습에서 손실 안정 하강. 미등록 category 목록 확인 후 6.7.1 표 확정, `n_max`·`max_degree` 재검토 |

각 단계는 앞 단계 테스트를 깨지 않는 것을 전제로 한다(pytest 누적).
**디코더(M6)와 평가 지표(M7)를 학습 루프(M8)보다 먼저 둔다.** 둘 다 학습 루프에 의존하지 않는
잎(leaf) 모듈이라 단독으로 만들고 테스트할 수 있다 — 디코더는 GT를 출력 형식으로 주입해 복원을 확인하고
(`test_decode`), 지표는 합성 예측·GT로 확인한다(`test_metric`). 검증 루프가 매 에폭 디코딩+평가를
하므로(9.1), 이 둘이 먼저 있어야 M8에서 `validation_step`을 한 번에 완성한다.

**M0~M9는 완료됐다** (`feat/reimplement`, 첫 학습 `log/260804_181614_base_full4gpu`).
위 표의 M1·M5·M6 완료 판정 문구는 당시(구 설계) 기준의 기록으로 남긴다.
9차 개정(선 단위 사슬)을 반영하는 다음 마일스톤:

| 단계      | 내용                                                                                       | 완료 판정                                                                                                       |
| ------- | ---------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| **M10** | **인코더 v2** (6.4절 — 선별 래스터·끝칸 미채움·3×3 순위 규칙·**`conn_dirs` 직접 저장**) + `test_encode` 재작성 + 합성 데이터 케이스 보강 | 새 불변식 9종 통과(반평행성·표적 검증 포함). `viz_gt.py` 육안 확인 — T접합·X교차·완만한 대각선·파선. 새 인코더 기준 통계(사슬 길이·소유권 손실·2셀 이하 선 비율)와 인코딩 시간 기록 |
| **M11** | **손실 v2** (8절 — `conn_dirs` 직접 매칭(유도 제거), $\mathcal{L}_{end}$ 신설, $\mathcal{L}_t$ 제거, `match_ambiguity`) + 모델 출력에 `end_logit` | 합성 1장 과적합에서 total → ~0, `match_ambiguity` → ~0, 끝 셀 end 확률 → 1                                               |
| **M12** | **디코더 v2** (`ChainDecoder`, 10절) + `test_decode` 재작성                                      | **SEED-MAP val GT 주입에서 인스턴스 F1 ≈ 1, 조각/GT ≈ 1.0** (구 설계 상한 0.69·1.7배를 뚫는 것이 재설계의 존재 이유). 합성 접합·교차 케이스 육안 확인 |
| **M13** | **split 폴더 재정리** — 파일 복사는 **완료**(6.7.2절, 8,979/1,282/2,567 검증 끝). 남은 것: 로더가 splits 폴더를 읽도록 수정 | 세 split 로드 건수가 `dataset.json`과 일치, 기존 학습 스모크 통과                                                              |
| **M14** | **재학습 + 스윕** — base config로 재학습, 학습된 체크포인트로 디코더 임계값 스윕, w=7 vs w=9 비교                     | `val/inst/f1`이 coverage에 근접(조각 벌점 소멸 확인). 스윕 결과를 `DecodeConfig` 기본값에 반영                                      |

---

## 13. 결정 사항

### 확정 (사용자 결정 반영)

| #    | 항목                | 결정                                                                                                                                                                                             |
| ---- | ----------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1    | 연결 슬롯 수           | **$R = 2$ 확정 (K = 3, 10차 개정 — 사용자 결정).** GT 분기 수와 일치. Y자 분기도 "주도선 + 보조선" 두 인스턴스로 찾으면 되므로 셀이 세 방향을 낼 필요가 없다. 매칭 순열 2개, 무매칭 슬롯 없음. $R = 3$은 ablation(`exp_r3`)으로만                                |
| 2    | ~~$t$의 의미~~       | **폐기 (9차 개정).** 슬롯 종점 라벨 $t$는 끝과 다른 클래스 접합을 겸해 디코더 규약 충돌을 낳았다. 끝은 셀 단위 `end_map` 직접 감독(`end_logit`, 8.2절)으로, 다른 클래스 접합 간선은 인코딩에서 제거(6.4절)                                                    |
| 3    | 백본                | DINOv3 ViT-L/16 위성판(`sat493m`). HF 게이트 동의 진행                                                                                                                                                   |
| 4    | InternImage       | 미지원 (커스텀 CUDA 커널 배제)                                                                                                                                                                           |
| 5    | 매칭 비용             | **방향 정렬 + 존재 확률만.** 클래스 일관성 항은 상황이 복잡해 제외                                                                                                                                                      |
| 6    | `grid_stride`     | 4 (L = 192) 유지                                                                                                                                                                                 |
| 7    | 실데이터              | **SEED-MAP v1.1.** 폴리라인 JSON + 온라인 인코딩. 원본은 평평한 구조 + `dataset.json`이며, **`SEED_MAP_v1.1_splits/{train,val,test}/{image,label}`로 재정리해 읽는다** (6.7.2, M13)                                        |
| 7-a  | 학습 클래스            | **`category_id` 11종 + 배경 = 12** (논문 Table V와 일치, 6.7.1). `type_id`는 미사용, `599 others`는 제외                                                                                                      |
| 7-b  | `geometry_type`   | **`LINE_STRING`만 사용.** `POLYGON`(화살표·횡단보도 등 노면 기호, 샘플의 36%)은 전부 제외                                                                                                                             |
| 7-c  | 타일 경계             | `image_points`가 타일 밖까지 이어진다(전체 점의 49.8%). **경계에서 교점을 넣어 자르고, 잘린 끝은 그냥 선의 끝으로 취급**(별도 플래그 없음, 6.7.4)                                                                                            |
| 7-d  | `n_max`           | 8000 → **9500** (9차 개정). 전체 train 실측 최대 8,909 — 8000이면 GT 셀이 잘린다 (6.7.5)                                                                                                                       |
| 8    | batch_size        | **1로 확정** (9차 개정, 실측 종결). OOM 원인은 윈도우 어텐션 활성 — w=7 + 윈도우 층 checkpointing으로 bs=4까지 열렸지만 처리량 이득이 미미해 bs=1 + `accumulate=16` 유지, 메모리는 `n_max`에 사용 (9.3)                                            |
| 9    | 클래스 선택 방식         | config의 **`path` + `name` 문자열 + `build_instance` 단일 관문**(5절). 자동 registry·decorator registry·config에 클래스 객체 직접 삽입은 모두 배제 — 순환참조 없음과 import 목록 관리 없음을 우선했다                                      |
| 10   | `from_cfg` 시그니처   | `from_cfg(module_cfg, cfg, **kwargs)` — 자기 섹션 config와 전체 cfg를 함께 받는다. 부품이 많아질수록 named parameter 릴레이가 지저분해지기 때문                                                                                 |
| 11   | 패키지 관리            | `uv` + editable 설치. lockfile로 재현성 확보, torch는 CUDA 인덱스를 명시 (2절)                                                                                                                                 |
| 12   | 실행 재현             | **소스 전체 복사**(`results/*/src/`). git commit·diff 방식 배제 — 실험마다 커밋을 강제하지 않기 위함 (4.3절)                                                                                                             |
| 13   | shape 기호          | **`B`는 언제나 배치.** 연결 슬롯 수는 **`R`**, 노드당 토큰 수는 `K = R + 1` (6.1절 기호표)                                                                                                                            |
| 14   | 백본 클래스 구조         | **3층** — `Backbone` 베이스 / 라이브러리별 중간 인터페이스(`HuggingFaceBackbone`·`TimmBackbone`) / **모델 계열별 클래스**(`Dinov3Backbone`·`SwinBackbone` …). 계열 안의 스케일(L/B/S)은 `pretrained` 문자열이 정하고 클래스는 하나 (7.2절)    |
| 15   | 손실 구조             | **종류별 모듈 3개 + 조립** — `HeatmapLoss`·`SelfSlotLoss`·`ConnLoss`를 `StellaCriterion`이 합산. 각 모듈이 이름 붙은 손실 dict를 반환하고 그것이 그대로 로그가 된다 (8.0절)                                                           |
| 15-a | 손실 가중치            | **단일 층.** 최하위 손실 항목마다 가중치 하나($w_{hm}, w_{cls}, w_{coord}, w_{end}, w_e, w_{dir}$ — 총 6개)만 두고, 모듈 단위 상위 가중치는 두지 않는다. 실효 가중치가 두 값의 곱이 되면 파라미터만 늘고 실수하기 쉽기 때문. `focal_*`·`match_w_*`는 가중치가 아니다 (8.0절) |
| 16   | 로깅                | **에폭 단위.** 학습 1에폭 ↔ 검증 1에폭. 손실 dict 전체를 `train/*`·`val/*`로 기록. 검증에서만 인스턴스 지표(`val/inst/*`)와 시각 로그 추가 (9.4절)                                                                                    |
| 17   | 시각 로그             | 검증 **배치당 첫 샘플 1장**, 3종 PNG — heat(블렌딩) / class(셀 중심 2×2) / slot(self 점 + R·G·B 방향선). 그리기 함수는 Lightning과 분리 (9.5절)                                                                              |
| 18   | 객체 생성(디코딩)        | **사슬 확장 3단계 (9·10차 개정)** — 정점 추출 → **클래스 확률 국소 피크 시드에서 양방향으로**, 마주봄($\mathbf{c}\cdot\mathbf{n} \to -1$) + 정렬 + 클래스 확률 하한 게이트로 한 노드씩 확장, 완성 사슬은 순도 검사 → 후처리. 구 4단계(간선 후보·양방향 확인·경로 절단)는 간선 소실 증폭으로 폐기. **매 에폭 검증에서 실행** (10절) |
| 19   | GT 노드 좌표          | **(9차 개정) 선마다 따로 그리고, 셀 소유 선의 픽셀 무게중심**을 쓴다. 구 "클래스 단위로 합쳐 그리기"는 GT 그래프와 인스턴스 목록의 불일치(F1 상한 0.69)를 낳아 폐기. 이중선 두 인스턴스는 두 사슬로 남긴다 (6.4절)                                                          |
| 20   | 셀 소유권             | **그 셀에 픽셀이 더 많은 선**이 이긴다(클래스 단위 → 선 단위로 변경, 9차 개정). "통과선이 소유한다"가 규칙 없이 따라 나온다. 동점은 희소 클래스 → 앞선 인스턴스 순 (6.4절)                                                                                  |
| 21   | 연결성 표현            | **(9·10차 개정) 선별 사슬 + 방향 직접 저장.** 소유 셀을 순서대로 잇되(잃은 칸 건너뛰기), 저장하는 것은 이웃 좌표가 아니라 **자기 점 → 이웃 점 단위 방향 2개**(`conn_dirs`)다. 끝 규약: 양 끝칸 미채움, 끝 셀 `end_map = 1` + 끝점 방향. 구 간선 합집합·차수 유도·`conn_cells`·`end_point`는 폐기 (6.2·6.4절) |
| 22   | GT 캐시             | 기본은 **온라인 인코딩**(벡터 단계 증강 유지). **val/test split만 무조건 캐시**한다(증강이 없어 결정적). 학습 캐시는 **불필요 확정** — 실측 70 ms/샘플 vs 스텝 328 ms / workers 8 (6.4.1절)                                                     |
| 23   | ~~종점 셀의 클래스 손실~~  | **특례 폐기 (9차 개정).** 근거였던 "끝 셀의 라벨 모호성"이 끝칸 미채움 규약으로 소멸 — 클래스 손실을 전 양성 셀에 준다. 디코더 폴리라인 클래스는 다수결이 아니라 **시드 클래스**(순도 검사가 보증, 10차 개정) (8.2·10.3~10.4절)                                                |
| 24   | 거짓 양성 셀의 클래스 손실   | **배경(0) CE를 준다.** 클래스 0을 한 번도 학습하지 않으면 디코더의 배경 필터($\arg\max \ne 0$, 10.2절)가 무력해지기 때문 (8.2절)                                                                                                    |
| 25   | `architecture.md` | 과거 설계 문서로 두고 **정합을 맞추지 않는다.** 본 계획서가 단일 출처다                                                                                                                                                    |
| 26   | 연결 방향의 원점         | **자기 노드 점 $\mathbf{p}^{\mathrm{full}}$** (셀 중심에서 변경, 9차 개정). 방향은 선의 접선이므로 좌표 예측과 원점을 공유하는 것이 자연스럽고, 인코딩·매칭·디코딩이 같은 식을 쓴다 (6.1절)                                                                 |
| 27   | 윈도우 어텐션           | **이웃 gather 방식** ($N \times N$ 마스크 아님) + **`window_size = 7`** + **윈도우 층만 gradient checkpointing.** 실측: 기준 12.09 GiB/455 ms → 6.21 GiB/334 ms. w=7의 정확도 영향은 M14에서 w=9와 비교 (7.6절)               |
| 28   | 미등록 `category_id` | 전체 train 재집계로 표 확정 — `599`·`5011`·`None`(합계 0.2%) 전부 제외. **`5011`(17개) 제외 확정** (6.7.1절)                                                                                                       |
| 29   | 감시 지표 이름          | `switch_rate` → **`match_ambiguity`** — 정의가 "스텝 간 배정 변화"가 아니라 "한 스텝 안의 배정 모호 셀 비율"이므로 이름을 정직하게 바꾼다 (8.3절)                                                                                       |
| 30   | split 정리 방법       | **복사** (사용자 결정). `SEED_MAP_v1.1_splits/`에 완전 독립 사본(+16 GB) — 원본 무손상, 링크의 rsync/도커 마운트 주의사항 회피. 파일 정리 완료, 로더 수정은 M13 (6.7.2절)                                                                       |
| 31   | 끝칸 미채움 범위         | **모든 끝(접합·자유·타일 경계)에 일괄 적용** (사용자 결정). 선이 짧아지지 않는다 — 끝방향 슬롯이 실제 끝점을 향하도록 감독되고, 디코더가 `end_extend`만큼 연장점을 찍어 복원한다. 3칸 선도 1셀 사슬 + 양방향 연장으로 살아남는다 (6.2·10.3절)                                        |
| 32   | 끝 예측 주체           | **`end_logit` 신설 확정** (사용자 결정). self 토큰이 셀 단위로 직접 예측, $t$ 슬롯은 완전 폐기 — 결정 2 최종 확정 (7.1·7.7·8.2절)                                                                                                 |
| 33   | 평행 겹침(연속 소유권 손실)  | **무시** (사용자 결정 — 매우 드문 케이스). 보정 규칙을 두지 않는다. M10 통계에 빈도만 기록해 둔다 (6.4절)                                                                                                                           |
| 34   | target 형태         | **모델 출력과 같은 형태로 (10차 개정, 사용자 결정 — 구 방침 폐기).** 인코더가 셀마다 **연결 방향 2개를 직접 저장**(`conn_dirs`, 자기 점 → 이웃 점). `conn_cells`(이웃 좌표)·`end_point`·criterion 유도를 전부 폐지 — 분기가 항상 2로 고정되면서 간접층의 근거가 사라졌다 (6.2절)  |
| 35   | 디코더 시드·클래스 관리     | **(10차 개정, 사용자 결정)** 시드 = **클래스 확률 국소 피크**(양방향 확장), 사슬 클래스 = 시드 클래스 고정. 확장 게이트에 사슬 클래스 확률 하한 `min_class_prob = 0.1`(일시적 하락 허용), 완성 사슬은 **순도 검사** `purity_thresh = 0.6` 초과 필수 — 미달 시 정점 반환 후 다른 시드에서 재시작 (10.3절) |

### 남은 확인 (구현하며 확정)

**측정해서 정할 것 (9차 개정 후)**

- **새 인코더 통계 (M10)**: 선 단위 인코딩 기준으로 재집계 — 사슬 길이 분포, 소유권으로 잃는 셀
  비율(연속 손실 길이 포함 — 결정 33의 "드물다" 확인용), 1셀 사슬이 되는 선의 비율,
  인코딩 시간(선별 그리기 비용).
- **디코더 하이퍼파라미터** (`DecodeConfig`의 `radius`·`align_thresh`·`opp_thresh`·`end_thresh`·
  `exist_thresh`·`min_class_prob`·`purity_thresh`·`end_extend`): 설계는 확정했고 **값은 학습된
  체크포인트로** 검증 셋에서 스윕한다(M14) — GT 주입에는 걸러낼 오탐이 없고 확률이 0/1이라
  필터 강도를 판단할 수 없다(구 `mutual` 스윕의 교훈).
- **w=7 vs w=9** (M14): 메모리·속도는 실측 완료(7.6절), 정확도 비교가 남았다.
- **평가 지표 값**: ① 버퍼 $\rho$를 차선 간격 실측 후 확인(현재 12 px, GSD 0.1278 m/px 기준 1.5 m).
  ② $\theta_{cov} = 0.5$ / $\theta_{cor} = 0.9$를 검증 셋에서 확인. (파선 GT 보정은 도입하지 않는 것으로
  종결 — 11.4절.)

**해결됨 (9차 개정에서 종결)** — GT 인코딩 캐시(불필요 실측 확정, 6.4.1), 전체 데이터 재집계(6.7.5),
미등록 `category_id`(6.7.1 확정), batch_size(1 확정, 9.3), `n_max`(9500, 7.1), 분기점 통과 규칙
(사슬 확장으로 개념 자체가 소멸, 10절), 파선 GT 규약(보정 불필요, 11.4).

> `architecture.md`는 과거 설계 문서이므로 정합을 맞추지 않는다.

---

## 14. 남은 의문점 (9차 개정 후)

9차 개정에서 설계가 갈리던 네 가지 — ① split 정리 방법(→ 복사), ② 끝칸 미채움 범위(→ 일괄 적용
+ 디코더 끝 연장), ③ 끝 예측 주체(→ `end_logit` 신설), ④ 평행 겹침(→ 무시) — 는 **전부 사용자
결정으로 종결**됐다(13절 결정 30~33). 남은 것은 결정이 급하지 않은 확인 항목들이다.

- **$R = 3$ ablation** (급하지 않음): $R = 2$는 확정됐다(결정 1). 여분 슬롯 1개가 오히려
  도움이 되는지(예: 분기 근처에서 두 후보를 동시에 들고 있다가 매칭이 고르게 하는 효과)만
  나중에 `exp_r3`로 확인해 볼 수 있다 (4.2절 예시).
- **`end_extend` 값** (스윕): 끝 연장 길이 기본 1셀은 기하 추정이다(끝점이 미채움 이웃 칸 안).
  M14 디코더 스윕에 포함한다.
- **w=7 vs w=9** (M14): 메모리·속도는 실측 완료(7.6절), 정확도 비교가 남았다.
- **자기 교차 선**: 한 선이 자기 자신과 교차하면 같은 셀이 사슬에 두 번 나올 수 있다.
  드물어서(라벨 특성상 거의 없음) 뒤 등장을 무시하는 것으로 두고, M10 통계에서 빈도만 확인한다.
- **끝 셀 클래스 불균형**: $\mathcal{L}_{end}$의 양성(끝 셀)이 약 2.5%다. pos_weight 없이
  시작하되 end 재현율이 낮으면 조정한다(8.2절).

# 구조와 조립 규칙 (1~5절)

저장소의 뼈대다 — 설계 원칙, 기술 스택, 폴더 구조, config 체계, 객체 조립 방식.
전체 색인과 문서 작성 원칙은 [design.md](design.md)에 있다.

---

## 1. 설계 원칙 — 기존 단점을 이렇게 극복한다

이 저장소는 선행 저장소 STELLA2026을 고치는 대신 처음부터 다시 구현한 것이다.
아래 표가 그 결정의 근거이자 지금 코드가 지키는 규칙이다.

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
| 9   | left/right 용어 혼란, 단위벡터 vs 오프셋 불일치(TODO)                         | 슬롯은 무순서(매칭으로 배정)라 prev/next가 없다. 연결은 GT도 예측도 **단위 방향 벡터 하나**로 같은 형태다(6.2절) — 표현이 하나뿐이라 불일치가 생길 자리가 없다                                                                                               |
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
| 지표 집계    | `torchmetrics` ≥ 1.4                    | `InstanceCCQ`(11절)·`CellDiagnostics`(11.5절)가 `Metric`을 상속 — DDP `all_gather` 위임                                                                                                                                    |
| 수치       | `numpy`, `einops`, (`scipy`)            | scipy는 매칭 단위테스트의 대조 구현(LSA)용                                                                                                                                                                                         |
| 시각화(개발용) | `opencv-python`                         | GT 인코딩 확인 스크립트                                                                                                                                                                                                       |
| 품질       | `ruff`, `pytest`                        | 포매팅+린트+테스트                                                                                                                                                                                                           |

---

## 3. 폴더/파일 구조

계획 당시의 트리에서 실제로 갈라진 부분이 있다 — 특히 `decode/`가 `graph.py` 하나에서 5개로,
`eval/`이 `ccq.py` 하나에서 3개로 늘었다. 아래는 **현재 구조**다.

```
stella/                         # 저장소 루트 (패키지명 "stella", editable 설치)
├── pyproject.toml              # 패키지·의존성·ruff·pytest 설정
├── README.md
├── configs/
│   ├── schema.py               # ★ 모든 config dataclass 정의 (단일 파일)
│   ├── base.py                 # get_config() -> ExperimentConfig (기본 실험 — SEED-MAP + ConvNeXtV2 + FPNLite)
│   ├── unit.py                 # 개선 루프 "단위 실험(U)" 규격 — 1 GPU·서브샘플·짧은 에폭 (research 스킬)
│   └── exp_*.py                # 변형 실험: base를 불러와 필드만 수정 (dinov3/r3/synthetic/vit_sfp)
├── stella/
│   ├── __init__.py             # 비워 둔다 (import 목록을 관리하지 않는다)
│   ├── builder.py              # resolve / build_instance / check_all — 클래스 선택 단일 관문 (5절)
│   ├── config_io.py            # config 로드·override·저장된 config 재적용 — Lightning 비의존 (4.3절)
│   ├── data/
│   │   ├── types.py            # GridDatasetBase(출력 계약 docstring 포함) + collate_fn + CLASS_COLOR
│   │   ├── encode.py           # 폴리라인 → 격자 GT 인코더 (6.4절)
│   │   ├── synthetic.py        # SyntheticDataset — 개발용 합성 데이터 (6.6절)
│   │   ├── augment.py          # 벡터 단계 증강 (flip/rot90 + 임의 회전·스케일 옵션) + 색상 증강
│   │   └── seedmap.py          # SeedMapDataset — SEED-MAP 로더·경계 자르기 (6.7절)
│   ├── model/
│   │   ├── backbone.py         # Backbone 베이스 + HuggingFaceBackbone/TimmBackbone + 계열 클래스 (7.2절)
│   │   ├── neck.py             # Neck 베이스 + SFP / FPNLite → (256,L,L) (7.3절)
│   │   ├── heatmap.py          # 보조 히트맵 헤드 + 노드 선택(thresh/topk) (7.4절)
│   │   ├── rope.py             # 2D axial RoPE (7.6절)
│   │   ├── blocks.py           # slot self-attn / cross-attn(전역·윈도우) / FFN
│   │   ├── heads.py            # self 헤드·연결 슬롯 헤드 (7.7절)
│   │   ├── stella.py           # StellaModel(from_cfg 포함) + ModelOutput 정의 (7.1절)
│   │   └── inject.py           # GT를 ModelOutput 형식으로 주입 — 천장 측정·디코더/손실 검증용 (7.1절)
│   ├── loss/
│   │   ├── matching.py         # 셀별 슬롯 배정 — R! 순열 완전탐색 벡터화 (8.3절)
│   │   ├── heatmap.py          # HeatmapLoss (8.1절)
│   │   ├── self_slot.py        # SelfSlotLoss — 클래스·좌표·끝(end) (8.2절)
│   │   ├── conn.py             # ConnLoss — 매칭 + 존재·방향 (8.3~8.4절)
│   │   └── criterion.py        # StellaCriterion — 위 셋을 조립·가중합 (8.0절)
│   ├── decode/
│   │   ├── vertices.py         # ① 정점 추출 + 시드 순서 (10.2절)
│   │   ├── graph.py            # ChainDecoder 본체 — ② 사슬 확장 오케스트레이션 (10.3절)
│   │   ├── postprocess.py      # ③ 후처리 — 조각 병합(ChainMerger)·RDP 단순화 (10.4절)
│   │   ├── stats.py            # ChainStats — 디코더 정지 사유·순도 탈락·병합 수 카운터 (10.6절)
│   │   ├── cache.py            # 모델 예측을 fp16 희소 npz로 저장/복원 — GPU 없이 디코더 스윕 (10.6절)
│   │   └── sweep.py            # evaluate_decode — 캐시된 예측 + 디코더 설정 → 지표. 튜닝 스크립트가 공유
│   ├── eval/
│   │   ├── ccq.py              # InstanceCCQ — 커버리지 중심 인스턴스 F1 (11.1~11.2절)
│   │   ├── cellstat.py         # CellDiagnostics — 셀 단위 진단 지표 22종 (11.5절)
│   │   └── geometry.py         # 점-폴리라인 거리·버퍼·리샘플 등 기하 유틸
│   └── train/
│       ├── module.py           # StellaTrainModule (LightningModule, 얇게)
│       ├── optim.py            # param group 4개 분리·워밍업+코사인 스케줄
│       ├── viz.py              # 시각 로그 그리기 — 순수 함수, Lightning 무관 (9.5절)
│       ├── callbacks.py        # VizCallback — 검증 배치마다 첫 샘플 저장 (9.5절)
│       └── train.py            # 진입점 + ★최상위 조립 배선 (5절)
├── scripts/
│   ├── viz_gt.py               # GT 인코딩·합성 데이터 육안 확인
│   ├── stat_labels.py          # SEED-MAP 라벨 통계 — 6.7.5절 표를 재생성
│   ├── dump_predictions.py     # 체크포인트 추론(또는 GT 주입)을 예측 캐시(npz)로 저장
│   ├── eval_decode.py          # 캐시된 예측으로 CPU만으로 디코딩+평가 (단일 설정/축 스윕)
│   ├── tune_decoder.py         # DecodeConfig 여러 축을 좌표 하강으로 튜닝
│   ├── run_experiments.py      # 여러 실험 arm을 GPU별로 동시 스케줄링 (개선 루프 무인 실행)
│   ├── summarize_runs.py       # 여러 실행 폴더의 metrics.csv를 표로 비교
│   ├── loss_balance.py         # 손실 항목 스케일 균형 점검 + 가중치 조정 제안
│   ├── class_confusion.py      # 예측 캐시 vs GT class_map — 클래스 혼동행렬 분석
│   └── show_run.py             # 단일 실행 폴더의 metrics.csv를 표로 출력
└── tests/
    ├── helpers.py              # 테스트 공용 헬퍼(합성 배치·GT 주입 재수출 등)
    ├── test_build.py           # ① 전 config의 path/name 해석 (빠름) ② 전체 조립 스모크 (느림, 5절)
    ├── test_encode.py          # GT 인코더 불변식 9종 검증 (6.4절)
    ├── test_augment.py         # 벡터 단계 증강(flip/rot90/회전·스케일)의 좌표 변환 검증
    ├── test_rope.py            # RoPE 상대위치 성질 검증
    ├── test_selector.py        # 노드 선택(thresh/topk, 최소 1노드 보장) 검증 (7.4절)
    ├── test_matching.py        # 순열 매칭을 scipy LSA와 대조 검증
    ├── test_decode.py          # GT를 모델 출력 형식으로 넣으면 원본 폴리라인이 복원되는지 (10절)
    ├── test_postprocess.py     # 조각 병합(ChainMerger)·RDP 단순화 검증 (10.4절)
    ├── test_metric.py          # 인스턴스 CCQ: 완전복원=F1 1, 조각 예측의 TP/redundant FP 판정 (11절)
    ├── test_cellstat.py        # CellDiagnostics 22종 지표 검증 (11.5절)
    ├── test_viz.py             # 시각 로그 함수의 shape·색상 규약 (9.5절)
    └── test_model.py           # shape 테스트 + 1-이미지 과적합 테스트
```

파일 수를 일부러 적게 유지한다. 한 파일 = 한 책임. `util/misc.py` 같은 잡동사니 파일은 만들지 않는다.
**계열 하나 = 파일 하나**로 둔다(베이스와 구현체를 같은 파일에). `backbone.py` 하나에 `Backbone`·`HuggingFaceBackbone`·`Dinov3Backbone`·`TimmBackbone`·`ConvNeXtBackbone` 등이 함께 있는 식이다.
`decode/`·`eval/`은 예외다 — 정점 추출·사슬 확장·후처리·진단이 각자 단위 테스트 대상이라 파일을 쪼갰다(위 목록의 절 번호 참고). 그래도 "여러 클래스가 협력해 한 알고리즘을 이룬다"는 계열 하나의 성격은 유지한다.
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
    name: str  # 클래스 이름 예: "Dinov3Backbone"
```

`kw_only=True`는 취향이 아니라 **필수**다. dataclass 상속은 하위 클래스 필드를 베이스 필드 **뒤에** 붙이는데,
일반 dataclass는 "기본값 있는 필드 뒤에 기본값 없는 필드"를 허용하지 않는다. `ModuleConfig`가 이미 기본값을 가지므로,
하위 config에 기본값 없는 필드를 하나라도 추가하는 순간 `TypeError: non-default argument follows default argument`가
**import 시점에** 터진다. `kw_only=True`면 생성자가 `def __init__(self, *, ...)`가 되어 이 제약 자체가 사라진다.
덤으로 위치 인자 생성이 막혀서, 필드 순서를 바꿔도 기존 코드가 조용히 어긋나지 않는다.

```python
@dataclass(kw_only=True)
class DataConfig(ModuleConfig):
    path: str = "stella.data.seedmap"
    name: str = "SeedMapDataset"  # 합성 데이터는 "stella.data.synthetic" / "SyntheticDataset" (6.6절)
    # 원본 SEED_MAP_v1.1을 {train,val,test}/{image,label} 구조로 재정리한 사본 (6.7.2절)
    root: str = "/media/humpback/.../Ongoing/2026_stella/SEED_MAP_v1.1_splits"
    image_size: int = 768  # SEED-MAP 원본 크기와 동일 — 리사이즈 없음
    grid_stride: int = 4  # 격자 배율 s. L = image_size // grid_stride = 192
    num_classes: int = 12  # 0 = background + 차선 11종 (6.7.1절)
    batch_size: int = 1  # 확정 — bs=2는 처리량 +16%뿐, accumulate가 유효 배치를 만든다 (9.3절)
    num_workers: int = 8
    max_degree: int = 2  # D: 셀당 GT 분기 수. 선 단위 사슬이라 **항상 정확히 2** (6.4절)
    encode_supersample: int = 1  # GT 래스터화 배율. 1 = 픽셀 해상도 (6.4절 A단계)
    cache_gt: str = "val_test"  # GT 캐시: "none" | "val_test"(기본) | "all" (6.4.1절)
    cache_dir: str = "/media/humpback/.../Ongoing/2026_stella/gt_cache"  # 데이터셋 폴더 옆에 따로 둔다
    augment: bool = True  # 학습 split에만 적용 (6.7.6절)
    # 격자 대칭 외의 기하 증강 (개선 루프 가설 백로그). 0이면 끈다 — 기본은 기존 동작.
    aug_rotate_deg: float = 0.0  # ± 이 각도까지 임의 회전. 타일 밖은 검게 채운다
    aug_scale_jitter: float = 0.0  # 1 ± 이 비율까지 등방 스케일
    limit: int = 0  # >0이면 split당 앞에서 N개만 사용 (스모크·단위 실험용, `configs/unit.py`)

    @property
    def grid_size(self) -> int:
        return self.image_size // self.grid_stride


@dataclass(kw_only=True)
class BackboneConfig(ModuleConfig):
    path: str = "stella.model.backbone"
    # "Dinov3Backbone" | "SwinBackbone" | "ConvNeXtBackbone" | "HrnetBackbone" | "TimmVitBackbone"
    # 기본값이 DINOv3가 아닌 이유: sat493m 저장소가 HF 게이트라 승인 전에는 받을 수 없다 (14절).
    name: str = "ConvNeXtBackbone"
    pretrained: str = "convnextv2_base.fcmae_ft_in22k_in1k_384"  # HF/timm 모델 ID
    lr_mult: float = 0.1  # optim.py가 읽는다 (__init__ 인자 아님)
    freeze: bool = False
    # 5레벨 백본(HRNet 등)에서 FPNLite가 쓰는 stride 4/8/16/32만 고른다.
    out_indices: tuple = ()  # 비우면 timm 기본값
    img_size: int = 0  # 고정 입력 크기 백본(Swin)에만. 0이면 지정하지 않는다


@dataclass(kw_only=True)
class NeckConfig(ModuleConfig):
    path: str = "stella.model.neck"
    name: str = "FPNLite"  # "SFP"(ViT 단일 스케일) | "FPNLite"(멀티스케일, 기본)
    out_blocks: int = 1  # FPNLite 출력단 3×3 블록 수 — 격자 위 국소 문맥의 양 (가설 백로그)


@dataclass(kw_only=True)
class ModelConfig(ModuleConfig):
    path: str = "stella.model.stella"
    name: str = "StellaModel"
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    neck: NeckConfig = field(default_factory=NeckConfig)
    d_model: int = 256  # neck·블록·헤드가 공유 — 여기 한 곳에만 둔다
    num_heads: int = 8
    num_conn_slots: int = 2  # R = 2 확정 (K = 3) — GT 분기 수와 일치 (결정 1)
    layers: tuple[str, ...] = ("global", "window", "window", "window", "window", "window")
    window_size: int = 7  # w. 9 → 7 (실측: peak 12.09 → 9.72 GiB, step 455 → 291 ms, 7.6절)
    ffn_dim: int = 1024
    dropout: float = 0.0
    grad_checkpoint: bool = True  # 윈도우 층만 재계산 — 활성의 대부분이 거기 있다 (7.6절)
    head_hidden: int = 1  # 출력 헤드 MLP의 은닉 블록 수. 1 = 계획서 원안("2층 MLP", 7.7절)
    share_slot_weights: bool = True  # 연결 슬롯 R개가 헤드 MLP를 공유하는가 (가설 백로그)
    node_sampling: str = "gt+pred"  # 학습 노드 선택: "gt+pred"(기본) | "gt" (7.4절)
    n_max: int = 9500  # 노드 수 상한. 전체 train 8,979장 실측 최대 8,909 (p99 5,893, 6.7.5절)
    heatmap_thresh: float = 0.3  # τ_h
    dilate: int = 3  # 예측 마스크 팽창: 0 | 3 | 5
    # "thresh" = 확률 > τ_h(기본) | "topk" = 확률 상위 K개 — thresh는 보정에 흔들린다(7.4절 실측).
    select_mode: str = "thresh"
    n_topk: int = 4000  # topk 모드의 K. 600장 서브샘플 재측정 근거는 6.7.5절 표 참고


# --- 손실: 종류별 모듈 config + 조립 config (8절) ---
@dataclass(kw_only=True)
class HeatmapLossConfig(ModuleConfig):
    path: str = "stella.loss.heatmap"
    name: str = "HeatmapLoss"
    w_heatmap: float = 1.0  # 총합에 그대로 곱해지는 유일한 가중치
    focal_alpha: float = 0.75  # 가중치가 아니라 focal 형태 파라미터. 0.75는 실측(f1 +22.4%)
    focal_gamma: float = 2.0


@dataclass(kw_only=True)
class SelfSlotLossConfig(ModuleConfig):
    path: str = "stella.loss.self_slot"
    name: str = "SelfSlotLoss"
    w_class: float = 1.0
    w_coord: float = 1.0
    w_end: float = 1.0  # 끝 셀 BCE — end_map 직접 감독 (8.2절)
    # 끝 셀 양성이 전체 양성의 ~2.5%라 로짓이 음수로 눌린다 (가설 백로그, 14절).
    end_pos_weight: float = 1.0  # 1.0 = 가중 없음
    class_bg_weight: float = 1.0  # 선택 셀의 ~70%가 배경이라 클래스 CE가 배경에 지배당한다 (가설 백로그)
    # 희소 클래스 3종이 검증 200장에서 0회 예측됐다. 전경을 빈도^(-power)로 가중한다 (가설 백로그).
    class_freq_power: float = 0.0  # 0.0 = 가중 없음


@dataclass(kw_only=True)
class ConnLossConfig(ModuleConfig):
    path: str = "stella.loss.conn"
    name: str = "ConnLoss"
    w_exist: float = 1.0
    w_dir: float = 1.0  # 연결 방향 손실 (1 - 내적)
    match_w_dir: float = 1.0  # λ_dir — 손실 가중치가 아니라 **매칭 비용** 계수 (8.3절)
    match_w_exist: float = 1.0  # λ_e   — 〃
    exist_pos_weight: float = 1.0  # 거짓 양성 셀이 압도적일 때 양성 쪽을 든다 (가설 백로그)
    dir_loss: str = "cosine"  # "cosine" = 1 - cos(기본) | "angle" = acos/π — 작은 오차에서 기울기가 산다


@dataclass(kw_only=True)
class LossConfig(ModuleConfig):
    path: str = "stella.loss.criterion"
    name: str = "StellaCriterion"
    heatmap: HeatmapLossConfig = field(default_factory=HeatmapLossConfig)
    self_slot: SelfSlotLossConfig = field(default_factory=SelfSlotLossConfig)
    conn: ConnLossConfig = field(default_factory=ConnLossConfig)


# --- 디코딩(객체 생성) + 평가 + 로깅 ---
@dataclass(kw_only=True)
class DecodeConfig(ModuleConfig):  # 사슬 확장 디코더 (10절)
    path: str = "stella.decode.graph"
    name: str = "ChainDecoder"
    heatmap_thresh: float = 0.3  # τ_h — 노드 후보 (추론 경로, 7.4절)
    exist_thresh: float = 0.3  # τ_e — 연결 슬롯 존재
    end_thresh: float = 0.5  # τ_end — 끝 셀 판정 (σ(end_logit)), 사슬 정지 조건
    radius: int = 5  # 탐색 반경(셀). 차선 간격 11.8 px 의 절반 이하로 잡는다 (10.3절)
    align_thresh: float = 0.95  # 내 슬롯 방향과 실제 상대 방향의 코사인 하한 (c·u_ab)
    opp_thresh: float = 0.7  # 마주봄 하한 — -(c·n) ≥ 이 값 (10.3절)
    w_opp: float = 1.0  # 후보 비용에서 마주봄 항의 비중
    min_class_prob: float = 0.2  # 확장 게이트 — 후보의 사슬 클래스 softmax 확률 하한 (10.3절)
    purity_thresh: float = 0.6  # 사슬 순도 하한 — argmax 클래스 일치 비율. 이하면 사슬 폐기 (10.3절)
    end_extend: float = 1.0  # 끝 셀에서 끝방향 슬롯으로 연장하는 길이(셀) — 10.3절 끝 연장
    min_points: int = 8  # 이보다 짧은 폴리라인은 버린다 (연장점 포함)
    min_points_short: int = 2  # 짧은 종류에만 적용하는 별도 하한 (10.3절)
    short_classes: tuple = (9, 10, 6)  # stop_line · safety_zone · path_change_restriction_line
    min_chain_score: float = 0.0  # 사슬 평균 점수 하한. 0이면 무동작 — 실측에서 이득이 없었다
    simplify_tol: float = 0.0  # >0이면 RDP 단순화 (픽셀)
    # --- 알고리즘 변형 (가설 백로그). 기본값은 전부 M12까지의 기존 동작이다 (10.6절) ---
    seed_mode: str = "class_peak"  # "class_peak"(기본) | "end_peak" — 선의 끝에서 먼저 시작
    stop_needs_nocand: bool = False  # True면 끝 확률 + 후보 없음을 모두 만족해야 정지
    merge_gap: float = 24.0  # 끝점 간 이 거리(픽셀) 안의 조각을 병합 (10.4절 ChainMerger)
    merge_align: float = 0.8  # 병합 정렬 하한 — 두 조각이 서로를 향하는 정도
    align_mode: str = "cosine"  # "cosine" = 각도 게이트(기본) | "perp" = 직선의 수직 이탈 게이트
    perp_thresh: float = 0.7  # perp 모드의 수직 이탈 상한 (셀 단위)
    w_dist: float = 0.072  # 후보 비용의 거리 항 계수 — **반경과 함께 정해진다** (10.3절)
    max_turn_deg: float = 45.0  # 연속한 두 스텝 사이의 방향 변화 상한(도)
    # 임계값들은 학습된 체크포인트로 검증 셋에서 스윕해 확정한다 (`scripts/tune_decoder.py`, 14절).
    # 구 GraphDecoder의 mutual·max_conn_dist·t_thresh는 폐기 — 10절 참고.


@dataclass(kw_only=True)
class MetricConfig(ModuleConfig):  # 인스턴스 평가 지표 (11절)
    path: str = "stella.eval.ccq"
    name: str = "InstanceCCQ"
    buffer_rho: float = 12.0  # ρ (픽셀). 실측 차선 간 거리 중앙값과 정확히 같다 — 11.1절 한계 참고
    cov_thresh: float = 0.5  # θ_cov — 커버리지(완전성) 하한, 관대
    cor_thresh: float = 0.9  # θ_cor — 정확성 하한, 엄격 (모든 GT 대상)
    angle_gate: float = 30.0  # 매칭 시 접선 방향 차 상한(도) — 수직 교차 배제
    sample_step: float = 2.0  # 길이 계산용 폴리라인 샘플 간격(픽셀)
    max_instances: int = 400  # 샘플당 평가 인스턴스 상한 (안전장치)
    # frag는 조금이라도 겹치는 예측을 다 세어 정확성이 높을수록 부풀려진다. frag_strict는 그 GT를
    # 이 비율 이상 덮는 조각만 세는, 조각남의 더 정직한 측정치다 (11.1절).
    frag_min_cov: float = 0.1


@dataclass(kw_only=True)
class CellDiagConfig(ModuleConfig):  # 셀 단위 진단 지표 22종 (11.5절, 개선 루프 전용)
    path: str = "stella.eval.cellstat"
    name: str = "CellDiagnostics"


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
    epochs: int = 100  # 아래는 pl.Trainer가 읽는다 (__init__ 인자 아님)
    accumulate: int = 16  # 유효 배치 = batch_size × accumulate × GPU 수
    grad_clip: float = 0.1
    precision: str = "bf16-mixed"
    devices: str = "auto"  # pl.Trainer devices. "1"이면 단일 GPU 고정(`configs/unit.py`가 이렇게 쓴다)
    limit_val_batches: float = 1.0
    seed: int = 42  # train.py가 읽는다
    # 체크포인트 — 과거 실행에서 last.ckpt가 도중에 멈춘 사고가 있어 명시적으로 둔다.
    ckpt_monitor: str = "val/inst/f1"  # 손실이 아니라 최종 지표를 기준으로 남긴다
    ckpt_mode: str = "max"
    ckpt_top_k: int = 3
    find_unused_parameters: bool = False  # 최소 1노드 보장(7.4절) 덕에 미사용 파라미터가 없다
    output_root: str = "/media/humpback/.../Ongoing/2026_stella/log"  # 〃


@dataclass(kw_only=True)
class ExperimentConfig:  # 이것 자체는 build 대상이 아니다
    cpu: CpuConfig = field(default_factory=CpuConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    decode: DecodeConfig = field(default_factory=DecodeConfig)
    eval: MetricConfig = field(default_factory=MetricConfig)
    cell_diag: CellDiagConfig = field(default_factory=CellDiagConfig)
    log: LogConfig = field(default_factory=LogConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
```

**config 필드가 전부 `__init__` 인자인 것은 아니다.** 상당수는 다른 곳이 읽는다.

| 필드                                                                                     | 읽는 곳                          |
| --------------------------------------------------------------------------------------- | ----------------------------- |
| `BackboneConfig.lr_mult`                                                                | `optim.py` param group (9.2절) |
| `DataConfig.batch_size`, `num_workers`                                                  | `DataLoader` (train/val이 **각각** 워커 풀을 만들어 실제 워커는 2배다) |
| `CpuConfig.*`                                                                           | `CpuBudget.apply()` — 진입 직후 (아래) |
| `TrainConfig.epochs`, `accumulate`, `grad_clip`, `precision`, `devices`, `limit_val_batches` | `pl.Trainer` (9.3절)       |
| `TrainConfig.ckpt_monitor`, `ckpt_mode`, `ckpt_top_k`                                   | `ModelCheckpoint` (9.3절)      |
| `TrainConfig.find_unused_parameters`                                                    | DDP strategy 선택 (9.3절)        |
| `TrainConfig.seed`, `output_root`                                                       | `train.py`                    |
| `ModuleConfig.path`, `name`                                                             | `builder.py`                  |

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


# configs/exp_vit_sfp.py — 백본 교체 예시 (실제 파일). path 기본값이 있어 name만 바꾸면 된다
def get_config():
    cfg = get_base()
    cfg.model.backbone.name = "TimmVitBackbone"  # 게이트 없는 timm ViT — SFP 경로 검증용
    cfg.model.backbone.pretrained = "vit_base_patch16_224.augreg_in21k"
    cfg.model.neck.name = "SFP"
    return cfg
```

장점: IDE에서 "이 필드를 어디서 바꾸는지"가 참조 검색으로 다 나온다. 오타는 실행 중 `AttributeError`가 아니라 **정적 검사**에서 잡힌다.
`path`·`name` 문자열만은 정적 검사가 안 되므로, 4.3의 `check_all`과 5절의 `test_config_resolves`가 그 자리를 맡는다.

### 4.3. 로드 규칙

- 진입점은 `--config <모듈 이름>`을 받는다(예: `--config configs.exp_r3`) + `--override 점.경로=값`
  0개 이상(예: `--override train.epochs=25 data.batch_size=1`). 로드는 `stella/config_io.py`의
  `load_config(module_name, overrides)`가 맡는다 — `importlib.import_module(...).get_config()`로
  base cfg를 만들고, override 문자열마다 `apply_override`가 **현재 값의 타입**(`cast_like`)으로
  캐스팅해 덮어쓴다. 이 모듈은 Lightning·DataLoader를 끌고 오지 않는다 — 학습 진입점(`train.py`)뿐
  아니라 GPU 없이 도는 디코더 튜닝 스크립트(`scripts/eval_decode.py`·`tune_decoder.py`)도 같은
  함수로 config를 읽는다. `apply_saved_config(cfg, saved_json)`도 같은 파일에 있다 — 실행 폴더의
  `config.json`을 현재 스키마 위에 되먹여 "그 실행이 무슨 설정이었나"를 재현할 때 스크립트가 쓴다.
  `configs`가 설치된 패키지이므로(2절) 파일 경로 대신 모듈 이름을 받는 편이 오타 시 에러가 명확하다.

- **config를 로드한 직후 `check_all(cfg)`를 부른다**(5절). 백본 가중치 다운로드·CUDA 초기화 전에
  모든 `path`/`name` 오타가 여기서 걸린다.

- 학습 시작 시 출력 폴더(`{train.output_root}/{YYMMDD_HHMMSS}_{config}[_{tag}]/`)에 다음을 남긴다.
  DDP는 같은 스크립트를 rank 수만큼 다시 실행하는데, 폴더는 rank 0만 만들고 환경변수로 경로를
  자식 프로세스에 물려준다.
  
  - `config.json` — `dataclasses.asdict(cfg)`.
  
  - **`src/` — 소스 전체 복사.** git commit·diff 방식을 쓰지 않는다. 실험마다 커밋을 강제하면
    "시작하자마자 끄고 숫자 하나 바꿔 다시 돌리는" 작업 방식에서 커밋 이력이 못 쓰게 되고,
    복원하려면 그 커밋이 살아 있는 저장소가 필요하다. 소스 전체는 수백 KB라 체크포인트에 비하면 무시할 크기이고,
    결과 폴더만으로 자기완결적이다.
    
    ```python
    shutil.copytree(
        repo_root,
        out_dir / "src",
        ignore=shutil.ignore_patterns(
            "__pycache__", ".git", ".venv", "results", "data", "*.pyc", "viz_gt_out", "docs"
        ),
    )
    ```
    
    제외 패턴은 필수다 — 그냥 복사하면 `data/`·`results/`까지 끌려온다.
  
  - `git_sha.txt` — git 저장소일 때만, `git rev-parse HEAD`와 dirty 여부 한 줄. 커밋을 강제하지 않으며
    소스 복사를 대체하지도 않는다. "대략 어느 시점 코드인가"를 훑을 때만 쓴다.

- 체크포인트가 하나도 없는 실행 폴더를 지우는 정리 스크립트는 **아직 없다** — 수동으로 지운다(백로그).

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


def main() -> None:
    cfg = load_config(args.config, args.override)  # stella/config_io.py (4.3절)
    check_all(cfg)  # ← 오타는 여기서 전부 걸린다

    parts = {
        "model": build_instance(cfg.model, cfg),
        "criterion": build_instance(cfg.loss, cfg),
        "decoder": build_instance(cfg.decode, cfg),
        "metric": build_instance(cfg.eval, cfg),
        "cell_diag": build_instance(cfg.cell_diag, cfg),  # CellDiagnostics (11.5절)
    }
    module = build_instance(cfg.train, cfg, **parts)
    train_set = build_instance(cfg.data, cfg, base=GridDatasetBase, split="train")
    val_set = build_instance(cfg.data, cfg, base=GridDatasetBase, split="val")
    ...
```

모든 줄이 같은 모양이라 "이 부품은 어떻게 만들더라"를 다시 확인할 일이 없다. 실제 배선은
`decoder`·`metric`·`cell_diag`까지 5개 부품을 `StellaTrainModule`에 한 번에 넘긴다(9.1절) —
디코더·평가 지표(M6·M7 당시 잎 모듈이었던 것)와 셀 진단(개선 루프에서 추가)이 전부 여기 모인다.

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

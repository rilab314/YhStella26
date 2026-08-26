# 학습·디코딩·평가 (9~11절)

학습 루프와 로깅, 셀 예측을 폴리라인 객체로 바꾸는 디코더, 인스턴스 단위 성능 지표.
전체 색인과 문서 작성 원칙은 [0_design.md](0_design.md)에 있다.

---

## 9. 학습 파이프라인 — `stella/train/`

### 9.1. `StellaTrainModule` (LightningModule, 얇게 유지)

```python
class StellaTrainModule(pl.LightningModule):
    def __init__(self, *, model: StellaModel, criterion: StellaCriterion,
                 decoder: ChainDecoder, metric: InstanceCCQ, cell_diag: CellDiagnostics,
                 lr: float, weight_decay: float, warmup_steps: int,
                 backbone_lr_mult: float, batch_size: int): ...

    def training_step(self, batch):          # forward → criterion → 손실 dict 로깅 → total 반환
    def validation_step(self, batch):        # forward → criterion·cell_diag 로깅 → 디코딩(10절) → 지표 누적
    def on_validation_epoch_start(self):     # 디코더 정지 사유 통계(ChainStats, 10.6절) 리셋
    def on_validation_epoch_end(self):       # 인스턴스·셀·디코더 지표를 집계·로깅 (val/inst, val/cell, val/dec)
    def on_train_epoch_start(self):          # param group별 lr을 lr/{group}로 로깅 (9.4절)
    def configure_optimizers(self):          # optim.py 호출
```

- 받는 것은 `model`·`criterion`·`decoder`·`metric`·`cell_diag`와 옵티마이저 값 몇 개뿐(`backbone_lr_mult`·
  `batch_size`는 `cfg.model.backbone.lr_mult`·`cfg.data.batch_size`를 `from_cfg`가 값으로 뽑아 넘긴 것).
  **전역 cfg를 들고 다니지 않는다** — 값만 받고 원본 dataclass는 참조하지 않는다.
- **`validation_step`은 디코딩까지 한다.** 매 에폭 검증마다 모델 출력을 폴리라인 객체로 만들고(10절),
  인스턴스 지표(11절)와 **셀 단위 진단 지표**(`cell_diag.update`, 11.5절)를 함께 누적한다. 여기서는
  **호출 지점과 누적 구조만** 다룬다.
- 시각 로그는 module이 아니라 **callback**이 맡는다(9.5절). 학습 로직과 그리기 로직을 섞지 않는다.

### 9.2. Optimizer / 스케줄 — `optim.py`

- AdamW. **param group 4개** — (백본인가) × (bias·norm인가)의 조합 전부: `backbone`(`lr × lr_mult`),
  `backbone_nodecay`(〃 + weight decay 0), `main`(기본 lr·wd), `main_nodecay`(기본 lr, weight decay 0).
  계획 당시의 "3개"에서 백본 안의 bias·norm도 별도 그룹으로 갈라졌다 — 백본 lr을 낮추면서도 그 안의
  bias·norm은 decay를 받지 않아야 하기 때문이다.
- 스케줄: **linear warmup(1000 step) + cosine decay** (step 단위), 코사인 바닥에 `MIN_LR_RATIO = 0.01`
  최저-lr 비율을 곱한다 — lr이 0으로 완전히 죽지 않게 한다.

### 9.3. Trainer 설정과 진입점 — `train.py`

5.3절의 배선 코드에 이어서:

```python
trainer = pl.Trainer(
    max_epochs=cfg.train.epochs,
    check_val_every_n_epoch=1,  # 학습 1에폭 ↔ 검증 1에폭 (9.4)
    precision=cfg.train.precision,  # bf16-mixed
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=_devices(cfg),  # cfg.train.devices — "auto" 기본, unit 규격은 "1"
    strategy=_strategy(cfg),  # GPU 1개면 "auto", 여러 개면 "ddp" | "ddp_find_unused_parameters_true"
    gradient_clip_val=cfg.train.grad_clip,
    accumulate_grad_batches=cfg.train.accumulate,
    limit_val_batches=cfg.train.limit_val_batches,
    callbacks=[
        ModelCheckpoint(
            monitor=cfg.train.ckpt_monitor,  # 기본 "val/inst/f1" — 손실이 아니라 최종 지표 기준
            mode=cfg.train.ckpt_mode,  # "max"
            save_top_k=cfg.train.ckpt_top_k,  # 3
            save_last=True,
            filename="epoch{epoch:03d}",
            auto_insert_metric_name=False,
        ),
        build_instance(cfg.log, cfg, out_dir=..., grid_stride=cfg.data.grid_stride),  # VizCallback (9.5)
        TQDMProgressBar(refresh_rate=20),
    ],
    logger=CSVLogger(save_dir=str(out_dir), name="", version=""),
    log_every_n_steps=10,
)
trainer.fit(module, train_loader, val_loader, ckpt_path=args.resume or None)
```

- **`ckpt_monitor`가 손실이 아니라 지표인 이유:** 과거 실행에서 `last.ckpt`가 도중에 멈춘 사고가
  있어, "마지막"이 아니라 "가장 좋았던 지표"를 명시적으로 남긴다(9.1절 결정).
- **배치 크기 확정 (13절 결정 8, 실측으로 종결):** bs=2가 OOM이던
  원인은 가중치가 아니라 윈도우 어텐션의 활성이었고(7.6절 — 파라미터·옵티마이저는 1.43 GiB로 전체의
  15%뿐), `window_size = 7` + 윈도우 층 checkpointing으로 해소됐다.
  확정 설정(w=7, ckpt, `n_max = 9500`) 실측(RTX 4090, 단일 GPU 마이크로벤치):

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
- **실제 학습 실행 속도 (M14, ConvNeXtV2-base 대역 백본):** 단일 GPU 4.8 it/s. 개선 루프 F 규격
  (4-GPU DDP, `configs/base.py` 그대로)은 에폭당 약 14분 — 위 표의 마이크로벤치는 디코딩·지표
  계산이 빠진 순수 forward/backward 스텝이고, 이 값은 검증 에폭(매 에폭 디코딩+평가 포함, 9.6절)까지
  포함한 실측이라 더 대표성이 있다.
- 출력 폴더: `{train.output_root}/{YYMMDD_HHMMSS}_{config}[_{tag}]/` — `config.json` + `src/`(소스
  전체 복사) + `git_sha.txt` + `checkpoints/` + `metrics.csv` + `viz/` (4.3절).
- EarlyStopping은 아직 붙이지 않았다 — `ckpt_monitor`(`val/inst/f1`)로 체크포인트만 지표 기준으로 남긴다.

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
| `train/self_slot/end`    | 끝 셀 BCE (원시)                                                             |
| `train/self_slot/total`  | $w_{cls}\mathcal{L}_{cls} + w_{coord}\mathcal{L}_{coord} + w_{end}\mathcal{L}_{end}$ |
| `train/conn/exist`       | 연결 존재 BCE (원시)                                                           |
| `train/conn/dir`         | 연결 방향 오차 (원시. 기본 곡선은 acos/π — 8.4절)                                      |
| `train/conn/match_ambiguity` | **손실 아님.** 배정 모호 셀 비율 — 매칭 불안정성 감시(8.3, 구 `switch_rate`)             |
| `train/conn/total`       | $w_{e}\mathcal{L}_{e} + w_{dir}\mathcal{L}_{dir}$                        |

`*/total`은 그 모듈의 가중합이고 `train/total`은 그것들의 **단순 합**이다(8.0절 — 상위 가중치가 없다).
개별 항목은 **원시 값**으로 남기므로, 가중치를 조정할 때 항목별 실제 크기를 그대로 비교할 수 있다.

**학습률 로깅은 `training_step`이 아니라 `on_train_epoch_start`에서 한다.** 옵티마이저 param group이
4개(9.2절)이므로 키도 4개다: `lr/backbone`, `lr/backbone_nodecay`, `lr/main`, `lr/main_nodecay`
(계획 당시의 `lr`·`lr_backbone` 2키 서술은 낡았다 — 그룹이 갈라진 만큼 늘었다).

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
  | `val/inst/correctness`                 | 예측이 **GT 선 하나** 위에 머무는 비율 (11.1) — 갈아탐을 보는 지표. **판정에 쓴다**  |
  | `val/inst/rms`                         | 매칭 구간 RMS 횡오차 (11.2)                            |
  | `val/inst/frag`                        | GT당 예측 조각 수 — 연결성·작업량, 부풀려짐(11.1절 한계)          |
  | `val/inst/frag_strict`                 | `frag_min_cov` 이상 덮는 조각만 세는 더 정직한 조각남 (11.1절)  |
  
  주 지표는 **클래스별로 따로** 낸 뒤 전체 micro와 클래스 평균 macro로 묶는다(11.1). **`f1`과
  `coverage`의 격차가 조각남의 크기**이므로 둘을 나란히 본다(11.2).

- **셀 단위 진단 지표 (`val/cell/*`)** — 디코딩을 거치지 않고 격자(셀) 단계의 원시 예측을 GT와
  직접 대조한다. `self.cell_diag.update(output, batch)`를 검증 배치마다 부르고,
  `on_validation_epoch_end`에서 `cell_diag.compute()`를 `val/cell/{key}`로 로깅한 뒤 리셋한다.
  22종 지표의 전체 목록과 뜻은 **11.5절**에 있다 — `link_ok`·`chain_expect`가 특히 중요하다(10.6절).

- **디코더 진단 (`val/dec/*`)** — 사슬 확장이 왜 멈췄는지의 카운터. `on_validation_epoch_start`에서
  `decoder.stats.reset()`, `on_validation_epoch_end`에서 `decoder.stats.summary()`를 `val/dec/*`로
  로깅한다. `stella/decode/stats.py`의 `ChainStats`가 담당한다 — 10.6절 참고.

- **시각 로그** (9.5절) — 스칼라가 아니라 PNG 파일이다.

`logger=CSVLogger(save_dir=out_dir, name="", version="")` → `metrics.csv` 한 장에 에폭별 행이 쌓인다.
`ModelCheckpoint`의 `monitor`는 `cfg.train.ckpt_monitor`(기본 `"val/inst/f1"`, `mode="max"`)로 두고,
파일 이름에 `/`가 들어가지 않도록 `auto_insert_metric_name=False`를 준다(9.3절).

### 9.5. 시각 로그 — `train/viz.py` + `train/callbacks.py`

검증 중 예측을 눈으로 확인하기 위한 PNG를 남긴다. **배치마다 첫 번째 샘플 하나만** 그린다(전부 그리면 느리다).

```
{run}/viz/epoch{E:03d}/{sample_id}.png     # 여섯 페이지를 붙인 2×3 시트 한 장
```

**한 프레임 = 파일 하나다.** 여섯 페이지를 따로 저장하면 같은 장면을 보려고 뷰어에서 파일
여섯 개를 짝지어 열어야 했다. `viz.tile_pages`가 페이지마다 이름표를 얹어 2행 3열로 붙인다
(패널 사이 구분선 4픽셀, 시트 크기 2312×1540).

| | 1열 | 2열 | 3열 |
| --- | --- | --- | --- |
| **1행** | `heat` | `class` | `slot` |
| **2행** | `end` | `gt` | `inst` |

윗줄은 셀 단위 예측, 아랫줄은 폴리라인이다. **`gt`와 `inst`를 나란히 둔 것이 배치의 이유** —
조각남·오검출이 한눈에 대조된다.

`stella/train/viz.py`는 **Lightning을 모르는 순수 함수 모음**이다(`np.ndarray` in → `np.ndarray` out).
그래서 단위 테스트가 가능하고, 같은 함수로 **GT도 그릴 수 있다** — GT의
`class_map`·`coord_map`·`end_map`·`conn_dirs`는 모델 출력과 격자는 물론 **형태까지 같아서**
(설계 방침 1) 인자만 바꿔 넣으면 되고, 방향 유도 코드가 필요 없다.
여섯 페이지를 만들어 시트로 붙이는 일은 `viz.PageRenderer`가 맡는다 —
`stella/train/callbacks.py`의 `VizCallback(pl.Callback)`이 `on_validation_batch_end`에서 샘플 0을 꺼내
호출하고, `scripts/viz_cache.py`가 **예측 캐시에서 같은 렌더러로 같은 시트를 다시 그린다**
(디코더 설정을 바꿔 가며 GPU 없이 그릴 수 있다).

**여섯 가지 그림** (원본 이미지 768×768 위에 그린다. 계획 당시엔 앞의 세 개뿐이었다):

| 페이지     | 내용                                                                                                         |
| ------- | ---------------------------------------------------------------------------------------------------------- |
| `heat`  | 히트맵 확률 $\sigma(\text{heatmap\_logit})$를 **파랑→빨강** 컬러맵으로 칠하고 원본과 **반씩 블렌딩**. 192×192를 nearest로 768×768까지 확대 |
| `class` | 원본 위에 **4×4 셀마다 중심 2×2 픽셀**을 클래스 색으로 칠하기                                                                   |
| `slot`  | 원본 위에 **self 좌표 = 검은 점**, **연결 슬롯 방향 = R/G 선** (자기 점에서 시작, 6.1절 원점 규약)                                    |
| `end`   | `end_logit`을 `heat`와 같은 방식(파랑→빨강, 블렌딩)으로 그린 것 — 끝 셀 예측 확인용                                                |
| `inst`  | **디코딩 결과**(10절 `ChainDecoder` 출력 폴리라인)를 클래스 색으로 그린 것 — GT가 아니라 실제 파이프라인 최종 출력                              |
| `gt`    | GT `instances`(원본 폴리라인)를 같은 방식으로 그린 것 — `inst`와 나란히 보면 조각남·오검출을 육안으로 바로 비교할 수 있다                          |

`gt`·`inst`는 폴리라인의 **양 끝점에 반지름 4의 원**(속은 클래스 색, 테두리는 흰색)을 찍는다.
선만 그리면 어디서 끊겼는지가 보이지 않아 **한 선이 몇 조각으로 쪼개졌는지를 셀 수 없다** —
조각남(`frag`)이 지금의 주 병목이므로 끝점 표시가 진단의 핵심이다.

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

# slot: self 점 + 슬롯 방향선 (둘 다 자기 점에서 — 6.1 원점 규약)
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
- **`batch_size = 1`이면 "배치당 1장"이 곧 "전 이미지"다.** 검증 100장이면 에폭당 600개 PNG(6종 × 100)가
  쌓인다. `log.max_batches`(기본 20)로 상한을 두고, `log.every_n_epochs`로 간격을 조절한다.
- 그리기는 CPU에서 하고, 텐서는 `.detach().cpu().float()`로 옮긴 뒤 넘긴다.

### 9.6. 알려진 함정과 대비책

| 함정                                                              | 대비                                                                                                                                                 |
| --------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| 선(라벨)이 하나도 없는 타일: 노드 0개 → 토큰 스택이 안 돌아 DDP가 unused parameter로 죽음 | 최소 1노드 강제(7.4). 실데이터에서 빈 타일이 많으면 학습 셋에서 제외도 검토                                                                                                     |
| 전역 cross-attn의 attention 행렬 $KN \times N$ (추론 최악 12000×3000)    | 마스크 없는 전역 층은 SDPA flash/mem-efficient 커널이 잡아 실체화 안 됨. 윈도우 층은 **이웃 gather 방식이 기본**(7.6절). gather 활성이 커서 윈도우 층만 gradient checkpointing (7.6절 실측) |
| 가변 $N$ 때문에 `torch.compile`이 재컴파일 반복                             | v1은 compile 끔. 안정화 후 `dynamic=True`로 시도                                                                                                            |
| `F.normalize`의 0 벡터 (학습 초기 conn_dir 원시 출력이 0 근처)                | `eps` 지정 + 방향 손실이 매칭된 슬롯에만 걸리므로 발산하지 않음                                                                                                            |
| GT 연결 수 > $D$인 셀                                                | 인코더가 경고 + 절단 + 통계 출력(6.4). 실데이터에서 빈도 확인 후 $R$ 조정(ablation과 연결)                                                                                     |
| 매 에폭 검증마다 디코딩(10절)을 도는 비용                                       | 디코딩은 CPU·numpy 작업이라 GPU와 겹칠 수 있다. 느리면 검증 셋을 부분 샘플링                                                                    |

---

## 10. 객체 생성 (디코딩) — `stella/decode/`

모델 출력은 **셀 단위 예측**이다(7.1). 평가와 실사용에 필요한 것은 **폴리라인 객체 목록**이다.
이 절이 그 변환을 정의한다. `ChainDecoder`(`decode/graph.py`)는 **매 에폭 검증 단계에서 실행되고**
(9.1), 그 결과가 인스턴스 단위 성능 평가의 입력이 된다.

**입력** `ModelOutput` 한 샘플 + `DecodeConfig`.
**출력** `list[dict]` — `{"class": int, "points": float32 (P, 2) 픽셀 좌표, "score": float}`.
6.2절 `targets["instances"]`와 **같은 형식**이라 GT와 예측을 바로 비교할 수 있다.

**왜 사슬 확장인가.** 전역 그래프를 만들어 절단하는 방식(정점 → 간선 후보 → 양방향 확인 →
그래프 절단)은 쓰지 않는다. 그 방식은 GT 주입에서도 간선 재현율이 97.7%에 그쳤고, 성분당
간선이 평균 47개라 **2.3%의 간선 손실이 연결 성분 수 1.8배로 증폭**됐다(SEED-MAP val 12장
실측 — 소실 원인: 슬롯이 GT 아닌 셀을 지목 278, 상호 최선 탈락 211). 긴 사슬에서는 간선
하나만 끊겨도 인스턴스가 쪼개진다. 지금 디코더는 인코딩(6.4절)과 같은 모양으로 **사슬을 한
노드씩 확장**하며, 전역 그래프·상호 최선 확인·경로 절단 단계가 없다. 수용 기준은 **GT를 출력
형식으로 주입하면 인스턴스 F1 ≈ 1**이고, 당시 설정에서 실측 **0.976**으로 통과했다.
**천장은 설정마다 다시 재야 하는 값이다** — 그 뒤 평가 버퍼를 12 px에서 3 px로 조이고 탐색
반경을 24에서 5로 줄이면서 현행 천장은 **0.941**이다(둘 다 일부러 치른 대가이고, 같은 판에서
모델 성능은 함께 올랐다).

### 10.1. 3단계 개요 — `decode/vertices.py` + `decode/graph.py` + `decode/postprocess.py`

```
ModelOutput
 → ① 정점 추출        노드 셀 → (클래스·클래스 확률, 점, 점수, 끝 확률, 슬롯) 목록      [vertices.py]
                      + 선의 법선 방향 비최대 억제 (넓은 전경 띠 → 한 줄 능선)
 → ② 사슬 확장        클래스 확률 국소 피크에서 양방향으로, 마주봄 확인으로 한 노드씩 연장  [graph.py]
                      + 완성된 사슬의 클래스 순도 검사 (탈락 시 정점 반환)
 → ③ 후처리           조각 병합·최소 길이·RDP 단순화·점수                          [postprocess.py]
                      + 겹쳐 그려진 중복 선 정리                                 [dedup.py]
```

계획 당시엔 이 3단계가 `graph.py` 한 파일이었다. 지금은 ①이 `vertices.py`(정점 추출 + 시드 순서),
③의 조각 병합·단순화가 `postprocess.py`, 중복 정리가 `dedup.py`로 분리됐고, `graph.py`에는
`ChainDecoder` 본체(②의 확장 로직 + ①③ 호출 오케스트레이션)가 남는다 — 세 단계가 각자 단위
테스트(`test_decode.py`·`test_postprocess.py`) 대상이라 갈랐다(3절). 디코더 정지 사유 카운터(`stats.py`)·예측 캐시(`cache.py`)·
평가 코어(`sweep.py`)는 10.6절에 별도로 다룬다.

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

**법선 방향 비최대 억제 (`vertex_local_max`, 기본 켜짐).** 모델이 정답보다 **넓은 띠**를 전경으로
부르고(선택 셀이 정답 셀의 4배) 그 띠에서 정점이 두 줄 잡힌다 — 예측 선의 18.5%가 같은 클래스의
다른 선과 1.8 px 간격으로 나란히 그려지는 원인이다. 그래서 정점 점수($\sigma(\hat h)\cdot\max_c\pi_c$)를
**예측된 연결 방향에 수직인 쪽으로만** 비교해 진 것을 지운다. 진행 방향으로 억제하면 선이 끊기므로
법선 방향만 본다. **이 억제는 GT 주입 천장을 소수점 넷째 자리까지 그대로 둔다** — 정답에는 넓은
띠가 없어(선 하나당 칸 하나가 주인) 억제할 것이 없기 때문이다.

**전경 판정은 세 규칙 중 하나다.** 기본은 위 식의 $\arg\max \neq 0$이고, `bg_prob_max > 0`이면
배경 확률 문턱으로, `fg_thresh >= 0`이면 `fg_logit`(7.1절)의 이진 확률로 바뀐다.
**여기가 짧은 선이 새는 관문이다** — 정답 칸이 히트맵 문턱은 88.9% 통과하는데 이 관문을 지나면
29.1%만 남고, 70칸 이상 선(96.7% → 50.0%)보다 짧은 선이 유독 더 깎인다. 그런데 **문턱을 풀어도
재현율이 1.5%밖에 오르지 않는다** — 모델이 그 칸들을 자신 있게 배경이라 부르고 있어서, 처방은
디코더가 아니라 학습 쪽이다. 두 대안은 기본에서 꺼져 있고 민감도 ablation용으로 남긴다.

### 10.3. ② 사슬 확장 — 한 노드씩, 단방향으로

핵심 확인은 **"서로가 서로의 점을 향하는가"**다. 현재 정점 $a$의 확장 슬롯 방향을 $\mathbf{c}$,
후보 정점 $b$의 슬롯 방향을 $\mathbf{n}$이라 하면, 둘이 마주보면 $\mathbf{c} \cdot \mathbf{n} \to -1$이다.
**"상호 최선(mutual best)" 검사는 하지 않는다** — 한 방향씩 확장하며 그때그때 최선 후보를
붙인다. mutual best는 슬롯이 한 칸 건너 셀을 지목하는 사소한 오차만으로 간선을 통째로 버려
긴 사슬을 끊었다(GT 주입에서 소실 간선 489개 중 211개가 이 검사 탈락).

**시드 선정 — 클래스 확률 국소 피크에서 양방향으로 (기본 `seed_mode = "class_peak"`).**
자기 클래스 확률 $\max_c \pi_c$가 **정점 이웃($3\times3$) 중 최대인 정점**(국소 피크)을 확률
내림차순으로 시드로 삼는다. 모델이 가장 확신하는 지점에서 출발해야 사슬 클래스가 안정적으로
정해진다 — 끝 셀에서 출발하지 않는 이유는 끝 셀의 감독이 상대적으로 어렵고(양성의 4.12%,
6.7.5절) 예측이 흔들리면 시작점 자체가 틀어지기 때문이다. **사슬 클래스 $y^\ast$ = 시드의 argmax 클래스**로
고정하고, 시드의 활성 슬롯들을 따라 **양방향으로** 확장한 뒤 두 반쪽을 이어 붙인다. 국소 피크가
소진되면 남은 미사용 정점을 확률 내림차순으로 시드에 쓴다(안전망). `seed_mode = "end_peak"`(개선
루프 가설 백로그, 10.6절)는 끝 확률이 높은 정점을 먼저 시드로 쓰는 구 방식으로 되돌리는 실험용
스위치다 — 기본값은 여전히 위 근거대로 `class_peak`다.

**확장 한 스텝.** 정점 $a$, 미사용 슬롯 $k$ ($\sigma(\hat{e}_{a,k}) > \tau_e$)에서:

1. **후보 집합.** $a$의 셀에서 체비셰프 반경 `radius`(= 5) 안의 미사용 정점 $b$ 중
   **사슬 클래스 확률이 하한을 넘는 것**: $\pi_{b,y^\ast} \ge$ `min_class_prob`(기본 0.2).
   argmax 일치를 요구하지 않는다 — 중간에 잠깐 다른 클래스가 이길 수 있기 때문이다. 다만
   사슬 클래스 확률이 0에 가까운 정점을 붙이는 것은 위험하므로 하한으로 거른다(순도 검사가
   뒤에서 한 번 더 잡는다). 반경이 1칸보다 커야 하는 이유는 교차점에서 소유권으로 잃은 칸을
   건너뛰기 위함이고(6.4절 — GT 간선의 98.7%가 1칸, 1.3%가 2칸), 실제 모델은 GT 셀의 절반쯤만
   정점으로 만들어 빈칸이 더 자주 생긴다. **얼마나 커도 되는지는 아래 "반경은 차선 간격이
   정한다"가 정한다.**
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
+ w_{dist}\,\lVert \mathbf{p}_b - \mathbf{p}_a \rVert
$$

   여기서 $w_{dist}$는 셀 단위 거리에 곱하는 계수다(`decode.w_dist`, 기본 **0.072**).

   **이 값은 탐색 반경과 함께 정해진다** — 곱 $w_{dist} \times$ `radius`가 정렬 항
   $(1-\text{align})$의 범위 $[0, 0.3]$과 겨룰 만큼은 되어야 한다. 곱이 작으면 **디코더가
   정렬만 조금 나으면 여러 셀을 건너뛰며 정점을 지나쳐** 한 선이 여러 조각으로 쪼개진다.
   게이트가 이 곱의 하한 0.3을 강제한다. 현재 작동점은 $0.072 \times 5 = 0.36$이다.

   **반경은 차선 간격이 정한다.** 이웃 선까지의 중앙 거리가 11.8 px인데 반경이 그보다 크면
   옆 차선의 정점이 후보로 들어온다 — 반경 12셀(48 px)에서는 옆 차선이 각도로 14°라
   정렬 게이트(18°)를 그냥 통과했고, 예측 점의 6.0%가 옆 선 위에 얹혔다. 반경 5셀(20 px)에서는
   2.5%다. **반경이 바꾸는 것은 꺾임각이 아니라 도약 길이다** — 디코딩된 스텝 길이의 99%
   분위수가 반경 12에서 36.2 px(차선 세 칸), 5에서 16.5 px, 3에서 12.5 px다.

   방향이 같은 원거리 후보와의 동률은 **마주봄 항**도 함께 가른다(건너뛴 셀은 되가리키는
   슬롯이 없다).

   `align_mode`(개선 루프 가설 백로그, 기본 `"cosine"` = 위 3번의 각도 게이트)를 `"perp"`로 바꾸면
   비용·게이트가 달라진다 — 예측 방향 $\hat{\mathbf{d}}_{a,k}$가 그리는 직선에서 $b$가 벗어난
   **수직 거리**(셀 단위, $\text{align} = \hat{\mathbf{d}}_{a,k}\cdot\mathbf{u}_{ab}$일 때
   $\lVert \mathbf{p}_b - \mathbf{p}_a\rVert\sqrt{1-\text{align}^2}$)가 `perp_thresh` 이하인
   후보만 남긴다. 각도 게이트는 먼 후보에 관대하고 가까운 후보에 엄격한 반면, perp 게이트는 그
   반대라는 가설을 시험한다.
5. **이동.** $b$를 사슬에 붙이고, $b$의 되가리킴 슬롯 $\mathbf{n}_b$를 사용 처리한 뒤
   $b$의 **반대쪽 활성 슬롯**으로 계속 확장한다.

**정지 조건.** ① $b$의 끝 확률 $> \tau_{end}$ (사슬 끝 도달) ② 게이트(방향 + **사슬 클래스 확률
하한**)를 통과한 후보 없음 ③ 정점 수 상한(이상 동작 안전망). **고리 폐쇄는 독립 조건이 아니다** —
사슬의 시작 정점은 이미 "사용됨"으로 표시돼 있어 후보 집합에서 자동으로 빠지고, 결과적으로
②(후보 없음)로 흡수돼 정지한다. `stop_needs_nocand`(기본 False, 개선 루프 가설 백로그)를 True로
켜면 ①만으로는 정지하지 않고 **끝 확률 조건과 후보 없음을 모두** 만족해야 멈춘다 — 끝 확률이
높아도 근처에 이어 붙일 후보가 남아 있으면 계속 확장해 보는 실험용 스위치다.

**순도 검사.** 양방향 확장이 끝나 사슬이 완성되면, 구성 정점 중 **argmax 클래스가
사슬 클래스 $y^\ast$와 일치하는 비율**을 잰다. `purity_thresh`(기본 0.6) **이하면 사슬을 버린다** —
정점과 슬롯을 미사용으로 되돌리고(다른 시드가 다시 쓸 수 있게), 그 시드는 실패로 표시해 재시도
하지 않으며, 다음 시드로 넘어간다. `min_class_prob` 게이트가 느슨한 만큼(0.2 — 일시적 확률
하락 허용) 여기서 "사슬이 통째로 다른 클래스를 따라간" 경우를 걸러낸다.

**끝 연장 (사용자 결정 — "선이 짧아지지 않는다").** 사슬이 끝 셀에서 멈추면, 그 셀의 남은 활성
슬롯(끝방향 — GT가 실제 끝점을 향하도록 감독했다) 방향으로 **연장점을 하나 추가**한다:
$\mathbf{p}_{\text{ext}} = \mathbf{p} + \hat{\mathbf{d}} \cdot \texttt{end\_extend}$ (기본 1셀 — 끝점은
미채움 이웃 칸 안에 있으므로 평균 거리가 약 1셀이다. 정확한 값은 스윕). 끝칸 미채움으로 잘린
길이가 여기서 복원된다. **1셀 사슬**(3칸짜리 선 — 분기 2개가 모두 끝방향)은 양쪽으로 연장해
3점 폴리라인이 된다. `min_points`는 연장점을 포함한 점 수에 적용한다.

간선 비용에 클래스 불일치 벌점을 넣거나 종점 슬롯을 예외 처리하지 않는다 — 클래스는 **사슬 클래스 확률 하한 + 순도 검사**가 부드럽게 관리한다. T자 접합에서 본선은
곁가지 끝 셀을 후보로 만나도 방향 게이트가 거르고, 곁가지는 자기 끝 셀에서 정지한다 —
**본선이 접합점에서 잘리는 문제가 구조적으로 없다.**

### 10.4. ③ 후처리 — `decode/postprocess.py`

0. **조각 병합 (`ChainMerger`).** 사슬 확장이 다 끝난 폴리라인 목록에서, `merge_gap`(기본 24 px)이
   양수면 끝점끼리 이 거리 안에 있고 접선이 서로를 향하는 정도(`merge_align`, 기본 0.8)를 넘는
   조각 쌍을 하나로 잇는다. 후보 탐색은 `cKDTree`로 한다. **이 단계가 살아난 것은 탐색 반경을
   줄인 뒤부터다** — 반경이 크던 시절에는 조각들이 서로 **겹쳐** 있어 "끝점이 가깝고 마주봄"
   조건에 거의 걸리지 않았다(영상당 0.33건). 반경 5에서는 조각이 끝과 끝으로 맞닿아 실제로
   병합이 일어난다.
1. **폴리라인의 클래스 = 사슬 클래스 $y^\ast$(시드의 클래스).** 순도 검사(10.3)가 구성 정점의
   60% 초과 일치를 보증하므로 다수결과 항상 일치한다 — 별도 다수결 단계를 두지 않는다.
2. 정점 수 < `min_points`면 버린다(연장점 포함). `simplify_tol > 0`이면 RDP로 단순화한다(기본 0 = 안 함).
3. **점수.** 폴리라인의 점수 = 구성 정점 점수의 평균. 평가에서 confidence 임계값 스윕에 쓴다.

### 10.4.1. ③ 후처리 — 중복 정리 `decode/dedup.py`

**같은 차선 위에 두 번 그려진 선을 정리한다.** 예측 선의 18.5%가 같은 클래스의 다른 선과
**1.8 px 간격으로 나란히** 놓인다 — 겹친 쌍의 91%가 한 셀(4 px) 이내이므로 이웃 차선
(간격 11.8 px)이 아니라 **같은 차선 위의 이중 그리기**다. 원인은 10.2절과 같다(모델이 정답보다
넓은 띠를 전경으로 부른다). 디코더는 정점을 한 번씩만 쓰지만 **"이미 그린 선"이라는 개념이 없어**
나란한 중복을 막지 못한다.

**계약이 이 단계의 전부다.** 지우거나, 이미 겹쳐 있던 두 선을 끝점에서 잇기만 한다 —
**떨어진 두 선을 잇지 않고**(간격을 메우지 않고), 없던 인스턴스를 만들지 않으며, 클래스를 바꾸지
않는다. 그래서 인스턴스 수는 줄기만 하고 늘지 않는다. 이는 주장이 아니라 알고리즘의 성질이다.

**문턱을 정하는 규칙은 평가 버퍼와 같다** — 진짜 이웃 선의 하위 분위수보다 좁게 잡는다.
같은 클래스 이웃 선이 3 px 이내에 있는 경우가 0.8%뿐이므로 `dedup_high = 3.0`이고, 제거 대상
중복의 간격 중앙값 1.8 px이 그 안에 들어온다. **6 px로 잡았더니 진짜 선을 지워 GT 주입 천장이
0.946 → 0.908로 깎였다.** `dedup_keep_ratio`(기본 0.35)는 보수 장치다 — 겹치지 않는 구간이
원래 길이의 이 비율 이상이면 중복이 아니라고 보고 원본을 그대로 둔다.

**정점 단계 억제(10.2절)와 선 단계 정리는 서로를 대체하지 않고 더해진다** (val 400장):
둘 다 끄면 0.3603 · 억제만 0.3734 · 중복 정리만 0.3798 · **둘 다 0.3860**.

**이 규약은 모델이 아니라 디코더에만 있다.** 확장 방식을 바꾸고 싶으면 재학습 없이 여기만 고치면 된다.
`radius`·$\tau_{align}$·$\tau_{opp}$·$\tau_{end}$·$\tau_e$·`min_class_prob`·`purity_thresh`와
개선 루프가 덧붙인 `seed_mode`·`stop_needs_nocand`·`merge_gap`·`merge_align`·`align_mode`·
`perp_thresh`·`vertex_local_max`·`dedup_*`(4.1절)는 전부 학습된 체크포인트로 검증 셋에서
스윕해 확정한다(`scripts/tune_decoder.py`,
14절) — GT 주입만으로는 오탐 필터 강도를 정할 수 없다. GT 주입에서는 확률이 0/1이라
`min_class_prob`·`purity_thresh`가 자명하게 통과되므로, M12 수용 기준(F1 ≈ 1)에는 영향이 없다.

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
- 위 스니펫은 골자만 보여준다. 실제 `validation_step`은 같은 배치에서 `self.cell_diag.update(out, batch)`도
  부르고(11.5절), `on_validation_epoch_start`/`on_validation_epoch_end`에서 `self.decoder.stats`를
  리셋·로깅한다(`val/dec/*`, 10.6절, 9.4절).

### 10.6. 셀 단위 진단과 사슬 신뢰도 — `decode/stats.py`, `stella/eval/cellstat.py`

디코더 자체와 별개로, **디코더가 왜 실패하는지**를 두 층위에서 진단한다.

**`ChainStats`(`decode/stats.py`)는 디코더 실행 그 자체의 카운터다.** 사슬이 멈춘 사유
(`end`·`nocand`·`exist`·`slotused`)별 횟수, 순도 검사 탈락 수, 병합 횟수, 정점 사용률을 세고,
`summary()`가 비율 dict를 낸다 — 검증 에폭마다 `val/dec/*`로 로깅된다(9.4절). 디코더 파라미터를
바꿨을 때 "정지 사유 분포가 어떻게 바뀌는가"를 바로 보여준다(예: `stop_needs_nocand`를 켜면
`end` 정지 비율이 줄고 `nocand` 정지가 는다).

**`CellDiagnostics`(`stella/eval/cellstat.py`)는 디코딩을 거치지 않는, 한 단계 더 이른 진단이다** —
격자(셀) 단계의 원시 예측(히트맵 확률·`class_logit`·`self_coord`·`end_logit`·`conn_dir`·`exist_logit`)을
GT 텐서와 셀 단위로 직접 대조해 "어느 헤드가 문제인지"를 손실(8절)보다 해석 가능한 형태로 분해한다.
전체 22종 지표는 11.5절에 있다.

**사슬 신뢰도 계산 — `link_ok`가 왜 개선의 기준 지표인가.** 6.7.5절(M10 재집계)에서 사슬 평균
길이가 **48.2 셀**로 나왔다 — 사슬 하나가 처음부터 끝까지 온전히 이어지려면 링크(한 노드에서
다음 노드로의 연결 예측)가 **47번 연속 성공**해야 한다는 뜻이다. `link_ok`(11.5절 — 방향 오차가
디코더 정렬 게이트 `align_thresh` 이내인 비율)를 링크 하나의 성공 확률로 보면, 선 하나가
쪼개지지 않고 살아남을 확률은 $(\text{link\_ok})^{47}$이고, 선당 기대 조각 수는 근사적으로

$$
\mathbb{E}[\text{조각 수}] \approx 1 + 47\,(1 - \text{link\_ok})
$$

이다. `link_ok = 0.99`면 기대 조각 수 $\approx 1.47$, `link_ok = 0.95`면 $\approx 3.35$,
`link_ok = 0.90`이면 $\approx 5.7$로 급격히 나빠진다 — 사슬이 길수록 링크 하나의 실패가 비싸다.
**그래서 `link_ok`는 0.99가 목표다.** `chain_expect`(11.5절, $= 1/(1-\text{link\_ok})$)는 같은
계산을 링크 실패까지의 기대 사슬 길이로 뒤집어 보여준다. **이 계산이 이후 모든 연결성 개선의
기준이다** — `link_ok`를 올리는 변경(손실 가중치·모델 구조·인코딩)은 위 공식으로 곧장 "선당
조각 수가 몇 개 줄어드는가"로 환산된다.

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

**매칭.** 같은 클래스의 $(G, P)$ 쌍 중 두 조건을 통과한 것만 후보로 두고, $C_1 + C_2$ 내림차순으로
정렬해 **그리디로 선점**한다(`InstanceCCQ._greedy_match`) — 계획 당시엔 "일대일 최대가중 매칭"
(Hungarian/LSA급)을 상정했지만 실제 구현은 정확한 최대가중이 아니라 그 근사인 그리디다. 각
GT·예측이 최대 한 번만 매칭되는 일대일 성질은 그대로 유지된다. 매칭된 쌍 = TP, 남은 GT = FN,
남은 예측 = FP.

$\rho$가 인접 차선 간격의 절반보다 작으면 예측 위 한 점은 많아야 하나의 GT 버퍼에만 들 수 있어 $C_2$의
"모든 GT"가 중복 없이 잘 정의된다. **설계상 $\rho$는 차선 간격의 절반 이하여야 한다** — 한동안
기본값이 이 전제를 스스로 어기고 있었다. 아래 "버퍼 폭 $\rho$ — 재설정"을 반드시 읽어라.

**FP를 두 종류로 나눠 보고한다.** HD맵 보정 관점에서 비용이 다르기 때문이다.

- **redundant FP**: 매칭 안 됐지만 $C_2 \ge 0.9$. 실제 GT 위의 잉여 조각이다. 병합만 하면 된다.
- **spurious FP**: 매칭 안 됐고 $C_2 < 0.9$. GT 밖에 그린 선이다. 삭제하고 확인해야 한다.

둘 다 precision에는 FP로 센다 — 조각남을 벌하는 것이 이 지표의 의도다. 분해는 진단용이며 $C_2$에서 공짜로 갈린다.

$\text{precision} = TP/(TP+FP)$, $\text{recall} = TP/(TP+FN)$, $F1$은 조화평균이다. **매칭과 집계는
클래스별로 따로 한다** — 차선 11종 각각에 대해 precision·recall·F1을 내고, 그 위에 전체를 합친
micro와 클래스 단순평균 macro를 함께 보고한다. redundant/spurious FP 분해도 클래스별로 유지한다.

**버퍼 폭 $\rho$ — 재설정.** 원래 계획은 "차선 간격의 절반 이하"였는데 기본값
`buffer_rho = 12.0`이 그 규칙을 스스로 어기고 있었다. 전체 데이터 실측 결과 **차선 간 거리
중앙값이 11.8 px**라 $\rho$가 간격의 절반이 아니라 **간격 전체와 같았다.** 그 결과:

- 예측이 **중간에 이웃 차선으로 갈아타도** 매 순간 어떤 GT 버퍼 안에는 있어서 만점을 받았다.
  실측: 매칭 자격을 얻던 예측의 **17%가 갈아탄 것**이었다 (GT 주입에서는 1%).
- 같은 저장 예측을 고친 지표로 다시 채점하니 **F1 0.5385 → 0.2932**였다.

**그래서 두 가지를 고쳤다.**

1. $\rho$를 **3.0 px**로 줄였다. $\rho$는 **반경**이므로 실제 허용 폭은 6 px — 간격 11.8 px의
   절반이고, 이제 11.1절의 설계 전제를 지킨다.
2. $C_2$(정확성)의 정의를 바꿨다. **모든 GT의 합집합**까지의 거리가 아니라 **가장 잘 맞는 GT
   하나**에 대해 잰다 — "이 예측이 한 선에 머무는가"를 직접 검사한다.

**그래서 `correctness`는 이제 판정에 쓴다.** 갈아탐을 보는 유일한 지표다. 다만 **사슬이 짧으면
갈아탈 기회가 없어 높게 나오므로** `chain_len`·`coverage`와 함께 읽는다.

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

$C_1(G, P) > 0$ 조건은 "조금이라도 겹치면 조각으로 센다"는 뜻이라 위 11.1절 한계(이웃 버퍼 오염)의
영향을 그대로 받는다. **`frag_strict`**는 같은 식에서 $C_1(G, P) > 0$ 대신 $C_1(G, P) \ge$
`frag_min_cov`(기본 0.1)를 요구한다 — 스치기만 한 예측을 조각 수에서 빼는, 더 보수적이고
정직한 조각남 측정치다(`val/inst/frag_strict`, 9.4절).

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
  방향 게이트, 길이 샘플 간격, `max_instances`(샘플당 평가 인스턴스 상한, 안전장치), `frag_min_cov`
  (11.1·11.2절). 계산은 폴리라인을 일정 간격으로 샘플해 점-폴리라인 거리로 하며, 그래서
  성긴 점의 대각선 문제가 없다.
- 출력 dict가 그대로 `val/inst/*`로 로깅된다(9.4절).
- **파선 GT 규약 — 종결.** "GT 쪽 일대다 매칭 허용"이나 "평가 전 공선 GT 병합" 보정은
  **도입하지 않는다.** 실측에서 GT 주입 F1 상한(0.63~0.69)의 주범은 GT 라벨의 조각남이
  아니라(옛 인코더의 그래프 성분 451 < GT 인스턴스 504 — 오히려 병합하고 있었다) **디코더의 간선
  소실 증폭**이었다. 선 단위 인코딩(6.4절)은 라벨 인스턴스와 사슬이 1:1이라 지표와 정확히
  정합한다 — dash마다 별도 객체면 사슬도 별도라 매칭이 그대로 성립한다. 새 수용 기준:
  **GT 주입 인스턴스 F1 ≈ 1** (M12) — 당시 설정에서 **실측 0.976으로 통과했다.**
  현행 설정(버퍼 3 px·반경 5)의 천장은 **0.941**이고, 게이트가 이 값의 하한(0.93)을 지킨다.

### 11.5. 셀 단위 진단 지표 — `stella/eval/cellstat.py`의 `CellDiagnostics` (개선 루프 전용, 문서 신설)

`InstanceCCQ`(11.1~11.2절)는 디코딩까지 거친 **최종 결과**를 재므로 "성능이 나쁘다"는 말해줘도
"어느 헤드가 원인인지"는 말해주지 않는다. `CellDiagnostics`는 디코딩을 **거치지 않고** 격자(셀)
단계의 원시 예측을 GT 텐서(`class_map`·`coord_map`·`end_map`·`conn_dirs`)와 셀 단위로 직접
대조해 이 틈을 메운다. 손실(8절)과 달리 미분 불가능한 정답률·재현율 통계이고, `CellDiagConfig`
(4.1절, `build_instance(cfg.cell_diag, cfg)`)로 만들어 `StellaTrainModule`에 꽂는다(9.1절) —
인터페이스는 `update(output, targets)` / `compute() -> dict` / `reset()`으로 `InstanceCCQ`와
같고, 결과는 매 검증 에폭 `val/cell/*`로 로깅된다(9.4절).

**22종 지표.** 헤드별로 묶어서 본다.

| 헤드 | 지표 | 뜻 |
| --- | --- | --- |
| 히트맵 | `heat_recall` | GT 양성 셀 중 노드 선택 파이프라인(임계+dilation+n_max 전체)에 걸린 비율 |
| 〃 | `heat_precision` | 선택된 노드 중 실제 GT 양성인 비율 |
| 〃 | `heat_pos` / `heat_neg` | GT 양성/음성 셀에서의 평균 히트맵 확률(임계 무관, 순위 품질) |
| 〃 | `node_per_img` | 이미지당 평균 선택 노드 수 |
| 클래스 | `class_acc` | 선택∩GT양성 셀 중 클래스 정답률(분모 = 선택된 GT 셀) |
| 〃 | `class_fg` | 같은 분모에서 "배경 아님"만 맞춘 비율(클래스 종류는 틀려도 됨) |
| 〃 | `class_recall` | 분모를 **전체 GT 셀**로 바꾼 클래스 정답률(실행 간 비교용 — 선택 수 차이에 안전) |
| 〃 | `vertex_recall` | 같은 분모에서 "배경 아님" 비율 — 디코더가 실제로 받는 정점 재현율 |
| 〃 | `class_bg_recall` | 거짓 양성 셀(선택됐지만 GT 음성)을 배경으로 맞게 예측한 비율 |
| 좌표 | `coord_err_px` | self 좌표 예측 오차의 L2 노름 평균(픽셀 단위) |
| 끝 | `end_recall` / `end_precision` | 끝 셀 판정의 재현율·정밀도 |
| 〃 | `end_pos` / `end_neg` | GT 끝/비끝 셀에서의 평균 end 확률(임계 무관) |
| 연결 존재 | `exist_pos` / `exist_neg` | GT 양성/거짓 양성 셀 연결 슬롯의 평균 존재 확률 |
| 연결 방향 | `dir_err_deg` / `dir_err_p90` | 매칭된 슬롯의 방향 오차 평균·90퍼센타일(도) |
| 사슬 신뢰도 | `link_ok` | 방향 오차가 디코더 정렬 게이트(`decode.align_thresh`, 각도 환산) 이내인 비율 — 한 링크가 성공할 확률 |
| 〃 | `link_ok_20deg` | 방향 오차 20도(느슨한 고정 기준) 이내인 비율 |
| 〃 | `chain_expect` | $= 1/(1-\text{link\_ok})$ — 링크 실패까지의 기대 사슬 길이(10.6절 계산의 근거) |

`link_ok`·`chain_expect`는 `stella/loss/matching.py`의 배정 로직을 재사용해 "디코더를 실제로
돌리지 않고" 사슬 성공 가능성을 대수적으로 근사한다 — 10.6절의 "선당 기대 조각 수" 계산이 여기서 나온다.

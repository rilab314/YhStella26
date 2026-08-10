# 모델과 손실 (7~8절)

백본·neck·히트맵·어텐션 스택·출력 헤드, 그리고 그 출력을 감독하는 손실 세 모듈.
전체 색인과 문서 작성 원칙은 [design.md](design.md)에 있다.

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
| `end_logit`     | `float32 (192, 192)`       | logit    | self 슬롯의 "**이 셀이 사슬의 끝**"일 확률 — `end_map` 직접 감독                |
| `exist_logit`   | `float32 (192, 192, R)`    | logit    | 슬롯별 연결 존재 확률                                                  |
| `conn_dir`      | `float32 (192, 192, R, 2)` | 단위벡터     | 슬롯별 연결 방향. `F.normalize`. 원점 = **자기 노드 점**(6.1절), 목표 = 상대 노드 점 |

슬롯별 `t_logit`(연결 대상이 종점/다른 클래스 접합)은 두지 않는다 — 끝 판정은 셀 단위
`end_logit`이 맡고, 다른 클래스 접합 간선은 이 인코딩에 없다(6.4절). *(사용자 확정 — 결정 32.)*

**GT와 모델 출력은 같은 형태다 (설계 방침 1).** heatmap↔`class_map > 0`,
`class_logit`↔`class_map`, `self_coord`↔`coord_map`, `end_logit`↔`end_map`,
`conn_dir`↔`conn_dirs`가 짝을 이룬다. $R = D = 2$(결정 1)라 개수까지 같고, 남는 차이는
**슬롯 순서**(무순서)뿐이다. 그래서 criterion이 하는 일은 유도가 아니라 **매칭**(8.3)과
손실 계산뿐이다.

**`stella/model/inject.py` — GT를 이 계약 형식으로 직접 채운다.** `gt_model_output(targets, ...)`가
`class_map`·`coord_map`·`end_map`·`conn_dirs`를 로짓이 포화된(`HIGH_LOGIT`) `ModelOutput`으로
바꿔 "완벽한 예측"을 만든다. 실제 학습·추론 경로에서는 쓰이지 않고, **파이프라인의 다른 부분을
독립적으로 검증**하는 테스트·측정 전용 유틸리티다 — 디코더가 GT를 그대로 복원하는지(M12,
10절), 손실이 매칭된 완벽한 입력에서 0으로 수렴하는지(M11, 8절), 그리고 개선 루프에서 "모델이
완벽해지면 지표가 얼마까지 오를 수 있는가"(천장, GT 주입 인스턴스 F1 0.976)를 잰다.

### 7.2. Backbone — `model/backbone.py`

여러 백본을 비교 실험할 것이므로 **3층 구조**로 둔다. 모델 계열마다 출력 형태가 다르므로 **계열별로 클래스 하나**를
만들고, 라이브러리 공통 작업(가중치 로드·전처리 상수·특징 추출 호출 규약)은 **중간 인터페이스 클래스**로 모은다.

```
Backbone(nn.Module, Buildable)          # 계약: forward(x) -> list[Tensor], out_channels, strides
├── HuggingFaceBackbone                 # transformers 공통: AutoModel/AutoImageProcessor 로드,
│   │                                   #   processor에서 mean/std 추출, 게이트 토큰 처리
│   └── Dinov3Backbone                  #   ViT 패치 토큰 → 1레벨 맵 (게이트 승인 대기, 14절)
└── TimmBackbone                        # timm 공통: create_model(pretrained=True),
    │                                   #   default_cfg에서 mean/std 추출
    ├── ConvNeXtBackbone                #   4레벨 (stride 4/8/16/32) — 현재 기본값
    ├── SwinBackbone                    #   4레벨, img_size 고정 입력
    ├── HrnetBackbone                   #   5레벨 → out_indices로 4/8/16/32만 선택
    └── TimmVitBackbone                 #   features_only=False, ViT 패치 토큰 → 1레벨 (SFP 경로 검증용)
```

`PerceptionEncoderBackbone`은 계획에는 있었지만 아직 만들지 않았다 — 필요해지면 계열 클래스 하나
추가로 확장한다(구조는 이미 이를 위해 설계돼 있다).

**계열 안의 스케일 변화(large/base/small/tiny)는 한 클래스가 처리한다.** 클래스를 고르는 것은 `name`이고,
스케일은 `pretrained` 문자열이 정한다. 채널 수·레이어 수는 로드한 모델에서 읽어 `out_channels`에 채운다.

```python
# configs/exp_dinov3.py — 게이트 승인 전에는 check_all은 통과하지만 생성 시 GatedRepoError가 난다
cfg.model.backbone.name = "Dinov3Backbone"
cfg.model.backbone.pretrained = "facebook/dinov3-vitl16-pretrain-sat493m"
cfg.model.neck.name = "SFP"  # ViT는 1레벨만 내므로 SFP와 짝짓는다
```

| 층                     | 책임                                                                                            |
| --------------------- | --------------------------------------------------------------------------------------------- |
| `Backbone`            | 계약만 정의. `out_channels: tuple[int,...]`, `strides: tuple[int,...]`, `pixel_mean/std` 버퍼        |
| `HuggingFaceBackbone` | `AutoModel.from_pretrained` 로드, `AutoImageProcessor`에서 정규화 상수 추출, `freeze` 처리                 |
| `TimmBackbone`        | `timm.create_model(pretrained=True, ...)` 로드, `default_cfg`에서 정규화 상수 추출. 계층형은 `features_only=True`, ViT류(`TimmVitBackbone`)는 `False`로 불러 패치 토큰을 직접 다룬다 |
| 계열 클래스                | **출력 형태를 `list[Tensor]`(stride 오름차순)로 맞추는 일.** ViT류는 패치 토큰 → `(B, C, h, w)` reshape, 계층형은 그대로 |

정규화 상수를 백본이 들고 있으므로 데이터셋은 `[0,1]` RGB만 내놓으면 되고(6.2절), 백본을 바꿔도
`StellaModel` 바깥은 불변이다.

| 우선순위       | 클래스 / 모델                                                                                 | 출력                          | 비고                                                 |
| ---------- | ---------------------------------------------------------------------------------------- | --------------------------- | -------------------------------------------------- |
| 1 (목표, 대기) | `Dinov3Backbone` — **DINOv3 ViT-L/16 위성 사전학습** `facebook/dinov3-vitl16-pretrain-sat493m` | 패치 토큰 → `(B, 1024, 48, 48)` | 위성 영상 4.9억 장 사전학습 dense 특화. **HF 게이트 미승인 — 아직 못 씀**(14절). `configs/exp_dinov3.py`로 전환 준비는 끝났다 |
| **2 (현재 기본)** | `ConvNeXtBackbone` — `convnextv2_base.fcmae_ft_in22k_in1k_384`                        | 4레벨 (stride 4/8/16/32)      | `configs/base.py` 기본값. 게이트 없음. DINOv3 승인 전 대역 백본 |
| 3          | `SwinBackbone` — SwinV2-L 등                                                              | 4레벨, `img_size` 고정 입력 필요    | 게이트 없음. FPNLite 경로 대조군                              |
| 4          | `TimmVitBackbone` — `vit_base_patch16_224.augreg_in21k` 등                                | 패치 토큰 → 1레벨                  | 게이트 없는 ViT. `configs/exp_vit_sfp.py` — SFP 경로 검증용  |
| 5          | `HrnetBackbone`                                                                          | 5레벨 → `out_indices`로 4개 선택   | 고해상도 유지형 대조군                                        |
| 6          | `PerceptionEncoderBackbone` 등                                                            | 계열마다 다름                     | **아직 미구현.** 필요할 때 계열 클래스 하나 추가로 확장                             |

- 백본 추가 = **계열 클래스 하나 추가**. 라이브러리가 이미 있으면 중간 클래스를 상속해 `forward` 출력 정리만 하면 된다.
- InternImage는 **지원하지 않는다**(13절 결정 4 — DCNv3 커스텀 CUDA 커널이 원칙 #6 위반).
- **`timm`·`transformers` import는 중간 인터페이스 클래스 안에서 한다.** 모듈 최상단에 두면 한쪽 라이브러리만
  깔린 환경에서 `check_all`(5.1절)이 통째로 죽는다.
- 1레벨만 내는 백본(ViT)은 `SFP`와, 4레벨을 내는 백본은 `FPNLite`와 짝을 이룬다. `Neck.from_cfg`가
  `backbone.out_channels` 길이를 보고 맞지 않으면 즉시 에러를 낸다.

### 7.3. Neck — `model/neck.py`

`Neck` 베이스클래스의 하위로 구현하고(config `model.neck`으로 고른다, 5절), 백본이 무엇이든
**공통으로 `(B, 256, 192, 192)`** = `(B, d_model, L, L)`를 낸다. 이 격자가 이후 전부(히트맵·노드 선택·토큰 임베딩)의 좌표계다.
**기본 config는 `FPNLite`다** — 현재 기본 백본 `ConvNeXtBackbone`(7.2절)이 4레벨을 내기 때문이다.
`SFP`는 ViT류(`Dinov3Backbone`·`TimmVitBackbone`)와 짝지을 때 `neck.name = "SFP"`로 켠다.

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
                              [3×3 Conv(256→256) + GN] × out_blocks
                                                    ↓
                                          (B, 256, 192, 192)
```

- lateral은 전부 `1×1 Conv + GroupNorm(32)`. 상위 레벨은 `F.interpolate(mode="nearest")`로 2배 올려 더한다.
  nearest를 쓰는 이유는 bilinear가 얇은 선을 흐리기 때문이다.
- 출력단 `3×3 Conv + GN`은 top-down 덧셈이 만드는 계단 현상(aliasing)을 없앤다. FPN의 output conv와 같은
  역할이다. **`out_blocks`(기본 1)로 이 블록 수를 조절한다** — 격자 위 국소 문맥을 얼마나 더 섞을지의
  손잡이로, 개선 루프 가설 백로그다(4.1절). 기본값 1은 계획서 원안과 동일하다.
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
- **추론 시 선택:** `select_mode`가 고른다(4.1절, 개선 루프 가설 백로그). 기본 `"thresh"`는
  $\sigma(\text{logit}) > \tau_h$ → `max_pool2d` dilation(3×3) → $N_{\max}$ 상한
  (낮은 임계값 + dilation으로 재현율을 우선한다 — 놓친 셀은 뒤에서 복구할 수 없다). `"topk"`는 확률 상위 `n_topk`개를
  그대로 뽑는다 — 실측(REF-F)에서 `thresh` 모드의 `heat_recall`(11.5절)이 에폭마다 0.0001~0.75로
  요동쳤는데, 이는 임계값이 로짓 스케일에 민감해 보정(calibration)이 흔들리면 통째로 무너지기
  때문이다. `topk`는 보정과 무관하게 상위 확률만 취하므로 이 불안정에서 자유롭다.
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
목적이 아니라 **슬롯 쿼리가 주변 노드를 훑는 것**이 목적이므로 고정 memory로 충분하다.

- **MHA는 직접 구현**한다(선형 qkv + `F.scaled_dot_product_attention` + 출력 프로젝션, 약 30줄). `nn.MultiheadAttention`은 RoPE를 끼워 넣을 수 없다.
- **윈도우 층은 $N \times N$ 마스크가 아니라 이웃 gather 방식이다.**
  셀당 노드가 최대 하나이므로 $w \times w$ 격자 오프셋을 그대로 gather 하면 결과가 같으면서
  어텐션 행렬이 $(N, K, w^2)$로 줄어든다.
- **`window_size = 7` (실측 근거).** 활성 메모리가 $w^2$에 비례하는데
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
| self ($k=0$)                | 2층 MLP | `end_logit` 1    | 없음                | logit                |
| 연결 ($k \ge 1$, 슬롯 간 가중치 공유) | 2층 MLP | `exist_logit` 1  | 없음                | logit                |
| 〃                           | 〃      | `conn_dir` 2     | **`F.normalize`** | **자기 점** 기준 **단위 방향** |

- 연결 슬롯은 **방향만 예측한다.** 상대 노드까지의 거리·좌표는 예측하지 않고, 감독도 방향 차이(1−내적)로만 준다(8.4).
  디코딩에서 상대 정점을 고를 때는 **양쪽 슬롯의 마주봄**($\mathbf{c}\cdot\mathbf{n} \to -1$)과
  실제 상대 방향 정렬로 비용을 만든다(10.3절).
- **끝 판정은 자기 셀의 $\hat{\mathrm{end}}$가 담당한다** — 이웃 셀의 슬롯에 맡기지 않는다. 끝 셀도 분기 2개(안쪽 + 끝방향)를 정상적으로 예측한다(6.2 끝 규약).
- **"2층 MLP"는 `head_hidden`(기본 1) 개의 은닉 블록 + 출력 선형층**을 뜻한다 — `head_hidden=1`이면
  `Linear→GELU→Linear`로 가중치 행렬이 정확히 2개다. 늘리면 은닉 블록이 그만큼 반복된다(4.1절,
  개선 루프 가설 백로그). `share_slot_weights`(기본 True)는 연결 헤드의 MLP를 $R$개 슬롯이 공유할지
  결정한다 — 끄면 슬롯마다 별도 가중치를 갖는다.

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
    # end_pos_weight·class_bg_weight·class_freq_power는 개선 루프 가설 백로그(4.1절).
    # 기본값(1.0 / 1.0 / 0.0)은 전부 "가중 없음"이다.
    # num_classes는 손실 config가 아니라 data config에 있어 from_cfg가 끌어온다.
    def __init__(
        self, *, w_class: float, w_coord: float, w_end: float,
        end_pos_weight: float, class_bg_weight: float,
        class_freq_power: float, num_classes: int,
    ): ...
    def forward(self, output, targets) -> dict[str, Tensor]:
        return {
            "class": l_cls,
            "coord": l_coord,
            "end": l_end,
            "total": self.w_class * l_cls + self.w_coord * l_coord + self.w_end * l_end,
        }


class ConnLoss(nn.Module, Buildable):
    # exist_pos_weight·dir_loss도 가설 백로그 — 기본값(1.0·"cosine")은 아래 수식 그대로
    def __init__(
        self,
        *,
        num_conn_slots: int,
        w_exist: float,
        w_dir: float,
        match_w_dir: float,
        match_w_exist: float,
        exist_pos_weight: float,
        dir_loss: str,
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

**손실을 주는 범위.** 셀 집합마다 감독이 다르다. 끝 셀 특례는 없다 — 끝칸을 채우지 않으므로
모든 양성 셀의 감독이 균일하다.

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

- **"종점 셀은 클래스 손실에서 뺀다"는 특례를 두지 않는다.** 그런 특례의 근거는 "선의 끝 셀은
  다른 차선이 지나가는 자리일 수 있어 라벨이 모호하다"인데, 끝 규약(6.2)이 그 다툼 셀 자체를
  채우지 않으므로 **모호한 셀이 애초에 존재하지 않는다.** 모든 양성 셀의 클래스는 소유 선이
  유일하게 정한다.
- **거짓 양성 셀은 배경(0)으로 감독한다.** 클래스 0을 한 번도 학습하지 않으면 `argmax`가 0을 낼 이유가
  없어져 **디코더의 배경 필터(10.2절 $\arg\max \neq 0$)가 무력해진다.** 히트맵 임계값 하나에만 의존하는
  대신 두 번째 걸름망을 만든다.
- **클래스 CE는 클래스별 가중 벡터 $\mathbf{w} \in \mathbb{R}^{C}$ 를 받는다.** 감소는
  $\sum_i w_{y_i} \ell_i / \sum_i w_{y_i}$ 이므로 **가중이 전부 1이면 단순 평균과 같다.**
  손잡이 둘이 이 벡터를 만든다 (둘 다 개선 루프 가설 백로그).
  - **`class_bg_weight`(기본 1.0)** — 배경 성분 $w_0$. 선택된 셀의 ~70%가 배경이라 CE가 배경에
    지배될 수 있다는 가설을 시험한다.
  - **`class_freq_power`(기본 0.0)** — 전경 성분. 인스턴스 빈도(`CLASS_INSTANCE_COUNT`)의
    `-power` 승으로 희소 클래스를 올리고, **전경 성분의 평균이 1이 되도록 정규화**한다.
    정규화가 있어야 `power`를 키워도 클래스 손실의 스케일이 변하지 않아 손실 균형이 유지된다.
    희소 클래스 3종(`bus_only_lane`·`safety_zone`·`bicycle_lane`)이 검증 200장에서 한 번도
    예측되지 않은 관측이 근거다. 빈도가 셀 수가 아니라 인스턴스 수라 선 길이만큼 근사가 섞인다.

**좌표 손실** — $\mathcal{P}$ 전체

$$
\mathcal{L}_{coord} = \frac{1}{|\mathcal{P}|} \sum_{(i,j) \in \mathcal{P}} \mathrm{SmoothL1}\!\left(\hat{\mathbf{c}}_{ij} - \mathbf{c}^{gt}_{ij}\right)
$$

**끝 손실** — $\mathcal{P}$ 전체. `end_map`을 직접 감독한다:

$$
\mathcal{L}_{end} = \frac{1}{|\mathcal{P}|} \sum_{(i,j) \in \mathcal{P}}
\mathrm{BCE}\!\left(\sigma(\hat{\mathrm{end}}_{ij}),\; \mathrm{end}_{ij}\right)
$$

끝 셀은 양성 셀의 **4.12%**(6.7.5절 M10 재집계 — 끝칸 미채움 반영)라 불균형이 있지만, 히트맵과
달리 수백:1이 아니므로 일단 `end_pos_weight = 1.0`(pos_weight 없음)으로 시작했다. 실제로 로짓이
음수로 눌리는 경향이 보이면 `end_pos_weight`(4.1절, 개선 루프 가설 백로그)를 올려 양성 쪽에
가중을 준다 — `BCEWithLogitsLoss(pos_weight=...)`에 그대로 전달된다.

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
   유도 계산이 없다.
2. 비용 텐서 `C: (P, R, D)` 계산. $R = D = 2$라 패딩이 없다
   ($R > D$인 ablation에서만 분기 축을 패딩하고 그 칸의 비용을 상수 0으로 둔다).
3. 슬롯 순열을 전부 나열: `perms (R!, R)`. $R = 2$면 **2개**(그대로 / 교차).
4. 순열별 총비용 `(P, R!)`을 gather+sum으로 만들고 `argmin` → 셀별 최적 배정.
5. 결과: `matched (P, R)` bool 마스크(기본 설정에서는 전부 참)와, 슬롯이 맡은 분기 인덱스.

$R \le 4$ 전제의 완전탐색이다 ($R \ge 5$가 필요해지면 그때 LSA로 교체).
`test_matching.py`는 무작위 비용에서 이 결과를 `scipy.optimize.linear_sum_assignment`와 대조한다.

**감시 지표 `match_ambiguity`.** 매칭 불안정성을 **한 스텝 안에서** 잰다 — **최적 순열과 차선
순열의 총비용 차가 0.05 미만인 셀의 비율**(= 다음 스텝에 뒤집히기 쉬운 배정). "배정이 스텝 간에
얼마나 바뀌는지"를 직접 재지 않는 이유는 스텝마다 배치 이미지가 달라 같은 셀을 추적할 수 없기
때문이다. 실측에서 의도한 신호를 준다: 합성 과적합 1.000 → 0.098, 실데이터 0.503(ep0) →
0.099(ep22). 모든 셀의 분기가 2라 "유효 분기 $\le 1$인 셀이 항상 모호로 세어져 지표가 바닥에
붙는" 문제도 없다. 손실이 아니라 감시 지표이고,
로깅 경로가 손실과 같아서 따로 배선할 필요가 없다(9.4절).

### 8.4. 매칭 후 연결 손실 (`ConnLoss` 본체)

$N_{match} = 2\,|\mathcal{P}|$ 는 매칭된 쌍의 총수다 (모든 양성 셀의 분기가 2개).

**존재 손실** — 감독 범위가 셀 종류마다 다르다:

- $\mathcal{P}$의 셀: 매칭된 슬롯 1. $R = D = 2$에서는 **모든 슬롯이 매칭되므로 전 슬롯 1**이다
  (무매칭 슬롯 0 감독은 $R > D$ ablation에서만 나타난다).
- $\mathcal{S} \setminus \mathcal{P}$의 셀(거짓 양성): **전 슬롯 0** ("존재하지 않음만 학습").

즉 기본 설정에서 exist의 변별 신호는 거짓 양성 셀에서만 나온다 — 사실상 "이 셀이 진짜 노드인가"의
셀 단위 신호가 슬롯별로 복제된 것이다. 디코더의 슬롯 게이트($\sigma(\hat e) > \tau_e$, 10.3절)로는
여전히 쓰이므로 유지한다. 거짓 양성 셀이 압도적으로 많은 실행에서는 `exist_pos_weight`(기본 1.0,
개선 루프 가설 백로그)로 양성(매칭된) 쪽에 가중을 준다 — `BCEWithLogitsLoss(pos_weight=...)`.

$$
\mathcal{L}_{e} = \frac{1}{|\mathcal{S}| \cdot R} \sum_{(i,j) \in \mathcal{S}} \sum_{k=1}^{R}
\mathrm{BCE}\!\left(\sigma(\hat{e}_k),\; \mathbf{1}[k \text{ matched}]\right)
$$

**방향 손실** — 매칭된 쌍에만. 크기·좌표 감독 없이 **방향 차이만** 학습한다. `dir_loss`(기본
`"cosine"`)가 형태를 고른다. 기본값의 값 범위는 $[0, 2]$:

$$
\mathcal{L}_{dir} = \frac{1}{N_{match}} \sum_{\text{matched}\,(k,m)} \left(1 - \hat{\mathbf{d}}_k \cdot \mathbf{d}^{gt}_m\right)
$$

`dir_loss = "angle"`(가설 백로그)이면 $\frac{1}{\pi}\arccos(\hat{\mathbf{d}}_k \cdot \mathbf{d}^{gt}_m)$의
평균으로 바뀐다 — 코사인 항은 오차가 작을수록 기울기가 0에 가까워지는데(1 근처에서 평평), 각도
항은 작은 오차에서도 기울기가 살아 있다는 가설을 시험한다.

끝 셀의 끝방향 분기도 똑같이 존재 1 + 방향으로 감독된다 — "선이 이쪽으로 끝났다"를 슬롯이
말하게 하고, 셀이 끝이라는 사실은 $\mathcal{L}_{end}$(8.2)가 따로 말한다.
슬롯 종점 손실 $\mathcal{L}_t$는 두지 않는다(6.2절). 매칭 안 된 슬롯의 방향에는 손실을 주지 않는다.

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

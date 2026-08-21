"""실험 config 스키마 — 모든 dataclass를 이 파일 하나에 둔다 (design 4.1).

이 모듈은 `stella` 패키지를 import 하지 않는다. config는 순수한 데이터이므로
코드 쪽으로 의존이 생기지 않고, 따라서 순환참조가 구조적으로 불가능하다.
"""

from dataclasses import dataclass, field


@dataclass(kw_only=True)
class ModuleConfig:
    """build_instance로 만드는 부품 config의 공통 베이스 (design 5절).

    하위 클래스가 두 필드에 기본값을 준다.
    """

    path: str  # 모듈 경로   예: "stella.model.backbone"
    name: str  # 클래스 이름 예: "Dinov3Backbone"


@dataclass(kw_only=True)
class DataConfig(ModuleConfig):
    path: str = "stella.data.seedmap"
    name: str = "SeedMapDataset"
    # 원본 SEED_MAP_v1.1을 {train,val,test}/{image,label} 구조로 재정리한 사본 (6.7.2절, M13)
    root: str = (
        "/media/humpback/435806fd-079f-4ba1-ad80-109c8f6e2ec0/Ongoing/2026_stella"
        "/SEED_MAP_v1.1_splits"
    )
    image_size: int = 768  # SEED-MAP 원본 크기와 동일 — 리사이즈 없음
    grid_stride: int = 4  # 격자 배율 s. L = image_size // grid_stride = 192
    num_classes: int = 12  # 0 = background + 차선 11종 (6.7.1절)
    batch_size: int = 1  # 확정 — bs=2는 처리량 +16%뿐, accumulate가 유효 배치를 만든다 (9.3절)
    num_workers: int = 8
    max_degree: int = 2  # D: 셀당 GT 분기 수. 선 단위 사슬이라 항상 정확히 2 (6.4절)
    encode_supersample: int = 1  # GT 래스터화 배율. 1 = 픽셀 해상도 (6.4절 A단계)
    # 연결 방향 GT를 **몇 칸 앞** 점으로 만들지. 1 = 이웃 칸(기존 동작).
    # **★ 08-15 기각: 1 이외의 값을 쓰지 마라.** GT 주입 천장이 k=1 0.977 → k=2 0.849 →
    # k=4 0.838 → k=8 0.807 로 무너진다(완벽한 예측인데 사슬이 46.4 → 25 로 반토막).
    # 기전은 **미해명**이다 — 마주봄 게이트(`opp_thresh`·`w_opp`)와 정렬 게이트(`align_thresh`)를
    # 각각 완전히 풀어도 `chain_len` 이 24~27 에서 꿈쩍하지 않는다. "k칸씩 건너뛴다"면
    # 길이가 46.4/k 여야 하는데 k=2·4·8 에서 26.0·25.1·23.9 로 포화하므로 그것도 아니다.
    # 손잡이는 남겨 둔다(기본값 1은 옛 인코딩과 바이트 단위로 같고 테스트로 고정돼 있다).
    # 이웃 칸 접선은 스텝마다 흔들린다 — 인코더로 직접 실측(val 40장·사슬 1,475개):
    # 평균 5.9도 · 90%분위 14.0도 · **>20도가 4.4%**. 스텝당 4.4%라도 길이 45 사슬에서는
    # 사슬당 평균 2회 끊길 확률이고, 디코더는 이 방향을 반경 끝까지 외삽한다.
    # 4칸이면 평균 1.6도 · >20도 0.5%. 모델의 방향 오차(7.3도)가 이 요동과 비슷한 수준이라
    # **목표 자체의 잡음이 하한을 만들고 있을 수 있다** (E23에서 검증).
    # **바꾸면 GT 캐시가 낡는다** — `seedmap`이 캐시 폴더 이름에 이 값을 붙여 분리한다.
    conn_lookahead: int = 1
    cache_gt: str = "val_test"  # "none" | "val_test" | "all" (6.4.1절)
    # 데이터셋 폴더를 건드리지 않도록 캐시는 그 옆에 둔다. 빈 문자열이면 {root}/gt_cache.
    cache_dir: str = (
        "/media/humpback/435806fd-079f-4ba1-ad80-109c8f6e2ec0/Ongoing/2026_stella/gt_cache"
    )
    augment: bool = True  # 학습 split에만 적용 (6.7.6절)
    # 격자 대칭 외의 기하 증강 (가설 백로그). 0이면 끈다 — 기본은 기존 동작.
    aug_rotate_deg: float = 0.0  # +-이 각도까지 임의 회전. 타일 밖은 검게 채운다
    aug_scale_jitter: float = 0.0  # 1 +- 이 비율까지 등방 스케일
    limit: int = 0  # >0이면 split당 앞에서 N개만 사용 (스모크·디버깅용)

    @property
    def grid_size(self) -> int:
        return self.image_size // self.grid_stride


@dataclass(kw_only=True)
class BackboneConfig(ModuleConfig):
    path: str = "stella.model.backbone"
    # "Dinov3Backbone" | "SwinBackbone" | "ConvNeXtBackbone" | "TimmVitBackbone"
    # 기본값이 DINOv3가 아닌 이유: sat493m 저장소가 HF 게이트라 승인 전에는 받을 수 없다.
    name: str = "ConvNeXtBackbone"
    pretrained: str = "convnextv2_base.fcmae_ft_in22k_in1k_384"  # HF/timm 모델 ID
    lr_mult: float = 0.1  # optim.py가 읽는다 (__init__ 인자 아님)
    freeze: bool = False
    # 5레벨 백본(HRNet·ResNet·MaxViT)에서 FPNLite가 쓰는 stride 4/8/16/32만 고른다.
    out_indices: tuple = ()  # 비우면 timm 기본값
    img_size: int = 0  # 고정 입력 크기 백본(Swin)에만. 0이면 지정하지 않는다


@dataclass(kw_only=True)
class NeckConfig(ModuleConfig):
    path: str = "stella.model.neck"
    name: str = "FPNLite"  # "SFP"(ViT 단일 스케일) | "FPNLite"(멀티스케일)
    out_blocks: int = 1  # FPNLite 출력단 3x3 블록 수 — 격자 위 국소 문맥의 양 (가설 백로그 C4)


@dataclass(kw_only=True)
class ModelConfig(ModuleConfig):
    path: str = "stella.model.stella"
    name: str = "StellaModel"
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    neck: NeckConfig = field(default_factory=NeckConfig)
    d_model: int = 256  # neck·블록·헤드가 공유 — 여기 한 곳에만 둔다
    num_heads: int = 8
    num_conn_slots: int = 2  # R = 2 (K = 3) — GT 분기 수와 일치 (결정 1, 10차 개정)
    layers: tuple[str, ...] = ("global", "window", "window", "window", "window", "window")
    # 활성 메모리가 w^2 에 비례하고 실제 연결은 디코더 탐색 반경 2셀 안에서 일어난다.
    # 실측(n_max 6000, bs 1): 9 -> 7 에서 peak 12.09 -> 9.72 GiB, step 455 -> 291 ms (7.6절).
    window_size: int = 7  # w
    # 디코더는 `argmax != 0`(전경이냐)만 쓰는데 12지 CE가 그 판정과 종류 분류를 겸한다.
    # 손실의 대부분은 정점을 살리는 종류 혼동이 먹고, 치명적인 "배경이라 부름"과 섞여 경쟁한다.
    # True면 전경 로짓 1채널을 따로 둔다 (E12). False면 파라미터가 아예 생기지 않는다.
    fg_head: bool = False
    ffn_dim: int = 1024
    dropout: float = 0.0
    grad_checkpoint: bool = True  # 윈도우 층만 재계산 (활성의 대부분이 거기 있다, 7.6절)
    head_hidden: int = 1  # 헤드 MLP의 은닉층 수. 1 = 계획서 원안(2층) (가설 백로그 C5)
    share_slot_weights: bool = True  # 연결 슬롯 R개가 MLP를 공유하는가 (가설 백로그 C5)
    node_sampling: str = "gt+pred"  # "gt+pred"(기본) | "gt" (7.4절)
    # 전체 train 8979장 실측: 평균 2121, p90 3681, p99 5893, 최대 8909 (6.7.5절).
    n_max: int = 9500
    heatmap_thresh: float = 0.3  # tau_h
    dilate: int = 3  # 예측 마스크 팽창: 0 | 3 | 5
    # "thresh" = 확률 > tau_h (기본) | "topk" = 확률 상위 K개 (백로그 C8)
    # — 보정에 흔들리지 않는다.
    # REF-F 실측: thresh 모드의 heat_recall이 에폭마다 0.0001~0.75로 요동쳤다.
    select_mode: str = "thresh"
    # topk 모드의 K. 실측 GT 노드 수는 평균 1,826 / p90 3,181 / p99 4,978 (E03)이라
    # 4,000이면 대부분의 타일에서 GT를 덮으면서 학습 토큰 수가 현재와 비슷하게 유지된다.
    n_topk: int = 4000


@dataclass(kw_only=True)
class HeatmapLossConfig(ModuleConfig):
    path: str = "stella.loss.heatmap"
    name: str = "HeatmapLoss"
    w_heatmap: float = 1.0
    # 가중치가 아니라 focal 형태 파라미터. 0.25 → 0.75는 E08 실측 — U 규격에서 f1 +22.4%
    # (0.1737 → 0.2126), heat_recall 0.8286 → 0.9545. 양성 쪽을 더 들어 정점 재현율을 산다.
    focal_alpha: float = 0.75
    focal_gamma: float = 2.0
    # 양성 셀의 가중을 그 셀이 속한 선의 길이에 반비례하게 (0 = 무동작).
    # **08-22 기각.** 분류 손실에는 같은 처방이 짧은 선 정답률 +20.5% 였는데 히트맵에는
    # 이득이 없다 — 길이 구간별 TP 가 0~20칸 0.200 -> 0.193 · 70칸 이상 0.440 -> 0.433
    # (0.5 배는 40~70칸 −8.5%). **히트맵은 이미 정답 칸의 96%를 고르고 있어 살릴 여지가 없다.**
    # 중복 문제(정점 억제 + 선 정리)처럼 두 층이 더해지는 구조가 아니었다.
    # 손잡이는 ablation 용으로 남긴다.
    length_power: float = 0.0


@dataclass(kw_only=True)
class SelfSlotLossConfig(ModuleConfig):
    path: str = "stella.loss.self_slot"
    name: str = "SelfSlotLoss"
    w_class: float = 1.0
    w_coord: float = 1.0
    w_end: float = 1.0  # 끝 셀 BCE — end_map 직접 감독 (8.2절, 9차 개정)
    # 끝 셀 양성이 전체 양성의 ~2.5%라 로짓이 음수로 눌린다 (가설 백로그).
    end_pos_weight: float = 1.0  # 1.0 = 가중 없음
    # 선택 셀의 ~70%가 배경이라 클래스 CE가 배경에 지배당한다 (가설 백로그 B6). 1.0 = 가중 없음
    class_bg_weight: float = 1.0
    # 희소 클래스 3종(bus_only_lane·safety_zone·bicycle_lane)이 200장에서 한 번도 예측되지
    # 않았다 (E07). 전경 가중을 인스턴스 빈도의 -power 승으로 준다. 0.0 = 가중 없음 (E09)
    # **08-19 채택: 0.5.** 반경을 5로 줄이면서 약해진 희소 종류를 학습 쪽에서 되찾는다.
    # 같은 U 규격 대조군 대비 **종류별 평균 f1 +3.4%** (0.2420 -> 0.2502, 판정 밴드 ±2%),
    # 안전지대 f1 0.0081 -> 0.0872 · 버스전용차로 0.0263 -> 0.0752. 대가는 없다
    # (precision −1.4% · correctness +0.9% · coverage +0.5%, 전부 잡음 범위).
    class_freq_power: float = 0.5
    # 전경 가중의 정규화 방식. "mean" = 평균 1 (E09 의 재분배 — 흔한 클래스를 1 아래로 누른다)
    # "floor" = 1 아래로 내리지 않음 (희소만 올린다). **E09 기각의 원인은 축이 아니라 정규화였다** —
    # `power=0.5` 가 `lane_line` 가중을 0.42 로 눌러 전경 인식이 무너졌다(class_fg -42%).
    # 08-19 사용자 결정: 반경 축소로 약해진 희소 종류는 이 손잡이로 보완한다.
    # **08-19 채택: "floor".** E09 가 이 축을 기각한 원인은 축이 아니라 "mean"(재분배)이었다 —
    # 희소를 올린 만큼 흔한 클래스를 눌러 `lane_line` 가중이 0.42 가 됐고 전경 인식이
    # 42% 무너졌다. "floor" 로 바꾸니 `class_fg` 가 0.483 -> 0.471 로 멀쩡하다.
    class_freq_norm: str = "floor"
    # 셀 가중을 그 셀이 속한 **선의 길이에 반비례**하게 (0.0 = 무동작). 손실은 셀 단위인데
    # 지표는 선 단위라, 100칸 선은 100표 7칸 선은 7표를 갖는다. 실측(08-20): 20칸 미만 선이
    # 정답 선의 46%인데 셀로는 13.6%뿐이고 정점 검출률이 0.339 로 70칸 이상(0.595)의 57%다.
    # **08-21 채택: 0.3.** 대조군(같은 코드·길이 역가중만 없음)과 같은 조건에서 길이 구간별
    # TP 비율: 0~20칸 0.166 -> **0.200 (+20.5%)** · 20~40 +4.7% · 40~70 +5.1% ·
    # 70칸 이상 −1.3%(밴드 안). 종류별 평균 f1 +4.0% · recall +3.8% · 전체 f1 제자리.
    # **세기 반응이 뚜렷하다** — 0.3 거의 다 오름 · 0.5 짧은 선 얻고 긴 선 −15.5% ·
    # 1.0 붕괴(차선 칸 인식 −36%). 재분배이므로 과하면 긴 선을 버린다.
    length_power: float = 0.3
    # 전경/배경 이진 BCE (E12). `model.fg_head=True`와 함께 쓴다. 0.0 = 항 자체를 계산하지 않는다.
    w_fg: float = 0.0
    fg_pos_weight: float = 1.0  # 선택 셀의 ~70%가 배경이라 양성 쪽을 들 수 있게 열어 둔다


@dataclass(kw_only=True)
class ConnLossConfig(ModuleConfig):
    path: str = "stella.loss.conn"
    name: str = "ConnLoss"
    w_exist: float = 1.0
    w_dir: float = 1.0
    match_w_dir: float = 1.0  # lambda_dir — 손실 가중치가 아니라 매칭 비용 계수 (8.3절)
    match_w_exist: float = 1.0  # lambda_e
    exist_pos_weight: float = 1.0  # 거짓 양성 셀이 압도적일 때 양성 쪽을 든다 (가설 백로그 B3)
    # "cosine" = 1 - cos (기본) | "angle" = acos/pi (백로그 B4)
    # — 작은 오차에서 기울기가 살아 있다.
    # **08-19 채택: "angle".** cosine 은 오차 0 근처에서 기울기가 sin θ 로 죽는다(6도에서 0.105).
    # angle 은 1/π 로 **상수**라 그 구간에서 계속 배운다. 대조군 대비 f1 +2.0% ·
    # dir_err_deg 6.611 -> 6.242 (−5.6%) · 상위 10% 오차 12.50 -> 11.50 (−8.0%), 손해 본 지표 없음.
    # **가중을 올리는 길은 정반대로 해롭다** — 같은 라운드의 `w_dir=20` 은 f1 −21.4% 였다.
    # 방향 항이 커지자 정점 검출이 굶어 정점이 1,217 -> 953 개/장으로 줄었다.
    # 손실 균형(SKILL 8절 A)의 처방은 "가중을 키워라"인데, 이 항에서는 **곡선을 고치는 것**이
    # 맞았다 — 같은 균형 개선을 스케일 6배로 얻으면서 다른 항을 굶기지 않는다.
    dir_loss: str = "angle"


@dataclass(kw_only=True)
class LossConfig(ModuleConfig):
    path: str = "stella.loss.criterion"
    name: str = "StellaCriterion"
    heatmap: HeatmapLossConfig = field(default_factory=HeatmapLossConfig)
    self_slot: SelfSlotLossConfig = field(default_factory=SelfSlotLossConfig)
    conn: ConnLossConfig = field(default_factory=ConnLossConfig)


@dataclass(kw_only=True)
class DecodeConfig(ModuleConfig):
    """사슬 확장 디코더 (9차 개정, 10절).

    임계값들은 학습된 체크포인트로 검증 셋에서 스윕해 확정한다 (13절 남은 확인).
    구 GraphDecoder의 mutual·w_dist·max_conn_dist·t_thresh는 폐기 — 10절 참고.
    """

    path: str = "stella.decode.graph"
    name: str = "ChainDecoder"
    heatmap_thresh: float = 0.3  # tau_h — 노드 후보 (추론 경로, 7.4절)
    # tau_e — 연결 슬롯 존재. **실측으로 정했다.** 0.5에서는 사슬의 27%가 "연결 없음"
    # 예측에 끊겼다(GT 주입은 0%). 0.3에서 stop_exist 0.268 → 0.001이 되고 그 아래로는
    # 포화한다. chain_len +40%.
    exist_thresh: float = 0.3
    end_thresh: float = 0.5  # tau_end — 끝 셀 판정 (sigmoid(end_logit)), 사슬 정지 조건
    # 탐색 반경(셀). **실측으로 정했다** — 2는 f1을 1.6배 깎고 있었다.
    # 현재 기본값 모델(α=0.75) 예측 200장, 기본 동작점: r=2 0.2173 · 8 0.2932 · 16 0.3436 ·
    # **24 0.3589** · 32 0.3617. 24가 비용/이득의 무릎이다(40장 디코딩 114초 vs 32의 129초).
    # 사슬이 "다음 후보 없음"으로 멈추던 것이 원인 — chain_len 4.40 → 8.90 (GT 평균 48.2셀).
    # **08-14 재선택**: 24는 ±96 px 이고 **99.9%의 선이 그 안에 다른 선을 갖는다**(차선 간격
    # 중앙값 11.8 px). 옛 지표(합집합 correctness · rho=12)가 갈아탐을 공짜로 봐 줘서 24가
    # 이겼다. 새 지표에서 12가 최적이고 `correctness`가 반경에 **단조 감소**한다
    # (r=4 0.872 · r=12 0.838 · r=24 0.726). 12는 사용자가 지정한 탐색 상한이기도 하다 —
    # f1 은 그 경계에서 아직 오르는 중이므로 상한을 풀면 다시 재야 한다.
    # **08-19 사용자 지시로 3~5 만 탐색해 5 로 확정** (20 px). 12(48 px)는 차선 간격 11.8 px 의
    # 네 배라 옆 차선·옆 진입로를 건너뛰고 이어 버렸다. 반경이 바꾸는 것은 꺾임각이 아니라
    # **도약 길이**다 — 디코딩된 스텝 길이 99% 가 반경 12 에서 36.2 px(차선 세 칸), 5 에서
    # 16.5 px, 3 에서 12.5 px 다. 전체 val 400장 실측(min_points 8 일정): 갈아탐 점 비율
    # 6.0% -> 2.5% · correctness 0.809 -> 0.861 · f1 0.350 -> 0.360 · GT 주입 천장은 유지
    # (0.964 -> 0.946). 3 은 갈아탐이 가장 적지만 f1 이 6% 낮다.
    # **대가**: 희소 3종(자전거도로·버스전용차로·안전지대)이 나빠진다 — 반경이 크면 띄엄띄엄한
    # 정점을 건너뛰어 이어 주던 것이 사라지기 때문이다. 그 보완은 **학습 쪽 클래스 가중**으로
    # 한다(사용자 결정 08-19). `min_points_short` 로는 안 된다 — 허위 조각이 늘어 더 나빴다.
    radius: int = 5
    # 내 슬롯 방향과 실제 상대 방향의 코사인 하한 (c . u_ab). **w_dist와 함께 정해진다** —
    # 거리 항이 죽어 있던 동안에는 조이면 손해였다(0.7이 최적). w_dist=0.03에서는
    # 0.85~0.9가 평평한 최적이고 0.7 대비 f1 +4.7%다.
    # **08-14 재선택**: 0.85는 옛 지표에서 정한 값이다. 새 지표(rho=4·단일 GT)에서는 조일수록
    # 좋아져 0.95~0.97이 평평하다 — 0.95가 f1 최고(0.3180), 0.97이 correctness 최고(0.8503).
    align_thresh: float = 0.95
    opp_thresh: float = 0.7  # 마주봄 하한 — -(c . n) >= 이 값 (10.3절)
    w_opp: float = 1.0  # 후보 비용에서 마주봄 항의 비중
    # 후보 비용의 거리 항 계수(셀 단위 거리에 곱한다). **실측으로 정했다 — 최대 개선폭.**
    # 0.001은 반경 2 시절의 동률 해소용 상수였고(최대 기여 0.004), 반경 24에서도 0.034라
    # 정렬 항 (1-align, 0~0.3) 앞에서 무시됐다. 그 결과 **디코더가 정렬만 조금 나으면
    # 20셀을 건너뛰며 정점을 지나쳤다** — 한 선이 4~5조각으로 쪼개진 주원인이다.
    # 0.03에서 f1 +33% · precision +64% · chain_len +130% · frag −38%(캐시 4개 200장 일정).
    # 0.03~0.04가 평평한 최적. 10.3절이 배제한 w_dist=0.3은 반경 2에서 잰 값이라 무효다.
    # 08-19: 반경을 12 -> 5 로 줄이면서 곱(`w_dist * radius`)을 0.36 으로 유지했다.
    # 게이트 불변식이 요구하는 하한 0.3 을 넘기고, 0.03 고정보다 f1 이 0.3~0.8% 낫다.
    w_dist: float = 0.072
    # 확장 게이트 — 후보의 사슬 클래스 softmax 확률 하한 (10.3절).
    # **08-14 재선택**: 새 지표에서 0.2가 캐시 3개에서 일관되게 낫다(+1.8% · +2.0% · +1.4%).
    # 판정 밴드(±10%) 안이라 "무효"지만 방향이 세 번 일치하고 대가가 없어 채택한다.
    min_class_prob: float = 0.2
    # 정점의 배경 필터. 음수면 기존 방식(`argmax != 0`),
    # 0 이상이면 전경 로짓 확률의 하한을 쓴다 (E12).
    # 정점을 **선의 법선 방향으로만** 비최대 억제해 넓은 전경 띠를 한 줄 능선으로 줄인다.
    # 중복 선을 **선 단계에서 지우는 것**(dedup)과 **정점 단계에서 안 만드는 것** 중
    # **08-20 실측으로 정했다 — 둘 다 켠다.** 서로를 대체하지 않고 더해진다(400장):
    #   둘 다 끔 0.3603 · 억제만 0.3734(+3.6%) · 중복 정리만 0.3798(+5.4%) · 둘 다 0.3860(+7.1%)
    # **억제는 GT 주입 천장을 전혀 깎지 않는다**(0.9414 = 기준선과 소수점 넷째 자리까지 동일) —
    # 정답에는 넓은 전경 띠가 없어(선 하나당 칸 하나가 주인) 억제할 것이 없기 때문이다.
    # 즉 표현 단계의 처방은 정답 위에서 아무것도 건드리지 않는다. 후처리는 −0.5% 를 치른다.
    vertex_local_max: bool = True
    fg_thresh: float = -1.0
    purity_thresh: float = 0.6  # 사슬 순도 하한 — argmax 클래스 일치 비율. 이하면 사슬 폐기
    end_extend: float = 1.0  # 끝 셀에서 끝방향 슬롯으로 연장하는 길이(셀) — 10.3절 끝 연장
    # 이보다 짧은 폴리라인은 버린다 (연장점 포함). **08-18 사용자 결정: 일단 6, 다음에 실험으로
    # 검증한다.** 버퍼 4px 시절 측정은 6 -> f1 0.377 · 8 -> 0.382 로 8이 근소하게 높았으나
    # 버퍼를 3px 로 줄인 뒤 다시 재지 않았다. 짧은 종류는 아래 `min_points_short` 가 보호한다.
    # **08-19 재선택: 8.** 반경 5 에서 전체 f1 은 10까지 계속 오르지만(0.358 -> 0.362)
    # **종류별 평균 f1 이 8에서 봉우리를 찍고 꺾인다**(0.2808 -> 0.2800) — 그 위로는 드문
    # 종류를 버려서 점수를 버는 것뿐이다. 반경 12 에서도 봉우리는 같은 8이다.
    min_points: int = 8
    # **짧은 차선 종류에만 적용하는 별도 하한** (사용자 지시 08-18 — 기준을 종류마다 따로 두지
    # 말고 "짧은 종류" 한 묶음에만 하나 더). val 실측 중앙 길이(칸): stop_line 7.5 ·
    # safety_zone 10.9 · path_change_restriction_line 14.1 · 그다음이 19.8 로 확 뛴다.
    # 일반 하한을 6 으로 올리면 정지선의 38% 가 통째로 사라지므로 이 셋만 낮게 둔다.
    min_points_short: int = 2
    short_classes: tuple = (9, 10, 6)  # stop_line · safety_zone · path_change_restriction_line
    # 사슬 평균 점수 하한. `min_points`와 같은 목적(허위 조각 제거)인데 **길이를 안 본다** —
    # GT 선의 13.3%가 6셀 미만이라 길이로 거르면 그만큼을 구조적으로 포기한다.
    min_chain_score: float = 0.0  # 0이면 무동작
    simplify_tol: float = 0.0  # >0이면 RDP 단순화 (픽셀)
    # --- 알고리즘 변형 (가설 백로그 A). 기본값은 전부 기존 동작이다 ---
    seed_mode: str = "class_peak"  # "class_peak" | "end_peak" (선의 끝에서 시작, A5)
    stop_needs_nocand: bool = False  # True면 끝 확률 + 후보 없음을 모두 만족해야 정지 (A4)
    # **08-19: 0 -> 24.** 반경 12 시절에는 조각들이 서로 **겹쳐** 있어 "끝점이 가깝고 마주봄"
    # 조건에 거의 걸리지 않았다(영상당 0.33건, f1 +-0.3%). 반경 5 에서는 조각이 끝과 끝으로
    # 맞닿아 실제로 터진다 — f1 0.3579 -> 0.3603, 종류별 평균 0.2808 -> 0.2828.
    merge_gap: float = 24.0  # >0이면 끝점 간 이 거리(픽셀) 안의 조각을 병합 (A2)
    merge_align: float = 0.8  # 병합 정렬 하한 — 두 조각이 서로를 향하는 정도
    # "cosine" = 각도 게이트(기본) | "perp" = 예측 방향 직선에서의 수직 이탈 게이트 (A6)
    # **cosine 게이트는 거리에 무력하다** — 같은 "한 차선 옆"이라도 앞 12셀이면 45도라 막히고
    # 앞 60셀이면 11도라 통과한다. 멀리 볼수록 이웃 차선이 싸게 들어온다. perp 는 이탈을
    # 셀 단위 절대값으로 재므로 거리에 비례해 엄격해진다. A6 의 기각 판정은 w_dist=0.001
    # 시절 값이라 무효다 — 거리 항이 죽어 있으면 어떤 게이트를 씌워도 사슬이 튄다.
    align_mode: str = "cosine"
    perp_thresh: float = 0.7  # perp 모드의 수직 이탈 상한 (셀 단위). 차선 간격 11.8px = 2.95셀
    # 연속한 두 스텝 사이의 방향 변화 상한(도). 180이면 무동작(기존 동작).
    # align_thresh 는 후보 방향과 **모델이 예측한 슬롯 방향**만 비교하므로 사슬이 매 스텝
    # 31.8도씩 꺾여도 아무것도 막지 않았고, 그 꺾임이 누적돼 시각적으로 무너졌다.
    # **08-14 실측**: `align_thresh`가 0.85일 때는 20도가 f1 +3.5%였으나, 0.95로 조인 뒤에는
    # 25~180도가 전부 평평하다(±0.3%). **두 게이트가 같은 것을 재고 있었다.**
    # 그리고 20도는 **GT 주입 천장을 0.981 → 0.777로 깎는다** — 정점이 인접 셀(4 px)이라
    # 직선 위에서도 스텝 방향이 양자화 잡음으로 20도를 넘기 때문이다. 천장 절벽은 25↔30 사이.
    # 45도는 물리적으로 불가능한 꺾임만 막는 값싼 안전장치다(천장 0.9772, 모델 f1 최고 타이).
    # **현재 작동점에서 측정 가능한 기여는 없다.**
    max_turn_deg: float = 45.0
    # --- 중복 정리 후처리 (10.5절, 08-20 신설). `dedup_high <= 0` 이면 무동작 ---
    # 예측 선의 18.5%가 같은 클래스의 다른 선과 **1.8 px 간격으로 나란히** 그려진다
    # (겹친 쌍의 91%가 한 셀 이내 — 이웃 차선 11.8 px 이 아니라 **같은 차선 위 이중 그리기**).
    # 디코더는 정점을 한 번씩만 쓰지만 "이미 그린 선"이라는 개념이 없어 이것을 막지 못한다.
    # **계약**: 지우거나 이미 겹친 선끼리 잇기만 한다 — 떨어진 선을 잇지 않고
    # 없던 인스턴스를 만들지 않는다. 인스턴스 수는 줄기만 하고 늘지 않는다.
    # **문턱 선정 규칙(08-20)**: 진짜 같은 클래스 이웃 선의 하위 분위수보다 좁게 잡는다 —
    # 실측 6px 이내 3.4% · 4px 이내 1.6% · **3px 이내 0.8%**. 제거 대상 중복은 간격 중앙 1.8px.
    # 6px 로 잡았더니 진짜 선을 지워 천장이 0.946 -> 0.908 로 깎였다. 평가 버퍼를 차선 간격의
    # 절반 이하로 정한 것과 같은 꼴의 규칙이다.
    dedup_high: float = 3.0  # 가로 거리가 이보다 크면 '겹치지 않음'(px)
    dedup_low: float = 1.5  # 이력 문턱의 아래쪽 — 이보다 가까우면 확실히 겹침
    dedup_min_free: float = 8.0  # 자유 구간이 이보다 짧으면 버린다(px)
    dedup_bridge: float = 0.0  # 자유 구간 사이 이보다 짧은 겹침 끊김은 메운다(px)
    dedup_min_diverge: float = 8.0  # 강한 이탈이 이 길이 이상 지속돼야 자유로 인정(px)
    dedup_join_gap: float = 6.0  # 자유 구간을 붙일 끝점 탐색 반경(px). 작게 둬야 간격을 안 메운다
    dedup_step: float = 2.0  # 재표본 간격(px)
    # 자유 구간이 원래 길이의 이 비율 이상이면 **중복이 아니라고 보고 원본을 그대로 둔다**.
    # 1.0 에 가까울수록 보수적(거의 포함된 선만 지운다). 0 이면 항상 잘라 낸다 —
    # 자르는 판은 f1 +5.5% 지만 recall −3.9% · 천장 0.946 -> 0.908 이라 **진짜 선을 지운다**.
    dedup_keep_ratio: float = 0.35


@dataclass(kw_only=True)
class MetricConfig(ModuleConfig):
    path: str = "stella.eval.ccq"
    name: str = "InstanceCCQ"
    # rho (픽셀). **실측으로 정했다.** 11.1절의 규칙은 "차선 간격의 절반 이하"인데
    # 옛 기본값 12.0 이 그 규칙을 스스로 어기고 있었다 — val 80장·선 2,896개에서 이웃 선까지의
    # 중앙 거리가 **11.8 px** 다(58%는 12 px 안에 이웃이 있다). 버퍼가 이웃 차선을 삼켜
    # **옆 차선으로 갈아탄 예측이 만점을 받았다.** 4 px = 간격의 1/3, GT 주입 천장은
    # 97.1%로 유지된다(12 px에서 97.5%) — 엄격하게 해도 도달 가능한 목표다.
    buffer_rho: float = 3.0
    cov_thresh: float = 0.5  # theta_cov — 커버리지 하한, 관대
    cor_thresh: float = 0.9  # theta_cor — 정확성 하한, 엄격
    angle_gate: float = 30.0  # 매칭 시 접선 방향 차 상한(도)
    sample_step: float = 2.0  # 길이 계산용 폴리라인 샘플 간격(픽셀)
    max_instances: int = 400  # 샘플당 평가 인스턴스 상한 (안전장치)
    # frag는 "조금이라도 겹치는" 예측을 다 세어 정확성이 높을수록 부풀려진다(E00에서 확인).
    # frag_strict는 그 GT를 이 비율 이상 덮는 조각만 센다 — 조각남의 정직한 측정치다.
    frag_min_cov: float = 0.1


@dataclass(kw_only=True)
class CellDiagConfig(ModuleConfig):
    """셀 단위 진단 (research 스킬 · 셀 단위 진단). 임계값은 `decode`에서 가져다 쓴다."""

    path: str = "stella.eval.cellstat"
    name: str = "CellDiagnostics"


@dataclass(kw_only=True)
class LogConfig(ModuleConfig):
    path: str = "stella.train.callbacks"
    name: str = "VizCallback"
    every_n_epochs: int = 1
    max_batches: int = 20  # 에폭당 최대 배치 수 (배치당 1장, 9.5절)
    heat_alpha: float = 0.5
    slot_line_len: float = 6.0
    exist_thresh: float = 0.5
    class_thresh: float = 0.5


@dataclass(kw_only=True)
class TrainConfig(ModuleConfig):
    path: str = "stella.train.module"
    name: str = "StellaTrainModule"
    lr: float = 1e-4
    weight_decay: float = 0.05
    warmup_steps: int = 1000
    epochs: int = 100  # 아래 넷은 pl.Trainer가 읽는다 (__init__ 인자 아님)
    accumulate: int = 16  # 유효 배치 = batch_size x accumulate x GPU 수
    grad_clip: float = 0.1
    precision: str = "bf16-mixed"
    devices: str = "auto"  # pl.Trainer devices
    limit_val_batches: float = 1.0
    seed: int = 42  # train.py가 읽는다
    # 체크포인트 — 구 실행에서 last.ckpt가 13 에폭에 멈춘 사고가 있어 명시적으로 둔다.
    ckpt_monitor: str = "val/inst/f1"  # 손실이 아니라 최종 지표를 기준으로 남긴다
    ckpt_mode: str = "max"
    ckpt_top_k: int = 3
    # 최소 1노드 보장(7.4절) 덕에 미사용 파라미터가 없다 — 끄면 매 스텝 그래프 순회가 사라진다.
    find_unused_parameters: bool = False
    output_root: str = (
        "/media/humpback/435806fd-079f-4ba1-ad80-109c8f6e2ec0/Ongoing/2026_stella/log"
    )


@dataclass(kw_only=True)
class CpuConfig(ModuleConfig):
    """이 프로세스가 쓸 CPU 예산 — 부하를 예측 가능하게 만드는 단일 출처.

    torch는 기본적으로 프로세스마다 코어를 **전부** 잡는다. 그래서 arm 두셋만 겹쳐도
    러너블 스레드가 폭증해 부하가 튀고(실측 5.7~18.6), 같은 기계를 쓰는 사람이 불편해진다.
    부하 숫자를 쫓아다니는 대신 **쓸 코어를 미리 떼어 두는 방식**을 쓴다.
    """

    path: str = "stella.runtime.cpu"
    name: str = "CpuBudget"
    torch_threads: int = 2  # 프로세스당 intra-op 스레드. 0이면 torch 기본(= 전체 코어)
    interop_threads: int = 1
    cores: str = ""  # "0-14" 처럼 주면 그 코어에만 붙는다. 비면 reserved_cores로 계산
    # 사람 몫으로 남길 코어 수. 32코어 중 17을 남겨 **이 프로젝트는 0~14 (15코어) 만 쓴다**
    # (사용자 지시 08-14). 같은 기계에서 사용자 본인 추론 작업이 함께 도는 것을 실측했다.
    reserved_cores: int = 17


@dataclass(kw_only=True)
class ExperimentConfig:
    """이것 자체는 build 대상이 아니다."""

    cpu: CpuConfig = field(default_factory=CpuConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    decode: DecodeConfig = field(default_factory=DecodeConfig)
    eval: MetricConfig = field(default_factory=MetricConfig)
    cell_diag: CellDiagConfig = field(default_factory=CellDiagConfig)
    log: LogConfig = field(default_factory=LogConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

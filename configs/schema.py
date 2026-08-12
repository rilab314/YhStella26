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
    class_freq_power: float = 0.0
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
    dir_loss: str = "cosine"


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
    radius: int = 24
    # 내 슬롯 방향과 실제 상대 방향의 코사인 하한 (c . u_ab). **w_dist와 함께 정해진다** —
    # 거리 항이 죽어 있던 동안에는 조이면 손해였다(0.7이 최적). w_dist=0.03에서는
    # 0.85~0.9가 평평한 최적이고 0.7 대비 f1 +4.7%다.
    align_thresh: float = 0.85
    opp_thresh: float = 0.7  # 마주봄 하한 — -(c . n) >= 이 값 (10.3절)
    w_opp: float = 1.0  # 후보 비용에서 마주봄 항의 비중
    # 후보 비용의 거리 항 계수(셀 단위 거리에 곱한다). **실측으로 정했다 — 최대 개선폭.**
    # 0.001은 반경 2 시절의 동률 해소용 상수였고(최대 기여 0.004), 반경 24에서도 0.034라
    # 정렬 항 (1-align, 0~0.3) 앞에서 무시됐다. 그 결과 **디코더가 정렬만 조금 나으면
    # 20셀을 건너뛰며 정점을 지나쳤다** — 한 선이 4~5조각으로 쪼개진 주원인이다.
    # 0.03에서 f1 +33% · precision +64% · chain_len +130% · frag −38%(캐시 4개 200장 일정).
    # 0.03~0.04가 평평한 최적. 10.3절이 배제한 w_dist=0.3은 반경 2에서 잰 값이라 무효다.
    w_dist: float = 0.03
    min_class_prob: float = 0.1  # 확장 게이트 — 후보의 사슬 클래스 softmax 확률 하한 (10.3절)
    # 정점의 배경 필터. 음수면 기존 방식(`argmax != 0`),
    # 0 이상이면 전경 로짓 확률의 하한을 쓴다 (E12).
    fg_thresh: float = -1.0
    purity_thresh: float = 0.6  # 사슬 순도 하한 — argmax 클래스 일치 비율. 이하면 사슬 폐기
    end_extend: float = 1.0  # 끝 셀에서 끝방향 슬롯으로 연장하는 길이(셀) — 10.3절 끝 연장
    min_points: int = 2  # 이보다 짧은 폴리라인은 버린다 (연장점 포함)
    simplify_tol: float = 0.0  # >0이면 RDP 단순화 (픽셀)
    # --- 알고리즘 변형 (가설 백로그 A). 기본값은 전부 기존 동작이다 ---
    seed_mode: str = "class_peak"  # "class_peak" | "end_peak" (선의 끝에서 시작, A5)
    stop_needs_nocand: bool = False  # True면 끝 확률 + 후보 없음을 모두 만족해야 정지 (A4)
    merge_gap: float = 0.0  # >0이면 끝점 간 이 거리(픽셀) 안의 조각을 병합 (A2)
    merge_align: float = 0.8  # 병합 정렬 하한 — 두 조각이 서로를 향하는 정도
    # "cosine" = 각도 게이트(기본) | "perp" = 예측 방향 직선에서의 수직 이탈 게이트 (A6)
    align_mode: str = "cosine"
    perp_thresh: float = 0.7  # perp 모드의 수직 이탈 상한 (셀 단위)


@dataclass(kw_only=True)
class MetricConfig(ModuleConfig):
    path: str = "stella.eval.ccq"
    name: str = "InstanceCCQ"
    buffer_rho: float = 12.0  # rho (픽셀). 차선 간격의 절반 이하 (11.1절)
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
    """셀 단위 진단 (improve-loop 스킬 · 셀 단위 진단). 임계값은 `decode`에서 가져다 쓴다."""

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
    cores: str = ""  # "0-21" 처럼 주면 그 코어에만 붙는다. 비면 reserved_cores로 계산
    reserved_cores: int = 10  # 사람 몫으로 남길 코어 수 (32코어 중 10 → 학습은 0~21만 쓴다)


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

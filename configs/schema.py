"""실험 config 스키마 — 모든 dataclass를 이 파일 하나에 둔다 (impl_plan 4.1).

이 모듈은 `stella` 패키지를 import 하지 않는다. config는 순수한 데이터이므로
코드 쪽으로 의존이 생기지 않고, 따라서 순환참조가 구조적으로 불가능하다.
"""

from dataclasses import dataclass, field


@dataclass(kw_only=True)
class ModuleConfig:
    """build_instance로 만드는 부품 config의 공통 베이스 (impl_plan 5절).

    하위 클래스가 두 필드에 기본값을 준다.
    """

    path: str  # 모듈 경로   예: "stella.model.backbone"
    name: str  # 클래스 이름 예: "Dinov3Backbone"


@dataclass(kw_only=True)
class DataConfig(ModuleConfig):
    path: str = "stella.data.seedmap"
    name: str = "SeedMapDataset"
    root: str = (
        "/media/humpback/435806fd-079f-4ba1-ad80-109c8f6e2ec0/Ongoing/2026_stella/SEED_MAP_v1.1"
    )
    image_size: int = 768  # SEED-MAP 원본 크기와 동일 — 리사이즈 없음
    grid_stride: int = 4  # 격자 배율 s. L = image_size // grid_stride = 192
    num_classes: int = 12  # 0 = background + 차선 11종 (6.7.1절)
    batch_size: int = 1
    num_workers: int = 8
    max_degree: int = 3  # D: 셀당 연결 이웃 저장 칸 수. num_conn_slots(R)와 같게
    encode_supersample: int = 1  # GT 래스터화 배율. 1 = 픽셀 해상도 (6.4절 A단계)
    cache_gt: str = "val_test"  # "none" | "val_test" | "all" (6.4.1절)
    # 데이터셋 폴더를 건드리지 않도록 캐시는 그 옆에 둔다. 빈 문자열이면 {root}/gt_cache.
    cache_dir: str = (
        "/media/humpback/435806fd-079f-4ba1-ad80-109c8f6e2ec0/Ongoing/2026_stella/gt_cache"
    )
    augment: bool = True  # 학습 split에만 적용 (6.7.6절)
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


@dataclass(kw_only=True)
class NeckConfig(ModuleConfig):
    path: str = "stella.model.neck"
    name: str = "FPNLite"  # "SFP"(ViT 단일 스케일) | "FPNLite"(멀티스케일)


@dataclass(kw_only=True)
class ModelConfig(ModuleConfig):
    path: str = "stella.model.stella"
    name: str = "StellaModel"
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    neck: NeckConfig = field(default_factory=NeckConfig)
    d_model: int = 256  # neck·블록·헤드가 공유 — 여기 한 곳에만 둔다
    num_heads: int = 8
    num_conn_slots: int = 3  # R = 3 (K = 4)
    layers: tuple[str, ...] = ("global", "window", "window", "window", "window", "window")
    window_size: int = 9  # w
    ffn_dim: int = 1024
    dropout: float = 0.0
    node_sampling: str = "gt+pred"  # "gt+pred"(기본) | "gt" (7.4절)
    # 전체 train 8979장 실측: 평균 2121, p90 3681, p99 5893, **최대 8909**.
    # 계획서의 8000(샘플 300장 기준)으로는 최대치를 못 담아 GT 셀이 잘린다.
    n_max: int = 9500
    heatmap_thresh: float = 0.3  # tau_h
    dilate: int = 3  # 예측 마스크 팽창: 0 | 3 | 5


@dataclass(kw_only=True)
class HeatmapLossConfig(ModuleConfig):
    path: str = "stella.loss.heatmap"
    name: str = "HeatmapLoss"
    w_heatmap: float = 1.0
    focal_alpha: float = 0.25  # 가중치가 아니라 focal 형태 파라미터
    focal_gamma: float = 2.0


@dataclass(kw_only=True)
class SelfSlotLossConfig(ModuleConfig):
    path: str = "stella.loss.self_slot"
    name: str = "SelfSlotLoss"
    w_class: float = 1.0
    w_coord: float = 1.0


@dataclass(kw_only=True)
class ConnLossConfig(ModuleConfig):
    path: str = "stella.loss.conn"
    name: str = "ConnLoss"
    w_exist: float = 1.0
    w_dir: float = 1.0
    w_t: float = 1.0
    match_w_dir: float = 1.0  # lambda_dir — 손실 가중치가 아니라 매칭 비용 계수 (8.3절)
    match_w_exist: float = 1.0  # lambda_e


@dataclass(kw_only=True)
class LossConfig(ModuleConfig):
    path: str = "stella.loss.criterion"
    name: str = "StellaCriterion"
    heatmap: HeatmapLossConfig = field(default_factory=HeatmapLossConfig)
    self_slot: SelfSlotLossConfig = field(default_factory=SelfSlotLossConfig)
    conn: ConnLossConfig = field(default_factory=ConnLossConfig)


@dataclass(kw_only=True)
class DecodeConfig(ModuleConfig):
    path: str = "stella.decode.graph"
    name: str = "GraphDecoder"
    heatmap_thresh: float = 0.3  # tau_h — 노드 후보 (추론 경로)
    exist_thresh: float = 0.5  # tau_e — 연결 슬롯 존재
    t_thresh: float = 0.5  # tau_t — 종점 판정
    # 아래 셋은 SEED-MAP val 6장에 GT를 주입한 스윕으로 정했다 (계획서 초기값은 3.0 / 0.3).
    # 반경 2.0: 3.0은 평행 차선의 후보를 너무 많이 끌어들여 간선 정확도 0.985 -> 0.967.
    # w_dist 0: 거리 항을 조금이라도 켜면 0.985 -> 0.953 으로 떨어진다. 방향이 유일한 신호다.
    max_conn_dist: float = 2.0  # 셀 단위. 연결 후보 탐색 반경
    cos_thresh: float = 0.7  # 방향 코사인 하한
    w_cos: float = 1.0
    w_dist: float = 0.0
    w_class: float = 1.0  # 클래스 불일치 (종점 슬롯은 면제, 10.3절)
    mutual: bool = True  # 양방향 확인 요구 여부 (10.4절)
    min_points: int = 2
    simplify_tol: float = 0.0  # >0이면 RDP 단순화 (픽셀)


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
    output_root: str = (
        "/media/humpback/435806fd-079f-4ba1-ad80-109c8f6e2ec0/Ongoing/2026_stella/log"
    )


@dataclass(kw_only=True)
class ExperimentConfig:
    """이것 자체는 build 대상이 아니다."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    decode: DecodeConfig = field(default_factory=DecodeConfig)
    eval: MetricConfig = field(default_factory=MetricConfig)
    log: LogConfig = field(default_factory=LogConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

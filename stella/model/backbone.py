"""백본 3층 구조 (design 7.2절).

    Backbone                      # 계약: forward(x) -> list[Tensor], out_channels, strides
    ├── HuggingFaceBackbone       # transformers 공통
    │   └── Dinov3Backbone
    └── TimmBackbone              # timm 공통
        ├── SwinBackbone
        ├── ConvNeXtBackbone
        └── TimmVitBackbone

계열 안의 스케일(L/B/S)은 `pretrained` 문자열이 정하고 클래스는 하나다.
`timm`·`transformers` import는 중간 인터페이스 클래스 안에서 한다 — 모듈 최상단에 두면
한쪽 라이브러리만 깔린 환경에서 `check_all`이 통째로 죽는다.
"""

import torch
from torch import nn

from stella.builder import Buildable

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class Backbone(nn.Module, Buildable):
    """계약만 정의한다. 정규화 상수를 백본이 들고 있으므로 데이터셋은 [0,1] RGB만 내놓으면 된다."""

    def __init__(self, *, pixel_mean: tuple[float, ...], pixel_std: tuple[float, ...]):
        super().__init__()
        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(1, 3, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(1, 3, 1, 1), False)

    @property
    def out_channels(self) -> tuple[int, ...]:
        raise NotImplementedError

    @property
    def strides(self) -> tuple[int, ...]:
        raise NotImplementedError

    def normalize(self, image: torch.Tensor) -> torch.Tensor:
        return (image - self.pixel_mean) / self.pixel_std

    def forward(self, image: torch.Tensor) -> list[torch.Tensor]:
        raise NotImplementedError

    def freeze_parameters(self) -> None:
        for param in self.parameters():
            param.requires_grad_(False)


class HuggingFaceBackbone(Backbone):
    """transformers 공통: AutoModel 로드, AutoImageProcessor에서 정규화 상수 추출."""

    def __init__(self, *, pretrained: str, freeze: bool):
        mean, std = self._load_normalization(pretrained)
        super().__init__(pixel_mean=mean, pixel_std=std)
        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(pretrained)
        self.patch_size = self._read_patch_size()
        if freeze:
            self.freeze_parameters()

    @staticmethod
    def _load_normalization(pretrained: str) -> tuple[tuple, tuple]:
        from transformers import AutoImageProcessor

        try:
            processor = AutoImageProcessor.from_pretrained(pretrained)
            return tuple(processor.image_mean), tuple(processor.image_std)
        except Exception:  # processor가 없는 저장소도 있다
            return IMAGENET_MEAN, IMAGENET_STD

    def _read_patch_size(self) -> int:
        config = self.model.config
        return int(getattr(config, "patch_size", 16))


class Dinov3Backbone(HuggingFaceBackbone):
    """ViT 패치 토큰 -> 1레벨 맵 (B, C, H/p, W/p)."""

    @property
    def out_channels(self) -> tuple[int, ...]:
        return (int(self.model.config.hidden_size),)

    @property
    def strides(self) -> tuple[int, ...]:
        return (self.patch_size,)

    def forward(self, image: torch.Tensor) -> list[torch.Tensor]:
        height, width = image.shape[-2:]
        output = self.model(pixel_values=self.normalize(image))
        tokens = output.last_hidden_state
        grid_h, grid_w = height // self.patch_size, width // self.patch_size
        patches = tokens[:, tokens.shape[1] - grid_h * grid_w :]
        return [patches.transpose(1, 2).reshape(image.shape[0], -1, grid_h, grid_w)]


class TimmBackbone(Backbone):
    """timm 공통: create_model(features_only=True) 로드, default_cfg에서 정규화 상수 추출.

    `out_indices` — 5레벨을 내는 백본(HRNet·ResNet·MaxViT 등)에서 FPNLite가 기대하는
    stride 4/8/16/32 네 레벨만 고른다. `img_size` — 고정 입력 크기 백본(Swin)에 실제 타일
    크기를 알려 준다(위치 임베딩은 timm이 보간한다).
    """

    def __init__(
        self,
        *,
        pretrained: str,
        freeze: bool,
        out_indices: tuple = (),
        img_size: int = 0,
        features_only: bool = True,
    ):
        import timm

        options = {"features_only": features_only, **self.extra_kwargs()}
        if out_indices:
            options["out_indices"] = tuple(int(index) for index in out_indices)
        if img_size:
            options["img_size"] = int(img_size)
        model = timm.create_model(pretrained, pretrained=True, **options)
        cfg = model.default_cfg if hasattr(model, "default_cfg") else model.pretrained_cfg
        super().__init__(pixel_mean=tuple(cfg["mean"]), pixel_std=tuple(cfg["std"]))
        self.model = model
        if freeze:
            self.freeze_parameters()

    @staticmethod
    def extra_kwargs() -> dict:
        return {}

    @property
    def out_channels(self) -> tuple[int, ...]:
        return tuple(self.model.feature_info.channels())

    @property
    def strides(self) -> tuple[int, ...]:
        return tuple(self.model.feature_info.reduction())

    def forward(self, image: torch.Tensor) -> list[torch.Tensor]:
        features = self.model(self.normalize(image))
        return [_to_nchw(f, c) for f, c in zip(features, self.out_channels)]


class ConvNeXtBackbone(TimmBackbone):
    """4레벨 (stride 4/8/16/32). CNN 대조군이자 게이트 없는 기본 백본."""


class SwinBackbone(TimmBackbone):
    """4레벨 (stride 4/8/16/32). timm swin은 NHWC로 내므로 `_to_nchw`가 정리한다.

    입력 크기가 고정된 계열이라 `img_size`를 반드시 준다 (768).
    """


class HrnetBackbone(TimmBackbone):
    """고해상도 병렬 분기 CNN — 얇은 선형 구조에 유리하다는 가설로 넣는다.

    stride 2 레벨을 함께 내므로 `out_indices=(1,2,3,4)`로 FPNLite 규격에 맞춘다.
    """


class TimmVitBackbone(TimmBackbone):
    """단일 스케일 ViT — SFP 경로 검증용 (게이트 없는 DINOv3 대역)."""

    def __init__(
        self, *, pretrained: str, freeze: bool, out_indices: tuple = (), img_size: int = 0
    ):
        super().__init__(pretrained=pretrained, freeze=freeze, features_only=False)
        self.patch_size = int(self.model.patch_embed.patch_size[0])
        self.embed_dim = int(self.model.embed_dim)

    @staticmethod
    def extra_kwargs() -> dict:
        return {"num_classes": 0, "dynamic_img_size": True}

    @property
    def out_channels(self) -> tuple[int, ...]:
        return (self.embed_dim,)

    @property
    def strides(self) -> tuple[int, ...]:
        return (self.patch_size,)

    def forward(self, image: torch.Tensor) -> list[torch.Tensor]:
        height, width = image.shape[-2:]
        tokens = self.model.forward_features(self.normalize(image))
        grid_h, grid_w = height // self.patch_size, width // self.patch_size
        if tokens.dim() == 4:  # dynamic_img_size는 NHWC로 낸다
            return [tokens.permute(0, 3, 1, 2).contiguous()]
        patches = tokens[:, tokens.shape[1] - grid_h * grid_w :]
        return [patches.transpose(1, 2).reshape(image.shape[0], -1, grid_h, grid_w)]


def _to_nchw(feature: torch.Tensor, channels: int) -> torch.Tensor:
    if feature.shape[1] == channels:
        return feature
    if feature.shape[-1] == channels:
        return feature.permute(0, 3, 1, 2).contiguous()
    raise ValueError(
        f"백본 출력에서 채널 축을 못 찾았다: shape={tuple(feature.shape)}, C={channels}"
    )

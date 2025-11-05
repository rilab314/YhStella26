# predictor.py
import torch
import numpy as np
from typing import Any, Dict, List, Optional, Union
import sys, os
sys.path.append(os.path.abspath('/workspace/SatelliteDet/SatelliteDet2025'))

torch.set_float32_matmul_precision("medium")

from configs.config import CfgNode
from util.misc import build_instance, NestedTensor
from util.target_logit_visualizer import TargetLogitVisualizer


class Predictor:
    def __init__(self, cfg: Any, model: torch.nn.Module, device: Optional[torch.device] = None, use_amp: bool = False):
        """역할: 이미 생성된 모델을 지정 device로 올리고 eval 모드로 전환."""
        self.cfg = cfg
        self.model = model.to(device if device is not None else torch.device(cfg.runtime.device)).eval()
        self.device = next(self.model.parameters()).device
        self.use_amp = use_amp

    @classmethod
    def from_cfg(
        cls,
        cfg: Union[str, Any],
        ckpt_path: Optional[str] = None,
        state_dict: Optional[Dict[str, torch.Tensor]] = None,
        map_location: Union[str, torch.device] = "cpu",
        strict: bool = False,
    ) -> "Predictor":
        """역할: cfg 로드 → core_model 빌드 → state_dict 로드 → Predictor 반환."""
        model = build_instance(cfg.lightning_model.module_name, cfg.lightning_model.class_name, cfg)

        if ckpt_path:
            raw = torch.load(ckpt_path, map_location=map_location)
            sd = raw.get("state_dict", raw)
        elif state_dict is not None:
            sd = state_dict
        else:
            raise ValueError("ckpt_path 또는 state_dict 중 하나는 필요합니다.")

        missing, unexpected = model.load_state_dict(sd, strict=strict)
        if missing or unexpected:
            print(f"[load_state_dict] missing={len(missing)} unexpected={len(unexpected)}")

        use_amp = bool(getattr(cfg.runtime, "amp", False))
        device = torch.device(cfg.runtime.device)
        return cls(cfg, model, device=device, use_amp=use_amp)

    @torch.no_grad()
    def predict(
        self,
        images: Union[np.ndarray, torch.Tensor, List[Union[np.ndarray, torch.Tensor]], Dict[str, torch.Tensor]],
        apply_softmax: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """역할: 입력 전처리 → (B,C,H,W) 배치 구성 → 모델 순전파 → (옵션) softmax."""
        if isinstance(images, dict):
            img = images.get("img", None)
            if img is None:
                raise ValueError("images dict는 'img': (B,C,H,W) 텐서를 포함해야 합니다.")
            batch = img.to(self.device) if torch.is_tensor(img) else img
        else:
            batch = self._to_batch(images).to(self.device, non_blocking=True)

        samples = self._to_samples(batch)
        if self.use_amp:
            with torch.autocast(device_type=str(self.device).split(':')[0], dtype=torch.float16):
                outputs = self.model(samples)
        else:
            outputs = self.model(samples)

        if apply_softmax and isinstance(outputs, dict) and "pred_logits" in outputs:
            outputs["pred_probs"] = outputs["pred_logits"].softmax(-1)
        return outputs

    @staticmethod
    def _to_tensor(x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """역할: 단일 이미지를 (C, H, W) float32 [0..1] 텐서로 변환."""
        if isinstance(x, np.ndarray):
            t = torch.from_numpy(x)
            t = (t.float() / 255.0) if t.dtype == torch.uint8 else t.float()
            return t.permute(2, 0, 1).contiguous()
        if isinstance(x, torch.Tensor):
            if x.ndim == 3 and x.shape[-1] == 3:
                return x.permute(2, 0, 1).contiguous().float()
            if x.ndim == 3 and x.shape[0] in (1, 3):
                return x.float()
            raise ValueError("torch.Tensor 입력은 (C,H,W) 또는 (H,W,3)이어야 합니다.")
        raise TypeError("지원하지 않는 입력 타입입니다.")

    def _to_batch(self, imgs: Union[np.ndarray, torch.Tensor, List[Union[np.ndarray, torch.Tensor]]]) -> torch.Tensor:
        """역할: 단일/리스트 입력을 배치 텐서로 변환."""
        if isinstance(imgs, (np.ndarray, torch.Tensor)):
            return self._to_tensor(imgs).unsqueeze(0)
        if isinstance(imgs, list):
            return torch.stack([self._to_tensor(x) for x in imgs], dim=0)
        raise TypeError("배치 입력 타입이 올바르지 않습니다.")

    def _to_samples(self, batch: torch.Tensor) -> NestedTensor:
        """역할: (B,C,H,W) → NestedTensor(B,C,H,W, mask) 변환."""
        if batch.ndim == 3:
            batch = batch.unsqueeze(0)
        b, _, h, w = batch.shape
        mask = torch.zeros((b, h, w), dtype=torch.bool, device=batch.device)
        return NestedTensor(batch, mask)


if __name__ == "__main__":
    import cv2
    import os
    from tqdm import tqdm

    ckpt_dir = '/workspace/SatelliteDet/tblog_251031/checkpoints'
    ckpt_list = os.listdir(ckpt_dir)
    img_dir = '/workspace/SatelliteDet/dataset/satellite_lane/validation/image'
    img_list = [os.path.join(img_dir, i) for i in os.listdir(img_dir)]

    for ckpt_path in ckpt_list:
        ckpt_path = os.path.join(ckpt_dir, ckpt_path)
        save_path = ckpt_path.split('-')[0] if 'last' not in ckpt_path else ckpt_path.replace('.ckpt', '')
        if not os.path.isfile(ckpt_path):
            continue
        os.makedirs(save_path, exist_ok=True)

        cfg = CfgNode.from_file("satellite_detr")
        print(ckpt_path, 'pred')
        predictor = Predictor.from_cfg(cfg, ckpt_path=ckpt_path)

        for img_path in tqdm(img_list):
            img_bgr = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            output = predictor.predict(img_rgb, apply_softmax=True)
            output = output[0]
            visualizer = TargetLogitVisualizer(cfg.dataset.labels)
            result_img = visualizer.create_visualization_panel(output, 'output', img_bgr)
            cv2.imwrite(os.path.join(save_path, os.path.basename(img_path)), result_img)

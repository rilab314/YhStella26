# predictor.py
import torch
import numpy as np
from typing import Any, Dict, List, Optional, Union
import sys, os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

torch.set_float32_matmul_precision("medium")

from configs.config import CfgNode
from util.misc import build_instance
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

        if self.use_amp:
            with torch.autocast(device_type=str(self.device).split(':')[0], dtype=torch.float16):
                outputs = self.model(batch)
        else:
            outputs = self.model(batch)

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


def main():
    ckpt_path = '/home/gorilla/kyh_workspace/project/results/tblog_260117_2012/checkpoints/last.ckpt'
    img_path = '/home/gorilla/kyh_workspace/project/dataset/satellite_lane/validation/image'

    vis_type_list = ['output', 'arrow', 'accurracy']
    save_path = ckpt_path.replace('ckpt', '')
    os.makedirs(save_path+'_pt', exist_ok=True)
    for vis_type in vis_type_list:
        os.makedirs(save_path+'_'+vis_type, exist_ok=True)

    cfg = CfgNode.from_file("satellite_detr")
    predictor = Predictor.from_cfg(cfg, ckpt_path=ckpt_path)
    visualizer = TargetLogitVisualizer(cfg.dataset.labels)

    for img_name in os.listdir(img_path):
        img_bgr = cv2.imread(os.path.join(img_path, img_name))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        output = predictor.predict(img_rgb, apply_softmax=True)

        torch.save(output[0], os.path.join(save_path+'_pt', img_name.replace('.png', '.pt')))

        # for vis_type in vis_type_list:
        #     result_img = visualizer.create_visualization_panel(copy.deepcopy(output[0]), vis_type, img_bgr)
        #     cv2.imwrite(os.path.join(save_path+'_'+vis_type, img_name), result_img)



if __name__ == "__main__":
    import cv2
    import os
    from tqdm import tqdm
    import copy

    main()

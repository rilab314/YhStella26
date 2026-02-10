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
    ckpt_path = '/home/gorilla/kyh_workspace/project/results/log_260209_0750/checkpoints/last.ckpt'
    
    img_path = '/home/gorilla/kyh_workspace/project/dataset/satellite_lane/validation/image'
    gt_json_dir = '/home/gorilla/kyh_workspace/project/dataset/satellite_lane/validation/json'

    vis_type_list = ['output', 'arrow', 'accurracy']
    save_path = ckpt_path.replace('.ckpt', '') if 'last' in ckpt_path else ckpt_path.split('-')[0]
    pred_pt_dir = save_path + '_pt'
    pred_instance_dir = save_path + '_instance'
    os.makedirs(pred_pt_dir, exist_ok=True)
    os.makedirs(pred_instance_dir, exist_ok=True)
    for vis_type in vis_type_list:
        os.makedirs(save_path+'_'+vis_type, exist_ok=True)

    cfg = CfgNode.from_file("stella_cfg")
    predictor = Predictor.from_cfg(cfg, ckpt_path=ckpt_path)
    visualizer = TargetLogitVisualizer(cfg.dataset.labels)
    instance_generator = build_instance(cfg.postprocessors.line.module_name, cfg.postprocessors.line.class_name, cfg)

    print("[1/3] Running prediction and saving .pt/.json instances...")
    img_names = sorted([n for n in os.listdir(img_path) if n.lower().endswith(".png")])
    for img_name in tqdm(img_names, desc="predict"):
        img_bgr = cv2.imread(os.path.join(img_path, img_name))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        output = predictor.predict(img_rgb, apply_softmax=True)

        stem = img_name.replace('.png', '')
        torch.save(output[0], os.path.join(pred_pt_dir, stem + '.pt'))
        pred_instances = instance_generator([output[0]])[0]
        instance_generator.save_points_to_json(pred_instances, os.path.join(pred_instance_dir, stem + '.json'))

    print("[2/3] Running F1 sweep (calculate_f1 logic)...")
    from util.calculate_f1 import (
        build_label_maps,
        build_thresholds,
        evaluate_dataset,
        infer_pred_format,
        prf_from_counts,
    )
    import csv

    id2name, cat2id = build_label_maps(cfg.dataset.labels)
    img_h = int(getattr(cfg.dataset, "image_height", 768))
    img_w = int(getattr(cfg.dataset, "image_width", img_h))
    img_size = [img_w, img_h]
    thickness = 5
    iou_th = 0.3
    bbox_pad = max(3, thickness * 2)
    thresholds = build_thresholds(0.5, 0.05, 0.95)
    f1_out_dir = save_path + "_f1_results"
    os.makedirs(f1_out_dir, exist_ok=True)
    pred_format = infer_pred_format(pred_pt_dir)

    for conf_threshold in thresholds:
        total, class_total = evaluate_dataset(
            gt_dir=gt_json_dir,
            pred_dir=pred_pt_dir,
            pred_format=pred_format,
            labels=cfg.dataset.labels,
            cat2id=cat2id,
            img_size=img_size,
            thickness=thickness,
            iou_th=iou_th,
            bbox_pad=bbox_pad,
            conf_threshold=conf_threshold,
            limit=None,
            print_per_image=False,
            show_progress=True,
        )

        tp, fp, fn = total["tp"], total["fp"], total["fn"]
        p, r, f1 = prf_from_counts(tp, fp, fn)
        th_str = f"{conf_threshold:.2f}".rstrip("0").rstrip(".")
        csv_path = os.path.join(f1_out_dir, f"thres_{th_str}.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["scope", "id", "name", "tp", "fp", "fn", "precision", "recall", "f1"])
            writer.writerow(["micro", "-", "-", tp, fp, fn, f"{p:.6f}", f"{r:.6f}", f"{f1:.6f}"])
            for lid in sorted(class_total.keys()):
                ctp = class_total[lid]["tp"]
                cfp = class_total[lid]["fp"]
                cfn = class_total[lid]["fn"]
                cp, cr, cf1 = prf_from_counts(ctp, cfp, cfn)
                class_name = id2name.get(lid, f"class_{lid}")
                writer.writerow(["class", lid, class_name, ctp, cfp, cfn, f"{cp:.6f}", f"{cr:.6f}", f"{cf1:.6f}"])
        print(f"[SAVED] {csv_path}")

    print("[3/3] Running IoU summary (compare_iou logic)...")
    from util.compare_iou import (
        LABELS,
        build_file_triplets,
        evaluate_one_image,
        load_json,
        parse_gt_instances,
        parse_pred_instances,
        prf_from_counts as iou_prf_from_counts,
        save_metrics_csv,
    )
    from collections import defaultdict

    iou_out_dir = save_path + "_iou_results"
    os.makedirs(iou_out_dir, exist_ok=True)
    for thickness, iou_th in ((3, 0.3), (5, 0.3), (3, 0.4), (5, 0.4)):
        class_total = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
        bbox_pad = max(3, thickness * 2)
        triplets = build_file_triplets(img_path, gt_json_dir, pred_instance_dir, img_ext=".png")
        total = {"tp": 0, "fp": 0, "fn": 0}

        for stem, _, gt_path, pred_path in tqdm(triplets, desc=f"iou thk={thickness} th={iou_th}"):
            if not os.path.exists(gt_path):
                continue
            gt_data = load_json(gt_path)
            pred_data = load_json(pred_path) if os.path.exists(pred_path) else []
            gt_instances = parse_gt_instances(gt_data, img_size=img_size)
            pred_instances = parse_pred_instances(pred_data, img_size=img_size)
            micro, per_class = evaluate_one_image(
                gt_instances=gt_instances,
                pred_instances=pred_instances,
                img_size=img_size,
                thickness=thickness,
                iou_th=iou_th,
                bbox_pad=bbox_pad,
            )
            for lid, stats in per_class.items():
                class_total[lid]["tp"] += stats["tp"]
                class_total[lid]["fp"] += stats["fp"]
                class_total[lid]["fn"] += stats["fn"]
            total["tp"] += micro["tp"]
            total["fp"] += micro["fp"]
            total["fn"] += micro["fn"]

        tp, fp, fn = total["tp"], total["fp"], total["fn"]
        p, r, f1 = iou_prf_from_counts(tp, fp, fn)
        print("==============================")
        print(f"[TOTAL] IoU>={iou_th:.2f}, thickness={thickness}, bbox_pad={bbox_pad}")
        print(f"MICRO TP={tp} FP={fp} FN={fn} | P={p:.4f} R={r:.4f} F1={f1:.4f}")
        for lid in sorted(class_total.keys()):
            ctp = class_total[lid]["tp"]
            cfp = class_total[lid]["fp"]
            cfn = class_total[lid]["fn"]
            cp, cr, cf1 = iou_prf_from_counts(ctp, cfp, cfn)
            class_name = next((l["name"] for l in LABELS if l["id"] == lid), f"class_{lid}")
            print(
                f"[{lid:2d}] {class_name:30s} | "
                f"TP={ctp:6d} FP={cfp:6d} FN={cfn:6d} | "
                f"P={cp:.4f} R={cr:.4f} F1={cf1:.4f}"
            )

        csv_name = f"iou(thk:{thickness}, th:{iou_th}).csv"
        csv_path = os.path.join(iou_out_dir, csv_name)
        save_metrics_csv(csv_path, total=total, class_total=class_total)
        print(f"[SAVED] {csv_path}")



if __name__ == "__main__":
    import cv2
    import os
    from tqdm import tqdm

    main()

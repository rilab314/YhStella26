import os
from collections import defaultdict
import torch
import pytorch_lightning as pl
from torch.optim.lr_scheduler import StepLR
from typing import Dict

from util.misc import get_sizes_and_ids, build_instance
from util.target_logit_visualizer import TargetLogitVisualizer
import cv2
from model.instance_generator import GeneratePolylineInstances
from util.compare_iou import compute_iou_metrics


def match_name_keywords(n, name_keywords):
    out = False
    for b in name_keywords:
        if b in n:
            break
    return out


class LightningModel(pl.LightningModule):
    vlog_frame_interval = 50

    @staticmethod
    def build_from_cfg(cfg):
        model = build_instance(cfg.core_model.module_name, cfg.core_model.class_name, cfg)
        criterion = build_instance(cfg.criterion.module_name, cfg.criterion.class_name, cfg)
        postproc_cfg = cfg.postprocessors.to_dict()
        postprocessors = {}
        for key, val in postproc_cfg.items():
            postproc = build_instance(val['module_name'], val['class_name'], cfg)
            postprocessors[key] = postproc
        model = LightningModel(cfg, model, criterion)
        device = torch.device(cfg.runtime.device)
        model.to(device)
        return model

    def __init__(self, cfg, model=None, criterion=None, postprocessors=None):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.postprocessors = postprocessors
        self.cfg = cfg
        self.loss_weights = {k: v for k, v in cfg.losses.to_dict().items() if k.endswith('_loss')}
        self.save_hyperparameters(ignore=['model', 'criterion'])
        self.instance_generator = GeneratePolylineInstances(cfg.dataset.labels)
        self.visualizer = TargetLogitVisualizer(self.cfg.dataset.labels)
        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[LightningModel] Number of params: {n_parameters}")
        for name, module in self.model.named_modules():
            if name in ("backbone.0._model.norm", "backbone.0._model.head"):
                for p in module.parameters():
                    p.requires_grad = False
        self.val_results = None

    def forward(self, samples, auxin=None):
        return self.model(samples, auxin)

    def training_step(self, batch):
        samples, targets = batch
        outputs = self(samples)
        loss_dict = self.criterion(outputs, targets)
        for k, v in loss_dict.items():
            factor = self.loss_weights.get(k, 1.0)
            self.log(f"train_{k}", v * factor, prog_bar=False, batch_size=self.cfg.training.batch_size)
        total_loss = sum(loss_dict[k] * self.loss_weights.get(k, 0) for k in loss_dict)
        self.log(f"train_total_loss", total_loss, prog_bar=False, batch_size=self.cfg.training.batch_size)
        return total_loss

    def on_after_backward(self):
        # 역전파(backward) 직후에 실행되어 그래디언트 상태를 점검함
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is None:
                print(f"--- Unused Parameter: {name} ---")

    def validation_step(self, batch, batch_idx):
        samples, targets = batch
        outputs = self(samples, {'split': 'val', 'batch_idx': batch_idx, 'filename':targets[0]['filename']})
        loss_dict = self.criterion(outputs, targets)
        for k, v in loss_dict.items():
            factor = self.loss_weights.get(k, 1.0)
            self.log(f"val_{k}", v * factor, prog_bar=False, batch_size=self.cfg.training.batch_size, sync_dist=True)
        total_loss = sum(loss_dict[k] * self.loss_weights.get(k, 0) for k in loss_dict)
        self.log(f"val_total_loss", total_loss, prog_bar=False, batch_size=self.cfg.training.batch_size, sync_dist=True)
        _, line_instances = self.eval_step(outputs, targets, self.current_epoch)
        # TODO: how to compute loss between instance(model output) and np data(label)
        if (batch_idx + 1) % self.vlog_frame_interval == 0:
            self.save_visual_log(outputs[0], targets[0], line_instances[0], self.current_epoch)
            self.save_json_log(outputs[0], targets[0], self.current_epoch)
        return total_loss
    
    def save_visual_log(self, output, target, line_instance, epoch):
        image_name = target['filename']
        with_img = cv2.imread(image_name)

        visualized_image = self.visualizer.visualize(
            output,
            target,
            line_instance,
            with_img=with_img,
        )
        img_save_path = os.path.join(self.cfg.runtime.output_dir, 'vlog', f'ep{epoch}_{image_name.split("/")[-1]}')
        cv2.imwrite(img_save_path, visualized_image)
    
    def eval_step(self, outputs, targets, epoch):
        batch_total = {"tp": 0, "fp": 0, "fn": 0}
        batch_per_class: Dict[int, Dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
        line_instances = []

        for output, target in zip(outputs, targets):
            pred_instances = self.instance_generator([output])[0]
            target_instances = target['instances']
            micro, per_class = compute_iou_metrics(
                gt_data=target_instances,
                pred_data=pred_instances,
                img_size=[self.cfg.dataset.image_width, self.cfg.dataset.image_height],
                thickness=3,
                iou_th=0.3,
            )
            for k in ("tp", "fp", "fn"):
                batch_total[k] += int(micro[k])
            for lid, stats in per_class.items():
                batch_per_class[int(lid)]["tp"] += int(stats["tp"])
                batch_per_class[int(lid)]["fp"] += int(stats["fp"])
                batch_per_class[int(lid)]["fn"] += int(stats["fn"])

            line_instances.append({
                "pred_instances": pred_instances,
                "gt_instances": target_instances,
            })

        if self.val_results is None:
            self.val_results = {
                "total": dict(batch_total),
                "per_class": {int(lid): dict(stats) for lid, stats in batch_per_class.items()},
            }
        else:
            for k in ("tp", "fp", "fn"):
                self.val_results["total"][k] += int(batch_total[k])
            for lid, stats in batch_per_class.items():
                if lid not in self.val_results["per_class"]:
                    self.val_results["per_class"][lid] = {"tp": 0, "fp": 0, "fn": 0}
                self.val_results["per_class"][lid]["tp"] += int(stats["tp"])
                self.val_results["per_class"][lid]["fp"] += int(stats["fp"])
                self.val_results["per_class"][lid]["fn"] += int(stats["fn"])

        metrics = {
            "total": dict(batch_total),
            "per_class": {int(lid): dict(stats) for lid, stats in batch_per_class.items()},
        }
        return metrics, line_instances

    def save_json_log(self, output, target, epoch):
        json_name = target['filename']
        pred_instances = self.instance_generator([output])[0]
        json_save_path = os.path.join(self.cfg.runtime.output_dir, 'vlog_json', f'ep{epoch}_{json_name.split("/")[-1]}'.replace('.png', '.json'))
        self.instance_generator.save_points_to_json(pred_instances, json_save_path)

    def on_validation_epoch_start(self):
        label_ids = sorted({int(x["id"]) for x in self.cfg.dataset.labels if "id" in x})
        self.val_results = {
            "total": {"tp": 0, "fp": 0, "fn": 0},
            "per_class": {lid: {"tp": 0, "fp": 0, "fn": 0} for lid in label_ids},
        }

    def on_validation_epoch_end(self):
        if self.val_results is None:
            return

        label_ids = sorted({int(x["id"]) for x in self.cfg.dataset.labels if "id" in x})
        local_total = torch.tensor(
            [
                float(self.val_results["total"]["tp"]),
                float(self.val_results["total"]["fp"]),
                float(self.val_results["total"]["fn"]),
            ],
            device=self.device,
        )

        local_per_class = torch.zeros((len(label_ids), 3), dtype=torch.float32, device=self.device)
        for i, lid in enumerate(label_ids):
            stats = self.val_results["per_class"].get(lid, {"tp": 0, "fp": 0, "fn": 0})
            local_per_class[i, 0] = float(stats["tp"])
            local_per_class[i, 1] = float(stats["fp"])
            local_per_class[i, 2] = float(stats["fn"])

        gathered_total = self.all_gather(local_total)
        gathered_per_class = self.all_gather(local_per_class)

        if gathered_total.ndim == 1:
            global_total = gathered_total
        else:
            global_total = gathered_total.sum(dim=0)

        if gathered_per_class.ndim == 2:
            global_per_class = gathered_per_class
        else:
            global_per_class = gathered_per_class.sum(dim=0)

        tp, fp, fn = [int(x.item()) for x in global_total]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        self.log("val_tp", float(tp), prog_bar=False, sync_dist=False)
        self.log("val_fp", float(fp), prog_bar=False, sync_dist=False)
        self.log("val_fn", float(fn), prog_bar=False, sync_dist=False)
        self.log("val_precision", float(precision), prog_bar=True, sync_dist=False)
        self.log("val_recall", float(recall), prog_bar=True, sync_dist=False)
        self.log("val_f1", float(f1), prog_bar=True, sync_dist=False)

        if self.trainer.is_global_zero:
            print(f"[VAL] TP={tp} FP={fp} FN={fn} | P={precision:.4f} R={recall:.4f} F1={f1:.4f}")

    def configure_optimizers(self):
        lr = self.cfg.training.lr
        lr_backbone = self.cfg.training.lr_backbone
        lr_backbone_names = self.cfg.training.lr_backbone_names
        lr_linear_proj_names = self.cfg.training.lr_linear_proj_names
        lr_linear_proj_mult = self.cfg.training.lr_linear_proj_mult
        weight_decay = self.cfg.training.weight_decay
        lr_drop = self.cfg.training.lr_drop
        sgd = getattr(self.cfg.training, "sgd", False)

        param_dicts = [
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if not match_name_keywords(n, lr_backbone_names)
                       and not match_name_keywords(n, lr_linear_proj_names)
                       and p.requires_grad
                ],
                "lr": lr,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if match_name_keywords(n, lr_backbone_names) and p.requires_grad
                ],
                "lr": lr_backbone,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if match_name_keywords(n, lr_linear_proj_names) and p.requires_grad
                ],
                "lr": lr * lr_linear_proj_mult,
            },
        ]

        if sgd:
            optimizer = torch.optim.SGD(param_dicts, lr=lr, momentum=0.9, weight_decay=weight_decay)
        else:
            optimizer = torch.optim.AdamW(param_dicts, lr=lr, weight_decay=weight_decay)

        lr_scheduler = StepLR(optimizer, step_size=lr_drop, gamma=0.1)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "monitor": "val_total_loss"  # or "coco/AP"
            }
        }

    def setup(self, stage: str):
        if stage == "fit":
            os.makedirs(os.path.join(self.cfg.runtime.output_dir, 'vlog'), exist_ok=True)
            os.makedirs(os.path.join(self.cfg.runtime.output_dir, 'vlog_json'), exist_ok=True)

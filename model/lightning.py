import os
import json
import torch
import pytorch_lightning as pl
from torch.optim.lr_scheduler import StepLR
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from util.misc import get_sizes_and_ids, build_instance
from util.target_logit_visualizer import TargetLogitVisualizer
import cv2
from model.instance_generator import GeneratePolylineInstances


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
        
        # TODO: how to compute loss between instance(model output) and np data(label)
        if (batch_idx + 1) % self.vlog_frame_interval == 0:
            self.save_visual_log(outputs[0], targets[0], self.current_epoch)
            self.save_json_log(outputs[0], targets[0], self.current_epoch)

        return total_loss
    
    def save_visual_log(self, output, target, epoch):
        image_name = target['filename']
        with_img = cv2.imread(image_name)
        visualized_image = self.visualizer.visualize(output, target, with_img=with_img)
        img_save_path = os.path.join(self.cfg.runtime.output_dir, 'vlog', f'ep{epoch}_{image_name.split("/")[-1]}')
        cv2.imwrite(img_save_path, visualized_image)
    
    def save_json_log(self, output, target, epoch):
        json_name = target['filename']
        pred_instances = self.instance_generator(output)
        json_save_path = os.path.join(self.cfg.runtime.output_dir, 'vlog_json', f'ep{epoch}_{json_name.split("/")[-1]}'.replace('.png', '.json'))
        self.instance_generator.save_points_to_json(pred_instances, json_save_path)


    def on_validation_epoch_end(self):
        # TODO implement performance eval
        # TODO eval per frame performance
        self.model.parameters()
        
        pass

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

# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import torch
import torch.nn.functional as F
from torch import nn
from typing import List

from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid,
                       build_instance)

from .segmentation import (dice_loss, sigmoid_focal_loss)
import copy


class SegmentationCriterion(nn.Module):
    @staticmethod
    def build_from_cfg(cfg):
        matcher = build_instance(cfg.matcher.module_name, cfg.matcher.class_name, cfg)
        losses = [k for k, v in cfg.losses.to_dict().items() if 'loss' in k]
        gamma = getattr(cfg.losses, 'focal_gamma', 2.0)
        neg_smooth_scale = getattr(cfg.losses, 'neg_smooth_scale', 0.5)
        min_neg_w = getattr(cfg.losses, 'min_neg_w', 1e-3)
        min_alpha_pos = getattr(cfg.losses, 'min_alpha_pos', 1e-2)
        return SegmentationCriterion(
            num_classes=cfg.dataset.num_classes,
            matcher=matcher,
            loss_names=losses,
            focal_alpha=cfg.losses.focal_alpha,
            focal_gamma=gamma,
            neg_smooth_scale=neg_smooth_scale,
            min_neg_w=min_neg_w, 
            min_alpha_pos=min_alpha_pos
        )

    def __init__(self, num_classes, matcher, loss_names: List[str],
                 focal_alpha=0.25, focal_gamma: float = 2.0,
                 neg_smooth_scale: float = 0.5, min_neg_w: float = 1e-3, 
                 min_alpha_pos: float = None):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.loss_names = loss_names
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.neg_smooth_scale = neg_smooth_scale
        self.min_neg_w = min_neg_w
        self.min_alpha_pos = min_alpha_pos

    def forward(self, outputs, targets):
        """
        outputs: List[Dict[str, Tensor]]
            - 각 dict 안에 'pred_points', 'pred_headtail', 'pred_logits' 키 존재
            - 각각 shape: (N, C, 6), (N, 2), (N, 1)
        
        targets: List[Dict[str, Tensor]]
            - 각 dict 안에 'points', 'headtail', 'labels' 키 존재
            - 각각 shape: (N, 6), (N, 2), (N, 1)
        """
        loss = {loss_name: 0 for loss_name in self.loss_names}
        for i, (output, target) in enumerate(zip(outputs, targets)):
            if 'cls_loss' in self.loss_names:
                loss['cls_loss'] += self.classification_loss(output, target)

            mask = (target['segm_label'] > 0).squeeze(-1)
            target_loss = target.copy()
            output_loss = output.copy()
            for key in target:
                if key not in ['size', 'image_id', 'filename']:
                    target_loss[key] = target[key][mask]  # (H, W, C) -> (N,  C)
            for key in output:
                output_loss[key] = output[key][mask]

            straight_match = self.matcher(output_loss, target_loss)
            if 'end_loss' in self.loss_names:
                loss['end_loss'] += self.endness_loss(output_loss, target_loss, straight_match)
            if 'point_loss' in self.loss_names:
                loss['point_loss'] += self.point_loss(output_loss, target_loss, straight_match)

        return loss
    
    def classification_loss(self, output, target):
        y = target['segm_label'].squeeze(-1)
        p = torch.softmax(output['segm_logit'], dim=-1)
        H, W, C = p.shape
        classes = torch.arange(self.num_classes, device=p.device, dtype=y.dtype).view(1, 1, C)
        oh = (y.unsqueeze(-1) == classes).to(p.dtype)
        pt = (p * oh).sum(-1)
        focal = (1.0 - pt).clamp_min(0) ** self.focal_gamma
        loss_map = - focal * torch.log(pt.clamp_min(1e-8))
        ignore_mask = (y == 0)
        loss_map = loss_map * ignore_mask * 0.1 + loss_map * (ignore_mask == 0) * 0.9
        return loss_map.mean()

    def endness_loss(self, output, target, straight_match: torch.Tensor):
        stacked_original = torch.stack([output['left_end_logit'], output['right_end_logit']], dim=1)  # (N, 2, 2)
        stacked_swapped = torch.stack([output['right_end_logit'], output['left_end_logit']], dim=1)
        condition = (straight_match == 1).view(output['left_end_logit'].shape[0], 1, 1)
        aligned_logits = torch.where(condition, stacked_original, stacked_swapped)
        stacked_target = torch.stack([target['left_end'], target['right_end']], dim=1)
        loss = F.binary_cross_entropy_with_logits(aligned_logits, stacked_target, reduction='mean')
        return loss

    def point_loss(self, output, target, straight_match: torch.Tensor):
        """
        output['xxx_point']: (N,2)
        staight_match: (N,)
        """
        center_loss = F.smooth_l1_loss(output['center_point'], target['center_point'], reduction='mean')
        
        stacked_original = torch.stack([output['left_point'], output['right_point']], dim=1)  # (N, 2, 2)
        stacked_swapped = torch.stack([output['right_point'], output['left_point']], dim=1)
        condition = (straight_match == 1).view(output['center_point'].shape[0], 1, 1)  # (N, 1, 1)
        aligned_points = torch.where(condition, stacked_original, stacked_swapped)
        stacked_target = torch.stack([target['left_point'], target['right_point']], dim=1)
        side_loss = F.smooth_l1_loss(aligned_points, stacked_target, reduction='mean')
        return center_loss + side_loss

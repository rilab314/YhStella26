# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import copy
from typing import Optional, List
import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_

from model.ops.modules import MSDeformAttn
from util.print_util import print_model, print_data


class DeformableTransformerEncoderOnly(nn.Module):
    @staticmethod
    def build_from_cfg(cfg):    
        return DeformableTransformerEncoderOnly(
            num_classes=cfg.dataset.num_classes,
            d_model=cfg.transformer.hidden_dim,
            nhead=cfg.transformer.nheads,
            num_encoder_layers=cfg.transformer.enc_layers,
            dim_feedforward=cfg.transformer.dim_feedforward,
            dropout=cfg.transformer.dropout,
            num_feature_levels=cfg.transformer.num_feature_levels,
            enc_n_points=cfg.transformer.enc_n_points,
        )

    def __init__(self, num_classes: int, d_model=256, nhead=8, num_encoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu", num_feature_levels=4, enc_n_points=4):
        super().__init__()
        self.num_classes = num_classes
        self.d_model = d_model
        self.nhead = nhead

        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, enc_n_points)
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)
        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        normal_(self.level_embed)

    def forward(self, srcs, pos_embeds, masks=None):
        '''
        assume that cfg.backbone.output_layers=['layer1', 'layer2', 'layer3', 'layer4'],
        srcs: list of tensors, [[B, C, H/4, W/4], [B, C, H/8, W/8], [B, C, H/16, W/16], [B, C, H/32, W/32]], C=256
        masks: list of tensors, [[B, H/4, W/4], [B, H/8, W/8], [B, H/16, W/16], [B, H/32, W/32]]
        pos_embeds: list of tensors, [[B, C, H/4, W/4], [B, C, H/8, W/8], [B, C, H/16, W/16], [B, C, H/32, W/32]], C=128
        '''
        # prepare input for encoder
        src_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, pos_embed) in enumerate(zip(srcs, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)  # [B, C, H*W] -> [B, H*W, C]
            pos_embed = pos_embed.flatten(2).transpose(1, 2)  # [B, C, H*W] -> [B, H*W, C]
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)  # [B, H*W, C] + [C] -> [B, H*W, C]
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
        src_flatten = torch.cat(src_flatten, 1)  # [B, sum(H*W), C]
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)  # [B, sum(H*W), C]
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)  # [L, 2], L=num levels=4
        # level_start_index: [L,] [0, H0*W0, sum_i=0~1_Hi*Wi, sum_i=0~2_Hi*Wi, sum_i=0~3_Hi*Wi], e.g. [0, 9216, 11520, 12096]
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        if masks is not None:
            valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)  # [B, L, 2]
        else:
            valid_ratios = torch.tensor([1, 1], dtype=torch.float32, device=src_flatten.device)  # [2,]
            valid_ratios = torch.tile(valid_ratios, (src_flatten.shape[0], len(srcs), 1))  # [B, L, 2]

        # memory: [B, sum(H*W), C]
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten)
        return memory

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)  # [B,]
        valid_W = torch.sum(~mask[:, 0, :], 1)  # [B,]
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)  # [B, 2]
        return valid_ratio



class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        # self attention
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device), 
                                          indexing='ij')
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)

        return output


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
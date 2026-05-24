#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Binary IQA training with dynamic dual-branch adaptive feature fusion.

Dataset format expected:
  data_root/
    train/
      usable/
      unusable/
    val/
      usable/
      unusable/
    test/
      usable/
      unusable/

Label mapping is fixed as:
  usable   -> 1
  unusable -> 0

Example:
  python train_mendeley_binary_dynamic_alpha.py \
    --data_root data/mendeley_iqa_701515 \
    --ckpt_path checkpoints/best_resnet50_imagenet.pth \
    --out_dir runs_binary_iqa/dynamic_alpha_seed0 \
    --epochs 25 \
    --warmup_epochs 3 \
    --batch_size 32 \
    --lr 1e-4 \
    --seed 0
"""

import os
import json
import argparse
import random
from typing import List, Dict, Tuple, Optional

from tqdm import tqdm

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision
from torchvision import transforms

try:
    from sklearn.metrics import roc_auc_score
except Exception:
    roc_auc_score = None

# Safe load support for old checkpoints containing argparse.Namespace
try:
    torch.serialization.add_safe_globals([argparse.Namespace])
except Exception:
    pass


# -----------------------------
# Utilities
# -----------------------------
class ResNetBackbone(nn.Module):
    def __init__(self, base: torchvision.models.ResNet):
        super().__init__()
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

    def forward_stages(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        return f1, f2, f3, f4


# -----------------------------
# Lightweight modules
# -----------------------------
class DyT(nn.Module):
    """Dynamic Tanh replacement for LayerNorm."""
    def __init__(self, num_features: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        return self.gamma * torch.tanh(self.alpha * x) + self.beta


class MatmulFreeDense(nn.Module):
    """Ternary-quantized dense layer with STE."""
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        w_quant = self.weight.clamp(-1, 1).round()
        w_ste = self.weight + (w_quant - self.weight).detach()
        return F.linear(x, w_ste, self.bias)


class FourierKAN(nn.Module):
    """Lightweight Fourier-KAN activation layer."""
    def __init__(self, dim: int, grid_size: int = 8):
        super().__init__()
        self.dim = dim
        self.grid_size = grid_size
        self.a = nn.Parameter(torch.randn(dim, grid_size) * 0.1)
        self.b = nn.Parameter(torch.randn(dim, grid_size) * 0.1)
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        k = torch.arange(1, self.grid_size + 1, device=x.device).float()
        x_unsqueezed = x.unsqueeze(-1)
        kx = x_unsqueezed * k
        cos_term = torch.cos(kx)
        sin_term = torch.sin(kx)
        out = (self.a * cos_term + self.b * sin_term).sum(dim=-1)
        return self.scale * out


# -----------------------------
# Alpha generator
# -----------------------------
class AlphaGenerator(nn.Module):
    """
    Input: s4 and t4.
    Output: channel-level alpha in (0,1)^{C_l}.
    """
    def __init__(self, in_channels=2048, d_model=256, out_channels=2048, num_heads=4):
        super().__init__()
        self.out_channels = out_channels
        self.proj_src = nn.Conv2d(in_channels, d_model, kernel_size=1)
        self.proj_tgt = nn.Conv2d(in_channels, d_model, kernel_size=1)
        self.query_token = nn.Parameter(torch.randn(1, d_model) * 0.02)

        self.q_proj = MatmulFreeDense(d_model, d_model, bias=False)
        self.k_proj = MatmulFreeDense(d_model, d_model, bias=False)
        self.v_proj = MatmulFreeDense(d_model, d_model, bias=False)
        self.out_proj_attn = MatmulFreeDense(d_model, d_model, bias=False)

        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % num_heads == 0
        self.head_dim = d_model // num_heads

        self.norm1 = DyT(d_model)
        self.norm2 = DyT(d_model)

        self.ffn_linear1 = MatmulFreeDense(d_model, d_model * 4)
        self.kan = FourierKAN(d_model * 4)
        self.ffn_linear2 = MatmulFreeDense(d_model * 4, d_model)
        self.out_proj_alpha = MatmulFreeDense(d_model, out_channels)

    def forward(self, s4, t4):
        B = s4.size(0)
        src_tokens = self.proj_src(s4).flatten(2).transpose(1, 2)
        tgt_tokens = self.proj_tgt(t4).flatten(2).transpose(1, 2)

        K = torch.cat([src_tokens, tgt_tokens], dim=1)
        V = K
        Q = self.query_token.unsqueeze(0).expand(B, -1, -1)

        Q_proj = self.q_proj(Q)
        K_proj = self.k_proj(K)
        V_proj = self.v_proj(V)

        Q_proj = Q_proj.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K_proj = K_proj.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V_proj = V_proj.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(Q_proj, K_proj.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_out = torch.matmul(attn_weights, V_proj)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, 1, self.d_model)
        attn_out = self.out_proj_attn(attn_out)

        Q = Q + self.norm1(attn_out)

        ffn_out = self.ffn_linear1(Q)
        ffn_out = self.kan(ffn_out)
        ffn_out = self.ffn_linear2(ffn_out)
        Q = Q + self.norm2(ffn_out)

        alpha_logits = self.out_proj_alpha(Q).squeeze(1)
        alpha = torch.sigmoid(alpha_logits)
        return alpha


# -----------------------------
# Main model
# -----------------------------
class AdaptiveInterpDynamic(nn.Module):
    def __init__(self, num_classes: int, alpha_priors: List[float], channels: List[int]):
        super().__init__()
        src_resnet = torchvision.models.resnet50(weights=None)
        tgt_resnet = torchvision.models.resnet50(weights=None)
        self.src_bb = ResNetBackbone(src_resnet)
        self.tgt_bb = ResNetBackbone(tgt_resnet)

        self.alpha_generators = nn.ModuleList([
            AlphaGenerator(in_channels=2048, d_model=256, out_channels=channels[i])
            for i in range(4)
        ])
        self.channels = channels
        self.alpha_priors = alpha_priors

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x, return_alphas=False):
        with torch.no_grad():
            s1, s2, s3, s4 = self.src_bb.forward_stages(x)
        t1, t2, t3, t4 = self.tgt_bb.forward_stages(x)

        s_list = [s1, s2, s3, s4]
        t_list = [t1, t2, t3, t4]

        alphas = []
        fused = []
        for i in range(4):
            alpha = self.alpha_generators[i](s4, t4)
            alphas.append(alpha)
            alpha_reshaped = alpha.view(alpha.size(0), -1, 1, 1)
            f = t_list[i] + alpha_reshaped * (s_list[i] - t_list[i])
            fused.append(f)

        feat = self.pool(fused[3]).flatten(1)
        logits = self.fc(feat)

        if return_alphas:
            return logits, alphas
        return logits


# -----------------------------
# Loss
# -----------------------------

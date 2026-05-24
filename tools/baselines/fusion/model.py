#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import random
import copy
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

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

try:
    torch.serialization.add_safe_globals([argparse.Namespace])
except Exception:
    pass


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


class GatingFusion(nn.Module):
    def __init__(self, channels: List[int], reduction: int = 16):
        super().__init__()
        self.mlps = nn.ModuleList()
        for c in channels:
            hidden = max(c // reduction, 16)
            self.mlps.append(nn.Sequential(
                nn.Linear(c, hidden),
                nn.ReLU(inplace=True),
                nn.Linear(hidden, c),
                nn.Sigmoid()
            ))

    def forward(self, s_list, t_list):
        fused = []
        gates = []
        for i, (s, t) in enumerate(zip(s_list, t_list)):
            pooled = F.adaptive_avg_pool2d(s + t, 1).flatten(1)
            g = self.mlps[i](pooled).view(pooled.size(0), -1, 1, 1)
            f = g * s + (1.0 - g) * t
            fused.append(f)
            gates.append(g)
        return fused, gates


class LearnableInterpolation(nn.Module):
    def __init__(self, alpha_init: List[float]):
        super().__init__()
        init = torch.tensor(alpha_init, dtype=torch.float32).clamp(1e-4, 1 - 1e-4)
        self.alpha_logits = nn.Parameter(torch.log(init / (1 - init)))

    @property
    def alphas(self):
        return torch.sigmoid(self.alpha_logits)

    def forward(self, s_list, t_list):
        fused = []
        alphas = self.alphas
        for i, (s, t) in enumerate(zip(s_list, t_list)):
            a = alphas[i]
            fused.append(a * s + (1.0 - a) * t)
        return fused, alphas


class FixedInterpolation(nn.Module):
    def __init__(self, alpha_values: List[float]):
        super().__init__()
        assert len(alpha_values) == 4
        self.register_buffer("alphas", torch.tensor(alpha_values, dtype=torch.float32))

    def forward(self, s_list, t_list):
        fused = []
        for i, (s, t) in enumerate(zip(s_list, t_list)):
            a = self.alphas[i]
            fused.append(a * s + (1.0 - a) * t)
        return fused, self.alphas


class ResidualMixing(nn.Module):
    def forward(self, s_list, t_list):
        fused = []
        for s, t in zip(s_list, t_list):
            fused.append(s + t)
        return fused, None


class ConvAdapter(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 16)
        self.adapter = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )

    def forward(self, x):
        return x + self.adapter(x)


class AdapterAdaptation(nn.Module):
    def __init__(self, channels: List[int], reduction: int = 16):
        super().__init__()
        self.adapters = nn.ModuleList([
            ConvAdapter(c, reduction=reduction) for c in channels
        ])

    def forward(self, s_list, t_list):
        adapted = []
        for i, t in enumerate(t_list):
            adapted.append(self.adapters[i](t))
        return adapted, None


class FusionBaselineModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        fusion_mode: str,
        alpha_values: List[float],
        adapter_reduction: int = 16,
        gating_reduction: int = 16,
    ):
        super().__init__()

        src_resnet = torchvision.models.resnet50(weights=None)
        tgt_resnet = torchvision.models.resnet50(weights=None)

        self.src_bb = ResNetBackbone(src_resnet)
        self.tgt_bb = ResNetBackbone(tgt_resnet)

        self.fusion_mode = fusion_mode
        self.channels = [256, 512, 1024, 2048]

        if fusion_mode == "residual":
            self.fusion = ResidualMixing()
        elif fusion_mode == "fixed_interp":
            self.fusion = FixedInterpolation(alpha_values)
        elif fusion_mode == "learnable_interp":
            self.fusion = LearnableInterpolation(alpha_values)
        elif fusion_mode == "gating":
            self.fusion = GatingFusion(self.channels, reduction=gating_reduction)
        elif fusion_mode == "adapter":
            self.fusion = AdapterAdaptation(self.channels, reduction=adapter_reduction)
        else:
            raise ValueError(f"Unsupported fusion_mode: {fusion_mode}")

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x, return_aux=False):
        with torch.no_grad():
            s1, s2, s3, s4 = self.src_bb.forward_stages(x)

        t1, t2, t3, t4 = self.tgt_bb.forward_stages(x)

        s_list = [s1, s2, s3, s4]
        t_list = [t1, t2, t3, t4]

        fused, aux = self.fusion(s_list, t_list)

        feat = self.pool(fused[3]).flatten(1)
        logits = self.fc(feat)

        if return_aux:
            return logits, aux
        return logits



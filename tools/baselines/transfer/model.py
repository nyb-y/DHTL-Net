#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import random
from typing import Dict, Tuple, List

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


def sanitize_key(k: str) -> str:
    for p in ["module.", "model.", "backbone.", "encoder."]:
        if k.startswith(p):
            k = k[len(p):]
    return k


def load_encoder_from_ckpt(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        sd = ckpt["model"]
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict):
        sd = ckpt
    else:
        raise ValueError(f"Unsupported checkpoint type: {type(ckpt)}")

    sd = {sanitize_key(k): v for k, v in sd.items()}

    model_dict = model.state_dict()
    load_dict = {}

    for k, v in sd.items():
        if k.startswith("fc.") or k.startswith("classifier."):
            continue
        if k in model_dict and model_dict[k].shape == v.shape:
            load_dict[k] = v

    model_dict.update(load_dict)
    model.load_state_dict(model_dict)

    print(f"[INFO] Loaded encoder from: {ckpt_path}")
    print(f"[INFO] Loaded tensors: {len(load_dict)}")


def set_trainable_layers(model, mode, warmup=False):
    assert mode in ["frozen", "all", "l4", "l34", "l234"]

    for p in model.parameters():
        p.requires_grad = False

    for p in model.fc.parameters():
        p.requires_grad = True

    if warmup:
        return

    def unfreeze(module):
        for p in module.parameters():
            p.requires_grad = True

    if mode == "frozen":
        return
    elif mode == "all":
        for p in model.parameters():
            p.requires_grad = True
    elif mode == "l4":
        unfreeze(model.layer4)
    elif mode == "l34":
        unfreeze(model.layer3)
        unfreeze(model.layer4)
    elif mode == "l234":
        unfreeze(model.layer2)
        unfreeze(model.layer3)
        unfreeze(model.layer4)



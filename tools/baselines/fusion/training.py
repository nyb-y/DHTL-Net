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


from common import macro_f1_from_cm, bal_acc_from_cm, accuracy_from_cm, sensitivity_specificity_from_cm, confusion_matrix, roc_auc_score
from model import FusionBaselineModel

def sanitize_key(k: str) -> str:
    for p in ["module.", "model.", "backbone.", "encoder."]:
        if k.startswith(p):
            k = k[len(p):]
    return k


def load_source_weights(model: FusionBaselineModel, ckpt_path: str):
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

    src_state = model.src_bb.state_dict()
    filtered = {}

    for k, v in sd.items():
        if k.startswith("fc.") or k.startswith("classifier."):
            continue
        if k in src_state and src_state[k].shape == v.shape:
            filtered[k] = v

    model.src_bb.load_state_dict(filtered, strict=False)
    model.tgt_bb.load_state_dict(filtered, strict=False)

    for p in model.src_bb.parameters():
        p.requires_grad = False

    print(f"[INFO] Loaded source weights from: {ckpt_path}")
    print(f"[INFO] Loaded tensors: {len(filtered)}")


def set_trainable_params(model: FusionBaselineModel, warmup: bool, ft_mode: str):
    for p in model.src_bb.parameters():
        p.requires_grad = False

    for p in model.tgt_bb.parameters():
        p.requires_grad = False

    if not warmup:
        if ft_mode == "all":
            modules = [model.tgt_bb]
        elif ft_mode == "l4":
            modules = [model.tgt_bb.layer4]
        elif ft_mode == "l34":
            modules = [model.tgt_bb.layer3, model.tgt_bb.layer4]
        elif ft_mode == "l234":
            modules = [model.tgt_bb.layer2, model.tgt_bb.layer3, model.tgt_bb.layer4]
        else:
            raise ValueError(f"Unsupported ft_mode: {ft_mode}")

        for m in modules:
            for p in m.parameters():
                p.requires_grad = True

    for p in model.fusion.parameters():
        p.requires_grad = not warmup

    for p in model.fc.parameters():
        p.requires_grad = True


def count_trainable_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes):
    model.eval()
    ys, ps, probs = [], [], []
    total_loss = 0.0
    n = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)
        prob = torch.softmax(logits, dim=1)
        pred = torch.argmax(logits, dim=1)

        total_loss += loss.item() * x.size(0)
        n += x.size(0)
        ys.append(y.cpu().numpy())
        ps.append(pred.cpu().numpy())
        probs.append(prob[:, 1].detach().cpu().numpy())

    y_true = np.concatenate(ys)
    y_pred = np.concatenate(ps)
    y_prob = np.concatenate(probs)

    cm = confusion_matrix(y_true, y_pred, num_classes)
    sens, spec = sensitivity_specificity_from_cm(cm)

    auc = float("nan")
    if roc_auc_score is not None and len(np.unique(y_true)) == 2:
        auc = float(roc_auc_score(y_true, y_prob))

    return {
        "loss": total_loss / max(n, 1),
        "macro_f1": macro_f1_from_cm(cm),
        "balanced_acc": bal_acc_from_cm(cm),
        "accuracy": accuracy_from_cm(cm),
        "auc": auc,
        "sensitivity": sens,
        "specificity": spec,
        "cm": cm,
    }


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    n = 0

    pbar = tqdm(loader, desc="Training", leave=False)

    for x, y in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        n += x.size(0)

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / max(n, 1)



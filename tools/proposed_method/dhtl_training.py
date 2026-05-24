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
from dhtl_common import macro_f1_from_cm, bal_acc_from_cm, accuracy_from_cm, sensitivity_specificity_from_cm, confusion_matrix, roc_auc_score
from dhtl_model import AdaptiveInterpDynamic

def compute_loss(
    logits,
    targets,
    alphas,
    alpha_priors,
    class_weights=None,
    lambda_prior=0.1,
    lambda_smooth=0.05,
    lambda_ent=0.01,
):
    ce_loss = F.cross_entropy(logits, targets, weight=class_weights)

    prior_loss = 0.0
    smooth_loss = 0.0
    ent_loss = 0.0

    for i, alpha in enumerate(alphas):
        prior = torch.full_like(alpha, alpha_priors[i])
        prior_loss = prior_loss + F.mse_loss(alpha, prior)
        smooth_loss = smooth_loss + alpha.var(dim=1).mean()
        eps = 1e-8
        ent = -(alpha * torch.log(alpha + eps) + (1 - alpha) * torch.log(1 - alpha + eps))
        ent_loss = ent_loss + ent.mean()

    total_loss = ce_loss + lambda_prior * prior_loss + lambda_smooth * smooth_loss + lambda_ent * ent_loss
    return total_loss, ce_loss, prior_loss, smooth_loss, ent_loss


# -----------------------------
# Load source weights
# -----------------------------
def sanitize_key(k: str) -> str:
    for p in ["module.", "model.", "backbone.", "encoder."]:
        if k.startswith(p):
            k = k[len(p):]
    return k


def load_source_weights(model: AdaptiveInterpDynamic, ckpt_path: str):
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

    missing, unexpected = model.src_bb.load_state_dict(filtered, strict=False)
    model.tgt_bb.load_state_dict(filtered, strict=False)

    for p in model.src_bb.parameters():
        p.requires_grad = False

    print(f"Loaded source weights from: {ckpt_path}")
    print(f"Loaded tensors: {len(filtered)}")
    print(f"Missing keys in src_bb: {len(missing)}, unexpected: {len(unexpected)}")


# -----------------------------
# Trainable controls
# -----------------------------
def set_trainable_params(model: AdaptiveInterpDynamic, warmup: bool, ft_mode: str):
    # Source branch always frozen.
    for p in model.src_bb.parameters():
        p.requires_grad = False

    # Target branch freeze all first.
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

    # Alpha generators frozen during warmup.
    for p in model.alpha_generators.parameters():
        p.requires_grad = not warmup

    # Classification head always trainable.
    for p in model.fc.parameters():
        p.requires_grad = True


def count_trainable_params(model: nn.Module):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


# -----------------------------
# Train / evaluate
# -----------------------------
@torch.no_grad()
def evaluate(model, loader, device, num_classes, alpha_priors, class_weights,
             lambda_prior, lambda_smooth, lambda_ent):
    model.eval()
    ys, ps, probs = [], [], []
    total_loss = 0.0
    n = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits, alphas = model(x, return_alphas=True)
        loss, _, _, _, _ = compute_loss(
            logits, y, alphas, alpha_priors, class_weights,
            lambda_prior, lambda_smooth, lambda_ent
        )
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
    f1 = macro_f1_from_cm(cm)
    bacc = bal_acc_from_cm(cm)
    acc = accuracy_from_cm(cm)
    sens, spec = sensitivity_specificity_from_cm(cm)
    auc = float("nan")
    if roc_auc_score is not None and len(np.unique(y_true)) == 2:
        auc = float(roc_auc_score(y_true, y_prob))

    return {
        "loss": total_loss / max(n, 1),
        "macro_f1": f1,
        "balanced_acc": bacc,
        "accuracy": acc,
        "sensitivity": sens,
        "specificity": spec,
        "auc": auc,
        "cm": cm,
    }


def train_one_epoch(model, loader, optimizer, device, alpha_priors, class_weights,
                    lambda_prior, lambda_smooth, lambda_ent):
    model.train()
    total_loss = 0.0
    n = 0
    pbar = tqdm(loader, desc="Training", leave=False)
    for x, y in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits, alphas = model(x, return_alphas=True)
        loss, ce, pl, sl, el = compute_loss(
            logits, y, alphas, alpha_priors, class_weights,
            lambda_prior, lambda_smooth, lambda_ent
        )
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "ce": f"{ce.item():.4f}"})
    return total_loss / max(n, 1)


# -----------------------------
# Data loading
# -----------------------------

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
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True




def parse_int_list(s: str) -> List[int]:
    if s is None or str(s).strip() == "":
        return []
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]

def macro_f1_from_cm(cm: np.ndarray) -> float:
    C = cm.shape[0]
    f1s = []
    for c in range(C):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        denom = (2 * tp + fp + fn)
        f1 = (2 * tp / denom) if denom > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s))


def bal_acc_from_cm(cm: np.ndarray) -> float:
    C = cm.shape[0]
    recalls = []
    for c in range(C):
        denom = cm[c, :].sum()
        rec = (cm[c, c] / denom) if denom > 0 else 0.0
        recalls.append(rec)
    return float(np.mean(recalls))


def accuracy_from_cm(cm: np.ndarray) -> float:
    total = cm.sum()
    return float(np.trace(cm) / total) if total > 0 else 0.0


def sensitivity_specificity_from_cm(cm: np.ndarray) -> Tuple[float, float]:
    """For binary classes: 0=unusable, 1=usable."""
    if cm.shape != (2, 2):
        return float("nan"), float("nan")
    tn, fp = cm[0, 0], cm[0, 1]
    fn, tp = cm[1, 0], cm[1, 1]
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # usable recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # unusable recall
    return float(sensitivity), float(specificity)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


# -----------------------------
# Dataset
# -----------------------------

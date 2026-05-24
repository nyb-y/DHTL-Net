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


def parse_float_list(s: str) -> List[float]:
    s = str(s).strip().strip('"').strip("'")
    s = s.lstrip("[").rstrip("]")
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def macro_f1_from_cm(cm: np.ndarray) -> float:
    f1s = []
    for c in range(cm.shape[0]):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        denom = 2 * tp + fp + fn
        f1s.append(2 * tp / denom if denom > 0 else 0.0)
    return float(np.mean(f1s))


def bal_acc_from_cm(cm: np.ndarray) -> float:
    recalls = []
    for c in range(cm.shape[0]):
        denom = cm[c, :].sum()
        recalls.append(cm[c, c] / denom if denom > 0 else 0.0)
    return float(np.mean(recalls))


def accuracy_from_cm(cm: np.ndarray) -> float:
    return float(np.trace(cm) / cm.sum()) if cm.sum() > 0 else 0.0


def sensitivity_specificity_from_cm(cm: np.ndarray) -> Tuple[float, float]:
    tn, fp = cm[0, 0], cm[0, 1]
    fn, tp = cm[1, 0], cm[1, 1]
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return float(sensitivity), float(specificity)


def confusion_matrix(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm



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


from common import macro_f1_from_cm, bal_acc_from_cm, accuracy_from_cm, sensitivity_specificity_from_cm, confusion_matrix, roc_auc_score

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



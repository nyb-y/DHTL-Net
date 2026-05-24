#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Binary IQA training with dynamic dual-branch adaptive feature fusion.

Ablation variant: DHTL-Net w/o KMD.
Only the three lightweight/replacement modules are removed:
  1) DyT -> nn.LayerNorm
  2) MatmulFreeDense -> nn.Linear
  3) FourierKAN -> nn.GELU
All other architecture, loss, data loading, training, and evaluation settings remain unchanged.

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
class BinaryIQAFolderDataset(Dataset):
    """
    Reads a folder split with subdirectories: usable/ and unusable/.
    Fixed labels:
      unusable -> 0
      usable   -> 1
    """
    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    def __init__(self, split_dir: str, transform=None, return_path: bool = False):
        self.split_dir = split_dir
        self.transform = transform
        self.return_path = return_path
        self.class_to_idx = {"unusable": 0, "usable": 1}
        self.idx_to_class = {0: "unusable", 1: "usable"}
        self.samples = []

        for cls_name, cls_id in self.class_to_idx.items():
            cls_dir = os.path.join(split_dir, cls_name)
            if not os.path.isdir(cls_dir):
                raise FileNotFoundError(f"Missing class directory: {cls_dir}")
            for root, _, files in os.walk(cls_dir):
                for fn in files:
                    ext = os.path.splitext(fn)[1].lower()
                    if ext in self.IMG_EXTS:
                        self.samples.append((os.path.join(root, fn), cls_id))

        if len(self.samples) == 0:
            raise RuntimeError(f"No images found under {split_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, y = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        if self.return_path:
            return img, y, img_path
        return img, y

    def class_counts(self) -> Dict[str, int]:
        counts = {"unusable": 0, "usable": 0}
        for _, y in self.samples:
            counts[self.idx_to_class[y]] += 1
        return counts


# -----------------------------
# ResNet Backbone
# -----------------------------

def make_loaders(data_root, batch_size, num_workers, seed, use_weighted_sampler=False):
    tf_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(8),
        transforms.ColorJitter(brightness=0.10, contrast=0.10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tf_eval = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")
    test_dir = os.path.join(data_root, "test")

    ds_tr = BinaryIQAFolderDataset(train_dir, transform=tf_train)
    ds_va = BinaryIQAFolderDataset(val_dir, transform=tf_eval)
    ds_te = BinaryIQAFolderDataset(test_dir, transform=tf_eval)

    print("Dataset counts:")
    print("  Train:", ds_tr.class_counts())
    print("  Val:  ", ds_va.class_counts())
    print("  Test: ", ds_te.class_counts())

    g = torch.Generator()
    g.manual_seed(seed)

    sampler = None
    shuffle = True
    if use_weighted_sampler:
        labels = [y for _, y in ds_tr.samples]
        counts = np.bincount(labels, minlength=2).astype(np.float32)
        weights_per_class = 1.0 / np.maximum(counts, 1.0)
        sample_weights = [weights_per_class[y] for y in labels]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        shuffle = False
        print("Using WeightedRandomSampler. Class counts:", counts.tolist())

    dl_tr = DataLoader(
        ds_tr, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
        num_workers=num_workers, pin_memory=True, generator=g
    )
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    dl_te = DataLoader(ds_te, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    label2id = {"unusable": 0, "usable": 1}
    classes = ["unusable", "usable"]
    return dl_tr, dl_va, dl_te, label2id, classes, ds_tr.class_counts()


def make_class_weights(train_counts: Dict[str, int], device):
    # Weight order: [unusable, usable]
    n0 = train_counts.get("unusable", 0)
    n1 = train_counts.get("usable", 0)
    total = n0 + n1
    if total == 0 or n0 == 0 or n1 == 0:
        return None
    w0 = total / (2.0 * n0)
    w1 = total / (2.0 * n1)
    return torch.tensor([w0, w1], dtype=torch.float32, device=device)



# -----------------------------
# Profiling and qualitative outputs

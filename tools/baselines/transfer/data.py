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


class BinaryIQAFolderDataset(Dataset):
    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    def __init__(self, split_dir, transform=None, return_path: bool = False):
        self.split_dir = split_dir
        self.transform = transform
        self.return_path = return_path
        self.class_to_idx = {"unusable": 0, "usable": 1}
        self.idx_to_class = {0: "unusable", 1: "usable"}
        self.samples = self.collect_samples_from_class_root(split_dir, self.class_to_idx, self.IMG_EXTS)

        if len(self.samples) == 0:
            raise RuntimeError(f"No images found under {split_dir}")

    @staticmethod
    def collect_samples_from_class_root(class_root: str, class_to_idx: Dict[str, int], exts):
        samples = []
        for cls_name, cls_id in class_to_idx.items():
            cls_dir = os.path.join(class_root, cls_name)
            if not os.path.isdir(cls_dir):
                raise FileNotFoundError(f"Missing class directory: {cls_dir}")

            for root, _, files in os.walk(cls_dir):
                for fn in files:
                    ext = os.path.splitext(fn)[1].lower()
                    if ext in exts:
                        samples.append((os.path.join(root, fn), cls_id))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, y = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        if self.return_path:
            return img, y, path
        return img, y

    def class_counts(self) -> Dict[str, int]:
        counts = {"unusable": 0, "usable": 0}
        for _, y in self.samples:
            counts[self.idx_to_class[y]] += 1
        return counts

    def add_samples_from_class_root(self, class_root: str):
        self.samples.extend(
            self.collect_samples_from_class_root(class_root, self.class_to_idx, self.IMG_EXTS)
        )

    def add_samples(self, samples):
        self.samples.extend(samples)


def split_extra_samples_by_class(samples, seed, train_ratio, val_ratio, test_ratio):
    ratios = np.array([train_ratio, val_ratio, test_ratio], dtype=np.float64)
    if np.any(ratios < 0):
        raise ValueError("Split ratios must be non-negative.")
    if not np.isclose(ratios.sum(), 1.0):
        raise ValueError("Split ratios must sum to 1.0.")

    rng = np.random.default_rng(seed)
    by_class = {}
    for path, y in samples:
        by_class.setdefault(y, []).append((path, y))

    split_train, split_val, split_test = [], [], []

    for class_samples in by_class.values():
        class_samples = list(class_samples)
        rng.shuffle(class_samples)
        n = len(class_samples)

        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        if n_train + n_val > n:
            n_val = max(0, n - n_train)
        n_test = n - n_train - n_val

        split_train.extend(class_samples[:n_train])
        split_val.extend(class_samples[n_train:n_train + n_val])
        split_test.extend(class_samples[n_train + n_val:n_train + n_val + n_test])

    return split_train, split_val, split_test


def make_loaders(
    data_root,
    batch_size,
    num_workers,
    seed,
    use_weighted_sampler=False,
    train_aug="default",
    extra_train_root=None,
    extra_data_root=None,
    extra_split_train=0.7,
    extra_split_val=0.15,
    extra_split_test=0.15,
    test_root_override=None,
):
    tf_eval = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    if train_aug == "none":
        tf_train = tf_eval
    else:
        tf_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(8),
            transforms.ColorJitter(brightness=0.10, contrast=0.10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    ds_tr = BinaryIQAFolderDataset(os.path.join(data_root, "train"), transform=tf_train)
    ds_va = BinaryIQAFolderDataset(os.path.join(data_root, "val"), transform=tf_eval)
    if test_root_override:
        ds_te = BinaryIQAFolderDataset(test_root_override, transform=tf_eval)
    else:
        ds_te = BinaryIQAFolderDataset(os.path.join(data_root, "test"), transform=tf_eval)

    base_train_counts = ds_tr.class_counts()
    base_val_counts = ds_va.class_counts()
    base_test_counts = ds_te.class_counts()
    extra_train_counts = None
    extra_val_counts = None
    extra_test_counts = None

    if extra_train_root:
        extra_probe = BinaryIQAFolderDataset(extra_train_root, transform=tf_train)
        extra_train_counts = extra_probe.class_counts()
        ds_tr.add_samples_from_class_root(extra_train_root)

    if extra_data_root:
        extra_samples = BinaryIQAFolderDataset.collect_samples_from_class_root(
            extra_data_root, ds_tr.class_to_idx, ds_tr.IMG_EXTS
        )
        extra_tr, extra_va, extra_te = split_extra_samples_by_class(
            extra_samples,
            seed=seed,
            train_ratio=extra_split_train,
            val_ratio=extra_split_val,
            test_ratio=extra_split_test,
        )
        extra_train_counts = BinaryIQAFolderDataset(extra_data_root, transform=tf_train).class_counts()
        extra_val_counts = {"unusable": 0, "usable": 0}
        extra_test_counts = {"unusable": 0, "usable": 0}
        extra_split_train_counts = {"unusable": 0, "usable": 0}
        for _, y in extra_tr:
            extra_split_train_counts[ds_tr.idx_to_class[y]] += 1
        for _, y in extra_va:
            extra_val_counts[ds_tr.idx_to_class[y]] += 1
        for _, y in extra_te:
            extra_test_counts[ds_tr.idx_to_class[y]] += 1
        extra_train_counts = extra_split_train_counts
        ds_tr.add_samples(extra_tr)
        ds_va.add_samples(extra_va)
        ds_te.add_samples(extra_te)

    print("Dataset counts:")
    print("  Train:", ds_tr.class_counts())
    print("  Val:  ", ds_va.class_counts())
    print("  Test: ", ds_te.class_counts())

    if test_root_override:
        print("Test override:")
        print("  Base test:", base_test_counts)
        print("  Override test:", ds_te.class_counts())

    if extra_train_root:
        print("Extra training data:")
        print("  Base train:", base_train_counts)
        print("  Extra train:", extra_train_counts)
        print("  Merged train:", ds_tr.class_counts())

    if extra_data_root:
        print("Extra split data:")
        print("  Base train:", base_train_counts)
        print("  Base val:  ", base_val_counts)
        print("  Base test: ", base_test_counts)
        print("  Extra train:", extra_train_counts)
        print("  Extra val:  ", extra_val_counts)
        print("  Extra test: ", extra_test_counts)
        print("  Merged train:", ds_tr.class_counts())
        print("  Merged val:  ", ds_va.class_counts())
        print("  Merged test: ", ds_te.class_counts())

    sampler = None
    shuffle = True

    if use_weighted_sampler:
        labels = [y for _, y in ds_tr.samples]
        counts = np.bincount(labels, minlength=2).astype(np.float32)
        weights_per_class = 1.0 / np.maximum(counts, 1.0)
        sample_weights = [weights_per_class[y] for y in labels]

        g_sampler = torch.Generator()
        g_sampler.manual_seed(seed)

        sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
            generator=g_sampler,
        )
        shuffle = False

        print("Using WeightedRandomSampler.")
        print("Class counts:", counts.tolist())

    g = torch.Generator()
    g.manual_seed(seed)

    dl_tr = DataLoader(
        ds_tr,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        generator=g,
    )

    dl_va = DataLoader(
        ds_va,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    dl_te = DataLoader(
        ds_te,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return dl_tr, dl_va, dl_te, ds_tr.class_counts()


def make_class_weights(train_counts, device):
    n0 = train_counts.get("unusable", 0)
    n1 = train_counts.get("usable", 0)
    total = n0 + n1

    if total == 0 or n0 == 0 or n1 == 0:
        return None

    w0 = total / (2.0 * n0)
    w1 = total / (2.0 * n1)
    return torch.tensor([w0, w1], dtype=torch.float32, device=device)



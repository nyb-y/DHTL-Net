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
def try_compute_flops_and_params(model: nn.Module, device: torch.device, input_size: int = 224):
    """
    Computes approximate forward FLOPs for one image.
    Requires optional package:
      pip install thop
    If thop is unavailable, returns NaN values and training continues normally.
    """
    try:
        from thop import profile
        model.eval()
        dummy = torch.randn(1, 3, input_size, input_size, device=device)
        with torch.no_grad():
            flops, params = profile(model, inputs=(dummy,), verbose=False)
        return float(flops), float(params)
    except Exception as e:
        print(f"[WARN] FLOPs computation skipped: {e}")
        print("[WARN] If you need FLOPs, run: pip install thop")
        return float("nan"), float("nan")


@torch.no_grad()
def profile_batchsize_latency_vram(
    model: nn.Module,
    device: torch.device,
    batch_sizes: List[int],
    input_size: int,
    warmup_iters: int,
    profile_iters: int,
):
    """
    Measure inference latency and peak CUDA memory for selected batch sizes.
    This CSV supports:
      1) batch size vs latency/time
      2) VRAM (MB) vs latency
    """
    rows = []
    model.eval()

    for bs in batch_sizes:
        x = torch.randn(bs, 3, input_size, input_size, device=device)

        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

            for _ in range(warmup_iters):
                _ = model(x)
            torch.cuda.synchronize()

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(profile_iters):
                _ = model(x)
            end.record()
            torch.cuda.synchronize()

            total_ms = start.elapsed_time(end)
            peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        else:
            import time
            for _ in range(warmup_iters):
                _ = model(x)
            t0 = time.perf_counter()
            for _ in range(profile_iters):
                _ = model(x)
            total_ms = (time.perf_counter() - t0) * 1000.0
            peak_mb = float("nan")

        latency_ms_per_batch = total_ms / max(profile_iters, 1)
        row = {
            "batch_size": int(bs),
            "latency_ms_per_batch": float(latency_ms_per_batch),
            "latency_ms_per_image": float(latency_ms_per_batch / bs),
            "vram_peak_mb": float(peak_mb),
            "input_size": int(input_size),
            "profile_iters": int(profile_iters),
            "warmup_iters": int(warmup_iters),
        }
        rows.append(row)
        print(
            f"[PROFILE] batch_size={bs} | "
            f"latency={latency_ms_per_batch:.3f} ms/batch | "
            f"{latency_ms_per_batch/bs:.3f} ms/image | "
            f"peak_vram={peak_mb:.1f} MB"
        )

    return rows


def denormalize_to_uint8(t: torch.Tensor) -> Image.Image:
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=t.dtype, device=t.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=t.dtype, device=t.device).view(3, 1, 1)
    x = (t * std + mean).clamp(0, 1)
    x = (x.detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(x)


@torch.no_grad()
def save_qualitative_probability_examples(
    model: nn.Module,
    data_root: str,
    out_dir: str,
    device: torch.device,
    n_per_class: int = 2,
    input_size: int = 512,
    dpi: int = 600,
):
    """
    Save two usable and two unusable test images with predicted probabilities overlaid.
    Images are saved as high-quality JPG files with 600 DPI metadata.
    """
    qdir = os.path.join(out_dir, "qualitative_examples")
    os.makedirs(qdir, exist_ok=True)

    tf_eval = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_dir = os.path.join(data_root, "test")
    ds = BinaryIQAFolderDataset(test_dir, transform=tf_eval, return_path=True)
    idx_to_class = {0: "unusable", 1: "usable"}

    picked = {0: [], 1: []}
    for idx, (_, y) in enumerate(ds.samples):
        if len(picked[y]) < n_per_class:
            picked[y].append(idx)
        if all(len(v) >= n_per_class for v in picked.values()):
            break

    def get_font(size: int = 22):
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        ]
        for fp in candidates:
            if os.path.exists(fp):
                return ImageFont.truetype(fp, size=size)
        return ImageFont.load_default()

    font = get_font(size=22)
    small_font = get_font(size=20)

    model.eval()
    saved = []
    for y_cls, idxs in picked.items():
        for k, idx in enumerate(idxs):
            x, y, path = ds[idx]
            logits = model(x.unsqueeze(0).to(device))
            prob = F.softmax(logits, dim=1)[0].detach().cpu().numpy()
            pred = int(prob.argmax())

            img = denormalize_to_uint8(x).convert("RGB")
            draw = ImageDraw.Draw(img)
            true_name = idx_to_class[int(y)]
            pred_name = idx_to_class[pred]

            line1 = f"True: {true_name} | Pred: {pred_name}"
            line2 = f"P(unusable)={prob[0]:.3f}   P(usable)={prob[1]:.3f}"

            pad_x, pad_y = 12, 8
            line_gap = 4
            bbox1 = draw.textbbox((0, 0), line1, font=font)
            bbox2 = draw.textbbox((0, 0), line2, font=small_font)
            h1 = bbox1[3] - bbox1[1]
            h2 = bbox2[3] - bbox2[1]
            box_h = pad_y * 2 + h1 + line_gap + h2

            draw.rectangle([0, 0, img.width, box_h], fill=(0, 0, 0))
            draw.text((pad_x, pad_y), line1, fill=(255, 255, 255), font=font)
            draw.text((pad_x, pad_y + h1 + line_gap), line2, fill=(255, 255, 255), font=small_font)
            draw.rectangle([0, 0, img.width - 1, img.height - 1], outline=(255, 255, 255), width=2)

            save_name = (
                f"{true_name}_{k+1}_pred-{pred_name}_"
                f"punusable{prob[0]:.3f}_pusable{prob[1]:.3f}_512dpi{dpi}.jpg"
            )
            save_path = os.path.join(qdir, save_name)
            img.save(
                save_path,
                format="JPEG",
                quality=95,
                subsampling=0,
                dpi=(dpi, dpi),
                optimize=True,
            )
            saved.append(save_path)

    print(f"[INFO] Saved qualitative examples to: {qdir}")
    for item in saved:
        print(f"       {item}")

    return saved

# -----------------------------
# Main
# -----------------------------

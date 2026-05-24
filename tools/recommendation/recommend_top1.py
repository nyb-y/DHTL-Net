#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute scan-level Top-1 Recommendation Accuracy for all seed0 methods.

Modified version:
  1. Uses server paths by default.
  2. Adds progress bars.
  3. Only saves one summary CSV for 12 methods:
       runs_binary_iqa/top1_rec_acc_final_dataset/top1_rec_acc_all_methods_seed0.csv
  4. Does not save per-method frame scores, JSON files, or recommended-frame images.
"""

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms

from train_DAF import BinaryIQAFolderDataset, set_seed


METHOD_SPECS = [
    {"method_id": "DAF", "method_label": "DHTL-Net", "run_dir": "DAF_seed0", "family": "daf"},
    {"method_id": "DHTL_wo_KMD", "method_label": "DHTL-Net w/o KMD", "run_dir": "DHTL_wo_KMD_seed0", "family": "wo_kmd"},

    {"method_id": "fusion_residual", "method_label": "Residual", "run_dir": "fusion_residual_seed0", "family": "fusion", "fusion_mode": "residual"},
    {"method_id": "fusion_fixed_interp", "method_label": "FixedInterp", "run_dir": "fusion_fixed_interp_seed0", "family": "fusion", "fusion_mode": "fixed_interp"},
    {"method_id": "fusion_learnable_interp", "method_label": "LearnableInterp", "run_dir": "fusion_learnable_interp_seed0", "family": "fusion", "fusion_mode": "learnable_interp"},
    {"method_id": "fusion_gating", "method_label": "Gating", "run_dir": "fusion_gating_seed0", "family": "fusion", "fusion_mode": "gating"},
    {"method_id": "fusion_adapter", "method_label": "Adapter", "run_dir": "fusion_adapter_seed0", "family": "fusion", "fusion_mode": "adapter"},

    {"method_id": "transfer_frozen", "method_label": "Frozen", "run_dir": "transfer_frozen_seed0", "family": "transfer", "transfer_mode": "frozen"},
    {"method_id": "transfer_all", "method_label": "All", "run_dir": "transfer_all_seed0", "family": "transfer", "transfer_mode": "all"},
    {"method_id": "transfer_l4", "method_label": "FT-S4", "run_dir": "transfer_l4_seed0", "family": "transfer", "transfer_mode": "l4"},
    {"method_id": "transfer_l34", "method_label": "FT-S3-S4", "run_dir": "transfer_l34_seed0", "family": "transfer", "transfer_mode": "l34"},
    {"method_id": "transfer_l234", "method_label": "FT-S2-S4", "run_dir": "transfer_l234_seed0", "family": "transfer", "transfer_mode": "l234"},
]


def find_dir_by_name(search_root: str, dirname: str) -> Optional[str]:
    if not os.path.isdir(search_root):
        return None

    for root, dirs, _ in os.walk(search_root):
        if dirname in dirs:
            return os.path.join(root, dirname)

    return None


def find_file_by_name(search_root: str, filename: str) -> Optional[str]:
    if not os.path.isdir(search_root):
        return None

    matches = []

    for root, _, files in os.walk(search_root):
        if filename in files:
            matches.append(os.path.join(root, filename))

    if not matches:
        return None

    preferred = [
        path for path in matches
        if "runs_plane" in os.path.normpath(path).lower().split(os.sep)
    ]

    if preferred:
        return sorted(preferred, key=len)[0]

    return sorted(matches, key=len)[0]


def resolve_default_paths(args) -> None:
    archive1_root = os.path.normpath(args.archive1_root)

    if args.data_root is None:
        args.data_root = find_dir_by_name(archive1_root, "final_dataset")

    if args.train_data_root is None:
        args.train_data_root = find_dir_by_name(archive1_root, "mendeley_iqa_701515_grouped900")

    if args.source_ckpt_path is None:
        args.source_ckpt_path = find_file_by_name(archive1_root, "best_resnet50_imagenet.pth")

    missing = []

    for name in ["data_root", "train_data_root", "source_ckpt_path"]:
        value = getattr(args, name)

        if not value:
            missing.append(name)
        elif name.endswith("_root") and not os.path.isdir(value):
            missing.append(f"{name}={value}")
        elif name.endswith("_path") and not os.path.isfile(value):
            missing.append(f"{name}={value}")

    if missing:
        raise FileNotFoundError(
            "Could not resolve required paths: "
            + ", ".join(missing)
            + ". Pass them explicitly with --data_root, --train_data_root, and --source_ckpt_path."
        )

    args.data_root = os.path.normpath(args.data_root)
    args.train_data_root = os.path.normpath(args.train_data_root)
    args.source_ckpt_path = os.path.normpath(args.source_ckpt_path)
    args.code_root = os.path.normpath(args.code_root)
    args.runs_root = os.path.normpath(args.runs_root)
    args.out_dir = os.path.normpath(args.out_dir)


def build_eval_transform(input_size: int):
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def denormalize_to_uint8(t: torch.Tensor) -> Image.Image:
    mean = torch.tensor(
        [0.485, 0.456, 0.406],
        dtype=t.dtype,
        device=t.device,
    ).view(3, 1, 1)

    std = torch.tensor(
        [0.229, 0.224, 0.225],
        dtype=t.dtype,
        device=t.device,
    ).view(3, 1, 1)

    x = (t * std + mean).clamp(0, 1)
    x = (x.detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    return Image.fromarray(x)


def clean_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {
        k: v for k, v in state_dict.items()
        if not (k.endswith("total_ops") or k.endswith("total_params"))
    }


def parse_float_list_arg(value, default: List[float]) -> List[float]:
    if value is None:
        return default

    if isinstance(value, list):
        return [float(x) for x in value]

    text = str(value).strip().strip("[]")

    if not text:
        return default

    return [float(x.strip()) for x in text.split(",") if x.strip()]


def load_checkpoint_model(ckpt_path: str, device: torch.device, method_spec: Dict):
    ckpt = torch.load(ckpt_path, map_location="cpu")

    if "model_state_dict" not in ckpt:
        raise KeyError(f"Checkpoint does not contain model_state_dict: {ckpt_path}")

    state_dict = clean_state_dict(ckpt["model_state_dict"])
    family = method_spec["family"]

    if family == "daf":
        from train_DAF import AdaptiveInterpDynamic

        model = AdaptiveInterpDynamic(
            num_classes=2,
            alpha_priors=[0.85, 0.75, 0.55, 0.30],
            channels=[256, 512, 1024, 2048],
        )

    elif family == "wo_kmd":
        from run_DHTL_wo_KMD import AdaptiveInterpDynamic

        model = AdaptiveInterpDynamic(
            num_classes=2,
            alpha_priors=[0.85, 0.75, 0.55, 0.30],
            channels=[256, 512, 1024, 2048],
        )

    elif family == "fusion":
        from baselines import FusionBaselineModel

        ckpt_args = ckpt.get("args", {})

        model = FusionBaselineModel(
            num_classes=2,
            fusion_mode=method_spec["fusion_mode"],
            alpha_values=parse_float_list_arg(
                ckpt_args.get("alpha_values"),
                [0.85, 0.75, 0.55, 0.30],
            ),
            adapter_reduction=int(ckpt_args.get("adapter_reduction", 16)),
            gating_reduction=int(ckpt_args.get("gating_reduction", 16)),
        )

    elif family == "transfer":
        model = torchvision.models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)

    else:
        raise ValueError(f"Unsupported method family: {family}")

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    return model, ckpt


def strip_label_from_mha_id(value: str) -> str:
    value = os.path.basename(value.replace("\\", "/"))

    value = re.sub(
        r"\.(png|jpg|jpeg|bmp|tif|tiff)$",
        "",
        value,
        flags=re.IGNORECASE,
    )

    match = re.search(
        r"(.+?\.mha)(?:_[pn])?(?:_\d+)?$",
        value,
        flags=re.IGNORECASE,
    )

    return match.group(1) if match else value


def infer_original_mha_id(path: str) -> str:
    basename = os.path.basename(path)

    if basename.startswith("extra__"):
        match = re.match(r"^extra__(.+?)__", basename)
        if match:
            return strip_label_from_mha_id(match.group(1))

    match = re.search(
        r"([^\\/]+?\.mha)(?:_[pn])?(?:_\d+)?\.[^.]+$",
        path,
        flags=re.IGNORECASE,
    )

    if match:
        return match.group(1)

    return strip_label_from_mha_id(basename)


def infer_acquisition_id(
    path: str,
    split_dir: str,
    group_regex: Optional[str],
    filename_tokens_to_drop: int,
    group_mode: str,
) -> str:
    if group_mode == "original_mha":
        return infer_original_mha_id(path)

    rel = os.path.relpath(path, split_dir)
    parts = rel.split(os.sep)

    basename = os.path.basename(path)
    stem, _ = os.path.splitext(basename)

    if group_regex:
        match = re.search(group_regex, basename)

        if match is None:
            match = re.search(group_regex, stem)

        if match is None:
            match = re.search(group_regex, rel.replace("\\", "/"))

        if not match:
            raise ValueError(f"Path does not match --group_regex: {path}")

        return match.group(1) if match.groups() else match.group(0)

    if len(parts) > 2:
        return "/".join(parts[1:-1])

    tokens = re.split(r"[_\-\s]+", stem)

    if len(tokens) > filename_tokens_to_drop:
        return "_".join(tokens[:-filename_tokens_to_drop])

    return stem


def effective_area_score(image: Image.Image, threshold: int = 12) -> float:
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    return float((gray > threshold).mean()) if gray.size > 0 else 0.0


def to_gray_float(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("L"), dtype=np.float32)


def psnr_against_reference(image: Image.Image, reference: Image.Image) -> float:
    x = to_gray_float(image)
    y = to_gray_float(reference)

    mse = float(np.mean((x - y) ** 2))

    if mse <= 1e-12:
        return 100.0

    return float(20.0 * math.log10(255.0 / math.sqrt(mse)))


def ssim_against_reference(image: Image.Image, reference: Image.Image) -> float:
    x = to_gray_float(image)
    y = to_gray_float(reference)

    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2

    mu_x = float(x.mean())
    mu_y = float(y.mean())

    sigma_x = float(x.var())
    sigma_y = float(y.var())
    sigma_xy = float(((x - mu_x) * (y - mu_y)).mean())

    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)

    if abs(denominator) <= 1e-12:
        return 0.0

    return float(max(min(numerator / denominator, 1.0), -1.0))


def minmax_normalize(values: List[float], fill_value: float = 1.0) -> List[float]:
    arr = np.asarray(values, dtype=np.float32)

    if arr.size == 0:
        return []

    vmin = float(arr.min())
    vmax = float(arr.max())

    if vmax - vmin <= 1e-12:
        return [float(fill_value)] * len(values)

    return [float(x) for x in (arr - vmin) / (vmax - vmin)]


@torch.no_grad()
def collect_frame_predictions(
    model,
    split_dir: str,
    input_size: int,
    device: torch.device,
    group_regex: Optional[str],
    filename_tokens_to_drop: int,
    source_mode: str,
    group_mode: str,
) -> pd.DataFrame:
    dataset = BinaryIQAFolderDataset(
        split_dir,
        transform=build_eval_transform(input_size),
        return_path=True,
    )

    rows = []

    for idx in tqdm(
        range(len(dataset)),
        desc="Frame prediction",
        unit="frame",
        dynamic_ncols=True,
    ):
        x, y, path = dataset[idx]

        basename = os.path.basename(path)
        is_extra = basename.startswith("extra__")

        if source_mode == "extra" and not is_extra:
            continue

        if source_mode == "base" and is_extra:
            continue

        logits = model(x.unsqueeze(0).to(device))
        prob = F.softmax(logits, dim=1)[0].detach().cpu().numpy()

        image = denormalize_to_uint8(x).convert("RGB")

        acquisition_id = infer_acquisition_id(
            path=path,
            split_dir=split_dir,
            group_regex=group_regex,
            filename_tokens_to_drop=filename_tokens_to_drop,
            group_mode=group_mode,
        )

        rows.append({
            "path": path,
            "true_label": int(y),
            "usable_prob": float(prob[1]),
            "unusable_prob": float(prob[0]),
            "acquisition_id": acquisition_id,
            "image": image,
            "source_mode": "extra" if is_extra else "base",
        })

    if not rows:
        raise RuntimeError(f"No frames selected from {split_dir} with source_mode={source_mode}")

    return pd.DataFrame(rows)


def add_quality_priors(
    df: pd.DataFrame,
    w_prob: float,
    w_qres: float,
    w_qssim: float,
    w_qpsnr: float,
) -> pd.DataFrame:
    output_rows = []

    grouped = df.groupby("acquisition_id", sort=True)

    for _, group in tqdm(
        grouped,
        desc="Quality prior scoring",
        unit="scan",
        dynamic_ncols=True,
    ):
        group = group.copy()

        pseudo_ref_idx = int(group["usable_prob"].idxmax())
        pseudo_ref = group.loc[pseudo_ref_idx, "image"]

        group["qres"] = minmax_normalize([
            effective_area_score(img)
            for img in group["image"]
        ])

        group["qssim"] = minmax_normalize([
            ssim_against_reference(img, pseudo_ref)
            for img in group["image"]
        ])

        group["qpsnr"] = minmax_normalize([
            psnr_against_reference(img, pseudo_ref)
            for img in group["image"]
        ])

        denom = max(w_prob + w_qres + w_qssim + w_qpsnr, 1e-12)

        group["recommendation_score"] = (
            w_prob * group["usable_prob"]
            + w_qres * group["qres"]
            + w_qssim * group["qssim"]
            + w_qpsnr * group["qpsnr"]
        ) / denom

        output_rows.append(group)

    return pd.concat(output_rows, ignore_index=True)


def compute_top1_accuracy(scored_df: pd.DataFrame) -> Tuple[int, float]:
    top1_correct = []
    num_acquisitions = 0

    grouped = scored_df.groupby("acquisition_id", sort=True)

    for _, group in tqdm(
        grouped,
        desc="Computing Top-1 accuracy",
        unit="scan",
        dynamic_ncols=True,
    ):
        group = group.sort_values(
            by=["recommendation_score", "usable_prob", "qres", "qssim", "qpsnr"],
            ascending=False,
        )

        best = group.iloc[0]
        top1_correct.append(int(best["true_label"] == 1))
        num_acquisitions += 1

    top1_acc = float(np.mean(top1_correct)) if len(top1_correct) > 0 else float("nan")

    return num_acquisitions, top1_acc


def resolve_methods(methods_arg: str) -> List[Dict]:
    if methods_arg.strip().lower() == "all":
        return METHOD_SPECS

    wanted = {item.strip() for item in methods_arg.split(",") if item.strip()}

    specs = [
        spec for spec in METHOD_SPECS
        if spec["method_id"] in wanted or spec["run_dir"] in wanted
    ]

    found = {spec["method_id"] for spec in specs} | {spec["run_dir"] for spec in specs}
    missing = sorted(wanted - found)

    if missing:
        raise ValueError(f"Unknown methods: {', '.join(missing)}")

    return specs


def missing_row(method_spec: Dict, status: str, error: str) -> Dict:
    return {
        "method_id": method_spec["method_id"],
        "method_label": method_spec["method_label"],
        "status": status,
        "num_frames": 0,
        "num_acquisitions": 0,
        "top1_recommendation_accuracy": float("nan"),
        "error": error,
    }


def build_retrain_command(method_spec: Dict, args, run_dir: str) -> List[str]:
    common = [
        "--data_root", args.train_data_root,
        "--ckpt_path", args.source_ckpt_path,
        "--out_dir", run_dir,
        "--epochs", str(args.train_epochs),
        "--warmup_epochs", str(args.train_warmup_epochs),
        "--batch_size", str(args.train_batch_size),
        "--lr", str(args.train_lr),
        "--num_workers", str(args.train_num_workers),
        "--seed", str(args.seed),
        "--train_aug", "none",
        "--scheduler", "none",
        "--profile_batch_sizes", "",
    ]

    if method_spec["family"] == "fusion":
        return [
            sys.executable,
            os.path.join(args.code_root, "baselines.py"),
            *common,
            "--fusion_mode", method_spec["fusion_mode"],
            "--ft_mode", args.train_ft_mode,
            "--alpha_values", "[0.85,0.75,0.55,0.30]",
            "--save_ckpt",
        ]

    if method_spec["family"] == "transfer":
        return [
            sys.executable,
            os.path.join(args.code_root, "tradition.py"),
            *common,
            "--mode", method_spec["transfer_mode"],
        ]

    raise ValueError(f"Retraining is not defined for family={method_spec['family']}")


def persist_checkpoint_to_seed0(run_dir: str, ckpt_path: str) -> str:
    seed0_ckpt_path = os.path.join(run_dir, "best_model.pth")

    os.makedirs(run_dir, exist_ok=True)

    if os.path.exists(seed0_ckpt_path):
        return seed0_ckpt_path

    if os.path.exists(ckpt_path):
        if os.path.normcase(os.path.normpath(ckpt_path)) != os.path.normcase(os.path.normpath(seed0_ckpt_path)):
            shutil.copy2(ckpt_path, seed0_ckpt_path)

        return seed0_ckpt_path

    raise FileNotFoundError(f"Checkpoint not found for seed0 directory: {seed0_ckpt_path}")


def ensure_checkpoint(method_spec: Dict, args, run_dir: str, ckpt_path: str) -> None:
    seed0_ckpt_path = os.path.join(run_dir, "best_model.pth")

    if os.path.exists(seed0_ckpt_path):
        if os.path.exists(ckpt_path):
            return

        if os.path.normcase(os.path.normpath(ckpt_path)) != os.path.normcase(os.path.normpath(seed0_ckpt_path)):
            shutil.copy2(seed0_ckpt_path, ckpt_path)

        return

    if not args.retrain_missing:
        return

    if method_spec["family"] not in {"fusion", "transfer"}:
        return

    os.makedirs(run_dir, exist_ok=True)

    cmd = build_retrain_command(method_spec, args, run_dir)
    log_path = os.path.join(run_dir, "retrain_seed0.log")

    print(
        f"[TRAIN] {method_spec['method_id']} missing best_model.pth; log: {log_path}",
        flush=True,
    )

    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write("Command:\n")
        log_file.write(
            " ".join(
                f'"{part}"' if " " in str(part) else str(part)
                for part in cmd
            )
            + "\n\n"
        )
        log_file.flush()

        completed = subprocess.run(
            cmd,
            cwd=args.code_root,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )

    if completed.returncode != 0:
        raise RuntimeError(
            f"Training failed for {method_spec['method_id']}. See log: {log_path}"
        )

    persisted_path = persist_checkpoint_to_seed0(run_dir, ckpt_path)

    if not os.path.exists(persisted_path):
        raise RuntimeError(
            f"Training finished but checkpoint was not created in seed0 dir: {persisted_path}"
        )


def evaluate_method(method_spec: Dict, args, device: torch.device, split_dir: str) -> Dict:
    run_dir = os.path.join(args.runs_root, method_spec["run_dir"])
    seed0_ckpt_path = os.path.join(run_dir, "best_model.pth")
    ckpt_path = args.ckpt_path or seed0_ckpt_path

    ensure_checkpoint(method_spec, args, run_dir, ckpt_path)

    if not os.path.exists(seed0_ckpt_path):
        return missing_row(
            method_spec=method_spec,
            status="missing_checkpoint",
            error=f"Missing checkpoint: {seed0_ckpt_path}",
        )

    ckpt_path = persist_checkpoint_to_seed0(run_dir, ckpt_path)

    model, _ = load_checkpoint_model(
        ckpt_path=ckpt_path,
        device=device,
        method_spec=method_spec,
    )

    group_regex = args.group_regex if args.group_mode in {"regex", "copied_group"} else None

    frame_df = collect_frame_predictions(
        model=model,
        split_dir=split_dir,
        input_size=args.input_size,
        device=device,
        group_regex=group_regex,
        filename_tokens_to_drop=args.filename_tokens_to_drop,
        source_mode=args.source_mode,
        group_mode=args.group_mode,
    )

    scored_df = add_quality_priors(
        df=frame_df,
        w_prob=args.w_prob,
        w_qres=args.w_qres,
        w_qssim=args.w_qssim,
        w_qpsnr=args.w_qpsnr,
    )

    num_acquisitions, top1_acc = compute_top1_accuracy(scored_df)

    row = {
        "method_id": method_spec["method_id"],
        "method_label": method_spec["method_label"],
        "status": "ok",
        "num_frames": int(len(scored_df)),
        "num_acquisitions": int(num_acquisitions),
        "top1_recommendation_accuracy": float(top1_acc),
        "error": "",
    }

    return row


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--archive1_root", type=str, default="data")
    parser.add_argument("--code_root", type=str, default=".")

    parser.add_argument(
        "--data_root",
        type=str,
        default="data/final_dataset",
        help="Recommendation dataset root.",
    )

    parser.add_argument(
        "--train_data_root",
        type=str,
        default="data/mendeley_iqa_701515_grouped900",
        help="Training dataset root.",
    )

    parser.add_argument(
        "--source_ckpt_path",
        type=str,
        default="checkpoints/best_resnet50_imagenet.pth",
        help="Source pretrain checkpoint.",
    )

    parser.add_argument("--split", type=str, default="")
    parser.add_argument("--runs_root", type=str, default="runs_binary_iqa")
    parser.add_argument("--out_dir", type=str, default="runs_binary_iqa/top1_rec_acc_final_dataset")

    parser.add_argument(
        "--methods",
        type=str,
        default="all",
        help="Comma-separated method_id/run_dir list, or all.",
    )

    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="Optional checkpoint path for single-method debugging.",
    )

    parser.add_argument(
        "--retrain_missing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Retrain missing fusion/transfer seed0 checkpoints.",
    )

    parser.add_argument("--train_epochs", type=int, default=25)
    parser.add_argument("--train_warmup_epochs", type=int, default=3)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--train_lr", type=float, default=1e-4)
    parser.add_argument("--train_num_workers", type=int, default=8)

    parser.add_argument(
        "--train_ft_mode",
        type=str,
        default="l34",
        choices=["l4", "l34", "l234", "all"],
    )

    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--group_regex", type=str, default=r"^extra__(.+?)__")

    parser.add_argument(
        "--group_mode",
        type=str,
        default="original_mha",
        choices=["original_mha", "copied_group", "regex"],
    )

    parser.add_argument("--filename_tokens_to_drop", type=int, default=1)

    parser.add_argument(
        "--source_mode",
        type=str,
        default="all",
        choices=["all", "extra", "base"],
    )

    parser.add_argument("--w_prob", type=float, default=1.0)
    parser.add_argument("--w_qres", type=float, default=1.0)
    parser.add_argument("--w_qssim", type=float, default=1.0)
    parser.add_argument("--w_qpsnr", type=float, default=1.0)

    args = parser.parse_args()

    resolve_default_paths(args)
    set_seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    split_dir = (
        os.path.normpath(os.path.join(args.data_root, args.split))
        if args.split
        else args.data_root
    )

    selected_methods = resolve_methods(args.methods)

    print(f"[INFO] Device: {device}", flush=True)
    print(f"[INFO] Evaluation data: {split_dir}", flush=True)
    print(f"[INFO] Output dir: {args.out_dir}", flush=True)
    print(f"[INFO] Methods to evaluate: {len(selected_methods)}", flush=True)

    rows = []

    for method_spec in tqdm(
        selected_methods,
        desc="Evaluating methods",
        unit="method",
        dynamic_ncols=True,
    ):
        print(
            f"\n[START] {method_spec['method_id']} ({method_spec['method_label']})",
            flush=True,
        )

        try:
            row = evaluate_method(
                method_spec=method_spec,
                args=args,
                device=device,
                split_dir=split_dir,
            )

            rows.append(row)

            print(
                f"[DONE] {method_spec['method_id']} | "
                f"status={row['status']} | "
                f"Top1Acc={row['top1_recommendation_accuracy']:.6f}",
                flush=True,
            )

        except Exception as exc:
            row = missing_row(
                method_spec=method_spec,
                status="error",
                error=str(exc),
            )

            rows.append(row)

            print(
                f"[ERROR] {method_spec['method_id']}: {exc}",
                flush=True,
            )

    summary_csv = os.path.join(
        args.out_dir,
        "top1_rec_acc_all_methods_seed0.csv",
    )

    summary_df = pd.DataFrame(rows)

    summary_df.to_csv(
        summary_csv,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n===== Top-1 Recommendation Accuracy Summary =====")
    print(summary_df[[
        "method_id",
        "method_label",
        "status",
        "num_frames",
        "num_acquisitions",
        "top1_recommendation_accuracy",
        "error",
    ]].to_string(index=False))

    print(f"\nSaved summary CSV: {summary_csv}")


if __name__ == "__main__":
    main()
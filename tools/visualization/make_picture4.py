#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms

from picture2 import (
    build_model,
    choose_scan,
    effective_area_score,
    list_scan_counts,
    minmax_normalize,
    psnr_against_reference,
    ssim_against_reference,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = ROOT / "final_dataset_link"
DEFAULT_CKPT = ROOT / "runs_binary_iqa" / "DAF_seed0" / "best_model.pth"
DEFAULT_OUTPUT_DIR = ROOT / "picture4"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot frame-wise recommendation, usability, and image-quality prior curves."
    )
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--ckpt_path", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--scan_id", type=str, default="")
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--min_each_label", type=int, default=3)
    parser.add_argument("--top_k", type=int, default=6)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def infer_scan_id(name: str) -> str:
    match = re.match(r"^(.*\.mha)_[pn]_\d+\.[^.]+$", name, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Unexpected frame filename: {name}")
    return match.group(1)


def infer_frame_index(name: str) -> int:
    match = re.search(r"_[pn]_(\d+)\.[^.]+$", name, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Unexpected frame filename: {name}")
    return int(match.group(1))


def scan_stem(scan_id: str) -> str:
    return scan_id[:-4] if scan_id.lower().endswith(".mha") else scan_id


def build_transform(input_size: int):
    return transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def load_scan_curve_data(
    model,
    data_root: Path,
    scan_id: str,
    input_size: int,
    device: torch.device,
) -> pd.DataFrame:
    transform = build_transform(input_size)
    rows = []

    for label_name in ("usable", "unusable"):
        label_dir = data_root / label_name
        if not label_dir.is_dir():
            continue

        for image_path in sorted(label_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTS:
                continue
            if infer_scan_id(image_path.name) != scan_id:
                continue

            image = Image.open(image_path).convert("RGB")
            x = transform(image).unsqueeze(0).to(device)
            prob = F.softmax(model(x), dim=1)[0].detach().cpu().numpy()
            rows.append(
                {
                    "image_path": image_path,
                    "label_name": label_name,
                    "frame_idx": infer_frame_index(image_path.name),
                    "usable_prob": float(prob[1]),
                    "unusable_prob": float(prob[0]),
                    "image": image.copy(),
                }
            )

    if not rows:
        raise RuntimeError(f"No frames found for scan: {scan_id}")

    df = pd.DataFrame(rows).sort_values(["frame_idx", "label_name", "image_path"]).reset_index(drop=True)
    df["row_id"] = np.arange(len(df))
    pseudo_ref = df.loc[int(df["usable_prob"].idxmax()), "image"]

    df["qres"] = minmax_normalize([effective_area_score(img) for img in df["image"]])
    df["qssim"] = minmax_normalize([ssim_against_reference(img, pseudo_ref) for img in df["image"]])
    df["qpsnr"] = minmax_normalize([psnr_against_reference(img, pseudo_ref) for img in df["image"]])
    df["image_quality_prior_score"] = (df["qres"] + df["qssim"] + df["qpsnr"]) / 3.0
    df["recommendation_score"] = (
        df["usable_prob"] + df["qres"] + df["qssim"] + df["qpsnr"]
    ) / 4.0
    return df


def plot_curve(df: pd.DataFrame, scan_id: str, output_path: Path, dpi: int) -> pd.DataFrame:
    has_duplicate_frame_idx = bool(df["frame_idx"].duplicated().any())
    x_col = "plot_x"
    if has_duplicate_frame_idx:
        df = df.copy()
        df[x_col] = np.arange(len(df))
        x_label = "Ordered frame sample"
    else:
        df = df.copy()
        df[x_col] = df["frame_idx"]
        x_label = "Frame index / Slice index"

    top1_row = df.loc[df["recommendation_score"].idxmax()]
    top1_plot_x = float(top1_row[x_col])
    top1_frame = int(top1_row["frame_idx"])
    top1_score = float(top1_row["recommendation_score"])
    top1_row_id = int(top1_row["row_id"])

    fig, ax = plt.subplots(figsize=(12, 5.5), constrained_layout=True)

    ax.plot(
        df[x_col],
        df["recommendation_score"],
        color="#d04a02",
        linewidth=3.2,
        label="Final recommendation score",
        zorder=3,
    )
    ax.plot(
        df[x_col],
        df["usable_prob"],
        color="#1f77b4",
        linewidth=2.0,
        alpha=0.95,
        label="Usability probability",
        zorder=2,
    )
    ax.plot(
        df[x_col],
        df["image_quality_prior_score"],
        color="#2a9d8f",
        linewidth=2.0,
        alpha=0.95,
        label="Image-quality prior score",
        zorder=2,
    )

    ax.axvline(
        x=top1_plot_x,
        color="#d04a02",
        linestyle="--",
        linewidth=1.5,
        alpha=0.8,
        zorder=1,
    )
    ax.scatter(
        [top1_plot_x],
        [top1_score],
        marker="*",
        s=260,
        color="#d04a02",
        edgecolors="black",
        linewidths=0.8,
        zorder=4,
    )
    annotation_text = (
        "Top-1 recommended frame\n"
        f"Label: {top1_row['label_name']}\n"
        f"Score: {top1_score:.3f}"
    )

    ann_x = max(float(df[x_col].min()) + 1.2, top1_plot_x - 0.7)
    ann_y = 0.985

    ax.annotate(
        annotation_text,
        xy=(top1_plot_x, top1_score),
        xytext=(ann_x, ann_y),
        textcoords="data",
        ha="right",
        va="top",
        fontsize=10,
        multialignment="left",
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#666666", "alpha": 0.95},
        arrowprops={"arrowstyle": "->", "color": "#666666", "lw": 1.0},
        annotation_clip=False,
    )

    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel("Normalized recommendation score", fontsize=11)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlim(float(df[x_col].min()), float(df[x_col].max()))
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.legend(frameon=False, loc="best")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    export_df = df.drop(columns=["image"]).copy()
    export_df["image_path"] = export_df["image_path"].astype(str)
    export_df["is_top1_recommended"] = export_df["row_id"] == top1_row_id
    export_df["x_axis_mode"] = "ordered_frame_sample" if has_duplicate_frame_idx else "frame_idx"
    return export_df


def main():
    args = parse_args()

    if not args.data_root.is_dir():
        raise FileNotFoundError(f"Missing data root: {args.data_root}")
    if not args.ckpt_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {args.ckpt_path}")

    scan_counts = list_scan_counts(args.data_root)
    scan_id = choose_scan(
        scan_counts=scan_counts,
        scan_id=args.scan_id.strip(),
        top_k=args.top_k,
        min_each_label=args.min_each_label,
    )

    device_name = args.device.strip() or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    model = build_model(device=device, ckpt_path=args.ckpt_path)

    df = load_scan_curve_data(
        model=model,
        data_root=args.data_root,
        scan_id=scan_id,
        input_size=args.input_size,
        device=device,
    )

    stem = scan_stem(scan_id)
    output_dir = args.output_dir
    figure_path = output_dir / f"{stem}_recommendation_curve.png"
    csv_path = output_dir / f"{stem}_recommendation_curve.csv"

    export_df = plot_curve(df=df, scan_id=scan_id, output_path=figure_path, dpi=args.dpi)
    export_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    top1_row = export_df.loc[export_df["is_top1_recommended"]].iloc[0]
    print(f"scan_id={scan_id}")
    print(f"figure={figure_path}")
    print(f"csv={csv_path}")
    print(f"top1_frame_index={int(top1_row['frame_idx'])}")
    print(f"top1_score={float(top1_row['recommendation_score']):.6f}")
    if str(top1_row["x_axis_mode"]) == "ordered_frame_sample":
        print("warning=duplicate frame indices detected in exported PNG names; x-axis falls back to ordered frame sample.")


if __name__ == "__main__":
    main()

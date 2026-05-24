#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

import torch
import torch.nn.functional as F
from torchvision import transforms

from train_DAF import AdaptiveInterpDynamic


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = ROOT / "final_dataset_link"
DEFAULT_CKPT = ROOT / "runs_binary_iqa" / "DAF_seed0" / "best_model.pth"
DEFAULT_OUTPUT_DIR = ROOT / "picture2"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class FrameRecord:
    image_path: Path
    scan_id: str
    label_name: str
    usable_prob: float
    unusable_prob: float
    image: Image.Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export 6 single-frame images from the same scan with P and recommendation index."
    )
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--ckpt_path", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--scan_id", type=str, default="")
    parser.add_argument("--top_k", type=int, default=6)
    parser.add_argument("--min_each_label", type=int, default=3)
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--device", type=str, default="")
    return parser.parse_args()


def load_font(size: int):
    for font_name in ("arial.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(font_name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def build_model(device: torch.device, ckpt_path: Path):
    model = AdaptiveInterpDynamic(
        num_classes=2,
        alpha_priors=[0.85, 0.75, 0.55, 0.30],
        channels=[256, 512, 1024, 2048],
    )
    ckpt = torch.load(str(ckpt_path), map_location=device)
    state_dict = {
        key: value
        for key, value in ckpt["model_state_dict"].items()
        if not (
            key.endswith(".total_ops")
            or key.endswith(".total_params")
            or key in {"total_ops", "total_params"}
        )
    }
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def infer_scan_id(name: str) -> str:
    match = re.match(r"^(.*\.mha)_[pn]_\d+\.[^.]+$", name, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Unexpected frame filename: {name}")
    return match.group(1)


def scan_stem(scan_id: str) -> str:
    return scan_id[:-4] if scan_id.lower().endswith(".mha") else scan_id


def list_scan_counts(data_root: Path) -> pd.DataFrame:
    rows = []
    for label_name in ("usable", "unusable"):
        label_dir = data_root / label_name
        for image_path in sorted(label_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTS:
                continue
            rows.append(
                {
                    "scan_id": infer_scan_id(image_path.name),
                    "label_name": label_name,
                    "path": str(image_path),
                }
            )

    if not rows:
        raise RuntimeError(f"No images found under {data_root}")

    df = pd.DataFrame(rows)
    counts = (
        df.groupby("scan_id", as_index=False)
        .agg(
            total_frames=("path", "count"),
            usable_frames=("label_name", lambda s: int((s == "usable").sum())),
            unusable_frames=("label_name", lambda s: int((s == "unusable").sum())),
        )
        .sort_values(
            ["total_frames", "usable_frames", "unusable_frames", "scan_id"],
            ascending=[False, False, False, True],
        )
        .reset_index(drop=True)
    )
    return counts


def choose_scan(scan_counts: pd.DataFrame, scan_id: str, top_k: int, min_each_label: int) -> str:
    if scan_id:
        return scan_id

    preferred = scan_counts[
        (scan_counts["usable_frames"] >= min_each_label)
        & (scan_counts["unusable_frames"] >= min_each_label)
        & (scan_counts["total_frames"] >= top_k)
    ]
    if not preferred.empty:
        return str(preferred.iloc[0]["scan_id"])

    fallback = scan_counts[
        (scan_counts["usable_frames"] >= 1)
        & (scan_counts["unusable_frames"] >= 1)
        & (scan_counts["total_frames"] >= top_k)
    ]
    if not fallback.empty:
        return str(fallback.iloc[0]["scan_id"])

    raise RuntimeError("No scan contains at least one usable and one unusable frame with enough total frames.")


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


def load_scan_frames(
    model,
    data_root: Path,
    scan_id: str,
    input_size: int,
    device: torch.device,
) -> pd.DataFrame:
    transform = transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    rows = []
    for label_name in ("usable", "unusable"):
        label_dir = data_root / label_name
        candidates = sorted(label_dir.glob(f"{scan_id}_[pn]_*.png"))
        for image_path in candidates:
            image = Image.open(image_path).convert("RGB")
            x = transform(image).unsqueeze(0).to(device)
            prob = F.softmax(model(x), dim=1)[0].detach().cpu().numpy()
            rows.append(
                FrameRecord(
                    image_path=image_path,
                    scan_id=scan_id,
                    label_name=label_name,
                    usable_prob=float(prob[1]),
                    unusable_prob=float(prob[0]),
                    image=image.copy(),
                ).__dict__
            )

    if not rows:
        raise RuntimeError(f"No frames found for scan: {scan_id}")

    df = pd.DataFrame(rows)
    pseudo_ref_idx = int(df["usable_prob"].idxmax())
    pseudo_ref = df.loc[pseudo_ref_idx, "image"]

    df["qres"] = minmax_normalize([effective_area_score(img) for img in df["image"]])
    df["qssim"] = minmax_normalize([ssim_against_reference(img, pseudo_ref) for img in df["image"]])
    df["qpsnr"] = minmax_normalize([psnr_against_reference(img, pseudo_ref) for img in df["image"]])
    df["recommendation_score"] = (
        df["usable_prob"] + df["qres"] + df["qssim"] + df["qpsnr"]
    ) / 4.0
    df["pred_label"] = np.where(df["usable_prob"] >= df["unusable_prob"], "usable", "unusable")
    df["pred_prob"] = np.where(df["usable_prob"] >= df["unusable_prob"], df["usable_prob"], df["unusable_prob"])
    df = df.sort_values(
        ["recommendation_score", "usable_prob", "qres", "qssim", "qpsnr"],
        ascending=False,
    ).reset_index(drop=True)
    df["recommend_rank"] = np.arange(1, len(df) + 1)
    return df


def select_mixed_frames(scan_df: pd.DataFrame, top_k: int, min_each_label: int) -> pd.DataFrame:
    usable_df = scan_df[scan_df["label_name"] == "usable"].head(min_each_label)
    unusable_df = scan_df[scan_df["label_name"] == "unusable"].head(min_each_label)

    if len(usable_df) < 1 or len(unusable_df) < 1:
        raise RuntimeError("Selected scan does not contain both usable and unusable frames.")

    selected = pd.concat([usable_df, unusable_df], ignore_index=True)
    remaining = scan_df.loc[~scan_df["image_path"].isin(selected["image_path"])].copy()

    while len(selected) < top_k and not remaining.empty:
        selected = pd.concat([selected, remaining.head(1)], ignore_index=True)
        remaining = remaining.iloc[1:].copy()

    if len(selected) < top_k:
        raise RuntimeError(f"Only selected {len(selected)} frames, fewer than requested top_k={top_k}.")

    selected = selected.sort_values(
        ["recommendation_score", "usable_prob"],
        ascending=False,
    ).head(top_k).reset_index(drop=True)
    selected["export_order"] = np.arange(1, len(selected) + 1)
    return selected


def annotate_image(
    image: Image.Image,
    usable_prob: float,
    unusable_prob: float,
    ri_value: float,
) -> Image.Image:
    image = image.convert("RGB")
    band_h = max(150, image.height // 4)
    out = Image.new("RGB", (image.width, image.height + band_h), "white")
    out.paste(image, (0, band_h))

    draw = ImageDraw.Draw(out)
    text_font = load_font(max(30, image.width // 16))
    if usable_prob >= unusable_prob:
        prob_text = f"P(usable)={usable_prob:.3f}"
    else:
        prob_text = f"P(unusable)={unusable_prob:.3f}"
    ri_text = f"RI={ri_value:.3f}"

    draw.rectangle((0, 0, image.width, band_h), fill=(16, 16, 16))
    prob_bbox = draw.textbbox((0, 0), prob_text, font=text_font)
    ri_bbox = draw.textbbox((0, 0), ri_text, font=text_font)
    prob_w = prob_bbox[2] - prob_bbox[0]
    ri_w = ri_bbox[2] - ri_bbox[0]
    side_pad = 18
    gap = 24

    if side_pad + prob_w + gap + ri_w + side_pad <= image.width:
        prob_x = side_pad
        ri_x = image.width - side_pad - ri_w
        prob_y = 42
        ri_y = 42
    else:
        prob_x = side_pad
        ri_x = side_pad
        prob_y = 24
        ri_y = 78

    draw.text((prob_x, prob_y), prob_text, fill="white", font=text_font)
    draw.text((ri_x, ri_y), ri_text, fill="white", font=text_font)
    return out


def export_images(selected_df: pd.DataFrame, output_dir: Path) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    exported_paths = []

    for _, row in selected_df.iterrows():
        stem = Path(str(row["image_path"])).stem
        export_name = (
            f"{int(row['export_order']):02d}_"
            f"{row['label_name']}_"
            f"{stem}_P{float(row['usable_prob']):.3f}_RI{float(row['recommendation_score']):.3f}.png"
        )
        annotated = annotate_image(
            image=row["image"],
            usable_prob=float(row["usable_prob"]),
            unusable_prob=float(row["unusable_prob"]),
            ri_value=float(row["recommendation_score"]),
        )
        out_path = output_dir / export_name
        annotated.save(out_path)
        exported_paths.append(out_path)

    return exported_paths


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

    scan_df = load_scan_frames(
        model=model,
        data_root=args.data_root,
        scan_id=scan_id,
        input_size=args.input_size,
        device=device,
    )
    selected_df = select_mixed_frames(
        scan_df=scan_df,
        top_k=args.top_k,
        min_each_label=args.min_each_label,
    )

    output_dir = args.output_dir
    exported_paths = export_images(selected_df=selected_df, output_dir=output_dir)

    csv_path = output_dir / f"{scan_stem(scan_id)}_selected_frames.csv"
    export_df = selected_df.drop(columns=["image"]).copy()
    export_df["image_path"] = export_df["image_path"].astype(str)
    export_df["export_path"] = [str(path) for path in exported_paths]
    export_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print(f"scan_id={scan_id}")
    print(f"usable_frames={int((scan_df['label_name'] == 'usable').sum())}")
    print(f"unusable_frames={int((scan_df['label_name'] == 'unusable').sum())}")
    print(f"output_dir={output_dir}")
    print(f"csv={csv_path}")
    print()
    print(
        export_df[
            [
                "export_order",
                "label_name",
                "image_path",
                "usable_prob",
                "recommendation_score",
                "export_path",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()

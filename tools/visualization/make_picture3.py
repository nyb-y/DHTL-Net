#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps

import torch
import torch.nn.functional as F
from torchvision import transforms

from recommend import METHOD_SPECS, load_checkpoint_model, resolve_methods


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = ROOT / "final_dataset_link"
DEFAULT_RUNS_ROOT = ROOT / "runs_binary_iqa"
DEFAULT_OUTPUT_DIR = ROOT / "picture3"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
CLASS_NAMES = ["unusable", "usable"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export 4 images x 12 methods qualitative predictions for paper figures."
    )
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--runs_root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--methods", type=str, default="all")
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument(
        "--image_paths",
        nargs="*",
        default=[],
        help="Optional 4 explicit image paths. If omitted, 4 images are auto-selected.",
    )
    parser.add_argument(
        "--selection_method",
        type=str,
        default="DAF",
        help="Reference method_id used for auto-selection. DAF is DHTL-Net.",
    )
    parser.add_argument(
        "--make_contact_sheet",
        action="store_true",
        help="Also save one 3x4 contact sheet per selected image.",
    )
    parser.add_argument(
        "--selection_min_margin",
        type=float,
        default=0.0,
        help="Minimum required advantage of DHTL-Net GT probability over every other method.",
    )
    parser.add_argument(
        "--selection_pool_per_label",
        type=int,
        default=30,
        help="Number of top correct DHTL-Net candidates per label kept for cross-method comparison.",
    )
    parser.add_argument(
        "--selection_min_other_wrong",
        type=int,
        default=1,
        help="Preferred minimum number of non-DHTL-Net methods that must be wrong on a selected image.",
    )
    parser.add_argument(
        "--selection_max_rank",
        type=int,
        default=5,
        help="Maximum allowed DHTL-Net GT-probability rank among all methods for selected images.",
    )
    parser.add_argument(
        "--selection_allow_fallback",
        action="store_true",
        help="Allow fallback to weaker DHTL-Net-correct samples if strict selection cannot find 4 images.",
    )
    return parser.parse_args()


def load_font(size: int):
    for font_name in ("arial.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(font_name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def infer_scan_id(name: str) -> str:
    match = re.match(r"^(.*\.mha)_[pn]_\d+\.[^.]+$", name, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Unexpected frame filename: {name}")
    return match.group(1)


def infer_frame_index(name: str) -> int:
    match = re.search(r"_[pn]_(\d+)\.[^.]+$", name, flags=re.IGNORECASE)
    if not match:
        return -1
    return int(match.group(1))


def infer_gt_label(image_path: Path) -> str:
    parent = image_path.parent.name.lower()
    if parent in {"usable", "unusable"}:
        return parent

    name = image_path.name.lower()
    if "_p_" in name:
        return "usable"
    if "_n_" in name:
        return "unusable"

    raise ValueError(f"Cannot infer ground-truth label from path: {image_path}")


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


def collect_dataset_frames(data_root: Path) -> pd.DataFrame:
    rows = []
    for label_name in ("usable", "unusable"):
        label_dir = data_root / label_name
        if not label_dir.is_dir():
            continue
        for image_path in sorted(label_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTS:
                continue
            rows.append(
                {
                    "image_path": image_path,
                    "gt_label": label_name,
                    "scan_id": infer_scan_id(image_path.name),
                    "frame_idx": infer_frame_index(image_path.name),
                }
            )

    if not rows:
        raise RuntimeError(f"No images found under {data_root}")

    return pd.DataFrame(rows)


@torch.no_grad()
def score_images_for_selection(
    image_paths: Sequence[Path],
    ckpt_path: Path,
    input_size: int,
    device: torch.device,
    method_spec: Dict,
) -> pd.DataFrame:
    model, _ = load_checkpoint_model(str(ckpt_path), device=device, method_spec=method_spec)
    transform = build_transform(input_size)

    rows = []
    for image_path in image_paths:
        image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
        x = transform(image).unsqueeze(0).to(device)
        prob = F.softmax(model(x), dim=1)[0].detach().cpu().numpy()
        gt_label = infer_gt_label(image_path)
        pred_idx = int(prob.argmax())
        pred_label = CLASS_NAMES[pred_idx]
        gt_prob = float(prob[1]) if gt_label == "usable" else float(prob[0])
        rows.append(
            {
                "image_path": image_path,
                "gt_label": gt_label,
                "pred_label": pred_label,
                "usable_prob": float(prob[1]),
                "unusable_prob": float(prob[0]),
                "gt_prob": gt_prob,
                "is_correct": pred_label == gt_label,
            }
        )

    return pd.DataFrame(rows)


@torch.no_grad()
def score_images_across_methods(
    frame_df: pd.DataFrame,
    method_specs: Sequence[Dict],
    runs_root: Path,
    input_size: int,
    device: torch.device,
) -> pd.DataFrame:
    image_paths = frame_df["image_path"].tolist()
    merged = frame_df.copy()

    for method_spec in method_specs:
        ckpt_path = runs_root / method_spec["run_dir"] / "best_model.pth"
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Missing checkpoint for selection: {ckpt_path}")

        scored_df = score_images_for_selection(
            image_paths=image_paths,
            ckpt_path=ckpt_path,
            input_size=input_size,
            device=device,
            method_spec=method_spec,
        ).rename(
            columns={
                "pred_label": f"{method_spec['method_id']}_pred_label",
                "usable_prob": f"{method_spec['method_id']}_usable_prob",
                "unusable_prob": f"{method_spec['method_id']}_unusable_prob",
                "gt_prob": f"{method_spec['method_id']}_gt_prob",
                "is_correct": f"{method_spec['method_id']}_is_correct",
            }
        )
        scored_df = scored_df.drop(columns=["gt_label"])
        merged = merged.merge(scored_df, on="image_path", how="inner")

    return merged


def build_prefilter_candidates(
    frame_df: pd.DataFrame,
    selection_scored_df: pd.DataFrame,
    pool_per_label: int,
) -> pd.DataFrame:
    merged = frame_df.merge(selection_scored_df, on=["image_path", "gt_label"], how="inner")
    correct_df = merged[merged["is_correct"]].copy()

    if correct_df.empty:
        raise RuntimeError("No correctly predicted images found for DHTL-Net prefiltering.")

    candidate_parts = []
    for label_name in ("usable", "unusable"):
        label_df = correct_df[correct_df["gt_label"] == label_name].copy()
        if label_df.empty:
            continue
        label_df = label_df.sort_values(["gt_prob", "scan_id", "frame_idx"], ascending=[False, True, True])
        candidate_parts.append(label_df.head(pool_per_label))

    if len(candidate_parts) != 2:
        raise RuntimeError("Prefiltering requires both usable and unusable correct DHTL-Net samples.")

    candidate_df = pd.concat(candidate_parts, ignore_index=True)
    candidate_df = candidate_df.drop_duplicates(subset=["image_path"]).reset_index(drop=True)
    return candidate_df[["image_path", "gt_label", "scan_id", "frame_idx"]].copy()


def choose_four_images_auto(
    all_scores_df: pd.DataFrame,
    selection_method_id: str,
    compared_method_ids: Sequence[str],
    min_margin: float,
    min_other_wrong: int,
    max_rank: int,
    allow_fallback: bool,
) -> Tuple[List[Path], pd.DataFrame]:
    main_gt_prob_col = f"{selection_method_id}_gt_prob"
    main_correct_col = f"{selection_method_id}_is_correct"

    other_method_ids = [
        method_id
        for method_id in compared_method_ids
        if method_id != selection_method_id
    ]

    other_gt_prob_cols = [f"{method_id}_gt_prob" for method_id in other_method_ids]
    other_correct_cols = [f"{method_id}_is_correct" for method_id in other_method_ids]

    candidate_df = all_scores_df.copy()
    candidate_df["other_max_gt_prob"] = candidate_df[other_gt_prob_cols].max(axis=1)
    candidate_df["other_mean_gt_prob"] = candidate_df[other_gt_prob_cols].mean(axis=1)
    candidate_df["main_advantage"] = candidate_df[main_gt_prob_col] - candidate_df["other_max_gt_prob"]
    candidate_df["main_mean_advantage"] = candidate_df[main_gt_prob_col] - candidate_df["other_mean_gt_prob"]
    candidate_df["other_wrong_count"] = len(other_correct_cols) - candidate_df[other_correct_cols].sum(axis=1)
    candidate_df["main_rank_by_gt_prob"] = candidate_df[[main_gt_prob_col] + other_gt_prob_cols].rank(
        axis=1,
        method="min",
        ascending=False,
    )[main_gt_prob_col]
    candidate_df = candidate_df[candidate_df[main_correct_col]].copy()

    if candidate_df.empty:
        raise RuntimeError("No correctly predicted DHTL-Net images found after cross-method scoring.")

    phase1_df = candidate_df[
        (candidate_df["other_wrong_count"] >= min_other_wrong)
        & (candidate_df["main_mean_advantage"] >= min_margin)
        & (candidate_df["main_rank_by_gt_prob"] <= max_rank)
    ].copy()
    phase1_df["selection_reason"] = "other_methods_wrong"

    phase2_df = candidate_df[
        (candidate_df["main_rank_by_gt_prob"] <= max_rank)
        & (candidate_df["main_mean_advantage"] >= min_margin)
    ].copy()
    phase2_df["selection_reason"] = "topk_gt_prob_rank"

    fallback_df = candidate_df[
        (candidate_df["main_rank_by_gt_prob"] <= max_rank)
        & (candidate_df["main_mean_advantage"] >= min_margin)
    ].copy()
    fallback_df["selection_reason"] = "daf_correct_fallback"

    sort_cols = [
        "other_wrong_count",
        "main_rank_by_gt_prob",
        "main_mean_advantage",
        "main_advantage",
        main_gt_prob_col,
        "scan_id",
        "frame_idx",
    ]
    sort_asc = [False, True, False, False, False, True, True]

    def pick_two_per_label(label_name: str) -> pd.DataFrame:
        selected_rows = []
        seen_paths = set()
        seen_scans = set()

        source_dfs = [phase1_df, phase2_df]
        if allow_fallback:
            source_dfs.append(fallback_df)

        for source_df in source_dfs:
            label_df = source_df[source_df["gt_label"] == label_name].sort_values(sort_cols, ascending=sort_asc)

            for _, row in label_df.iterrows():
                path_key = str(row["image_path"])
                scan_key = str(row["scan_id"])
                if path_key in seen_paths or scan_key in seen_scans:
                    continue
                selected_rows.append(row)
                seen_paths.add(path_key)
                seen_scans.add(scan_key)
                if len(selected_rows) == 2:
                    break
            if len(selected_rows) == 2:
                break

        if len(selected_rows) < 2:
            for source_df in source_dfs:
                label_df = source_df[source_df["gt_label"] == label_name].sort_values(sort_cols, ascending=sort_asc)
                for _, row in label_df.iterrows():
                    path_key = str(row["image_path"])
                    if path_key in seen_paths:
                        continue
                    selected_rows.append(row)
                    seen_paths.add(path_key)
                    if len(selected_rows) == 2:
                        break
                if len(selected_rows) == 2:
                    break

        if len(selected_rows) < 2:
            raise RuntimeError(
                f"Could not find 2 selected images for label={label_name} under the current strict rules. "
                "Try increasing --selection_pool_per_label, relaxing --selection_max_rank, "
                "lowering --selection_min_other_wrong, or enabling --selection_allow_fallback."
            )

        return pd.DataFrame(selected_rows)

    usable = pick_two_per_label("usable")
    unusable = pick_two_per_label("unusable")

    selected = pd.concat([usable, unusable], ignore_index=True)
    selected = selected.sort_values(
        ["gt_label", "other_wrong_count", "main_rank_by_gt_prob", "main_mean_advantage", "main_advantage", main_gt_prob_col],
        ascending=[False, False, True, False, False, False],
    ).reset_index(drop=True)
    return [Path(p) for p in selected["image_path"].tolist()], selected


def resolve_selected_images(args, device: torch.device) -> Tuple[List[Path], str]:
    if args.image_paths:
        selected = [Path(p).resolve() for p in args.image_paths]
        if len(selected) != 4:
            raise ValueError("--image_paths must provide exactly 4 images.")
        missing = [str(path) for path in selected if not path.is_file()]
        if missing:
            raise FileNotFoundError("Missing image(s): " + ", ".join(missing))
        return selected, "manual"

    frame_df = collect_dataset_frames(args.data_root)
    method_specs = resolve_methods(args.selection_method)
    if len(method_specs) != 1:
        raise ValueError("--selection_method must resolve to exactly one method.")

    method_spec = method_specs[0]
    ckpt_path = args.runs_root / method_spec["run_dir"] / "best_model.pth"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint for auto-selection: {ckpt_path}")

    selection_scored_df = score_images_for_selection(
        image_paths=frame_df["image_path"].tolist(),
        ckpt_path=ckpt_path,
        input_size=args.input_size,
        device=device,
        method_spec=method_spec,
    )
    candidate_frame_df = build_prefilter_candidates(
        frame_df=frame_df,
        selection_scored_df=selection_scored_df,
        pool_per_label=args.selection_pool_per_label,
    )

    selection_specs = resolve_methods(args.methods)
    all_scores_df = score_images_across_methods(
        frame_df=candidate_frame_df,
        method_specs=selection_specs,
        runs_root=args.runs_root,
        input_size=args.input_size,
        device=device,
    )
    selected, selected_stats_df = choose_four_images_auto(
        all_scores_df=all_scores_df,
        selection_method_id=method_spec["method_id"],
        compared_method_ids=[spec["method_id"] for spec in selection_specs],
        min_margin=args.selection_min_margin,
        min_other_wrong=args.selection_min_other_wrong,
        max_rank=args.selection_max_rank,
        allow_fallback=args.selection_allow_fallback,
    )

    stats_cols = [
        "image_path",
        "gt_label",
        "scan_id",
        "frame_idx",
        f"{method_spec['method_id']}_gt_prob",
        "other_max_gt_prob",
        "other_mean_gt_prob",
        "main_mean_advantage",
        "main_advantage",
        "other_wrong_count",
        "main_rank_by_gt_prob",
        "selection_reason",
    ]
    selection_stats_path = args.output_dir / "selection_stats.csv"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_stats_df.loc[:, stats_cols].to_csv(selection_stats_path, index=False, encoding="utf-8-sig")
    return selected, f"auto:{method_spec['method_id']}_vs_all"


def annotate_prediction(
    image: Image.Image,
    method_label: str,
    gt_label: str,
    pred_label: str,
    prob: Sequence[float],
) -> Image.Image:
    image = image.convert("RGB")
    font = load_font(max(18, image.width // 22))
    title_font = load_font(max(22, image.width // 18))

    line1 = f"Method: {method_label}"
    line2 = f"GT: {gt_label} | Pred: {pred_label}"
    line3 = f"P(unusable)={float(prob[0]):.3f} | P(usable)={float(prob[1]):.3f}"

    draw_probe = ImageDraw.Draw(image)
    b1 = draw_probe.textbbox((0, 0), line1, font=title_font)
    b2 = draw_probe.textbbox((0, 0), line2, font=font)
    b3 = draw_probe.textbbox((0, 0), line3, font=font)

    text_h = (b1[3] - b1[1]) + (b2[3] - b2[1]) + (b3[3] - b3[1])
    pad_y = max(10, image.height // 60)
    band_h = text_h + pad_y * 4
    out = Image.new("RGB", (image.width, image.height + band_h), "white")
    out.paste(image, (0, band_h))

    draw = ImageDraw.Draw(out)
    header_color = (20, 20, 20)
    draw.rectangle((0, 0, image.width, band_h), fill=header_color)
    text_x = max(12, image.width // 40)
    y = pad_y
    draw.text((text_x, y), line1, fill="white", font=title_font)
    y += (b1[3] - b1[1]) + pad_y

    pred_ok = gt_label == pred_label
    gt_fill = (0, 200, 120) if gt_label == "usable" else (255, 170, 0)
    pred_fill = (0, 200, 120) if pred_ok else (255, 90, 90)

    draw.text((text_x, y), f"GT: {gt_label}", fill=gt_fill, font=font)
    gt_box = draw.textbbox((text_x, y), f"GT: {gt_label}", font=font)
    draw.text((gt_box[2] + 16, y), f"Pred: {pred_label}", fill=pred_fill, font=font)
    y += (b2[3] - b2[1]) + pad_y
    draw.text((text_x, y), line3, fill="white", font=font)
    return out


@torch.no_grad()
def run_inference_for_method(
    method_spec: Dict,
    image_paths: Sequence[Path],
    input_size: int,
    device: torch.device,
    runs_root: Path,
    output_dir: Path,
) -> List[Dict]:
    ckpt_path = runs_root / method_spec["run_dir"] / "best_model.pth"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    model, _ = load_checkpoint_model(str(ckpt_path), device=device, method_spec=method_spec)
    transform = build_transform(input_size)
    method_dir = output_dir / method_spec["method_id"]
    method_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for image_idx, image_path in enumerate(image_paths, start=1):
        original = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
        x = transform(original).unsqueeze(0).to(device)
        prob = F.softmax(model(x), dim=1)[0].detach().cpu()
        pred_idx = int(prob.argmax().item())
        pred_label = CLASS_NAMES[pred_idx]
        gt_label = infer_gt_label(image_path)

        annotated = annotate_prediction(
            image=original,
            method_label=method_spec["method_label"],
            gt_label=gt_label,
            pred_label=pred_label,
            prob=prob.tolist(),
        )

        export_name = (
            f"img{image_idx:02d}_"
            f"{image_path.stem}_"
            f"pred-{pred_label}_"
            f"punusable{float(prob[0]):.3f}_"
            f"pusable{float(prob[1]):.3f}.png"
        )
        export_path = method_dir / export_name
        annotated.save(export_path)

        rows.append(
            {
                "image_index": image_idx,
                "image_path": str(image_path),
                "method_id": method_spec["method_id"],
                "method_label": method_spec["method_label"],
                "gt_label": gt_label,
                "pred_label": pred_label,
                "punusable": float(prob[0]),
                "pusable": float(prob[1]),
                "export_path": str(export_path),
            }
        )

    return rows


def build_contact_sheet(image_index: int, manifest_df: pd.DataFrame, output_dir: Path) -> Path:
    subset = (
        manifest_df[manifest_df["image_index"] == image_index]
        .sort_values(["method_label", "method_id"], ascending=[True, True])
        .reset_index(drop=True)
    )
    if subset.empty:
        raise RuntimeError(f"No rows for image_index={image_index}")

    tiles = [Image.open(path).convert("RGB") for path in subset["export_path"]]
    cols = 3
    rows = math.ceil(len(tiles) / cols)
    tile_w = max(img.width for img in tiles)
    tile_h = max(img.height for img in tiles)
    canvas = Image.new("RGB", (cols * tile_w, rows * tile_h), "white")

    for idx, tile in enumerate(tiles):
        x = (idx % cols) * tile_w
        y = (idx // cols) * tile_h
        canvas.paste(tile.resize((tile_w, tile_h)), (x, y))

    out_path = output_dir / f"contact_sheet_img{image_index:02d}.png"
    canvas.save(out_path)
    return out_path


def main() -> None:
    args = parse_args()
    device_name = args.device.strip() or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)

    method_specs = resolve_methods(args.methods)
    if len(method_specs) != 12 and args.methods.strip().lower() == "all":
        raise RuntimeError(f"Expected 12 methods, got {len(method_specs)}")

    selected_images, selection_mode = resolve_selected_images(args, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: List[Dict] = []
    for method_spec in method_specs:
        manifest_rows.extend(
            run_inference_for_method(
                method_spec=method_spec,
                image_paths=selected_images,
                input_size=args.input_size,
                device=device,
                runs_root=args.runs_root,
                output_dir=args.output_dir,
            )
        )

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = args.output_dir / "manifest.csv"
    manifest_df.to_csv(manifest_path, index=False, encoding="utf-8-sig")

    selected_df = pd.DataFrame(
        {
            "image_index": list(range(1, len(selected_images) + 1)),
            "image_path": [str(path) for path in selected_images],
            "gt_label": [infer_gt_label(path) for path in selected_images],
            "selection_mode": selection_mode,
        }
    )
    selected_path = args.output_dir / "selected_images.csv"
    selected_df.to_csv(selected_path, index=False, encoding="utf-8-sig")

    contact_paths = []
    if args.make_contact_sheet:
        for image_index in range(1, len(selected_images) + 1):
            contact_paths.append(str(build_contact_sheet(image_index, manifest_df, args.output_dir)))

    print(f"selection_mode={selection_mode}")
    print(f"device={device}")
    print(f"num_methods={len(method_specs)}")
    print(f"num_images={len(selected_images)}")
    print(f"num_exports={len(manifest_df)}")
    print(f"output_dir={args.output_dir}")
    print(f"manifest={manifest_path}")
    print(f"selected_csv={selected_path}")
    if contact_paths:
        print("contact_sheets:")
        for path in contact_paths:
            print(path)
    print()
    print(selected_df.to_string(index=False))


if __name__ == "__main__":
    main()

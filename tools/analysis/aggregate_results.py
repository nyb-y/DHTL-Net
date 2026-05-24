#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggregate 11 binary-IQA experiment methods across multiple seeds and generate
paper-ready summaries/figures.

Expected run directory names:
  DAF_seed0 ... DAF_seed4
  fusion_residual_seed0 ... fusion_adapter_seed4
  transfer_frozen_seed0 ... transfer_l234_seed4

Artifacts consumed per run directory:
  - results.json
  - history.csv
  - flops_summary.json
  - profile_batchsize_latency_vram.csv

Figure policy:
  - batch size vs time: seed0 only, one curve per method
  - VRAM vs latency: seed0 only, one curve per method
  - FLOPs vs loss: seed0 only, one curve per method using epoch-wise history

Table policy:
  - F1 / BAcc mean+/-std use all available seeds

Confusion matrices:
  - exported as numeric tables, not plotted

Example:
  python aggregate_experiments_windows.py ^
    --runs_root runs_binary_iqa ^
    --out_dir docs/results/latest
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.family"] = "Times New Roman"

FIG_SIZE = (10, 5.5)
PLOT_FONTSIZE = 24

METHOD_SPECS = [
    {"method_id": "DHTL_wo_KMD", "label": "DHTL-Net", "run_prefix": "DHTL_wo_KMD_seed"},
    {"method_id": "DAF", "label": "DHTL-Net w/o KMD", "run_prefix": "DAF_seed"},
    {"method_id": "fusion_residual", "label": "Residual", "run_prefix": "fusion_residual_seed"},
    {"method_id": "fusion_fixed_interp", "label": "FixedInterp", "run_prefix": "fusion_fixed_interp_seed"},
    {"method_id": "fusion_learnable_interp", "label": "LearnableInterp", "run_prefix": "fusion_learnable_interp_seed"},
    {"method_id": "fusion_gating", "label": "Gating", "run_prefix": "fusion_gating_seed"},
    {"method_id": "fusion_adapter", "label": "Adapter", "run_prefix": "fusion_adapter_seed"},
    {"method_id": "transfer_frozen", "label": "Frozen", "run_prefix": "transfer_frozen_seed"},
    {"method_id": "transfer_all", "label": "All", "run_prefix": "transfer_all_seed"},
    {"method_id": "transfer_l4", "label": "FT-S4", "run_prefix": "transfer_l4_seed"},
    {"method_id": "transfer_l34", "label": "FT-S3-S4", "run_prefix": "transfer_l34_seed"},
    {"method_id": "transfer_l234", "label": "FT-S2-S4", "run_prefix": "transfer_l234_seed"},
]

PLOT_GROUPS = [
    {
        "group_id": "daf_vs_baselines",
        "title_suffix": "DHTL-Net vs. Fusion Baselines",
        "method_ids": [
            "DHTL_wo_KMD",
            "DAF",
            "fusion_residual",
            "fusion_fixed_interp",
            "fusion_learnable_interp",
            "fusion_gating",
            "fusion_adapter",
        ],
    },
    {
        "group_id": "daf_vs_tradition",
        "title_suffix": "DHTL-Net vs. Transfer Baselines",
        "method_ids": [
            "DHTL_wo_KMD",
            "transfer_frozen",
            "transfer_all",
            "transfer_l4",
            "transfer_l34",
            "transfer_l234",
        ],
    },
]

SEED_PATTERN = re.compile(r"seed(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs_root",
        type=str,
        default=r"runs_binary_iqa",
        help="Root directory containing DAF_seed*/fusion_*/transfer_* run folders.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=r"docs/results/latest",
        help="Directory to save aggregated tables and figures.",
    )
    parser.add_argument(
        "--loss_mode",
        type=str,
        default="val",
        choices=["val", "train"],
        help="Loss source for FLOPs-vs-loss curves. 'val' uses val_loss, 'train' uses loss.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI.")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_read_json(path: Path) -> Optional[dict]:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_read_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.is_file():
        return None
    return pd.read_csv(path)


def extract_seed(run_dir: Path) -> Optional[int]:
    match = SEED_PATTERN.search(run_dir.name)
    return int(match.group(1)) if match else None


def format_mean_std(mean: float, std: float, scale: float = 100.0, digits: int = 2) -> str:
    if pd.isna(mean):
        return "NA"
    spread = 0.0 if pd.isna(std) else std
    return f"{mean * scale:.{digits}f}+/-{spread * scale:.{digits}f}"


def build_color_map(method_labels: List[str]) -> Dict[str, tuple]:
    cmap = plt.get_cmap("tab20")
    return {label: cmap(i % cmap.N) for i, label in enumerate(method_labels)}


def get_marker(method_label: str) -> str:
    return "*" if method_label == "DHTL-Net" else "o"


def finalize_axes(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_xlabel(xlabel, fontsize=PLOT_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=PLOT_FONTSIZE)
    ax.tick_params(axis="both", labelsize=PLOT_FONTSIZE)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)


def add_inset_legend(ax) -> None:
    ax.legend(
        loc="best",
        fontsize=PLOT_FONTSIZE,
        frameon=True,
        framealpha=0.9,
        facecolor="white",
        edgecolor="0.8",
    )


def filter_methods(df: pd.DataFrame, method_ids: List[str]) -> pd.DataFrame:
    if len(df) == 0 or "method_id" not in df.columns:
        return pd.DataFrame(columns=df.columns)
    return df.loc[df["method_id"].isin(method_ids)].copy()


def compute_axis_limits(values: pd.Series, pad_ratio: float = 0.05) -> tuple[float, float]:
    numeric = pd.to_numeric(values, errors="coerce")
    numeric = numeric[np.isfinite(numeric)]
    if len(numeric) == 0:
        return (0.0, 1.0)

    vmin = float(numeric.min())
    vmax = float(numeric.max())
    if np.isclose(vmin, vmax):
        pad = max(abs(vmin) * pad_ratio, 1.0)
        return (vmin - pad, vmax + pad)

    span = vmax - vmin
    pad = span * pad_ratio
    return (vmin - pad, vmax + pad)


def build_shared_axes(
    seed0_profile_df: pd.DataFrame,
    seed0_history_df: pd.DataFrame,
    loss_mode: str,
) -> Dict[str, object]:
    axes: Dict[str, object] = {}

    profile_all = seed0_profile_df.copy()
    history_all = seed0_history_df.copy()

    profile_all["batch_size"] = pd.to_numeric(profile_all["batch_size"], errors="coerce")
    profile_all["latency_ms_per_batch"] = pd.to_numeric(profile_all["latency_ms_per_batch"], errors="coerce")
    profile_all["vram_peak_mb"] = pd.to_numeric(profile_all["vram_peak_mb"], errors="coerce")

    loss_col = "val_loss" if loss_mode == "val" and "val_loss" in history_all.columns else "loss"
    history_all["cumulative_train_estimated_tflops"] = pd.to_numeric(
        history_all["cumulative_train_estimated_tflops"], errors="coerce"
    )
    history_all[loss_col] = pd.to_numeric(history_all[loss_col], errors="coerce")

    batch_ticks = sorted(
        int(x) for x in profile_all["batch_size"].dropna().unique().tolist()
    )

    axes["batch_vs_time"] = {
        "xlim": compute_axis_limits(profile_all["batch_size"], pad_ratio=0.06),
        "ylim": compute_axis_limits(profile_all["latency_ms_per_batch"], pad_ratio=0.06),
        "xticks": batch_ticks,
    }
    axes["vram_vs_latency"] = {
        "xlim": compute_axis_limits(profile_all["latency_ms_per_batch"], pad_ratio=0.06),
        "ylim": compute_axis_limits(profile_all["vram_peak_mb"], pad_ratio=0.06),
    }
    axes["flops_vs_loss"] = {
        "xlim": compute_axis_limits(history_all["cumulative_train_estimated_tflops"], pad_ratio=0.04),
        "ylim": compute_axis_limits(history_all[loss_col], pad_ratio=0.06),
    }
    return axes


def collect_runs(
    runs_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, np.ndarray], List[str]]:
    run_rows: List[dict] = []
    profile_rows: List[dict] = []
    history_rows: List[dict] = []
    confusion_by_method: Dict[str, List[np.ndarray]] = {}
    warnings: List[str] = []

    for spec in METHOD_SPECS:
        method_id = spec["method_id"]
        method_label = spec["label"]
        run_prefix = spec["run_prefix"]
        run_dirs = sorted([p for p in runs_root.glob(f"{run_prefix}*") if p.is_dir()])

        if not run_dirs:
            warnings.append(f"[WARN] No run directories found for {method_id} under {runs_root}")
            continue

        confusion_by_method.setdefault(method_id, [])

        for run_dir in run_dirs:
            seed = extract_seed(run_dir)
            results = safe_read_json(run_dir / "results.json")
            history_df = safe_read_csv(run_dir / "history.csv")
            flops_info = safe_read_json(run_dir / "flops_summary.json")
            profile_df = safe_read_csv(run_dir / "profile_batchsize_latency_vram.csv")

            if results is None:
                warnings.append(f"[WARN] Missing results.json: {run_dir}")
                continue

            if flops_info is None:
                warnings.append(f"[WARN] Missing flops_summary.json: {run_dir}")
                flops_info = {}

            cm = np.asarray(
                results.get("test_confusion_matrix", [[np.nan, np.nan], [np.nan, np.nan]]),
                dtype=float,
            )
            confusion_by_method[method_id].append(cm)

            run_rows.append(
                {
                    "method_id": method_id,
                    "method_label": method_label,
                    "seed": seed,
                    "run_dir": str(run_dir),
                    "best_epoch": results.get("best_epoch", np.nan),
                    "test_loss": results.get("test_loss", np.nan),
                    "test_macro_f1": results.get("test_macro_f1", np.nan),
                    "test_balanced_acc": results.get("test_balanced_acc", np.nan),
                    "test_accuracy": results.get("test_accuracy", np.nan),
                    "test_auc": results.get("test_auc", np.nan),
                    "test_sensitivity_usable": results.get("test_sensitivity_usable", np.nan),
                    "test_specificity_unusable": results.get("test_specificity_unusable", np.nan),
                    "forward_flops_per_sample": flops_info.get("forward_flops_per_sample", np.nan),
                    "forward_gflops_per_sample": (
                        flops_info.get("forward_flops_per_sample", np.nan) / 1e9
                        if pd.notna(flops_info.get("forward_flops_per_sample", np.nan))
                        else np.nan
                    ),
                    "profiled_params": flops_info.get("profiled_params", np.nan),
                }
            )

            if history_df is None:
                warnings.append(f"[WARN] Missing history.csv: {run_dir}")
            else:
                history_df = history_df.copy()
                history_df["method_id"] = method_id
                history_df["method_label"] = method_label
                history_df["seed"] = seed
                history_df["run_dir"] = str(run_dir)

                if "cumulative_train_estimated_tflops" not in history_df.columns:
                    if "cumulative_train_estimated_flops" in history_df.columns:
                        history_df["cumulative_train_estimated_tflops"] = (
                            pd.to_numeric(history_df["cumulative_train_estimated_flops"], errors="coerce") / 1e12
                        )
                    else:
                        history_df["cumulative_train_estimated_tflops"] = np.nan

                history_rows.extend(history_df.to_dict(orient="records"))

            if profile_df is None:
                warnings.append(f"[WARN] Missing profile_batchsize_latency_vram.csv: {run_dir}")
            else:
                profile_df = profile_df.copy()
                profile_df["method_id"] = method_id
                profile_df["method_label"] = method_label
                profile_df["seed"] = seed
                profile_df["run_dir"] = str(run_dir)
                profile_rows.extend(profile_df.to_dict(orient="records"))

    aggregated_cms = {}
    for method_id, matrices in confusion_by_method.items():
        valid = [cm for cm in matrices if cm.shape == (2, 2) and np.isfinite(cm).all()]
        if valid:
            aggregated_cms[method_id] = np.sum(valid, axis=0)

    return (
        pd.DataFrame(run_rows),
        pd.DataFrame(profile_rows),
        pd.DataFrame(history_rows),
        aggregated_cms,
        warnings,
    )


def summarize_metrics(run_df: pd.DataFrame) -> pd.DataFrame:
    if len(run_df) == 0:
        return pd.DataFrame()

    grouped = run_df.groupby(["method_id", "method_label"], sort=False)
    summary_df = grouped.agg(
        n_runs=("seed", "count"),
        f1_mean=("test_macro_f1", "mean"),
        f1_std=("test_macro_f1", "std"),
        bacc_mean=("test_balanced_acc", "mean"),
        bacc_std=("test_balanced_acc", "std"),
        acc_mean=("test_accuracy", "mean"),
        acc_std=("test_accuracy", "std"),
        auc_mean=("test_auc", "mean"),
        auc_std=("test_auc", "std"),
    ).reset_index()

    summary_df["f1_mean_std"] = [
        format_mean_std(mean, std) for mean, std in zip(summary_df["f1_mean"], summary_df["f1_std"])
    ]
    summary_df["bacc_mean_std"] = [
        format_mean_std(mean, std) for mean, std in zip(summary_df["bacc_mean"], summary_df["bacc_std"])
    ]
    return summary_df


def select_seed0_df(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    if len(df) == 0 or "seed" not in df.columns:
        return pd.DataFrame(columns=df.columns)

    seed0_df = df.loc[df["seed"] == 0].copy()
    if len(seed0_df) > 0:
        return seed0_df

    sort_cols = ["method_id", "seed"] + [col for col in group_cols if col not in {"method_id", "seed"}]
    return df.sort_values(sort_cols).groupby(group_cols, as_index=False).head(1).copy()


def write_markdown_summary(summary_df: pd.DataFrame, out_path: Path) -> None:
    if len(summary_df) == 0:
        out_path.write_text("No summary data available.\n", encoding="utf-8")
        return

    table_df = summary_df[["method_label", "n_runs", "f1_mean_std", "bacc_mean_std"]].copy()
    table_df.columns = ["Method", "Runs", "Macro-F1 (mean+/-std, %)", "BAcc (mean+/-std, %)"]
    lines = [
        "| " + " | ".join(table_df.columns.tolist()) + " |",
        "| " + " | ".join(["---"] * len(table_df.columns)) + " |",
    ]
    for row in table_df.itertuples(index=False):
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex_summary(summary_df: pd.DataFrame, out_path: Path) -> None:
    if len(summary_df) == 0:
        out_path.write_text("% No summary data available.\n", encoding="utf-8")
        return

    lines = [
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Method & Macro-F1 (\%) & BAcc (\%) \\",
        r"\midrule",
    ]
    for _, row in summary_df.iterrows():
        lines.append(f"{row['method_label']} & {row['f1_mean_std']} & {row['bacc_mean_std']} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_confusion_matrices(
    summary_df: pd.DataFrame,
    cm_by_method: Dict[str, np.ndarray],
    out_csv: Path,
    out_json: Path,
) -> None:
    rows = []
    payload = {}

    for spec in METHOD_SPECS:
        method_id = spec["method_id"]
        if method_id not in cm_by_method:
            continue

        method_label = spec["label"]
        cm = np.asarray(cm_by_method[method_id], dtype=float)
        metric_row = summary_df.loc[summary_df["method_id"] == method_id]
        f1_text = metric_row.iloc[0]["f1_mean_std"] if len(metric_row) > 0 else None
        bacc_text = metric_row.iloc[0]["bacc_mean_std"] if len(metric_row) > 0 else None

        rows.append(
            {
                "method_id": method_id,
                "method_label": method_label,
                "macro_f1_mean_std": f1_text,
                "bacc_mean_std": bacc_text,
                "tn_true_unusable_pred_unusable": int(cm[0, 0]),
                "fp_true_unusable_pred_usable": int(cm[0, 1]),
                "fn_true_usable_pred_unusable": int(cm[1, 0]),
                "tp_true_usable_pred_usable": int(cm[1, 1]),
            }
        )
        payload[method_id] = {
            "method_label": method_label,
            "macro_f1_mean_std": f1_text,
            "bacc_mean_std": bacc_text,
            "class_order": ["unusable", "usable"],
            "confusion_matrix_sum_over_seeds": cm.astype(int).tolist(),
        }

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def plot_batch_vs_time(
    profile_df: pd.DataFrame,
    out_path: Path,
    dpi: int,
    title_suffix: str,
    axes_config: Dict[str, object],
) -> None:
    if len(profile_df) == 0:
        return

    method_labels = profile_df["method_label"].drop_duplicates().tolist()
    colors = build_color_map(method_labels)
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    for label in method_labels:
        sub = profile_df.loc[profile_df["method_label"] == label].copy()
        sub["batch_size"] = pd.to_numeric(sub["batch_size"], errors="coerce")
        sub["latency_ms_per_batch"] = pd.to_numeric(sub["latency_ms_per_batch"], errors="coerce")
        sub = sub.sort_values("batch_size")
        sub = sub.loc[np.isfinite(sub["batch_size"]) & np.isfinite(sub["latency_ms_per_batch"])]
        if len(sub) == 0:
            continue

        ax.plot(
            sub["batch_size"],
            sub["latency_ms_per_batch"],
            label=label,
            color=colors[label],
            marker=get_marker(label),
            linewidth=1.8,
            markersize=14 if label == "DHTL-Net" else 9,
        )

    finalize_axes(ax, f"Batch Size vs. Inference Time ({title_suffix})", "Batch Size", "Latency (ms / batch)")
    ax.set_xlim(*axes_config["xlim"])
    ax.set_ylim(*axes_config["ylim"])
    ax.set_xticks(axes_config["xticks"])
    add_inset_legend(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_vram_vs_latency(
    profile_df: pd.DataFrame,
    out_path: Path,
    dpi: int,
    title_suffix: str,
    axes_config: Dict[str, object],
) -> None:
    if len(profile_df) == 0:
        return

    method_labels = profile_df["method_label"].drop_duplicates().tolist()
    colors = build_color_map(method_labels)
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    for label in method_labels:
        sub = profile_df.loc[profile_df["method_label"] == label].copy()
        sub["latency_ms_per_batch"] = pd.to_numeric(sub["latency_ms_per_batch"], errors="coerce")
        sub["vram_peak_mb"] = pd.to_numeric(sub["vram_peak_mb"], errors="coerce")
        sub["batch_size"] = pd.to_numeric(sub["batch_size"], errors="coerce")
        sub = sub.sort_values("batch_size")
        sub = sub.loc[np.isfinite(sub["latency_ms_per_batch"]) & np.isfinite(sub["vram_peak_mb"])]
        if len(sub) == 0:
            continue

        ax.plot(
            sub["latency_ms_per_batch"],
            sub["vram_peak_mb"],
            label=label,
            color=colors[label],
            marker=get_marker(label),
            linewidth=1.8,
            markersize=14 if label == "DHTL-Net" else 9,
        )

    finalize_axes(ax, f"VRAM vs. Latency ({title_suffix})", "Latency (ms / batch)", "Peak VRAM (MB)")
    ax.set_xlim(*axes_config["xlim"])
    ax.set_ylim(*axes_config["ylim"])
    add_inset_legend(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_flops_vs_loss(
    history_df: pd.DataFrame,
    out_path: Path,
    dpi: int,
    loss_mode: str,
    title_suffix: str,
    axes_config: Dict[str, object],
) -> None:
    if len(history_df) == 0:
        return

    loss_col = "val_loss" if loss_mode == "val" and "val_loss" in history_df.columns else "loss"
    x_col = "cumulative_train_estimated_tflops"

    method_labels = history_df["method_label"].drop_duplicates().tolist()
    colors = build_color_map(method_labels)
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    for label in method_labels:
        sub = history_df.loc[history_df["method_label"] == label].copy()
        sub[x_col] = pd.to_numeric(sub[x_col], errors="coerce")
        sub[loss_col] = pd.to_numeric(sub[loss_col], errors="coerce")
        if "epoch" in sub.columns:
            sub["epoch"] = pd.to_numeric(sub["epoch"], errors="coerce")
            sub = sub.sort_values("epoch")
        else:
            sub = sub.sort_values(x_col)
        sub = sub.loc[np.isfinite(sub[x_col]) & np.isfinite(sub[loss_col])]
        if len(sub) == 0:
            continue

        ax.plot(
            sub[x_col],
            sub[loss_col],
            label=label,
            color=colors[label],
            marker=get_marker(label),
            linewidth=1.8,
            markersize=14 if label == "DHTL-Net" else 9,
        )

    ylabel = "Validation Loss" if loss_col == "val_loss" else "Training Loss"
    finalize_axes(ax, f"FLOPs vs. Loss ({title_suffix})", "Cumulative Training FLOPs (TFLOPs)", ylabel)
    ax.set_xlim(*axes_config["xlim"])
    ax.set_ylim(*axes_config["ylim"])
    add_inset_legend(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    runs_root = Path(args.runs_root)
    out_dir = Path(args.out_dir)
    figures_dir = out_dir / "figures"

    ensure_dir(out_dir)
    ensure_dir(figures_dir)

    run_df, profile_df, history_df, cm_by_method, warnings = collect_runs(runs_root)
    summary_df = summarize_metrics(run_df)
    seed0_run_df = select_seed0_df(run_df, ["method_id"])
    seed0_profile_df = select_seed0_df(profile_df, ["method_id", "batch_size"])
    seed0_history_df = select_seed0_df(history_df, ["method_id", "epoch"])
    shared_axes = build_shared_axes(seed0_profile_df, seed0_history_df, args.loss_mode)

    run_df.to_csv(out_dir / "per_run_metrics.csv", index=False)
    profile_df.to_csv(out_dir / "per_run_profiles.csv", index=False)
    history_df.to_csv(out_dir / "per_run_history.csv", index=False)
    summary_df.to_csv(out_dir / "method_summary.csv", index=False)
    seed0_run_df.to_csv(out_dir / "seed0_metrics.csv", index=False)
    seed0_profile_df.to_csv(out_dir / "seed0_profiles.csv", index=False)
    seed0_history_df.to_csv(out_dir / "seed0_history.csv", index=False)

    write_markdown_summary(summary_df, out_dir / "method_summary.md")
    write_latex_summary(summary_df, out_dir / "method_summary_latex.txt")
    export_confusion_matrices(
        summary_df,
        cm_by_method,
        out_dir / "confusion_matrices.csv",
        out_dir / "confusion_matrices.json",
    )

    for group in PLOT_GROUPS:
        group_id = group["group_id"]
        title_suffix = group["title_suffix"]
        method_ids = group["method_ids"]

        group_profile_df = filter_methods(seed0_profile_df, method_ids)
        group_history_df = filter_methods(seed0_history_df, method_ids)

        plot_batch_vs_time(
            group_profile_df,
            figures_dir / f"batch_vs_time_{group_id}.png",
            args.dpi,
            title_suffix,
            shared_axes["batch_vs_time"],
        )
        plot_vram_vs_latency(
            group_profile_df,
            figures_dir / f"vram_vs_latency_{group_id}.png",
            args.dpi,
            title_suffix,
            shared_axes["vram_vs_latency"],
        )
        plot_flops_vs_loss(
            group_history_df,
            figures_dir / f"flops_vs_loss_{group_id}.png",
            args.dpi,
            args.loss_mode,
            title_suffix,
            shared_axes["flops_vs_loss"],
        )

    manifest = {
        "runs_root": str(runs_root.resolve()),
        "out_dir": str(out_dir.resolve()),
        "loss_mode": args.loss_mode,
        "n_runs_loaded": int(len(run_df)),
        "n_methods_loaded": int(summary_df["method_id"].nunique()) if len(summary_df) > 0 else 0,
        "plot_data_policy": "Six figures are generated. Each uses seed0 only. FLOPs-vs-loss is an epoch-wise method curve.",
        "warnings": warnings,
    }
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    if warnings:
        print("\n".join(warnings))

    print(f"[OK] Saved aggregation outputs to: {out_dir.resolve()}")
    print(f"[OK] Loaded runs: {len(run_df)}")
    if len(summary_df) > 0:
        print(summary_df[["method_label", "n_runs", "f1_mean_std", "bacc_mean_std"]].to_string(index=False))


if __name__ == "__main__":
    main()

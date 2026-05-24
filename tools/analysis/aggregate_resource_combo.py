#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Generate combined Batch Size + Latency + VRAM figures from seed0 profile data.

This script does NOT modify aggregate_experiments_windows.py.

Default input:
  docs/results/latest2\seed0_profiles.csv

Default output directory:
  docs/results/latest2\figures_combo
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


plt.rcParams["font.family"] = "Times New Roman"

FIG_SIZE = (11.2, 6.4)
PLOT_FONTSIZE = 24
LINEWIDTH = 2.2


METHOD_SPECS = [
    {"method_id": "DHTL_wo_KMD", "label": "DHTL-Net"},
    {"method_id": "DAF", "label": "DHTL-Net w/o KMD"},
    {"method_id": "fusion_residual", "label": "Residual"},
    {"method_id": "fusion_fixed_interp", "label": "FixedInterp"},
    {"method_id": "fusion_learnable_interp", "label": "LearnableInterp"},
    {"method_id": "fusion_gating", "label": "Gating"},
    {"method_id": "fusion_adapter", "label": "Adapter"},
    {"method_id": "transfer_frozen", "label": "Frozen"},
    {"method_id": "transfer_all", "label": "All"},
    {"method_id": "transfer_l4", "label": "FT-S4"},
    {"method_id": "transfer_l34", "label": "FT-S3-S4"},
    {"method_id": "transfer_l234", "label": "FT-S2-S4"},
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary_dir",
        type=str,
        default=r"docs/results/latest2",
        help="Directory containing seed0_profiles.csv.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=r"docs/results/latest2\figures_combo",
        help="Directory to save combined figures.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI.")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def build_label_map() -> Dict[str, str]:
    return {spec["method_id"]: spec["label"] for spec in METHOD_SPECS}


def build_color_map(method_labels: List[str]) -> Dict[str, tuple]:
    cmap = plt.get_cmap("tab20")
    return {label: cmap(i % cmap.N) for i, label in enumerate(method_labels)}


def get_marker(method_label: str) -> str:
    return "*" if method_label == "DHTL-Net" else "o"


def compute_axis_limits(values: pd.Series, pad_ratio: float = 0.06) -> tuple[float, float]:
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


def finalize_axes(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_xlabel(xlabel, fontsize=PLOT_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=PLOT_FONTSIZE)
    ax.tick_params(axis="both", labelsize=PLOT_FONTSIZE)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.45)


def add_legends(ax, method_labels: List[str], colors: Dict[str, tuple]) -> None:
    method_handles = [
        Line2D(
            [0],
            [0],
            color=colors[label],
            marker=get_marker(label),
            linewidth=LINEWIDTH,
            markersize=14 if label == "DHTL-Net" else 8,
            label=label,
        )
        for label in method_labels
    ]
    type_handles = [
        Line2D([0], [0], color="black", linewidth=LINEWIDTH, marker="o", label="Latency"),
        Patch(facecolor="gray", edgecolor="black", alpha=0.28, label="VRAM"),
    ]

    legend_methods = ax.legend(
        handles=method_handles,
        loc="upper left",
        fontsize=PLOT_FONTSIZE - 4,
        frameon=True,
        framealpha=0.92,
        facecolor="white",
        edgecolor="0.8",
    )
    ax.add_artist(legend_methods)
    ax.legend(
        handles=type_handles,
        loc="upper right",
        fontsize=PLOT_FONTSIZE - 4,
        frameon=True,
        framealpha=0.92,
        facecolor="white",
        edgecolor="0.8",
    )


def plot_combo(
    profile_df: pd.DataFrame,
    out_path: Path,
    dpi: int,
    title_suffix: str,
    shared_latency_ylim: tuple[float, float],
    shared_vram_ylim: tuple[float, float],
    shared_batch_ticks: List[int],
) -> None:
    if len(profile_df) == 0:
        return

    method_labels = profile_df["method_label"].drop_duplicates().tolist()
    colors = build_color_map(method_labels)
    batch_sizes = sorted(int(x) for x in profile_df["batch_size"].dropna().unique().tolist())
    positions = np.arange(len(batch_sizes), dtype=float)
    n_methods = max(len(method_labels), 1)
    bar_width = min(0.78 / n_methods, 0.16)

    fig, ax_left = plt.subplots(figsize=FIG_SIZE)
    ax_right = ax_left.twinx()

    for idx, label in enumerate(method_labels):
        sub = profile_df.loc[profile_df["method_label"] == label].copy()
        sub["batch_size"] = pd.to_numeric(sub["batch_size"], errors="coerce")
        sub["latency_ms_per_batch"] = pd.to_numeric(sub["latency_ms_per_batch"], errors="coerce")
        sub["vram_peak_mb"] = pd.to_numeric(sub["vram_peak_mb"], errors="coerce")
        sub = sub.sort_values("batch_size")
        sub = sub.loc[
            np.isfinite(sub["batch_size"])
            & np.isfinite(sub["latency_ms_per_batch"])
            & np.isfinite(sub["vram_peak_mb"])
        ]
        if len(sub) == 0:
            continue

        batch_to_latency = dict(zip(sub["batch_size"].astype(int), sub["latency_ms_per_batch"]))
        batch_to_vram = dict(zip(sub["batch_size"].astype(int), sub["vram_peak_mb"]))
        line_x = positions + (idx - (n_methods - 1) / 2.0) * bar_width
        latency_y = [batch_to_latency.get(bs, np.nan) for bs in batch_sizes]
        vram_y = [batch_to_vram.get(bs, np.nan) for bs in batch_sizes]

        ax_right.bar(
            line_x,
            vram_y,
            width=bar_width * 0.92,
            color=colors[label],
            alpha=0.28,
            edgecolor=colors[label],
            linewidth=0.8,
            zorder=1,
        )
        ax_left.plot(
            line_x,
            latency_y,
            color=colors[label],
            marker=get_marker(label),
            linewidth=LINEWIDTH,
            markersize=14 if label == "DHTL-Net" else 8,
            zorder=3,
        )

    finalize_axes(
        ax_left,
        f"Batch Size vs. Latency and VRAM ({title_suffix})",
        "Batch Size",
        "Latency (ms / batch)",
    )
    ax_right.set_ylabel("Peak VRAM (MB)", fontsize=PLOT_FONTSIZE)
    ax_right.tick_params(axis="y", labelsize=PLOT_FONTSIZE)

    ax_left.set_xticks(positions)
    ax_left.set_xticklabels([str(x) for x in batch_sizes], fontsize=PLOT_FONTSIZE)
    ax_left.set_xlim(positions[0] - 0.6, positions[-1] + 0.6)
    ax_left.set_ylim(*shared_latency_ylim)
    ax_right.set_ylim(*shared_vram_ylim)

    add_legends(ax_left, method_labels, colors)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    summary_dir = Path(args.summary_dir)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    label_map = build_label_map()
    profile_df = read_csv(summary_dir / "seed0_profiles.csv").copy()
    profile_df["method_label"] = profile_df["method_id"].map(label_map).fillna(profile_df["method_label"])

    shared_latency_ylim = compute_axis_limits(profile_df["latency_ms_per_batch"], pad_ratio=0.06)
    shared_vram_ylim = compute_axis_limits(profile_df["vram_peak_mb"], pad_ratio=0.06)
    shared_batch_ticks = sorted(int(x) for x in profile_df["batch_size"].dropna().unique().tolist())

    for group in PLOT_GROUPS:
        group_df = profile_df.loc[profile_df["method_id"].isin(group["method_ids"])].copy()
        plot_combo(
            group_df,
            out_dir / f"batch_latency_vram_combo_{group['group_id']}.png",
            args.dpi,
            group["title_suffix"],
            shared_latency_ylim,
            shared_vram_ylim,
            shared_batch_ticks,
        )

    print(f"[OK] Saved combined figures to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()

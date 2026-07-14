#!/usr/bin/env python3
"""
Plot training curves from a train_raster() metrics.csv (isolated branch
training on GT rasters). Different CSV format from the COCO/all pipeline:
no train_mask_iou / val_junc_recall@Tpx, instead val_iou (region) or
val_precision/val_recall (line/point).

Usage:
    python plot_losses_raster.py                                   # auto-detect latest run under nl_brp_raster
    python plot_losses_raster.py /path/to/run/folder                # specific run
    python plot_losses_raster.py /path/to/run/folder --branch line  # force branch (else read from config.yml)
"""

import argparse
import glob
import os
import sys

import pandas as pd
import matplotlib.pyplot as plt
import yaml

LOSS_NAMES = ["loss_jloc", "loss_joff", "loss_mask", "loss_afm", "loss_remask"]

COLORS = {
    "loss_jloc"   : "#e41a1c",
    "loss_joff"   : "#377eb8",
    "loss_mask"   : "#4daf4a",
    "loss_afm"    : "#ff7f00",
    "loss_remask" : "#984ea3",
}

# which losses are actually non-zero for each isolated branch (see detector.py forward_train)
ACTIVE_LOSSES_BY_BRANCH = {
    "region":    ["loss_mask"],
    "line":      ["loss_afm"],    # afm_predictor (2ch dx/dy) + MAE against gt_afm (afm_op)
    "point":     ["loss_jloc", "loss_joff"],
    "point_sce": ["loss_jloc", "loss_joff"],  # same losses as "point", afm_head is frozen (no loss_afm)
}

BASE_DIR = "/mnt/DATA/IMANE/PLR-Net_output/PLR-Net/nl_brp_raster"


def find_latest_run():
    """Most recent timestamped run folder (any branch subfolder) with a metrics.csv.
    Sorted by file mtime, not path string, so the branch name doesn't affect the order."""
    pattern = os.path.join(BASE_DIR, "*", "*", "metrics.csv")
    runs = glob.glob(pattern)
    if not runs:
        return None
    latest = max(runs, key=os.path.getmtime)
    return os.path.dirname(latest)


def detect_branch(run_dir):
    """Read ACTIVE_BRANCH from the saved config.yml of this run."""
    config_path = os.path.join(run_dir, "config.yml")
    if not os.path.exists(config_path):
        return None
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg.get("MODEL", {}).get("ACTIVE_BRANCH", None)


def main():
    parser = argparse.ArgumentParser(
        description="Plot loss curves from a train_raster() metrics.csv file."
    )
    parser.add_argument("run_dir", nargs="?", default=None,
                        help="Run folder containing metrics.csv. If omitted, latest run is used.")
    parser.add_argument("--branch", choices=["region", "line", "point", "point_sce"], default=None,
                        help="Force branch type (else auto-detected from config.yml).")
    args = parser.parse_args()

    if args.run_dir is not None:
        run_dir = args.run_dir
    else:
        run_dir = find_latest_run()
        if run_dir is None:
            sys.exit(f"No runs found under {BASE_DIR}\nRun training first or pass the run folder as argument.")
        print(f"Auto-detected latest run: {run_dir}")

    csv_path = os.path.join(run_dir, "metrics.csv")
    if not os.path.exists(csv_path):
        sys.exit(f"File not found: {csv_path}")

    branch = args.branch or detect_branch(run_dir)
    if branch is None:
        sys.exit("Could not detect branch (region/line/point) — pass --branch explicitly.")
    print(f"Branch: {branch}")

    df = pd.read_csv(csv_path, na_values=["", " "])
    numeric_cols = ["epoch", "train_loss", "val_loss"] + \
                   ["w_" + k for k in LOSS_NAMES] + \
                   ["val_w_" + k for k in LOSS_NAMES] + \
                   ["val_iou", "val_precision", "val_recall"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    val_df = df.dropna(subset=["val_loss"])

    # -------- layout: 3 subplots --------
    #   [0] Total loss (train vs val)
    #   [1] Individual train losses (only the active branch's losses are non-zero)
    #   [2] Branch metric: IoU (region) or precision/recall (line/point)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- subplot 0 : total loss train vs val ---
    ax = axes[0]
    ax.plot(df["epoch"], df["train_loss"], color="steelblue", linewidth=1.4, label="Train loss")
    if len(val_df):
        ax.plot(val_df["epoch"], val_df["val_loss"], color="tomato", linewidth=1.4,
                marker="o", markersize=4, label="Val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Weighted total loss")
    ax.set_title(f"Total Loss (train vs val) — branch={branch}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- subplot 1 : weighted train losses per component ---
    # only plot the losses actually active for this branch — the other
    # columns are 0 for the whole run (branch isolation), plotting them
    # just adds flat lines at 0 with no information.
    ax = axes[1]
    for loss in ACTIVE_LOSSES_BY_BRANCH[branch]:
        col = "w_" + loss
        if col in df.columns:
            ax.plot(df["epoch"], df[col], color=COLORS[loss], linewidth=1.2, label=loss)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Weighted loss")
    ax.set_title(f"Train Losses (weighted) — active for {branch}")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # --- subplot 2 : branch-specific metric ---
    ax = axes[2]
    if branch == "region":
        if "val_iou" in df.columns:
            sub = df.dropna(subset=["val_iou"])
            if len(sub):
                ax.plot(sub["epoch"], sub["val_iou"], color="#2ca02c", linewidth=1.4,
                        marker="o", markersize=4, label="Val IoU")
                ax.set_ylim(0, 1)
        ax.set_ylabel("IoU")
        ax.set_title("Val Mask IoU (region branch)")
    else:
        for col, color, label in [("val_precision", "#d62728", "Precision"),
                                   ("val_recall", "#1f77b4", "Recall")]:
            if col in df.columns:
                sub = df.dropna(subset=[col])
                if len(sub):
                    ax.plot(sub["epoch"], sub[col], color=color, linewidth=1.4,
                            marker="o", markersize=4, label=label)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Score")
        ax.set_title(f"Val Precision/Recall ({branch} branch)")
    ax.set_xlabel("Epoch")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(run_dir, "loss_curves_raster.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved -> {out_path}")
    print(f"Epochs plotted : {int(df['epoch'].min())}-{int(df['epoch'].max())}  "
          f"|  Val checkpoints : {len(val_df)}")


if __name__ == "__main__":
    main()

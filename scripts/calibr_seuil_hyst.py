#!/usr/bin/env python3
"""
Calibrate hysteresis thresholds for the line (contour) branch.

This script collects line probability values from all validation patches,
separates them into foreground (true contour pixels) and background (pixels
far from any true contour), and prints their percentile statistics.
These statistics are then used to choose optimal low/high thresholds for
`apply_hysteresis_threshold` in `label_parcels()`.
"""

import os, sys
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

# Add the project root to the import search path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PLRNet.config import cfg
from PLRNet.detector import BuildingDetector
from inspect_branch_raster import run_branch, load_patch, load_gt

# ----- 1. Initialisation: model, config, checkpoint -----
config_file = 'config-files/PLR-Net_branch_all_raster.yaml'
checkpoint_file = '/mnt/DATA/IMANE/PLR-Net_output/PLR-Net/nl_brp_raster/all_raster/2026-07-14_21-27-08/checkpoints/best_val_loss.pth'

# Load and freeze the configuration
cfg.merge_from_file(config_file)
cfg.freeze()
device = cfg.MODEL.DEVICE
mean = list(cfg.DATASETS.IMAGE.PIXEL_MEAN)   # normalisation constants
std  = list(cfg.DATASETS.IMAGE.PIXEL_STD)

# Instantiate the model, load the checkpoint, and set to evaluation mode
model = BuildingDetector(cfg, test=True).to(device)
ckpt = torch.load(checkpoint_file, map_location=device)
state = ckpt.get('model', ckpt)                # handle both raw and wrapped checkpoints
state = {k.replace('module.', ''): v for k, v in state.items()}  # remove 'module.' prefix if present
model.load_state_dict(state, strict=False)
model.eval()

# ----- 2. List of validation patch stems -----
data_nl_patches = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data_nl', 'patches')
patches_root = os.path.join(data_nl_patches, 'train')
val_stems_file = os.path.join(data_nl_patches, 'val.txt')

with open(val_stems_file) as f:
    val_stems = [line.strip() for line in f if line.strip()]

# ----- 3. Collect line probability statistics over all validation patches -----
all_fg = []   # probabilities on true contour pixels
all_bg = []   # probabilities on unambiguous background pixels

for stem in val_stems:
    # Load image and run inference
    img_tensor, rgb = load_patch(stem, patches_root, mean, std)
    pred = run_branch(model, img_tensor, device, 'all_raster')
    gt_lines = load_gt(stem, patches_root, 'gt_lines')
    line_prob = pred['line']

    # Foreground: pixels that lie on a true contour (gt_lines == 1)
    fg_vals = line_prob[gt_lines > 0]

    # Background: pixels that are far (>5 px) from any true contour
    dist_to_gt = distance_transform_edt(gt_lines == 0)
    bg_vals = line_prob[dist_to_gt > 5]

    all_fg.append(fg_vals)
    all_bg.append(bg_vals)

# Merge all values into flat arrays
all_fg = np.concatenate(all_fg)
all_bg = np.concatenate(all_bg)

# Print percentile summaries for threshold calibration
print('fg (true contours):  p5=%.3f  median=%.3f  p95=%.3f' % (
    np.percentile(all_fg,5), np.median(all_fg), np.percentile(all_fg,95)))
print('bg (true background): p50=%.3f  p95=%.3f  p99=%.3f' % (
    np.percentile(all_bg,50), np.percentile(all_bg,95), np.percentile(all_bg,99)))
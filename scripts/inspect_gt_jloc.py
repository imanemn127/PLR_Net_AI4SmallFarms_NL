#!/usr/bin/env python3
"""
Inspect GT jloc statistics across the training dataset.

We want to know: how many junction pixels does each patch actually have?
If jloc_gt.sum() is tiny (e.g. < 5 pixels per patch on average), then
loss_joff is masked on almost no pixels and learns nothing.

Usage (from PLR-Net/ directory):
    /mnt/DATA/IMANE/ai4sf/bin/python scripts/inspect_gt_jloc.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader

from PLRNet.config import cfg
from PLRNet.dataset import build_train_dataset
from PLRNet.encoder import Encoder

cfg.merge_from_file("config-files/PLR-Net.yaml")
cfg.freeze()

print("Loading training dataset...")
train_dataset = build_train_dataset(cfg)
loader = DataLoader(
    train_dataset.dataset,
    batch_size=1,
    shuffle=False,
    num_workers=4,
    collate_fn=train_dataset.dataset.__class__.__getitem__.__func__
    if False else None,
)

encoder = Encoder(cfg)

# We iterate directly on the dataset (not the DataLoader) to avoid
# collate issues — we just need the raw annotations
dataset = train_dataset.dataset

n_junction_pixels = []   # total junction pixels (jloc_gt > 0) per patch
n_concave_pixels  = []   # class 1
n_convex_pixels   = []   # class 2
n_raw_junctions   = []   # number of junction coordinates in ann['junctions']
total_pixels      = 256 * 256

print(f"Scanning {len(dataset)} patches...\n")

for idx in range(len(dataset)):
    image, ann = dataset[idx]

    # build GT targets the same way the encoder does during training
    junctions = torch.tensor(ann['junctions'], dtype=torch.float32)
    junc_tag  = torch.tensor(ann['juncs_tag'],  dtype=torch.long)

    H, W = 256, 256
    jmap = torch.zeros((H, W), dtype=torch.long)

    if len(junctions) > 0:
        xint = junctions[:, 0].long().clamp(0, W - 1)
        yint = junctions[:, 1].long().clamp(0, H - 1)
        jmap[yint, xint] = junc_tag

    n_junc_px  = int((jmap > 0).sum())
    n_conc_px  = int((jmap == 1).sum())
    n_conv_px  = int((jmap == 2).sum())

    n_junction_pixels.append(n_junc_px)
    n_concave_pixels.append(n_conc_px)
    n_convex_pixels.append(n_conv_px)
    n_raw_junctions.append(len(ann['junctions']))

    if idx % 500 == 0:
        print(f"  [{idx:4d}/{len(dataset)}]  "
              f"raw_juncs={len(ann['junctions']):4d}  "
              f"junc_pixels={n_junc_px:4d}  "
              f"(concave={n_conc_px}, convex={n_conv_px})")

n_junction_pixels = np.array(n_junction_pixels)
n_raw_junctions   = np.array(n_raw_junctions)

print("\n" + "=" * 55)
print("GT JLOC STATISTICS (training set)")
print("=" * 55)
print(f"  Total patches scanned     : {len(dataset)}")
print()
print(f"  Raw junctions per patch")
print(f"    mean  : {n_raw_junctions.mean():.1f}")
print(f"    median: {np.median(n_raw_junctions):.1f}")
print(f"    min   : {n_raw_junctions.min()}")
print(f"    max   : {n_raw_junctions.max()}")
print()
print(f"  Junction pixels in jmap (jloc_gt > 0)")
print(f"    mean  : {n_junction_pixels.mean():.2f}  / {total_pixels} px")
print(f"    median: {np.median(n_junction_pixels):.2f}")
print(f"    min   : {n_junction_pixels.min()}")
print(f"    max   : {n_junction_pixels.max()}")
print(f"    % of image area (mean): {100 * n_junction_pixels.mean() / total_pixels:.4f}%")
print()

# How many patches have ZERO junction pixels? (void or no annotations)
n_zero = int((n_junction_pixels == 0).sum())
print(f"  Patches with 0 junction pixels : {n_zero} ({100*n_zero/len(dataset):.1f}%)")

# Key diagnosis: raw junctions vs jmap pixels
# If raw_juncs >> jmap_pixels, junctions are overwriting each other
n_collisions = n_raw_junctions - n_junction_pixels
n_collision_patches = int((n_collisions > 0).sum())
print()
print(f"  Pixel collisions (2 junctions on the same pixel)")
print(f"    patches with collisions : {n_collision_patches} ({100*n_collision_patches/len(dataset):.1f}%)")
if n_collision_patches > 0:
    mean_loss = n_collisions[n_collisions > 0].mean()
    print(f"    mean junctions lost per affected patch : {mean_loss:.1f}")

print()
print("INTERPRETATION")
print("-" * 55)
ratio = n_junction_pixels.mean() / total_pixels
if ratio < 0.005:
    print(f"  [!] Only {ratio*100:.3f}% of pixels are junctions.")
    print(f"      loss_joff mask is nearly empty — the L1 loss on")
    print(f"      sub-pixel offsets trains on almost nothing.")
    print(f"      -> Consider Gaussian heatmap instead of one-hot jmap.")
else:
    print(f"  [OK] {ratio*100:.3f}% junction pixels — loss_joff mask is dense enough.")
print("=" * 55)

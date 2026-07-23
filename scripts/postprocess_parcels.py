#!/usr/bin/env python3
"""
postprocess_parcels.py — turn raw region/line/point predictions from an
all_raster checkpoint into per-parcel simplified points
--> It cuts the region mask into individual parcels using the line
contours, assigns each detected point to its parcel, then merges points
that are close together within the same parcel.

Steps:
  1. region_bin AND NOT line_bin -> connected components -> parcel labels
  2. each point candidate (point_prob > POINT_THRESHOLD) gets the label
     of the parcel it falls on
  3. within each parcel, cluster points closer than CLUSTER_DIST_PX and
     replace each cluster by its centroid

This does not vectorize into polygons yet — it produces a per-parcel
point cloud, the input for the next vectorization step.

Usage (from PLR-Net/ directory):
  /mnt/DATA/IMANE/ai4sf/bin/python scripts/postprocess_parcels.py \
      --config     config-files/PLR-Net_branch_all_raster.yaml \
      --checkpoint /mnt/DATA/IMANE/PLR-Net_output/PLR-Net/nl_brp_raster/all_raster/2026-07-14_21-27-08/checkpoints/best_val_loss.pth \
      --output     /home/imane/DATA/PLR-Net_output/postprocess_parcels
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.cluster.hierarchy import fclusterdata
from skimage.filters import apply_hysteresis_threshold
from skimage.measure import label
from skimage.morphology import remove_small_objects, skeletonize
from skimage.segmentation import watershed
from scipy.ndimage import maximum_filter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PLRNet.config import cfg
from PLRNet.detector import BuildingDetector
from inspect_branch_raster import run_branch, load_patch, load_gt, ALL_RASTER_SUBBRANCHES

REGION_THRESHOLD  = 0.5
POINT_THRESHOLD   = 0.5
CLUSTER_DIST_PX   = 5.0
LINE_HYST_LOW     = 0.15  # hysteresis low threshold — reconnects weak-but-real contour pixels
LINE_HYST_HIGH    = 0.38  # hysteresis high threshold — seeds must be confident contour pixels
MIN_PARCEL_PX     = 20   # drops slivers cut off by line-branch noise


def label_parcels(region_prob, line_prob):
    """region_prob, line_prob: (H,W) float arrays in [0,1].
    Returns an (H,W) int label map: 0 = background, 1..N = parcel id.

    NMS keep only local maxima of line_prob, drops diffuse noise before thresholding.
    Hysteresis rejects isolated low-confidence noise while still reconnecting
    weak-but-real contour stretches, unlike a single fixed threshold.
    skeletonize restores the 1px width lost to hysteresis growing the contour
    into a blob.
    """
    region_bin = region_prob > REGION_THRESHOLD
    
    #local_max = (line_prob == maximum_filter(line_prob, size=3))
    #line_prob_nms = np.where(local_max, line_prob, 0.0)
    
    line_hyst  = apply_hysteresis_threshold(line_prob, LINE_HYST_LOW, LINE_HYST_HIGH)
    
    line_bin   = skeletonize(line_hyst)
   
    cut = region_bin & ~line_bin
    labels = label(cut, connectivity=1)
    labels = remove_small_objects(labels, max_size=MIN_PARCEL_PX)
    
    labels = watershed(line_prob, markers=labels, mask=region_bin)
    return labels


def extract_point_candidates(point_prob):
    """point_prob: (H,W) float array. Returns (N,2) array of (x,y) pixel coords."""
    ys, xs = np.where(point_prob > POINT_THRESHOLD)
    return np.stack([xs, ys], axis=1).astype(np.float32)


def assign_and_simplify(points_xy, parcel_labels, cluster_dist_px=CLUSTER_DIST_PX):
    """Assign each point to a parcel label, then merge points closer than
    cluster_dist_px within the same parcel into their centroid.
    Returns dict {parcel_id: (M,2) array of simplified (x,y) points}."""
    if len(points_xy) == 0:
        return {}

    xs = points_xy[:, 0].astype(int)
    ys = points_xy[:, 1].astype(int)
    h, w = parcel_labels.shape
    xs = np.clip(xs, 0, w - 1)
    ys = np.clip(ys, 0, h - 1)
    point_parcel_ids = parcel_labels[ys, xs]

    simplified = {}
    for parcel_id in np.unique(point_parcel_ids):
        if parcel_id == 0:
            continue  # background, not a real parcel
        parcel_points = points_xy[point_parcel_ids == parcel_id]
        if len(parcel_points) == 1:
            simplified[parcel_id] = parcel_points
            continue
        clusters = fclusterdata(parcel_points, t=cluster_dist_px, criterion='distance')
        centroids = np.array([parcel_points[clusters == c].mean(axis=0)
                              for c in np.unique(clusters)])
        simplified[parcel_id] = centroids

    return simplified


def save_figure(rgb, parcel_labels, raw_points, simplified_points_by_parcel, img_name, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=120)
    fig.patch.set_facecolor('#111111')

    axes[0].imshow(rgb)
    axes[0].set_title('RGB image', color='white', fontsize=10, pad=4)

    axes[1].imshow(parcel_labels, cmap='nipy_spectral', interpolation='nearest')
    if len(raw_points):
        axes[1].scatter(raw_points[:, 0], raw_points[:, 1], s=1, c='white', alpha=0.5)
    n_parcels = int(parcel_labels.max())
    axes[1].set_title(f'Parcel labels ({n_parcels} parcels) + raw points',
                      color='white', fontsize=10, pad=4)

    axes[2].imshow(rgb)
    all_simplified = (np.concatenate(list(simplified_points_by_parcel.values()))
                      if simplified_points_by_parcel else np.empty((0, 2)))
    if len(all_simplified):
        axes[2].scatter(all_simplified[:, 0], all_simplified[:, 1],
                        s=8, c='red', marker='+', linewidths=1.2)
    n_raw = len(raw_points)
    n_simplified = len(all_simplified)
    axes[2].set_title(f'Simplified points per parcel ({n_raw} -> {n_simplified})',
                      color='white', fontsize=10, pad=4)

    for ax in axes:
        ax.axis('off')

    plt.tight_layout(pad=0.5)
    out_path = os.path.join(out_dir, f'{img_name}.png')
    plt.savefig(out_path, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',     required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output',     default='/home/imane/DATA/PLR-Net_output/postprocess_parcels')
    parser.add_argument('--n-images',   type=int, default=6)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    cfg.merge_from_file(args.config)
    cfg.freeze()
    device = cfg.MODEL.DEVICE
    active_branch = cfg.MODEL.ACTIVE_BRANCH
    assert active_branch == 'all_raster', \
        f"postprocess_parcels.py needs a checkpoint trained with ACTIVE_BRANCH=all_raster, got {active_branch!r}"
    mean = list(cfg.DATASETS.IMAGE.PIXEL_MEAN)
    std  = list(cfg.DATASETS.IMAGE.PIXEL_STD)

    model = BuildingDetector(cfg, test=True).to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    state = ckpt.get('model', ckpt)
    state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval()
    epoch = ckpt.get('epoch', '?')
    print(f'Checkpoint loaded — epoch {epoch}')

    data_nl_patches = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data_nl', 'patches')
    patches_root = os.path.join(data_nl_patches, 'train')
    val_stems_file = os.path.join(data_nl_patches, 'val.txt')

    with open(val_stems_file) as f:
        val_stems = [line.strip() for line in f if line.strip()]

    step = max(1, len(val_stems) // args.n_images)
    selected = val_stems[::step][:args.n_images]

    print(f'\nProcessing {len(selected)} images -> {args.output}\n')
    print(f'{"image":<40} {"n_parcels":>10} {"n_raw_points":>13} {"n_simplified":>13}')
    print('-' * 80)

    for stem in selected:
        img_tensor, rgb = load_patch(stem, patches_root, mean, std)
        pred = run_branch(model, img_tensor, device, 'all_raster')

        parcel_labels = label_parcels(pred['region'], pred['line'])
        raw_points = extract_point_candidates(pred['point'])
        simplified = assign_and_simplify(raw_points, parcel_labels)

        out_path = save_figure(rgb, parcel_labels, raw_points, simplified, stem, args.output)
        n_simplified = sum(len(v) for v in simplified.values())
        print(f'{stem:<40} {int(parcel_labels.max()):>10} {len(raw_points):>13} {n_simplified:>13}')

    print(f'\nDone. Figures saved to: {args.output}')


if __name__ == '__main__':
    main()

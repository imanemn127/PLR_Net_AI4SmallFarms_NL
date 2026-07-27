#!/usr/bin/env python3
"""
vectorize_parcels.py — build one simplified polygon per parcel from an
all_raster checkpoint's region/line/point predictions.

Builds on postprocess_parcels.py's parcel labeling and per-parcel point
simplification. For each labeled parcel:
  1. extract its mask outline (cv2.findContours)
  2. simplify it with Douglas-Peucker (cv2.approxPolyDP, epsilon=2px)
The simplified per-parcel points from postprocess_parcels are drawn on
top for visual comparison only — they do not feed into the polygon itself.

Usage (from PLR-Net/ directory):
  /mnt/DATA/IMANE/ai4sf/bin/python scripts/vectorize_parcels.py \
      --config     config-files/PLR-Net_branch_all_raster.yaml \
      --checkpoint /mnt/DATA/IMANE/PLR-Net_output/PLR-Net/nl_brp_raster/all_raster/2026-07-14_21-27-08/checkpoints/best_val_loss.pth \
      --output     /home/imane/DATA/PLR-Net_output/vectorize_parcels
"""

import argparse
import os
import sys

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PLRNet.config import cfg
from PLRNet.detector import BuildingDetector
from inspect_branch_raster import run_branch, load_patch
from postprocess_parcels import label_parcels, extract_point_candidates, assign_and_simplify

DP_EPSILON_PX = 2.0
MIN_PARCEL_AREA_PX = 20  # drop slivers too small to be a real parcel
SNAP_DIST_PX=5.0

def min_dist_to_points(vertex, points):
    """vertex: (2,) array (x,y). points: (M,2) array. Returns the distance to the closest point, or inf if points is empty."""
    if len(points) == 0:
        return np.inf
    dists = np.sqrt(((points - vertex) ** 2).sum(axis=1))
    return dists.min()


def vectorize_parcel(mask, parcel_points, epsilon_px=DP_EPSILON_PX, snap_dist_px=SNAP_DIST_PX):
    """mask: (H,W) bool array for one parcel. parcel_points: (M,2) array of
    point-branch corners assigned to this parcel (possibly empty).
    Returns (N,2) polygon vertices in (x,y) pixel coords, or None."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    simplified = cv2.approxPolyDP(largest, epsilon_px, closed=True)
    pts = simplified.reshape(-1, 2)

    if len(parcel_points) == 0:
        return pts  # no points detected, we keep the contour as it is

    keep = np.array([min_dist_to_points(p, parcel_points) < snap_dist_px for p in pts])
    filtered = pts[keep]

    if len(filtered) < 3:
        return pts   # the filter removed too many points, otherwise the polygon becomes invalid

    return filtered


def vectorize_all_parcels(parcel_labels, simplified_points_by_parcel,
                          epsilon_px=DP_EPSILON_PX, min_area_px=MIN_PARCEL_AREA_PX,
                          snap_dist_px=SNAP_DIST_PX):
    """Returns dict {parcel_id: (N,2) polygon vertices}."""
    polygons = {}
    for parcel_id in np.unique(parcel_labels):
        if parcel_id == 0:
            continue
        mask = parcel_labels == parcel_id
        if mask.sum() < min_area_px:
            continue
        parcel_points = simplified_points_by_parcel.get(parcel_id, np.empty((0, 2)))
        poly = vectorize_parcel(mask, parcel_points, epsilon_px, snap_dist_px)
        if poly is not None and len(poly) >= 3:
            polygons[parcel_id] = poly
    return polygons


def save_figure(rgb, polygons, simplified_points_by_parcel, img_name, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), dpi=120)
    fig.patch.set_facecolor('#111111')

    for ax in axes:
        ax.imshow(rgb)
        ax.axis('off')

    for parcel_id, poly in polygons.items():
        closed = np.vstack([poly, poly[0]])
        axes[0].plot(closed[:, 0], closed[:, 1], '-', color='#00ff88', linewidth=1.0)
    axes[0].set_title(f'Vectorized polygons ({len(polygons)} parcels)',
                      color='white', fontsize=10, pad=4)

    for parcel_id, poly in polygons.items():
        closed = np.vstack([poly, poly[0]])
        axes[1].plot(closed[:, 0], closed[:, 1], '-', color='#00ff88', linewidth=0.8)
    all_points = (np.concatenate(list(simplified_points_by_parcel.values()))
                 if simplified_points_by_parcel else np.empty((0, 2)))
    if len(all_points):
        axes[1].scatter(all_points[:, 0], all_points[:, 1],
                        s=10, c='red', marker='+', linewidths=1.2)
    axes[1].set_title('Polygons + point-branch corners (overlay)',
                      color='white', fontsize=10, pad=4)

    plt.tight_layout(pad=0.5)
    out_path = os.path.join(out_dir, f'{img_name}.png')
    plt.savefig(out_path, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',     required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output',     default='/home/imane/DATA/PLR-Net_output/vectorize_parcels')
    parser.add_argument('--n-images',   type=int, default=6)
    parser.add_argument('--epsilon',    type=float, default=DP_EPSILON_PX,
                        help='Douglas-Peucker simplification tolerance in pixels')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    cfg.merge_from_file(args.config)
    cfg.freeze()
    device = cfg.MODEL.DEVICE
    active_branch = cfg.MODEL.ACTIVE_BRANCH
    assert active_branch == 'all_raster', \
        f"vectorize_parcels.py needs a checkpoint trained with ACTIVE_BRANCH=all_raster, got {active_branch!r}"
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
    print(f'{"image":<40} {"n_parcels":>10} {"n_polygons":>11} {"avg_vertices":>13}')
    print('-' * 80)

    for stem in selected:
        img_tensor, rgb = load_patch(stem, patches_root, mean, std)
        pred = run_branch(model, img_tensor, device, 'all_raster')

        parcel_labels = label_parcels(pred['region'], pred['line'])
        raw_points = extract_point_candidates(pred['point'])
        simplified_points = assign_and_simplify(raw_points, parcel_labels)
        polygons = vectorize_all_parcels(parcel_labels, simplified_points, epsilon_px=args.epsilon)

        out_path = save_figure(rgb, polygons, simplified_points, stem, args.output)
        avg_vertices = np.mean([len(p) for p in polygons.values()]) if polygons else 0.0
        print(f'{stem:<40} {int(parcel_labels.max()):>10} {len(polygons):>11} {avg_vertices:>13.1f}')

    print(f'\nDone. Figures saved to: {args.output}')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
vectorize_parcels_coverage.py — coverage-aware alternative to
vectorize_parcels.py. Instead of simplifying each parcel contour on its
own (cv2.approxPolyDP), which breaks shared edges between neighboring
parcels, this vectorizes the whole label raster in one pass and
simplifies it as a topology-preserving coverage.

Steps:
  1. label_parcels() (same as postprocess_parcels.py, unchanged)
  2. rasterio.features.shapes on the full label raster -> one polygon per
     connected component, following pixel edges exactly
  3. GeoDataFrame.simplify_coverage(tolerance) -> Visvalingam-Whyatt
     simplification that keeps shared edges identical across parcels

Usage (from PLR-Net/ directory):
  /mnt/DATA/IMANE/ai4sf/bin/python scripts/vectorize_parcels_coverage.py \
      --config     config-files/PLR-Net_branch_all_raster.yaml \
      --checkpoint /mnt/DATA/IMANE/PLR-Net_output/PLR-Net/nl_brp_raster/all_raster/2026-07-14_21-27-08/checkpoints/best_val_loss.pth \
      --output     /home/imane/DATA/PLR-Net_output/vectorize_parcels_coverage
"""

import argparse
import os
import sys

import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import rasterio.features
import torch
from shapely.geometry import shape

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PLRNet.config import cfg
from PLRNet.detector import BuildingDetector
from inspect_branch_raster import run_branch, load_patch
from postprocess_parcels import label_parcels

COVERAGE_TOLERANCE_PX = 2.0
MIN_PARCEL_AREA_PX = 20


def vectorize_labels_to_coverage(parcel_labels, tolerance_px=COVERAGE_TOLERANCE_PX,
                                  min_area_px=MIN_PARCEL_AREA_PX):
    """parcel_labels: (H,W) int array, 0 = background.
    Returns a GeoDataFrame(parcel_id, geometry) in pixel (x,y) coords."""
    mask = parcel_labels > 0
    shapes_iter = rasterio.features.shapes(
        parcel_labels.astype(np.int32), mask=mask, connectivity=4)

    records = [{'parcel_id': int(value), 'geometry': shape(geom)}
               for geom, value in shapes_iter]
    if not records:
        return gpd.GeoDataFrame(columns=['parcel_id', 'geometry'])

    gdf = gpd.GeoDataFrame(records)
    gdf = gdf[gdf.geometry.area >= min_area_px].reset_index(drop=True)
    if len(gdf) == 0:
        return gdf

    gdf['geometry'] = gdf.geometry.simplify_coverage(tolerance_px)
    return gdf


def save_figure(rgb, gdf, img_name, out_dir):
    fig, ax = plt.subplots(figsize=(8, 8), dpi=120)
    fig.patch.set_facecolor('#111111')

    ax.imshow(rgb)
    ax.axis('off')

    for geom in gdf.geometry:
        polys = geom.geoms if geom.geom_type == 'MultiPolygon' else [geom]
        for poly in polys:
            xs, ys = poly.exterior.xy
            ax.plot(xs, ys, '-', color='#00ff88', linewidth=1.0)

    ax.set_title(f'Coverage-simplified polygons ({len(gdf)} parcels)',
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
    parser.add_argument('--output',     default='/home/imane/DATA/PLR-Net_output/vectorize_parcels_coverage')
    parser.add_argument('--n-images',   type=int, default=6)
    parser.add_argument('--tolerance',  type=float, default=COVERAGE_TOLERANCE_PX,
                        help='simplify_coverage tolerance in pixels')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    cfg.merge_from_file(args.config)
    cfg.freeze()
    device = cfg.MODEL.DEVICE
    active_branch = cfg.MODEL.ACTIVE_BRANCH
    assert active_branch == 'all_raster', \
        f"vectorize_parcels_coverage.py needs a checkpoint trained with ACTIVE_BRANCH=all_raster, got {active_branch!r}"
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
        gdf = vectorize_labels_to_coverage(parcel_labels, tolerance_px=args.tolerance)

        out_path = save_figure(rgb, gdf, stem, args.output)
        avg_vertices = (np.mean([len(g.exterior.coords) - 1 for g in gdf.geometry])
                        if len(gdf) else 0.0)
        print(f'{stem:<40} {int(parcel_labels.max()):>10} {len(gdf):>11} {avg_vertices:>13.1f}')

    print(f'\nDone. Figures saved to: {args.output}')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
export_gt_geotiff.py — export the EXACT GT used during training as GeoTIFFs.

The 3 outputs are built using the same code as train_dataset.py + encoder.py,
so what you see in QGIS is exactly what the network learns from.

Output files (georeferenced, open directly in QGIS):
  <stem>_gt_region.tif   binary mask  (0=background, 1=parcel)
  <stem>_gt_nodes.tif    junction map (0=background, 1=corner pixel)
  <stem>_gt_lines.tif    boundary map (0=background, 1=boundary pixel)

In QGIS: drag and drop → Symbologie → Bande grise unique → Min=0, Max=1
Value 0 = black (background), Value 1 = white (feature)

Usage:
  /mnt/DATA/IMANE/ai4sf/bin/python scripts/export_gt_geotiff.py \
      --stem NL_train_z1_r000000_c002460 \
      --output /home/imane/DATA/PLR-Net_output/patch_gt_compare
"""

import argparse
import os
import sys
import json

import numpy as np
import rasterio
import cv2
from pycocotools import mask as maskUtils
from shapely.geometry import Polygon

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_NL    = '/mnt/DATA/IMANE/PLR-Net/data_nl'
PATCHES    = os.path.join(DATA_NL, 'patches')
TRAIN_COCO = os.path.join(DATA_NL, 'coco', 'train_coco.json')
VAL_COCO   = os.path.join(DATA_NL, 'coco', 'val_coco.json')


def find_patch(stem):
    for split in ['train', 'test']:
        p = os.path.join(PATCHES, split, 'images', stem + '.tif')
        if os.path.exists(p):
            return p, split
    return None, None


def find_coco_anns(stem):
    for coco_path in [TRAIN_COCO, VAL_COCO]:
        with open(coco_path) as f:
            coco = json.load(f)
        for img in coco['images']:
            if stem in img['file_name']:
                img_id = img['id']
                anns = [a for a in coco['annotations'] if a['image_id'] == img_id]
                return anns, img['height'], img['width']
    return [], 256, 256


def save_geotiff(array, path, crs, transform):
    """Save 2D float32 array as single-band GeoTIFF with patch georef."""
    arr = array.astype(np.float32)
    with rasterio.open(
        path, 'w',
        driver='GTiff',
        height=arr.shape[0],
        width=arr.shape[1],
        count=1,
        dtype=np.float32,
        crs=crs,
        transform=transform,
        compress='lzw',
    ) as dst:
        dst.write(arr, 1)


def build_gt_region(anns, H, W):
    """
    Exact same logic as train_dataset.py line 87:
      seg_mask += coco.annToMask(ann)
      seg_mask = np.clip(seg_mask, 0, 1)
    """
    seg_mask = np.zeros((H, W), dtype=np.float32)
    for ann in anns:
        rles = maskUtils.frPyObjects(ann['segmentation'], H, W)
        rle  = maskUtils.merge(rles)
        seg_mask += maskUtils.decode(rle).astype(np.float32)
    return np.clip(seg_mask, 0, 1)


def build_gt_nodes(anns, H, W):
    """
    Exact same logic as encoder.py:
      xint, yint = junctions[:,0].long(), junctions[:,1].long()
      jmap[yint, xint] = 1.0   (binary, after our BCE change)
    Corners extracted same way as train_dataset.py (convex_hull tagging).
    """
    jmap = np.zeros((H, W), dtype=np.float32)
    n_total = 0

    for ann in anns:
        coords = np.array(ann['segmentation'][0]).reshape(-1, 2)
        # remove closing duplicate (same as train_dataset.py points = segm[:-1])
        if np.allclose(coords[0], coords[-1]):
            coords = coords[:-1]

        poly = Polygon(coords)
        if poly.area <= 0:
            continue

        for pt in coords:
            xi = int(np.clip(pt[0], 0, W - 1))
            yi = int(np.clip(pt[1], 0, H - 1))
            jmap[yi, xi] = 1.0
            n_total += 1

    return jmap, n_total


def build_gt_lines(anns, H, W):
    """
    The AFM (line GT) is computed from parcel edges.
    Here we approximate it as the boundary pixels of the region mask
    (same pixels the AFM is built from in encoder.py via the C++ afm() op).
    boundary = region mask XOR eroded region mask
    """
    region = build_gt_region(anns, H, W)
    mask_u8 = (region * 255).astype(np.uint8)
    kernel  = np.ones((3, 3), np.uint8)
    eroded  = cv2.erode(mask_u8, kernel, iterations=1)
    boundary = ((mask_u8 > 0) & (eroded == 0)).astype(np.float32)
    return boundary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stem',   required=True, help='Patch name without extension')
    parser.add_argument('--output', default='/home/imane/DATA/PLR-Net_output/patch_gt_compare')
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    stem = args.stem

    # get patch georef (CRS + transform) from the image tif
    img_path, split = find_patch(stem)
    if img_path is None:
        print(f'ERROR: patch {stem} not found in train or test'); sys.exit(1)

    with rasterio.open(img_path) as src:
        crs       = src.crs
        transform = src.transform
        W, H      = src.width, src.height

    print(f'Patch  : {stem}')
    print(f'CRS    : {crs}')
    print(f'Size   : {W} x {H}')

    # load COCO annotations
    anns, _, _ = find_coco_anns(stem)
    print(f'Parcels: {len(anns)} annotations in COCO')

    # build the 3 GT maps (exact same code as training pipeline)
    gt_region          = build_gt_region(anns, H, W)
    gt_nodes, n_total  = build_gt_nodes(anns, H, W)
    gt_lines           = build_gt_lines(anns, H, W)

    n_node_px = int((gt_nodes > 0).sum())
    n_lost    = n_total - n_node_px

    print(f'\nGT region : {int(gt_region.sum())} parcel pixels  ({100*gt_region.mean():.1f}% of image)')
    print(f'GT nodes  : {n_total} corners in polygons → {n_node_px} unique pixels  ({n_lost} lost to pixel collisions, {100*n_lost/max(n_total,1):.1f}%)')
    print(f'GT lines  : {int(gt_lines.sum())} boundary pixels')

    # save georeferenced GeoTIFFs
    out_region = os.path.join(args.output, f'{stem}_gt_region.tif')
    out_nodes  = os.path.join(args.output, f'{stem}_gt_nodes.tif')
    out_lines  = os.path.join(args.output, f'{stem}_gt_lines.tif')

    save_geotiff(gt_region, out_region, crs, transform)
    save_geotiff(gt_nodes,  out_nodes,  crs, transform)
    save_geotiff(gt_lines,  out_lines,  crs, transform)

    print(f'\nSaved GeoTIFFs (drag into QGIS):')
    print(f'  {out_region}')
    print(f'  {out_nodes}')
    print(f'  {out_lines}')
    print(f'\nIn QGIS: Symbologie -> Bande grise unique -> Min=0, Max=1')
    print(f'Value 0 = black (background)   Value 1 = white (feature)')
    print(f'\nCompare gt_nodes with your coins_brp_raster.tif from QGIS.')
    print(f'If gt_nodes has many more white pixels -> rasterio.shapes creates fake corners.')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
eval_vectorize_coverage.py — quantitative evaluation of the post-processed
pipeline (all_raster -> label_parcels -> rasterio.shapes -> simplify_coverage)
against COCO ground truth, using the same vector-level metrics as eval_full.py:
  IoU     : pixel IoU between the union of predicted polygons and GT polygons
  PoLiS   : polygon similarity distance, matched GT-pred pairs at IoU > 0.5
  Junction recall @3/5/8px : fraction of GT corners matched by a predicted vertex

This does not touch the model or label_parcels() — it only measures how the
current post-processing chain performs, to compare against Table 10 numbers
already collected by eval_full.py for the original PLR-Net pipeline.

Usage (from PLR-Net/ directory):
  /mnt/DATA/IMANE/ai4sf/bin/python scripts/eval_vectorize_coverage.py \
      --config     config-files/PLR-Net_branch_all_raster.yaml \
      --checkpoint /mnt/DATA/IMANE/PLR-Net_output/PLR-Net/nl_brp_raster/all_raster/2026-07-14_21-27-08/checkpoints/best_val_loss.pth \
      --split test \
      --output /home/imane/DATA/PLR-Net_output/eval_vectorize_coverage
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
import rasterio
import torch
from pycocotools.coco import COCO
from pycocotools import mask as cocomask
from scipy.spatial.distance import cdist
from shapely.geometry import Point as ShapelyPoint, Polygon as ShapelyPolygon
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PLRNet.config import cfg
from PLRNet.detector import BuildingDetector
from PLRNet.utils.metrics.cIoU import calc_IoU
from inspect_branch_raster import run_branch, load_patch
from postprocess_parcels import label_parcels
from vectorize_parcels_coverage import vectorize_labels_to_coverage, COVERAGE_TOLERANCE_PX, MIN_PARCEL_AREA_PX

_BASE   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANN_DIR = os.path.join(_BASE, "data_nl", "coco")

RECALL_THRESHOLDS = [3, 5, 8]


def compute_junction_recall(gt_polys, pred_verts, thresholds=RECALL_THRESHOLDS):
    """gt_polys: list of (N,2) arrays. pred_verts: (M,2) array of all predicted
    polygon vertices in the image. Fraction of GT corners matched within T px."""
    if not gt_polys:
        return {t: float('nan') for t in thresholds}
    gt_corners = np.vstack(gt_polys)
    if len(pred_verts) == 0:
        return {t: 0.0 for t in thresholds}
    min_dists = cdist(gt_corners, pred_verts).min(axis=1)
    return {t: float((min_dists <= t).mean()) for t in thresholds}


def polis_one_side(coords, boundary):
    pts = list(coords)[:-1]
    if not pts:
        return 0.0
    return sum(boundary.distance(ShapelyPoint(c)) for c in pts) / (2 * len(pts))


def polis_distance(pts_a, pts_b):
    try:
        pa = ShapelyPolygon(pts_a)
        pb = ShapelyPolygon(pts_b)
        if not pa.is_valid or not pb.is_valid or pa.area < 1e-6 or pb.area < 1e-6:
            return None
        return polis_one_side(pa.exterior.coords, pb.exterior) + \
               polis_one_side(pb.exterior.coords, pa.exterior)
    except Exception:
        return None


def poly_bbox(pts):
    x, y = pts[:, 0], pts[:, 1]
    return [float(x.min()), float(y.min()),
            float(x.max() - x.min()), float(y.max() - y.min())]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',     required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--split',      default='test', choices=['test', 'val'])
    parser.add_argument('--output',     default='/home/imane/DATA/PLR-Net_output/eval_vectorize_coverage')
    parser.add_argument('--tolerance',  type=float, default=COVERAGE_TOLERANCE_PX)
    parser.add_argument('--min-area',   type=float, default=MIN_PARCEL_AREA_PX)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    cfg.merge_from_file(args.config)
    cfg.freeze()
    device = cfg.MODEL.DEVICE
    active_branch = cfg.MODEL.ACTIVE_BRANCH
    assert active_branch == 'all_raster', \
        f"eval_vectorize_coverage.py needs a checkpoint trained with ACTIVE_BRANCH=all_raster, got {active_branch!r}"
    mean = list(cfg.DATASETS.IMAGE.PIXEL_MEAN)
    std  = list(cfg.DATASETS.IMAGE.PIXEL_STD)

    model = BuildingDetector(cfg, test=True).to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    state = ckpt.get('model', ckpt)
    state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval()
    epoch = ckpt.get('epoch', '?')
    print(f'Checkpoint loaded — epoch {epoch}, split={args.split}')

    ann_file = os.path.join(ANN_DIR, f'{args.split}_coco.json')
    coco_gt = COCO(ann_file)
    img_ids = coco_gt.getImgIds()
    print(f'{len(img_ids)} images in {args.split} split\n')

    patches_root = os.path.join(_BASE, 'data_nl', 'patches')

    iou_list    = []
    polis_list  = []
    recall_lists = {t: [] for t in RECALL_THRESHOLDS}
    n_pred_polys_total = 0
    n_gt_polys_total   = 0

    for img_id in tqdm(img_ids, desc='Evaluating'):
        img_info = coco_gt.loadImgs(ids=[img_id])[0]
        h, w = img_info['height'], img_info['width']
        stem = os.path.splitext(os.path.basename(img_info['file_name']))[0]
        split_dir = 'test' if stem.startswith('NL_test') else 'train'

        img_tensor, _ = load_patch(stem, os.path.join(patches_root, split_dir), mean, std)
        pred = run_branch(model, img_tensor, device, 'all_raster')

        parcel_labels = label_parcels(pred['region'], pred['line'])
        gdf = vectorize_labels_to_coverage(parcel_labels, tolerance_px=args.tolerance,
                                            min_area_px=args.min_area)

        pred_mask = np.zeros((h, w), dtype=np.uint8)
        pred_polys = []
        for geom in gdf.geometry:
            polys = geom.geoms if geom.geom_type == 'MultiPolygon' else [geom]
            for poly in polys:
                pts = np.array(poly.exterior.coords[:-1])
                if len(pts) < 3:
                    continue
                pred_polys.append(pts)
                cv2.fillPoly(pred_mask, [pts.round().astype(np.int32)], 1)
        pred_mask = pred_mask.astype(bool)

        gt_anns  = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=[img_id]))
        gt_polys_raw = [ann['segmentation'][0] for ann in gt_anns]
        gt_polys = []
        for poly in gt_polys_raw:
            pts = np.array(poly).reshape(-1, 2)
            if len(pts) > 1 and np.allclose(pts[0], pts[-1], atol=0.5):
                pts = pts[:-1]
            gt_polys.append(pts)

        gt_mask = np.zeros((h, w), dtype=np.uint8)
        for ann in gt_anns:
            gt_mask = np.clip(gt_mask + coco_gt.annToMask(ann), 0, 1).astype(np.uint8)
        gt_mask = gt_mask.astype(bool)

        iou_list.append(calc_IoU(pred_mask, gt_mask))

        recalls = compute_junction_recall(gt_polys, np.vstack(pred_polys) if pred_polys else np.empty((0, 2)))
        for t in RECALL_THRESHOLDS:
            if not np.isnan(recalls[t]):
                recall_lists[t].append(recalls[t])

        if gt_polys and pred_polys:
            gt_bboxes = [poly_bbox(p) for p in gt_polys]
            dt_bboxes = [poly_bbox(p) for p in pred_polys]
            ious = cocomask.iou(dt_bboxes, gt_bboxes, [0] * len(gt_bboxes))
            for gi, gt_pts in enumerate(gt_polys):
                best_di  = int(np.argmax(ious[:, gi]))
                best_iou = float(ious[best_di, gi])
                if best_iou > 0.5:
                    d = polis_distance(gt_pts, pred_polys[best_di])
                    if d is not None:
                        polis_list.append(d)

        n_pred_polys_total += len(pred_polys)
        n_gt_polys_total   += len(gt_polys)

    mean_iou   = float(np.mean(iou_list)) * 100 if iou_list else float('nan')
    polis_mean = float(np.mean(polis_list)) if polis_list else float('nan')
    jr = {t: float(np.mean(recall_lists[t])) if recall_lists[t] else float('nan')
          for t in RECALL_THRESHOLDS}

    print('\n' + '=' * 60)
    print('POST-PROCESSED PIPELINE — VECTOR-LEVEL METRICS')
    print('=' * 60)
    print(f"{'IoU region (%)':<22} {mean_iou:>10.2f}")
    print(f"{'PoLiS':<22} {polis_mean:>10.3f}")
    print(f"{'Junction R@3px':<22} {jr[3]:>10.4f}")
    print(f"{'Junction R@5px':<22} {jr[5]:>10.4f}")
    print(f"{'Junction R@8px':<22} {jr[8]:>10.4f}")
    print('-' * 60)
    print(f"{'Images evaluated':<22} {len(iou_list):>10}")
    print(f"{'Pred polygons (total)':<22} {n_pred_polys_total:>10}")
    print(f"{'GT polygons (total)':<22} {n_gt_polys_total:>10}")
    print('=' * 60)

    summary = {
        'checkpoint': args.checkpoint,
        'epoch': epoch,
        'split': args.split,
        'tolerance_px': args.tolerance,
        'min_area_px': args.min_area,
        'n_images': len(iou_list),
        'iou_region_pct': mean_iou,
        'polis': polis_mean,
        'junction_recall': {f'@{t}px': jr[t] for t in RECALL_THRESHOLDS},
        'n_pred_polys_total': n_pred_polys_total,
        'n_gt_polys_total': n_gt_polys_total,
    }
    summary_path = os.path.join(args.output, 'summary_eval_vectorize_coverage.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nJSON summary → {summary_path}')


if __name__ == '__main__':
    main()

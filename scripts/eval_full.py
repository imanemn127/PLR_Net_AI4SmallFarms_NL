#!/usr/bin/env python3
"""
eval_full.py — Full evaluation of PLR-Net on the NL test set.

Computes all metrics from Table 10 of the article:
  APpoly / ARpoly   : COCO polygon AP/AR (segmentation)
  APbound / ARbound : Boundary IoU AP/AR
  PoLiS             : Polygon similarity metric
  IoU               : Pixel IoU on rasterised polygons (post-processing)

Inspired by:
  - eval_testA.py         : inference loop, image loading, IoU/junction recall
  - tools/test_pipelines.py : COCO detection JSON generation
  - tools/evaluation.py   : coco_eval / boundary_eval / polis_eval wrappers
  - PLRNet/utils/metrics/polis.py : PoLiS matching logic (IoU > 0.5 threshold)

Usage (from /mnt/DATA/IMANE/PLR-Net/):
  /mnt/DATA/IMANE/ai4sf/bin/python scripts/eval_full.py \
      --config  config-files/PLR-Net.yaml \
      --checkpoint /mnt/DATA/IMANE/PLR-Net_output/PLR-Net/nl_brp/YYYY-MM-DD_HH-MM-SS/checkpoints/best_val_loss.pth \
      --split test \
      --output /home/imane/DATA/PLR-Net_output/eval_full/nl_ep145
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

# Add repo root to path so PLRNet and tools packages are found
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rasterio
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as coco_mask_utils
from skimage.measure import label, regionprops
from scipy.spatial.distance import cdist
from shapely.geometry import Point as ShapelyPoint, Polygon as ShapelyPolygon

# boundary_iou: installed from github.com/bowenc0221/boundary-iou-api
from boundary_iou.coco_instance_api.coco import COCO as BCOCO
from boundary_iou.coco_instance_api.cocoeval import COCOeval as BCOCOeval

from PLRNet.config import cfg
from PLRNet.detector import BuildingDetector
from PLRNet.utils.metrics.cIoU import calc_IoU


_BASE     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_ROOT  = os.path.join(_BASE, "data_nl", "patches")   # images root
ANN_DIR   = os.path.join(_BASE, "data_nl", "coco")      # {train,val,test}_coco.json

RECALL_THRESHOLDS = [3, 5, 8]  # pixels, for junction recall


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load_image(img_path, cfg):
    """Read a Sentinel-2 GeoTIFF, normalize, and return a (1,3,H,W) tensor."""
    with rasterio.open(img_path) as src:
        arr = src.read([1, 2, 3]).astype(np.float32) / 10000.0  # scale to [0,1]
    mean = np.array(cfg.DATASETS.IMAGE.PIXEL_MEAN, dtype=np.float32)
    std  = np.array(cfg.DATASETS.IMAGE.PIXEL_STD,  dtype=np.float32)
    img_norm = (arr.transpose(1, 2, 0) - mean) / std
    return torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0)


# ---------------------------------------------------------------------------
# GT mask
# ---------------------------------------------------------------------------

def build_gt_mask(coco_obj, img_id, h, w):
    """Union of all COCO annotation masks for one image → binary (H,W)."""
    ann_ids = coco_obj.getAnnIds(imgIds=[img_id])
    anns    = coco_obj.loadAnns(ids=ann_ids)
    mask    = np.zeros((h, w), dtype=np.uint8)
    for ann in anns:
        mask = np.clip(mask + coco_obj.annToMask(ann), 0, 1).astype(np.uint8)
    return mask


# ---------------------------------------------------------------------------
# COCO detection format helpers (from test_pipelines.py)
# ---------------------------------------------------------------------------

def poly_bbox(pts):
    """Axis-aligned bounding box [x, y, w, h] from (N,2) point array."""
    x, y = pts[:, 0], pts[:, 1]
    return [float(x.min()), float(y.min()),
            float(x.max() - x.min()), float(y.max() - y.min())]


# ---------------------------------------------------------------------------
# Junction recall (from eval_testA.py)
# ---------------------------------------------------------------------------

def compute_junction_recall(gt_polys, pred_juncs, thresholds=RECALL_THRESHOLDS):
    """Fraction of GT polygon corners matched within T pixels by a predicted junction."""
    all_gt = []
    for poly in gt_polys:
        pts = np.array(poly).reshape(-1, 2)
        # Remove duplicate closing vertex if present
        if len(pts) > 1 and np.allclose(pts[0], pts[-1], atol=0.5):
            pts = pts[:-1]
        all_gt.append(pts)

    if not all_gt:
        return {t: float('nan') for t in thresholds}

    gt_corners = np.vstack(all_gt)
    if len(pred_juncs) == 0:
        return {t: 0.0 for t in thresholds}

    # Closest predicted junction for each GT corner
    min_dists = cdist(gt_corners, pred_juncs).min(axis=1)
    return {t: float((min_dists <= t).mean()) for t in thresholds}


# ---------------------------------------------------------------------------
# PoLiS distance (adapted from PLRNet/utils/metrics/polis.py)
# ---------------------------------------------------------------------------

def polis_one_side(coords, boundary):
    """Sum of distances from each vertex in coords to boundary, divided by 2*n."""
    pts = list(coords)[:-1]  # skip duplicate closing vertex
    if not pts:
        return 0.0
    return sum(boundary.distance(ShapelyPoint(c)) for c in pts) / (2 * len(pts))


def polis_distance(pts_a, pts_b):
    """Symmetric PoLiS distance between two polygons (None if invalid)."""
    try:
        pa = ShapelyPolygon(pts_a)
        pb = ShapelyPolygon(pts_b)
        if not pa.is_valid or not pb.is_valid or pa.area < 1e-6 or pb.area < 1e-6:
            return None
        return polis_one_side(pa.exterior.coords, pb.exterior) + \
               polis_one_side(pb.exterior.coords, pa.exterior)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     required=True,  help="YAML config file")
    parser.add_argument("--checkpoint", required=True,  help=".pth checkpoint file")
    parser.add_argument("--split",      default="test", choices=["test", "val"],
                        help="Which COCO split to evaluate on")
    parser.add_argument("--output",     default="/home/imane/DATA/PLR-Net_output/eval_full/nl")
    parser.add_argument("--skip-nodata", action="store_true",
                        help="Skip patches where >50%% of pixels are NaN (NoData)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    cfg.merge_from_file(args.config)
    cfg.freeze()
    device = cfg.MODEL.DEVICE

    # ---- Load model and checkpoint -----------------------------------------
    model = BuildingDetector(cfg, test=True).to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    epoch = torch.load(args.checkpoint, map_location='cpu').get('epoch', '?')
    state = ckpt.get("model", ckpt)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"\n[eval_full]  checkpoint : {args.checkpoint}  (epoch {epoch})")
    print(f"[eval_full]  split      : {args.split}")
    print(f"[eval_full]  output     : {args.output}\n")

    # ---- Load COCO annotations ---------------------------------------------
    ann_file = os.path.join(ANN_DIR, f"{args.split}_coco.json")
    coco_gt  = COCO(ann_file)
    # pycocotools COCOeval requires 'iscrowd' in every annotation; add it if missing
    for ann in coco_gt.dataset['annotations']:
        ann.setdefault('iscrowd', 0)
    img_ids  = coco_gt.getImgIds()
    cat_id   = coco_gt.getCatIds()[0]  # category 'field'
    print(f"[eval_full]  {len(img_ids)} images in '{args.split}' split\n")

    # ---- Inference loop ----------------------------------------------------
    coco_poly_results = []   # polygon detections → saved as JSON for COCO eval
    iou_poly_list     = []   # per-image IoU on rasterised polygons (post-processing)
    recall_lists      = {t: [] for t in RECALL_THRESHOLDS}
    polis_list        = []   # per matched GT-pred pair

    for img_id in tqdm(img_ids, desc="Inference"):
        img_info = coco_gt.loadImgs(ids=[img_id])[0]
        img_path = os.path.join(IMG_ROOT, img_info['file_name'])
        h, w     = img_info['height'], img_info['width']

        if not os.path.exists(img_path):
            print(f"  [SKIP] {img_path}")
            continue

        if args.skip_nodata:
            with rasterio.open(img_path) as _src:
                _arr = _src.read(1).astype(np.float32)
            if np.isnan(_arr).mean() > 0.5:
                continue

        tensor = load_image(img_path, cfg).to(device)

        with torch.no_grad():
            output, _ = model(tensor)

        mask_pred  = output['mask_pred'][0]          # (H,W) float
        juncs      = output['juncs_pred'][0]          # (N,2) junction coords
        pred_polys = output['polys_pred'][0] if output['polys_pred'] else []

        # (IoU computed after post-processing, see below)

        # --- GT polygon vertices for this image -----------------------------
        gt_anns  = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=[img_id]))
        gt_polys = [ann['segmentation'][0] for ann in gt_anns]

        # --- Junction recall ------------------------------------------------
        recalls = compute_junction_recall(gt_polys, juncs)
        for t in RECALL_THRESHOLDS:
            if not np.isnan(recalls[t]):
                recall_lists[t].append(recalls[t])

        # --- Build COCO detection entries (one per predicted polygon) --------
        # Score = mean predicted mask value inside the polygon region.
        # This follows test_pipelines.py generate_coco_mask() and gives COCO
        # a meaningful confidence ranking for the AP/AR computation.
        mask_float = mask_pred.astype(np.float32)  # raw sigmoid output [0,1]
        for poly in pred_polys:
            pts = np.array(poly).reshape(-1, 2)
            if len(pts) < 3:
                continue
            # Rasterise polygon to get a binary region mask
            import cv2 as _cv2
            region = np.zeros((h, w), dtype=np.uint8)
            _cv2.fillPoly(region, [pts.round().astype(np.int32)], 1)
            score = float(mask_float[region == 1].mean()) if region.sum() > 0 else 0.5
            coco_poly_results.append({
                'image_id':    img_id,
                'category_id': cat_id,
                'segmentation': [pts.ravel().tolist()],
                'bbox':         poly_bbox(pts),
                'score':        score,
            })

        # --- PoLiS: match each GT to closest predicted poly (IoU > 0.5) -----
        if gt_polys and pred_polys:
            gt_pts_list = [np.array(g).reshape(-1, 2) for g in gt_polys]
            dt_pts_list = [np.array(p).reshape(-1, 2) for p in pred_polys]
            gt_bboxes   = [poly_bbox(p) for p in gt_pts_list]
            dt_bboxes   = [poly_bbox(p) for p in dt_pts_list]
            # ious shape: (n_dt, n_gt)
            ious = coco_mask_utils.iou(dt_bboxes, gt_bboxes, [0] * len(gt_bboxes))
            for gi, gt_pts in enumerate(gt_pts_list):
                best_di  = int(np.argmax(ious[:, gi]))
                best_iou = float(ious[best_di, gi])
                if best_iou > 0.5:
                    d = polis_distance(gt_pts, dt_pts_list[best_di])
                    if d is not None:
                        polis_list.append(d)

    # ---- Save detection JSON -----------------------------------------------
    dt_file = os.path.join(args.output, "detections_poly.json")
    with open(dt_file, "w") as f:
        json.dump(coco_poly_results, f)
    print(f"\nDetections saved → {dt_file}  ({len(coco_poly_results)} polygons total)\n")

    # ---- APpoly / ARpoly — standard COCO segmentation eval -----------------
    print("=" * 60)
    print("COCO POLYGON  (APpoly / ARpoly)")
    print("=" * 60)
    ap_poly = ar_poly = float('nan')
    if coco_poly_results:
        coco_dt   = coco_gt.loadRes(dt_file)
        coco_eval = COCOeval(coco_gt, coco_dt, "segm")
        coco_eval.params.catIds = [cat_id]
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        # stats[0] = AP@[0.5:0.95], stats[8] = AR@[0.5:0.95] maxDets=100
        ap_poly = float(coco_eval.stats[0]) * 100
        ar_poly = float(coco_eval.stats[8]) * 100
    else:
        print("  No predictions — skipping.")

    # ---- APbound / ARbound — boundary IoU eval (dilation_ratio=0.02) -------
    print("\n" + "=" * 60)
    print("BOUNDARY IoU  (APbound / ARbound)")
    print("=" * 60)
    ap_bound = ar_bound = float('nan')
    if coco_poly_results:
        dilation_ratio = 0.02  # same as article default
        bcoco_gt = BCOCO(ann_file, get_boundary=True, dilation_ratio=dilation_ratio)
        # boundary_iou also needs iscrowd field
        for ann in bcoco_gt.dataset['annotations']:
            ann.setdefault('iscrowd', 0)
        bcoco_gt.createIndex()
        bcoco_dt = bcoco_gt.loadRes(dt_file)
        beval    = BCOCOeval(bcoco_gt, bcoco_dt, iouType="boundary",
                             dilation_ratio=dilation_ratio)
        beval.evaluate()
        beval.accumulate()
        beval.summarize()
        ap_bound = float(beval.stats[0]) * 100
        ar_bound = float(beval.stats[8]) * 100

    # ---- PoLiS -------------------------------------------------------------
    polis_mean = float(np.mean(polis_list)) if polis_list else float('nan')

    # ---- IoU on rasterised polygons (post-processing) — matches article ----
    # Rasterise predicted polygons from dt_file, compare pixel-by-pixel to GT.
    # Equivalent to compute_IoU_cIoU in PLRNet/utils/metrics/cIoU.py.
    from pycocotools import mask as cocomask
    iou_poly_list = []
    if coco_poly_results:
        coco_gti = COCO(ann_file)
        coco_dtp = coco_gti.loadRes(dt_file)
        for _img_id in coco_gti.getImgIds():
            _info = coco_gti.loadImgs(_img_id)[0]
            _h, _w = _info['height'], _info['width']
            # predicted mask from polygons
            _pred = np.zeros((_h, _w), dtype=bool)
            for _ann in coco_dtp.loadAnns(coco_dtp.getAnnIds(imgIds=[_img_id])):
                _rle = cocomask.frPyObjects(_ann['segmentation'], _h, _w)
                _pred |= cocomask.decode(_rle).reshape(_h, _w).astype(bool)
            # GT mask from polygons
            _gt = np.zeros((_h, _w), dtype=bool)
            for _ann in coco_gti.loadAnns(coco_gti.getAnnIds(imgIds=[_img_id])):
                _rle = cocomask.frPyObjects(_ann['segmentation'], _h, _w)
                _gt |= cocomask.decode(_rle).reshape(_h, _w).astype(bool)
            if _gt.sum() == 0 and _pred.sum() == 0:
                continue
            iou_poly_list.append(calc_IoU(_pred, _gt))
    mean_iou = float(np.mean(iou_poly_list)) * 100 if iou_poly_list else float('nan')

    # ---- Junction recall averages ------------------------------------------
    jr = {t: float(np.mean(recall_lists[t])) if recall_lists[t] else float('nan')
          for t in RECALL_THRESHOLDS}

    # ---- Print comparison table --------------------------------------------
    print("\n" + "=" * 60)
    print("COMPARISON WITH TABLE 10 (article)")
    print("=" * 60)
    print(f"{'Metric':<22} {'Yours':>10}  {'PLR-Net (article)':>18}")
    print("-" * 54)
    print(f"{'APpoly (%)':<22} {ap_poly:>10.1f}  {'47.1':>18}")
    print(f"{'ARpoly (%)':<22} {ar_poly:>10.1f}  {'55.5':>18}")
    print(f"{'APbound (%)':<22} {ap_bound:>10.1f}  {'41.5':>18}")
    print(f"{'ARbound (%)':<22} {ar_bound:>10.1f}  {'53.1':>18}")
    print(f"{'PoLiS':<22} {polis_mean:>10.3f}  {'1.51':>18}")
    print(f"{'IoU region (%)':<22} {mean_iou:>10.2f}  {'75.86':>18}")
    print("-" * 54)
    print(f"{'Junction R@3px':<22} {jr[3]:>10.4f}")
    print(f"{'Junction R@5px':<22} {jr[5]:>10.4f}")
    print(f"{'Junction R@8px':<22} {jr[8]:>10.4f}")
    print(f"{'Images evaluated':<22} {len(iou_poly_list):>10}")
    print("=" * 60)

    # ---- Save JSON summary -------------------------------------------------
    summary = {
        "checkpoint": args.checkpoint,
        "epoch": epoch,
        "split": args.split,
        "n_images": len(iou_poly_list),
        "APpoly":   ap_poly,
        "ARpoly":   ar_poly,
        "APbound":  ap_bound,
        "ARbound":  ar_bound,
        "polis":    polis_mean,
        "iou_region_pct": mean_iou,
        "junction_recall": {f"@{t}px": jr[t] for t in RECALL_THRESHOLDS},
        "article_table10_PLRNet": {
            "APpoly": 47.1, "ARpoly": 55.5,
            "APbound": 41.5, "ARbound": 53.1,
            "polis": 1.51, "iou": 75.86,
        },
    }
    summary_path = os.path.join(args.output, "summary_eval_full.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nJSON summary → {summary_path}")


if __name__ == "__main__":
    main()

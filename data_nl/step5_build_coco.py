#!/usr/bin/env python3
"""
step5_build_coco.py
-------------------
Build COCO JSON files from already-extracted 256x256 patches.
Polygons are derived from the BRP label raster masks (unique parcel IDs).

Input:
  data_nl/patches/train/images/  + masks/
  data_nl/patches/test/images/   + masks/
  data_nl/patches/train.txt      (train subset ~80%)
  data_nl/patches/val.txt        (val subset ~20%)

Output:
  data_nl/coco/train_coco.json
  data_nl/coco/val_coco.json
  data_nl/coco/test_coco.json

Usage:
    /mnt/DATA/IMANE/ai4sf/bin/python data_nl/step5_build_coco.py
"""

import json
import os
import numpy as np
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape, Polygon, MultiPolygon
from tqdm import tqdm

# ---- Paths ------------------------------------------------------------------
DATA_NL   = os.path.dirname(os.path.abspath(__file__))
PATCHES   = os.path.join(DATA_NL, "patches")
DST_ROOT  = os.path.join(DATA_NL, "coco")

TRAIN_TXT = os.path.join(PATCHES, "train.txt")
VAL_TXT   = os.path.join(PATCHES, "val.txt")

CATEGORY    = {"id": 1, "name": "field"}
MIN_AREA_PX = 100   # discard polygon slivers smaller than this (px²)

os.makedirs(DST_ROOT, exist_ok=True)


# ---- Helpers ----------------------------------------------------------------

def read_txt(path):
    with open(path) as f:
        return [l.strip() for l in f if l.strip()]


def mask_to_polygons(mask: np.ndarray) -> list:
    """
    Convert a label raster (unique int IDs per parcel, 0=background) into a
    list of Shapely Polygons in pixel coordinates.
    """
    polygons = []
    parcel_ids = np.unique(mask)
    parcel_ids = parcel_ids[parcel_ids != 0]

    for pid in parcel_ids:
        binary = (mask == pid).astype(np.uint8)
        for geom_dict, val in shapes(binary, mask=binary):
            if val == 0:
                continue
            geom = shape(geom_dict)
            if isinstance(geom, Polygon):
                polys = [geom]
            elif isinstance(geom, MultiPolygon):
                polys = list(geom.geoms)
            else:
                continue
            for p in polys:
                if p.area >= MIN_AREA_PX:
                    polygons.append(p)

    return polygons


def polygon_to_flat(polygon: Polygon) -> list:
    coords = []
    for x, y in polygon.exterior.coords:
        coords.extend([float(x), float(y)])
    return coords


def bbox_and_area(flat: list):
    pts = np.array(flat, dtype=np.float64).reshape(-1, 2)
    if np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    xs, ys = pts[:, 0], pts[:, 1]
    x_min, y_min = float(xs.min()), float(ys.min())
    w = float(xs.max()) - x_min
    h = float(ys.max()) - y_min
    n = len(pts)
    area = 0.0
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return [x_min, y_min, w, h], float(abs(area) / 2.0)


# ---- Core builder -----------------------------------------------------------

def build_split(stems: list, patch_split_dir: str, split_name: str) -> dict:
    """
    patch_split_dir : absolute path to patches/train/ or patches/test/
    split_name      : "train", "val", or "test"  (used only for file_name prefix)
    """
    img_dir  = os.path.join(patch_split_dir, "images")
    mask_dir = os.path.join(patch_split_dir, "masks")

    coco = {
        "info": {
            "year": 2020,
            "version": "1.0",
            "description": "PLR-Net NL Sentinel-2 BRP patches",
        },
        "categories": [CATEGORY],
        "images": [],
        "annotations": [],
    }

    img_id = 1
    ann_id = 1

    # "train" and "val" both live in patches/train/; "test" in patches/test/
    rel_split = "test" if split_name == "test" else "train"

    for stem in tqdm(stems, desc=f"  {split_name}", unit="patch"):
        img_path  = os.path.join(img_dir,  stem + ".tif")
        mask_path = os.path.join(mask_dir, stem + ".tif")

        if not os.path.isfile(img_path) or not os.path.isfile(mask_path):
            continue

        with rasterio.open(img_path) as src:
            res_x = float(abs(src.transform.a))
            res_y = float(abs(src.transform.e))
            tl_x  = float(src.transform.c)
            tl_y  = float(src.transform.f)
            w, h  = src.width, src.height

        with rasterio.open(mask_path) as msrc:
            mask = msrc.read(1)

        # file_name relative to root = data_nl/patches/
        file_name = f"{rel_split}/images/{stem}.tif"

        coco["images"].append({
            "id":         img_id,
            "file_name":  file_name,
            "image_path": file_name,
            "width":      w,
            "height":     h,
            "res_x":      round(res_x, 6),
            "res_y":      round(res_y, 6),
            "top_left":   [round(tl_x, 6), round(tl_y, 6)],
        })

        polys = mask_to_polygons(mask)
        for feat_id, poly in enumerate(polys):
            flat = polygon_to_flat(poly)
            bbox, area = bbox_and_area(flat)
            coco["annotations"].append({
                "feature_id":   feat_id,
                "id":           ann_id,
                "image_id":     img_id,
                "segmentation": [flat],
                "area":         round(area, 4),
                "bbox":         [round(v, 4) for v in bbox],
                "category_id":  CATEGORY["id"],
                "iscrowd":      0,
            })
            ann_id += 1

        img_id += 1

    return coco


# ---- Main -------------------------------------------------------------------

def main():
    train_stems = read_txt(TRAIN_TXT)
    val_stems   = read_txt(VAL_TXT)
    test_stems  = sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(os.path.join(PATCHES, "test", "images"))
        if f.endswith(".tif")
    )

    print(f"Stems — train: {len(train_stems)}, val: {len(val_stems)}, test: {len(test_stems)}")

    splits = [
        (train_stems, os.path.join(PATCHES, "train"), "train", "train_coco.json"),
        (val_stems,   os.path.join(PATCHES, "train"), "val",   "val_coco.json"),
        (test_stems,  os.path.join(PATCHES, "test"),  "test",  "test_coco.json"),
    ]

    for stems, patch_dir, split_name, out_file in splits:
        print(f"\n{'='*55}\nBuilding {split_name}  ({len(stems)} patches)")
        coco = build_split(stems, patch_dir, split_name)
        out_path = os.path.join(DST_ROOT, out_file)
        with open(out_path, "w") as f:
            json.dump(coco, f, indent=2)
        n_img = len(coco["images"])
        n_ann = len(coco["annotations"])
        print(f"  → {out_path}")
        print(f"  images: {n_img},  annotations: {n_ann},  avg: {n_ann/n_img if n_img else 0:.1f}/patch")

    print("\nDone.")


if __name__ == "__main__":
    main()

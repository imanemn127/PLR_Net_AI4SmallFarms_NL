"""
step3_extract_patches.py
------------------------
Extract 256x256 patches with 20% overlap (stride=205 px) from the
national Sentinel-2 mosaic and the BRP label raster.

Zone coordinates are in EPSG:28992 (RD New, metres), measured in QGIS
from the national mosaic, consistent with Figure 1 of the article.

Input  : mosaic_sentinel2/NL_mosaic_2020_10m.tif
         labels_raster/NL_BRP2020_10m_labels.tif
Output : patches/train/images/  patches/train/masks/
         patches/test/images/   patches/test/masks/

Usage:
    /mnt/DATA/IMANE/ai4sf/bin/python data_nl/step3_extract_patches.py
"""

import os
import numpy as np
import rasterio
from rasterio.windows import Window

# ---- Paths ------------------------------------------------------------------
DATA_NL     = os.path.dirname(os.path.abspath(__file__))
MOSAIC      = os.path.join(DATA_NL, "mosaic_sentinel2", "NL_mosaic_2020_10m.tif")
LABELS      = os.path.join(DATA_NL, "labels_raster",   "NL_BRP2020_10m_labels.tif")
PATCHES_DIR = os.path.join(DATA_NL, "patches")

# ---- Patch parameters (matching the article) --------------------------------
PATCH_SIZE = 256   # pixels
STRIDE     = 205   # 256 * 0.8 ≈ 205 px  →  ~20% overlap

# ---- Quality filters --------------------------------------------------------
# Skip patches where too many pixels are NoData (sea, mosaic border, Wadden islands).
MAX_NAN_FRAC = 0.30   # skip if >30% of band-1 pixels are NaN

# ---- Zone definitions (EPSG:28992, metres) ----------------------------------
# Format: (xmin, ymin, xmax, ymax)
# Zones inspired by Figure 1 of the article.
# Final patch counts depend on quality filters above (NaN + BRP coverage).
ZONES = {
    "test": [
        # Friesland interior
        (156886, 540284, 228886, 608817),
    ],
    "train": [
        # Flevoland / Gelderland / Overijssel (central-east)
        (148000, 386290, 261260, 518000),
        # Zuid-Holland / Utrecht / Noord-Brabant (central-west)
        (26540, 360590, 148000, 480000),
    ]
}

# ---- Helper: create output directories -------------------------------------
def make_dirs(split):
    img_dir  = os.path.join(PATCHES_DIR, split, "images")
    mask_dir = os.path.join(PATCHES_DIR, split, "masks")
    os.makedirs(img_dir,  exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    return img_dir, mask_dir


# ---- Helper: convert geographic bbox to pixel window -----------------------
def bbox_to_pixel_offsets(src, xmin, ymin, xmax, ymax):
    """Convert EPSG:28992 bounding box to pixel (col_off, row_off, width, height)."""
    t = src.transform
    col_off = max(0, int((xmin - t.c) / t.a))
    row_off = max(0, int((ymax - t.f) / t.e))  # t.e is negative (north-up)
    col_end = min(src.width,  int((xmax - t.c) / t.a))
    row_end = min(src.height, int((ymin - t.f) / t.e))
    return col_off, row_off, col_end - col_off, row_end - row_off


# ---- Main extraction loop --------------------------------------------------
def extract_patches():
    total = {"train": 0, "test": 0}

    with rasterio.open(MOSAIC) as src_img, \
         rasterio.open(LABELS) as src_lbl:

        for split, zone_list in ZONES.items():
            img_dir, mask_dir = make_dirs(split)
            print(f"\n{'='*55}")
            print(f"Split: {split}  ({len(zone_list)} zone(s))")

            for z_idx, (xmin, ymin, xmax, ymax) in enumerate(zone_list):
                col0, row0, zone_w, zone_h = bbox_to_pixel_offsets(
                    src_img, xmin, ymin, xmax, ymax
                )
                print(f"\n  Zone {z_idx+1}: "
                      f"col={col0} row={row0} w={zone_w} h={zone_h}  "
                      f"({zone_w*10/1000:.1f} x {zone_h*10/1000:.1f} km)")

                n_saved = 0
                for row in range(0, zone_h - PATCH_SIZE + 1, STRIDE):
                    for col in range(0, zone_w - PATCH_SIZE + 1, STRIDE):
                        win = Window(col0 + col, row0 + row, PATCH_SIZE, PATCH_SIZE)

                        # Read image (4 bands: B2, B3, B4, B8) and label
                        img_patch = src_img.read(window=win)       # (4, 256, 256)
                        lbl_patch = src_lbl.read(1, window=win)    # (256, 256)

                        # Skip patches that are mostly NoData (sea, mosaic border)
                        nan_frac = np.isnan(img_patch[0].astype(np.float32)).mean()
                        if nan_frac > MAX_NAN_FRAC:
                            continue

                        name = f"NL_{split}_z{z_idx+1}_r{row:06d}_c{col:06d}"

                        # Save image patch
                        img_meta = src_img.meta.copy()
                        img_meta.update({
                            "height": PATCH_SIZE, "width": PATCH_SIZE,
                            "transform": src_img.window_transform(win),
                            "compress": "lzw",
                        })
                        with rasterio.open(
                            os.path.join(img_dir, name + ".tif"), "w", **img_meta
                        ) as dst:
                            dst.write(img_patch)

                        # Save label patch
                        lbl_meta = src_lbl.meta.copy()
                        lbl_meta.update({
                            "height": PATCH_SIZE, "width": PATCH_SIZE,
                            "transform": src_lbl.window_transform(win),
                            "compress": "lzw",
                        })
                        with rasterio.open(
                            os.path.join(mask_dir, name + ".tif"), "w", **lbl_meta
                        ) as dst:
                            dst.write(lbl_patch, 1)

                        n_saved += 1

                print(f"  Zone {z_idx+1}: {n_saved} patches saved")
                total[split] += n_saved

    print("\n" + "="*55)
    print("DONE")
    print(f"  Train patches : {total['train']}")
    print(f"  Test patches  : {total['test']}")
    print(f"  Output        : {PATCHES_DIR}")


if __name__ == "__main__":
    extract_patches()

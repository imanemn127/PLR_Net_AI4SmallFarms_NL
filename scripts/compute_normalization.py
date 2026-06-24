"""
Compute per-channel mean and std of the NL BRP training patches (Sentinel-2).
Reads bands [1, 2, 3] (B2, B3, B4) from the GeoTIFFs produced by step3.
Values are raw uint16 reflectance (0-10000); stats are computed after /10000.

NoData masking: pixels where ALL 3 bands == 0 are mosaic border pixels and
are excluded from the statistics.

Usage:
    /mnt/DATA/IMANE/ai4sf/bin/python scripts/compute_normalization.py
    /mnt/DATA/IMANE/ai4sf/bin/python scripts/compute_normalization.py --max_images 2000
"""

import argparse
import json
import os
import numpy as np
import rasterio
from tqdm import tqdm

DATA_ROOT = "/mnt/DATA/IMANE/PLR-Net/data_nl"
COCO_JSON = os.path.join(DATA_ROOT, "coco", "train_coco.json")
IMG_ROOT  = os.path.join(DATA_ROOT, "patches")   # file_name is relative to this
MAX_PIXEL = 10000.0


def main(max_images=None):
    with open(COCO_JSON) as f:
        images = json.load(f)["images"]

    if max_images:
        rng = np.random.default_rng(0)
        images = rng.choice(images, size=min(max_images, len(images)), replace=False).tolist()

    print(f"Computing stats on {len(images)} patches (NoData=0 masked) …")

    # Welford online algorithm — vectorised per patch, no per-pixel loop
    n    = np.zeros(3, dtype=np.float64)
    mean = np.zeros(3, dtype=np.float64)
    M2   = np.zeros(3, dtype=np.float64)

    for img_info in tqdm(images):
        path = os.path.join(IMG_ROOT, img_info["file_name"])
        with rasterio.open(path) as src:
            data = src.read([1, 2, 3]).astype(np.float64) / MAX_PIXEL  # (3, H, W)

        # Mask NoData pixels: exclude positions where any band is NaN or
        # where all 3 bands are exactly 0 (mosaic border)
        valid = ~(np.any(np.isnan(data), axis=0) | np.all(data == 0.0, axis=0))

        for c in range(3):
            pixels = data[c][valid]
            pixels = pixels[~np.isnan(pixels)]  # extra safety
            if pixels.size == 0:
                continue
            # Welford batch update
            batch_n    = pixels.size
            batch_mean = pixels.mean()
            batch_M2   = ((pixels - batch_mean) ** 2).sum()

            new_n    = n[c] + batch_n
            delta    = batch_mean - mean[c]
            mean[c]  = (n[c] * mean[c] + batch_n * batch_mean) / new_n
            M2[c]   += batch_M2 + delta ** 2 * n[c] * batch_n / new_n
            n[c]     = new_n

    std = np.sqrt(M2 / (n - 1))

    print("\n=== Results (after /10000, NoData excluded) ===")
    print(f"PIXEL_MEAN: [{mean[0]:.4f}, {mean[1]:.4f}, {mean[2]:.4f}]")
    print(f"PIXEL_STD:  [{std[0]:.4f}, {std[1]:.4f}, {std[2]:.4f}]")
    print("\nPaste into config-files/PLR-Net.yaml under DATASETS.IMAGE:")
    print(f"    PIXEL_MEAN: [{mean[0]:.4f}, {mean[1]:.4f}, {mean[2]:.4f}]")
    print(f"    PIXEL_STD:  [{std[0]:.4f}, {std[1]:.4f}, {std[2]:.4f}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_images", type=int, default=None,
                        help="Random subset (seed=0) for speed. Default: all.")
    args = parser.parse_args()
    main(args.max_images)

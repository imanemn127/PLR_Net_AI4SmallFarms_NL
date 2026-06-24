"""
step1_merge_mosaic.py
---------------------
Merge all GeoTIFF sub-tiles exported from Google Earth Engine
into a single national mosaic covering the Netherlands.

GEE splits each quadrant into sub-tiles named like:
    S2_NL_2020_NL_NE-0000000000-0000000000.tif
    S2_NL_2020_NL_NE-0000000000-0000011776.tif
    ...
All sub-tiles share the same CRS (EPSG:28992), resolution (10 m),
and bands (B2, B3, B4, B8) — no reprojection needed, just merge.

Input  : all .tif files in mosaic_sentinel2/
Output : mosaic_sentinel2/NL_mosaic_2020_10m.tif

Usage:
    /mnt/DATA/IMANE/ai4sf/bin/python step1_merge_mosaic.py
"""

import os
import glob
import rasterio
from rasterio.merge import merge

# ---- Paths ------------------------------------------------------------------
DATA_NL    = os.path.dirname(os.path.abspath(__file__))
MOSAIC_DIR = os.path.join(DATA_NL, "mosaic_sentinel2")
OUTPUT     = os.path.join(MOSAIC_DIR, "NL_mosaic_2020_10m.tif")

# ---- Find all input sub-tiles (exclude the output file itself) --------------
tiles = sorted([
    f for f in glob.glob(os.path.join(MOSAIC_DIR, "*.tif"))
    if "NL_mosaic_2020_10m" not in f
])

if len(tiles) == 0:
    raise FileNotFoundError(
        f"No GeoTIFF tiles found in {MOSAIC_DIR}.\n"
        "Copy the files downloaded from Google Drive before running this script."
    )

print(f"Found {len(tiles)} sub-tile(s) to merge:")
for t in tiles:
    print(f"  {os.path.basename(t)}")

# ---- Open all tiles and merge into one raster -------------------------------
print("\nMerging (this may take several minutes for large datasets)...")
datasets      = [rasterio.open(t) for t in tiles]
mosaic, transform = merge(datasets)

# ---- Write output with metadata from first tile ----------------------------
meta = datasets[0].meta.copy()
meta.update({
    "driver":    "GTiff",
    "height":    mosaic.shape[1],
    "width":     mosaic.shape[2],
    "transform": transform,
    "compress":  "lzw",   # lossless compression to reduce file size
    "bigtiff":   "YES",   # required for files > 4 GB
})

for ds in datasets:
    ds.close()

print(f"\nWriting output: {OUTPUT}")
with rasterio.open(OUTPUT, "w", **meta) as dst:
    dst.write(mosaic)

print("\nDone.")
print(f"  Bands  : {mosaic.shape[0]}  (B2, B3, B4, B8)")
print(f"  Height : {mosaic.shape[1]} px  ({mosaic.shape[1]*10/1000:.1f} km)")
print(f"  Width  : {mosaic.shape[2]} px  ({mosaic.shape[2]*10/1000:.1f} km)")
print(f"  CRS    : {meta['crs']}")

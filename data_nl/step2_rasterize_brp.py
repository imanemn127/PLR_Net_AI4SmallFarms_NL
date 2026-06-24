"""
step2_rasterize_brp.py
-----------------------
Rasterize the BRP 2020 GeoPackage (agricultural parcel cadastre)
onto the exact same grid as the Sentinel-2 mosaic.

Each pixel receives the index of the parcel it belongs to (1-based),
or 0 if no parcel covers that pixel.
This produces a label raster perfectly aligned with the mosaic,
ready for patch extraction.

Input  : mosaic_sentinel2/NL_mosaic_2020_10m.tif  (reference grid)
         brp/brp_crop_plots_definitive_2020.gpkg   (BRP parcels)
Output : labels_raster/NL_BRP2020_10m_labels.tif

Usage:
    /mnt/DATA/IMANE/ai4sf/bin/python step2_rasterize_brp.py
"""

import os
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds

# ---- Paths ------------------------------------------------------------------
DATA_NL    = os.path.dirname(os.path.abspath(__file__))
MOSAIC     = os.path.join(DATA_NL, "mosaic_sentinel2", "NL_mosaic_2020_10m.tif")
BRP_FILE   = os.path.join(DATA_NL, "brp", "brp_crop_plots_definitive_2020.gpkg")
OUTPUT     = os.path.join(DATA_NL, "labels_raster", "NL_BRP2020_10m_labels.tif")

# ---- Read the mosaic to get the reference grid ------------------------------
print("Reading reference grid from mosaic...")
with rasterio.open(MOSAIC) as src:
    ref_transform = src.transform
    ref_crs       = src.crs
    ref_height    = src.height
    ref_width     = src.width

print(f"  CRS    : {ref_crs}")
print(f"  Size   : {ref_width} x {ref_height} px")
print(f"  Origin : ({ref_transform.c:.1f}, {ref_transform.f:.1f})")

# ---- Read BRP parcels -------------------------------------------------------
print("\nReading BRP 2020 parcels...")
brp = gpd.read_file(BRP_FILE)
print(f"  {len(brp)} parcels loaded")
print(f"  CRS: {brp.crs}")

# Reproject to match the mosaic if needed
if brp.crs != ref_crs:
    print(f"  Reprojecting from {brp.crs} to {ref_crs}...")
    brp = brp.to_crs(ref_crs)

# ---- Build (geometry, value) pairs for rasterization -----------------------
# Each parcel gets a unique integer ID (1-based).
# This allows identifying individual parcels in the label raster.
print("\nPreparing geometries for rasterization...")
shapes = [
    (geom, idx + 1)
    for idx, geom in enumerate(brp.geometry)
    if geom is not None and geom.is_valid
]
print(f"  {len(shapes)} valid geometries")

# ---- Rasterize onto the reference grid -------------------------------------
print("\nRasterizing (this may take 10-30 minutes for the full country)...")
label_raster = rasterize(
    shapes,
    out_shape=(ref_height, ref_width),
    transform=ref_transform,
    fill=0,           # background = 0 (no parcel)
    dtype=np.int32,   # int32 supports up to ~2 billion unique parcel IDs
    all_touched=False # only pixels whose center falls inside the polygon
)

# ---- Write output -----------------------------------------------------------
print(f"\nWriting output: {OUTPUT}")
with rasterio.open(
    OUTPUT, "w",
    driver="GTiff",
    height=ref_height,
    width=ref_width,
    count=1,
    dtype=np.int32,
    crs=ref_crs,
    transform=ref_transform,
    compress="lzw",
    bigtiff="YES",
) as dst:
    dst.write(label_raster, 1)

n_parcel_px = (label_raster > 0).sum()
coverage    = 100.0 * n_parcel_px / (ref_height * ref_width)
print("\nDone.")
print(f"  Parcel pixels : {n_parcel_px:,}  ({coverage:.1f}% of total area)")
print(f"  Background px : {(label_raster == 0).sum():,}")

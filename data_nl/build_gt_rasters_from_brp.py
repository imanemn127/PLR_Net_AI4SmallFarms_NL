"""
build_gt_rasters_from_brp.py
------------------------------
Build the 4 GT rasters (region, lines, nodes, afm) directly from the BRP
vector GeoPackage, aligned on the 256x256 / 10m patches already
extracted by step3_extract_patches.py. All rasters come from the same
source (the BRP polygons clipped per patch) so they are pixel-consistent
with each other, with no leftover artifacts from rasterize->vector
reconversion or additive mask merging.

Outputs per patch, same grid/CRS/transform as images/ :
  patches/<split>/gt_region/<stem>.tif   binary parcel mask {0,1}
  patches/<split>/gt_lines/<stem>.tif    1px boundary map {0,1}
  patches/<split>/gt_nodes/<stem>.tif    corners filtered by angle {0,1}
  patches/<split>/gt_afm/<stem>.tif      2-band (dx,dy) attraction field,
                                          computed with afm_op (CUDA) on
                                          the polygon edges (requires GPU)

Usage:
    /mnt/DATA/IMANE/ai4sf/bin/python data_nl/build_gt_rasters_from_brp.py
    /mnt/DATA/IMANE/ai4sf/bin/python data_nl/build_gt_rasters_from_brp.py --split test
    /mnt/DATA/IMANE/ai4sf/bin/python data_nl/build_gt_rasters_from_brp.py --limit 20
    /mnt/DATA/IMANE/ai4sf/bin/python data_nl/build_gt_rasters_from_brp.py --only afm
    /mnt/DATA/IMANE/ai4sf/bin/python data_nl/build_gt_rasters_from_brp.py --only region lines nodes
"""

import os
import sys
import argparse

import numpy as np
import rasterio
import geopandas as gpd
import torch
from rasterio.features import rasterize
from shapely.geometry import LineString

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PLRNet.csrc.lib.afm_op import afm

DATA_NL   = os.path.dirname(os.path.abspath(__file__))
PATCHES   = os.path.join(DATA_NL, "patches")
BRP_FILE  = os.path.join(DATA_NL, "brp", "brp_crop_plots_definitive_2020.gpkg")

ANGLE_THRESHOLD_DEG = 10.0


def filter_corners_by_angle(coords, angle_threshold_deg=ANGLE_THRESHOLD_DEG):
    """Keep a vertex only if its interior angle deviates from 180 deg
    by more than angle_threshold_deg (drops points aligned on straight edges)."""
    pts = np.asarray(coords, dtype=np.float64)
    if len(pts) > 1 and np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    n = len(pts)
    if n < 3:
        return pts

    kept = []
    for i in range(n):
        prev_pt = pts[i - 1]
        curr_pt = pts[i]
        next_pt = pts[(i + 1) % n]

        v1 = prev_pt - curr_pt
        v2 = next_pt - curr_pt
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-9 or n2 < 1e-9:
            continue

        cos_angle = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        interior_angle_deg = np.degrees(np.arccos(cos_angle))

        if abs(180.0 - interior_angle_deg) > angle_threshold_deg:
            kept.append(curr_pt)

    return np.array(kept) if kept else np.empty((0, 2))


def build_gt_region(polygons, out_shape, transform):
    if not polygons:
        return np.zeros(out_shape, dtype=np.float32)
    shapes = [(geom, 1) for geom in polygons]
    arr = rasterize(
        shapes, out_shape=out_shape, transform=transform,
        fill=0, dtype=np.uint8, all_touched=False,
    )
    return arr.astype(np.float32)


def build_gt_lines(polygons, out_shape, transform):
    line_geoms = []
    for geom in polygons:
        rings = [geom.exterior] + list(geom.interiors)
        for ring in rings:
            line_geoms.append(LineString(ring.coords))
    if not line_geoms:
        return np.zeros(out_shape, dtype=np.float32)
    shapes = [(geom, 1) for geom in line_geoms]
    arr = rasterize(
        shapes, out_shape=out_shape, transform=transform,
        fill=0, dtype=np.uint8, all_touched=True,
    )
    return arr.astype(np.float32)


def build_gt_nodes(polygons, out_shape, transform, angle_threshold_deg=ANGLE_THRESHOLD_DEG):
    H, W = out_shape
    jmap = np.zeros((H, W), dtype=np.float32)
    inv_transform = ~transform

    for geom in polygons:
        rings = [geom.exterior] + list(geom.interiors)
        for ring in rings:
            corners = filter_corners_by_angle(np.array(ring.coords), angle_threshold_deg)
            for x_map, y_map in corners:
                col, row = inv_transform * (x_map, y_map)
                ci, ri = int(np.floor(col)), int(np.floor(row))
                if 0 <= ri < H and 0 <= ci < W:
                    jmap[ri, ci] = 1.0
    return jmap


def polygons_to_pixel_segments(polygons, transform):
    """Convert polygon rings to (x1,y1,x2,y2) segments in pixel coords."""
    inv_transform = ~transform
    segments = []
    for geom in polygons:
        rings = [geom.exterior] + list(geom.interiors)
        for ring in rings:
            coords = np.array(ring.coords)
            if len(coords) > 1 and np.allclose(coords[0], coords[-1]):
                coords = coords[:-1]
            n = len(coords)
            if n < 2:
                continue
            pixel_coords = np.array([inv_transform * (x, y) for x, y in coords])
            for i in range(n):
                p1 = pixel_coords[i]
                p2 = pixel_coords[(i + 1) % n]
                segments.append([p1[0], p1[1], p2[0], p2[1]])
    return segments


def build_gt_afm(polygons, out_shape, transform):
    """(2, H, W) attraction field via the real afm_op CUDA operator,
    computed directly from BRP polygon edges (no raster round-trip)."""
    H, W = out_shape
    segments = polygons_to_pixel_segments(polygons, transform)
    if not segments:
        return np.zeros((2, H, W), dtype=np.float32)

    lines = torch.tensor(segments, dtype=torch.float32).cuda()
    shape_info = torch.IntTensor([[0, lines.size(0), H, W]]).cuda()
    afmap, _ = afm(lines, shape_info, H, W)
    return afmap[0].cpu().numpy().astype(np.float32)


def save_geotiff(array, path, crs, transform):
    """array: (H, W) single-band, or (C, H, W) multi-band."""
    array = array.astype(np.float32)
    if array.ndim == 2:
        array = array[None]
    n_bands, h, w = array.shape
    with rasterio.open(
        path, "w", driver="GTiff",
        height=h, width=w, count=n_bands,
        dtype=np.float32, crs=crs, transform=transform, compress="lzw",
    ) as dst:
        dst.write(array)


ALL_TARGETS = ("region", "lines", "nodes", "afm")


def process_split(split, brp_gdf, limit=None, only=ALL_TARGETS):
    img_dir = os.path.join(PATCHES, split, "images")
    out_dirs = {t: os.path.join(PATCHES, split, f"gt_{t}") for t in only}
    for d in out_dirs.values():
        os.makedirs(d, exist_ok=True)

    stems = sorted(f[:-4] for f in os.listdir(img_dir) if f.endswith(".tif"))
    if limit:
        stems = stems[:limit]

    print(f"\n{'='*55}\nSplit: {split}  ({len(stems)} patches)  targets: {list(only)}")

    sindex = brp_gdf.sindex

    for i, stem in enumerate(stems):
        img_path = os.path.join(img_dir, stem + ".tif")
        with rasterio.open(img_path) as src:
            crs = src.crs
            transform = src.transform
            H, W = src.height, src.width
            bounds = src.bounds

        cand_idx = list(sindex.intersection(bounds))
        flat_polys = []
        if cand_idx:
            cand = brp_gdf.iloc[cand_idx]
            clipped = gpd.clip(cand, [bounds.left, bounds.bottom, bounds.right, bounds.top])
            for g in clipped.geometry:
                if g is None or g.is_empty:
                    continue
                if g.geom_type == "Polygon":
                    flat_polys.append(g)
                elif g.geom_type == "MultiPolygon":
                    flat_polys.extend(list(g.geoms))

        if "region" in only:
            gt_region = build_gt_region(flat_polys, (H, W), transform)
            save_geotiff(gt_region, os.path.join(out_dirs["region"], stem + ".tif"), crs, transform)

        if "lines" in only:
            gt_lines = build_gt_lines(flat_polys, (H, W), transform)
            save_geotiff(gt_lines, os.path.join(out_dirs["lines"], stem + ".tif"), crs, transform)

        if "nodes" in only:
            gt_nodes = build_gt_nodes(flat_polys, (H, W), transform)
            save_geotiff(gt_nodes, os.path.join(out_dirs["nodes"], stem + ".tif"), crs, transform)

        if "afm" in only:
            gt_afm = build_gt_afm(flat_polys, (H, W), transform)
            save_geotiff(gt_afm, os.path.join(out_dirs["afm"], stem + ".tif"), crs, transform)

        if (i + 1) % 200 == 0 or (i + 1) == len(stems):
            print(f"  {i+1}/{len(stems)} patches done")

    print(f"Split {split}: done -> {list(out_dirs.values())}")


def main():
    global ANGLE_THRESHOLD_DEG

    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "test", "all"], default="all")
    parser.add_argument("--limit", type=int, default=None,
                         help="only process the first N patches per split")
    parser.add_argument("--angle-threshold", type=float, default=ANGLE_THRESHOLD_DEG)
    parser.add_argument("--only", nargs="+", choices=list(ALL_TARGETS), default=list(ALL_TARGETS),
                         help="which targets to (re)compute, e.g. --only afm, or --only region lines nodes")
    args = parser.parse_args()

    ANGLE_THRESHOLD_DEG = args.angle_threshold

    print("Loading BRP GeoPackage...")
    brp = gpd.read_file(BRP_FILE)
    print(f"  {len(brp)} parcels loaded, CRS={brp.crs}")

    splits = ["train", "test"] if args.split == "all" else [args.split]
    for split in splits:
        process_split(split, brp, limit=args.limit, only=args.only)


if __name__ == "__main__":
    main()

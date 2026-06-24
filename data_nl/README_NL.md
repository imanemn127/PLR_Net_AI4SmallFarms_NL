# Netherlands BRP Dataset Reconstruction — PLR-Net

Reconstruction of the Netherlands Sentinel-2 dataset used in the PLR-Net article:

> Mengmeng Li et al., *"Extracting vectorized agricultural parcels from high-resolution
> satellite images using a Point-Line-Region interactive multitask model"*,
> Computers and Electronics in Agriculture, 231 (2025) 109953.

---

## Motivation

The article reports **mask IoU = 75.86%** on a national Netherlands dataset
(~64,800 km², 6,940 training patches, 1,504 test patches at 256×256 px, 10 m/px).
This dataset is not publicly available.

A preliminary experiment on the `sentinel-2-nl` subset of Ai4SmallFarms
(87 tiles of ~1 km², same BRP labels) reached **mask IoU = 0.674**,
with the gap attributed to the 7× smaller training set (87 km² vs 64,800 km²).
This motivated reconstructing the full national dataset from open sources.

---

## Why the two datasets are incompatible

| Criterion | PLR-Net article | sentinel-2-nl (Ai4SmallFarms) |
|-----------|----------------|-------------------------------|
| Spatial extent | National mosaic ~64,800 km² (42,845×31,580 px) | 87 sparse tiles ~1×1 km each |
| Image source | PDOK official composite | Independent acquisition |
| Radiometric range | 0–~8,000 (BOA reflectance) | 0–~20,000 |
| Labels | Full BRP 2020 (all parcel types) | BRP subset (GeoPackage per tile) |
| Patch extraction | Sliding window on 3 geographic boxes | No patch extraction (tile = sample) |
| Train/test split | 2 red boxes (train) + 1 yellow box (test) | 60/13/14 tiles geographic split |

Results on `sentinel-2-nl` cannot be directly compared to the article.
They constitute a **geographic transfer evaluation** (model trained on NL national
data, tested on a different NL acquisition), not a reproduction.

---

## Data Sources

| Source | Description | URL |
|--------|-------------|-----|
| Sentinel-2 L2A | Cloud-free median composite, May–Oct 2020, 10 m, EPSG:28992 | Google Earth Engine |
| BRP 2020 | Dutch agricultural parcel cadastre (GeoPackage) | https://www.pdok.nl |

---

## Pipeline Overview

```
GEE export (4 quadrants)
       ↓
step1_merge_mosaic.py      → NL_mosaic_2020_10m.tif       (28,966 × 32,488 px)
       ↓
step2_rasterize_brp.py     → NL_BRP2020_10m_labels.tif    (parcel IDs raster)
       ↓
step3_extract_patches.py   → patches/train/  6,942 patches
                           → patches/test/   1,505 patches
       ↓
split_train_val.py         → train.txt (5,553)  val.txt (1,389)
       ↓
step5_build_coco.py        → coco/train_coco.json  (358,996 annotations)
                           → coco/val_coco.json    ( 89,838 annotations)
                           → coco/test_coco.json   ( 96,910 annotations)
       ↓
compute_normalization.py   → PIXEL_MEAN / PIXEL_STD for PLR-Net.yaml
```

---

## Directory Structure

```
data_nl/
├── mosaic_sentinel2/               # not in git
│   ├── S2_NL_2020_NL_{NW,NE,SW,SE}-*.tif   # GEE sub-tiles
│   └── NL_mosaic_2020_10m.tif               # merged (step 1 output)
├── brp/                            # not in git
│   └── brp_crop_plots_definitive_2020.gpkg
├── labels_raster/                  # not in git
│   └── NL_BRP2020_10m_labels.tif
├── patches/                        # not in git
│   ├── train/images/   (6,942 × 256×256 GeoTIFF)
│   ├── train/masks/    (6,942 × 256×256 GeoTIFF, parcel IDs)
│   ├── test/images/    (1,505 × 256×256 GeoTIFF)
│   ├── test/masks/     (1,505 × 256×256 GeoTIFF)
│   ├── train.txt       (5,553 stems, 80% split)
│   └── val.txt         (1,389 stems, 20% split)
├── coco/                           # not in git
│   ├── train_coco.json
│   ├── val_coco.json
│   └── test_coco.json
├── step1_merge_mosaic.py
├── step2_rasterize_brp.py
├── step3_extract_patches.py
├── split_train_val.py
├── step5_build_coco.py
└── README_NL.md                    # this file
```

Large files (`.tif`, `.gpkg`, patches, JSON) are excluded from git via `.gitignore`.

---

## Step 1 — Sentinel-2 Mosaic (Google Earth Engine)

GEE script: `scripts/gee_sentinel2_nl_mosaic.js`

- **Collection**: `COPERNICUS/S2_SR` (Sentinel-2 Level-2A)
- **Date range**: 2020-05-01 to 2020-10-01 (growing season, cloud reduction)
- **Cloud filter**: SCL band — values 3 (shadow), 8, 9, 10 (cloud/cirrus) masked
- **Composite**: median of valid pixels per scene < 50% cloud cover
- **Bands exported**: B2, B3, B4, B8 (blue, green, red, NIR — 10 m)
- **CRS**: EPSG:28992 (RD New — Dutch national projection)
- **Export**: 4 geographic quadrants (NW/NE/SW/SE), split into sub-tiles by GEE

After downloading all sub-tiles from Google Drive:

```bash
/mnt/DATA/IMANE/ai4sf/bin/python data_nl/step1_merge_mosaic.py
```

Output: `mosaic_sentinel2/NL_mosaic_2020_10m.tif` — 28,966 × 32,488 px, LZW compressed.

> **Note:** The article uses a PDOK official composite. Our GEE reconstruction
> produces slightly different radiometric values (median vs PDOK pipeline) and
> has cosmetic seam lines between quadrants that do not affect agricultural patches.

---

## Step 2 — Rasterize BRP 2020

Download `brp_crop_plots_definitive_2020.gpkg` from https://www.pdok.nl.

```bash
/mnt/DATA/IMANE/ai4sf/bin/python data_nl/step2_rasterize_brp.py
```

Reprojects to EPSG:28992 if needed, burns unique integer parcel IDs onto the
exact mosaic pixel grid. Output: `labels_raster/NL_BRP2020_10m_labels.tif`.
Pixel value = parcel ID (1-based integer) or 0 (background / non-agricultural).

---

## Step 3 — Extract 256×256 Patches

```bash
/mnt/DATA/IMANE/ai4sf/bin/python data_nl/step3_extract_patches.py
```

Sliding window: **256×256 px, stride=205 px (≈ 20% overlap)**, matching the article:

> *"We cropped the images and corresponding ground-truth parcel labels from each area
> into image patches of 256×256 pixels with a 20% overlap."*

### Zone coordinates (EPSG:28992)

The article's Figure 1 shows two red boxes (train) and one yellow box (test) but
does not publish their coordinates. They were estimated from the figure scale bar:

| Zone | Role | Coordinates (xmin, ymin, xmax, ymax) | Size | Patches |
|------|------|--------------------------------------|------|---------|
| Yellow | Test | (133870, 535340, 206130, 624000) | 72×89 km | 1,505 |
| Red 1 | Train | (148000, 386290, 261260, 518000) | 113×132 km | 3,520 |
| Red 2 | Train | (26540, 360590, 148000, 480000) | 122×119 km | 3,422 |

**Geographic location:**
- Yellow box → Friesland (north Netherlands, agricultural polders)
- Red box 1 → Flevoland / Gelderland / Overijssel (centre-east)
- Red box 2 → Zuid-Holland / Utrecht / Noord-Brabant (centre-west)

Red boxes 1 and 2 are side by side horizontally (x_max_red2 = x_min_red1 = 148,000)
with no pixel overlap. Yellow box is entirely north of both red boxes.

**`MIN_PARCEL_FRACTION = 0.0`**: all patches are kept, including mosaic border
patches (NoData encoded as `float64 NaN`). NaN pixels are replaced by 0 via
`np.nan_to_num` in the dataset loader before any normalisation. This matches the
article's protocol (mechanical sliding window, no filtering).

**Patch counts vs article:**

| Split | This reconstruction | Article |
|-------|---------------------|---------|
| Train | 6,942 | 6,940 |
| Test | 1,505 | 1,504 |

---

## Step 4 — Train / Val Split

```bash
/mnt/DATA/IMANE/ai4sf/bin/python data_nl/split_train_val.py
```

Random 80/20 split of training patches with `random.seed(42)` for reproducibility.
Writes `patches/train.txt` (5,553 stems) and `patches/val.txt` (1,389 stems).

---

## Step 5 — Build COCO JSON

```bash
/mnt/DATA/IMANE/ai4sf/bin/python data_nl/step5_build_coco.py
```

For each patch:
1. Read the label mask (parcel IDs raster)
2. For each unique parcel ID → binarize → vectorize via `rasterio.features.shapes()`
3. Convert polygon to pixel coordinates, compute COCO bbox and area (shoelace)
4. Discard polygons < 100 px² (slivers from rasterization)
5. Write standard COCO JSON (`images` + `annotations` + `categories`)

Output: `coco/{train,val,test}_coco.json`.

| Split | Images | Annotations | Avg ann/patch |
|-------|--------|-------------|---------------|
| train | 5,553 | 358,996 | 64.7 |
| val | 1,389 | 89,838 | 64.7 |
| test | 1,505 | 96,910 | 64.4 |

---

## Normalisation

```bash
/mnt/DATA/IMANE/ai4sf/bin/python scripts/compute_normalization.py
```

Computed on train patches (NaN/all-zero pixels excluded).
Values after dividing raw uint16 by 10,000:

| Channel | Mean | Std |
|---------|------|-----|
| B2 (blue) | 0.0545 | 0.0298 |
| B3 (green) | 0.0768 | 0.0314 |
| B4 (red) | 0.0668 | 0.0397 |

---

## Dataset Registration

Three entries added to `PLRNet/config/paths_catalog.py`:

```python
'nl_brp_train': { 'img_dir': '../data_nl/patches',
                  'ann_file': '../data_nl/coco/train_coco.json' },
'nl_brp_val':   { 'img_dir': '../data_nl/patches',
                  'ann_file': '../data_nl/coco/val_coco.json'   },
'nl_brp_test':  { 'img_dir': '../data_nl/patches',
                  'ann_file': '../data_nl/coco/test_coco.json'  },
```

Active configuration in `config-files/PLR-Net.yaml`:

```yaml
DATASETS:
  TRAIN: ("nl_brp_train",)
  TEST:  ("nl_brp_val",)      # val during training
  IMAGE:
    PIXEL_MEAN: [0.0545, 0.0768, 0.0668]
    PIXEL_STD:  [0.0298, 0.0314, 0.0397]
    TO_255: False
SOLVER:
  IMS_PER_BATCH: 8
  BASE_LR: 1e-4
  MAX_EPOCH: 150
OUTPUT_DIR: "/mnt/DATA/IMANE/PLR-Net_output/PLR-Net/nl_brp"
```

For final evaluation on the test split:

```bash
CUDA_VISIBLE_DEVICES=1 /mnt/DATA/IMANE/ai4sf/bin/python scripts/test.py \
    --config-file config-files/PLR-Net.yaml \
    DATASETS.TEST '("nl_brp_test",)'
```

---

## Training

```bash
CUDA_VISIBLE_DEVICES=1 /mnt/DATA/IMANE/ai4sf/bin/python -u scripts/train.py \
    --config-file config-files/PLR-Net.yaml
```

---

## Results (to be updated after training)

| Metric | This run | Article (PLR-Net) |
|--------|----------|-------------------|
| Mask IoU (%) | — | 75.86 |
| AP poly (%) | — | 47.1 |
| AR poly (%) | — | 55.5 |
| AP boundary (%) | — | 41.5 |
| AR boundary (%) | — | 53.1 |
| PoLiS (m) | — | 1.51 |

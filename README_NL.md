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

## Training History (NL BRP dataset)

All runs use Adam, `IMS_PER_BATCH=8`, `MAX_EPOCH=150`, `STEPS=(25,)` unless noted.
Junction loss is CrossEntropy over 3 classes (background/concave/convex) in runs NL1–NL6.
Junction recall metrics were added to the CSV starting from run NL5.

---

### NL1 — batch=1 smoke test (stopped at epoch 6)

First launch on the NL dataset to check the pipeline end-to-end. `IMS_PER_BATCH=1` was a
configuration mistake — throughput was too slow to be useful.

| Config | Value |
|--------|-------|
| `IMS_PER_BATCH` | **1** (bug) |
| `loss_jloc` weight | 8.0 |
| `loss_joff` weight | 0.25 |
| STEPS | (13,) |

Stopped at epoch 6 after confirming the pipeline runs without errors. No val metrics.

---

### NL2 — batch=8, `STEPS=(13,)` (stopped at epoch 67)

Fixed the batch size. Used the same LR schedule as the first AI4SmallFarms runs.

| Config | Value |
|--------|-------|
| `IMS_PER_BATCH` | 8 |
| `loss_jloc` weight | 8.0 |
| `loss_joff` weight | 0.25 |
| STEPS | (13,) |

| Loss (weighted) | Train (ep 1 → 67) | Val (ep 5 → 65) |
|------|-------------------|-----------------|
| total | 3.31 → 2.03 | 0.321 → 0.256 |
| `w_loss_jloc` | 1.88 → 1.03 | 0.167 → 0.128 |
| `w_loss_mask` | 0.33 → 0.17 | 0.031 → 0.023 |
| `w_loss_afm` | 0.41 → 0.25 | 0.041 → 0.033 |
| `w_loss_remask` | 0.64 → 0.58 | 0.081 → 0.072 |
| **`val_mask_iou`** | — | **0.463 (ep 5) → 0.783 (ep 65)** |

Stopped early at epoch 67 — LR drop at epoch 13 was too aggressive given that the NL dataset
is much larger than AI4SmallFarms (the model converges well before the drop and then stagnates).
Changed `STEPS` to `(25,)` for next run.

---

### NL3 — `STEPS=(25,)` (stopped at epoch 112)

| Config | Value |
|--------|-------|
| `loss_jloc` weight | 8.0 |
| `loss_joff` weight | 0.25 |
| STEPS | **(25,)** |

| Loss (weighted) | Train (ep 1 → 112) | Val (ep 5 → 110) |
|------|---------------------|------------------|
| total | 3.29 → 1.95 | 0.336 → 0.262 |
| `w_loss_jloc` | 1.87 → 0.98 | 0.158 → 0.127 |
| `w_loss_mask` | 0.33 → 0.16 | 0.042 → 0.024 |
| `w_loss_afm` | 0.41 → 0.24 | 0.043 → 0.035 |
| `w_loss_remask` | 0.64 → 0.58 | 0.092 → 0.077 |
| **`val_mask_iou`** | — | **0.594 (ep 5) → 0.690 (ep 110)** |

Stopped early at epoch 112 to reduce `loss_jloc` weight — mask IoU was still rising but
slowly, and the junction branch behaviour remained unknown without recall metrics.

---

### NL4 — `loss_jloc` 8.0→4.0 (150 epochs, complete)

Hypothesis: `loss_jloc` weight 8.0 dominates all other terms at 6,942 training patches.
Halving it might let the mask and AFM branches converge better.

| Change | Why |
|--------|-----|
| `loss_jloc` weight 8.0 → **4.0** | Rebalance gradient with mask and AFM branches |

| Loss (weighted) | Train (ep 1 → 150) | Val (ep 150 only) |
|------|-------------------|-------------------|
| total | 2.93 → 1.56 | 0.221 |
| `w_loss_jloc` | 1.28 → 0.65 | 0.089 |
| `w_loss_mask` | 0.40 → 0.14 | 0.030 |
| `w_loss_afm` | 0.57 → 0.29 | 0.039 |
| `w_loss_remask` | 0.60 → 0.49 | 0.063 |
| **`val_mask_iou`** | — | **0.763** |

First good result: val IoU at 76.3% at epoch 150, very close to the article (75.86%).
However, `loss_joff` weight was still 0.25 and junc recall was not tracked yet — the junction
branch behaviour was unknown at this point.

---

### NL5 — add junction recall tracking, `loss_joff` 0.25→1.0, `class_weights=[0.1, 5, 5]` (stopped at epoch 46)

After NL4 showed good mask IoU, the question became: why does APpoly stay near 0?
Added `val_junc_recall@{3,5,8}px` to the CSV. Also added `JLOC_CLASS_WEIGHTS=[0.1, 5.0, 5.0]`
to up-weight concave and convex corners (which are ~0.5% of pixels) in the CrossEntropy loss.
Raised `loss_joff` to 1.0 (same reasoning as the AI4SmallFarms experiment).

| Change | Why |
|--------|-----|
| Junction recall added to CSV | Need to quantify how well the point branch learns |
| `JLOC_CLASS_WEIGHTS=[0.1, 5.0, 5.0]` | Up-weight corner pixels in CrossEntropy (background ~99.5%) |
| `loss_joff` weight 0.25 → **1.0** | Offset head starved of gradient at 0.25 |

| Loss (weighted) | Train (ep 1 → 46) | Val (ep 5 → 45) |
|------|-------------------|-----------------|
| total | 3.85 → 3.34 | 0.462 → 0.414 |
| `w_loss_jloc` | 2.56 → 2.22 | 0.307 → 0.278 |
| `w_loss_mask` | 0.29 → 0.23 | 0.035 → 0.029 |
| `w_loss_afm` | 0.43 → 0.35 | 0.051 → 0.043 |
| `w_loss_remask` | 0.57 → 0.53 | 0.069 → 0.064 |
| **`val_mask_iou`** | — | **0.675 (ep 5) → 0.736 (ep 45)** |
| val junc recall @5px | — | 0.750 (ep 5) → 0.727 (ep 45) |
| val junc recall @8px | — | 0.896 (ep 5) → 0.865 (ep 45) |

**Diagnosis:** Junction recall @5px starts very high at epoch 5 (75%) and actually *decreases*
as training continues (72.7% at epoch 45). The class weights overfit the training distribution
of corner types — the model predicts many spurious corners early on (high recall, low precision).
`val_w_loss_jloc` is around 0.28–0.31, which looks reasonable, but this is the CrossEntropy
value on 3 classes — it does not mean the predictions are spatially accurate.

Stopped at epoch 46: the high initial recall (coming mostly from the class weights) was
misleading, and mask IoU (73.6% at ep 45) was slightly worse than NL4.

---

### NL6 — same config as NL5 + RDP simplification of GT polygons (150 epochs, complete)

Hypothesis: the GT junctions from step5 contain many collinear vertices along straight
parcel edges (artefacts of the rasterio.shapes reconversion). Maybe simplifying GT polygons
with RDP before computing jloc would give cleaner corner targets.

| Change | Why |
|--------|-----|
| RDP simplification of GT polygons | Remove collinear vertices from rasterio.shapes artefacts |

| Loss (weighted) | Train (ep 1 → 150) | Val (ep 5 → 150) |
|------|-------------------|-----------------|
| total | 2.79 → 1.79 | 1.580 → 2.191 |
| `w_loss_jloc` | 1.56 → 0.89 | 1.415 → 2.037 |
| `w_loss_mask` | 0.28 → 0.13 | 0.034 → 0.033 |
| `w_loss_afm` | 0.36 → 0.26 | 0.062 → 0.057 |
| `w_loss_remask` | 0.56 → 0.49 | 0.068 → 0.063 |
| **`val_mask_iou`** | — | **0.697 (ep 5) → 0.759 (ep 150)** |
| val junc recall @5px | — | 0.719 (ep 5) → 0.688 (ep 150) |
| val junc recall @8px | — | 0.889 (ep 5) → 0.850 (ep 150) |

**Diagnosis:** `val_w_loss_jloc` keeps increasing (1.42 at ep 5 → 2.04 at ep 150) even as
train jloc loss decreases — clear overfitting on the junction branch after RDP. RDP reduces
the number of GT corner pixels, which makes the training targets sparser but also more
sensitive to mismatch. Junction recall @5px decreases over epochs (0.72 → 0.69), suggesting
the model is fitting the RDP-simplified corners but those corners diverge from the true
geometry visible in the val images.

Val IoU at 75.9% is almost identical to NL4, so RDP had no benefit for segmentation and
made the junction branch worse.

---

### NL7 — CrossEntropy → BCE binary, `pos_weight=50` (150 epochs, complete)

Reading the article section 3.4 more carefully: the point branch predicts a single binary
heatmap (0=background, 1=any corner), not a 3-class map. The article makes no distinction
between concave and convex corners in the loss — both are target=1. This is different from
what runs NL1–NL6 implemented (CrossEntropy over 3 classes with class weights).

| Change | Why |
|--------|-----|
| `nn.CrossEntropyLoss` 3-class → `nn.BCEWithLogitsLoss` binary | Article uses a single binary heatmap, not concave/convex classification |
| `JLOC_CLASS_WEIGHTS=[0.1,5,5]` → `JLOC_POS_WEIGHT=50.0` | BCE equivalent for handling ~0.5% corner pixel imbalance |
| `jloc_predictor` 3 channels → **1 channel** | Binary output; no class distinction |
| `jmap` in encoder: int 3-class → float binary (0.0/1.0) | GT consistent with new loss |
| `loss_joff` weight reset to 0.25 | RDP result showed raising joff to 1.0 hurts mask IoU |

| Loss (weighted) | Train (ep 1 → 150) | Val (ep 5 → 150) |
|------|-------------------|-----------------|
| total | 8.86 → 4.90 | 0.851 → 0.817 |
| `w_loss_jloc` | 7.16 → 3.85 | 0.689 → 0.681 |
| `w_loss_mask` | 0.41 → 0.20 | 0.037 → 0.030 |
| `w_loss_afm` | 0.60 → 0.33 | 0.055 → 0.042 |
| `w_loss_remask` | 0.61 → 0.52 | 0.069 → 0.063 |
| **`val_mask_iou`** | — | **0.674 (ep 5) → 0.753 (ep 150)** |
| val junc recall @3px | — | 0.379 (ep 5) → 0.427 (ep 150) |
| val junc recall @5px | — | 0.532 (ep 5) → **0.559 (ep 150)** |
| val junc recall @8px | — | 0.676 (ep 5) → 0.680 (ep 150) |

**Diagnosis:** Mask IoU converges to 75.3% at epoch 150, almost matching the article (75.86%).
Junction recall @5px reaches 55.9% and is still slowly rising — in Fig. 12 of the article,
the NL APpoly curve continues rising until epoch 100+, which is consistent with the model
needing more epochs to sharpen corner localisation.

The `w_loss_jloc` absolute values are high (7.16 → 3.85) compared to the CrossEntropy runs
because `pos_weight=50` inflates the BCE loss on corner pixels. This is expected — the
relative balance with other losses is what matters, and val loss decreases steadily.

APpoly stays near 0% despite the improved junction recall. The issue is not the loss function
anymore — see the GT quality analysis below.

---

### NL8 — Branch point isolated training (stopped at epoch 81)

Train each of the 3 branches (region, line, point) independently to locate
where the network fails. Config: `config-files/PLR-Net_branch_point.yaml`,
`ACTIVE_BRANCH="point"`. The region and line losses are zeroed, and the SCE cross-attention
between branches is disabled — the point branch trains only on its own backbone features.


| Loss (weighted) | Train (ep 1 → 80) | Val (ep 5 → 80) |
|------|-------------------|-----------------|
| `w_loss_jloc` | 7.21 → 4.50 | 0.703 → 0.601 |
| `w_loss_mask` | 0.0 (disabled) | 0.0 (disabled) |
| `w_loss_afm` | 0.0 (disabled) | 0.0 (disabled) |
| val junc recall @5px | — | **0.539 (ep 5) → 0.539 (ep 80)** |
| val junc recall @8px | — | 0.690 (ep 5) → 0.660 (ep 80) |

**Diagnosis:** The `w_loss_mask = 0.0` and `w_loss_afm = 0.0` at every single epoch confirm
the isolation works — only the point branch gradient flows. Junction recall plateaus
immediately at ~54% @5px from epoch 5 and does not improve with more training. The training
loss (`w_loss_jloc`) decreases (7.21 → 4.50) which means the network is fitting the GT
heatmap — but recall does not improve on val, which means the GT heatmap itself is wrong.
This confirms the hypothesis: the bottleneck is the GT corner quality, not the network's
capacity to learn from it.

---

## GT Corner Quality Issue

### Why the GT junctions are wrong

The COCO JSON annotations (and therefore all GT junctions) come from reconverting the parcel
ID raster back to vectors using `rasterio.features.shapes()`. This function traces pixel
boundary contours — it outputs a vertex at every pixel-level direction change, not at real
parcel corners. Two types of bad corners result:

1. **Rasterisation artefacts**: a diagonal parcel edge in pixel space becomes a staircase
   contour with one vertex every 1–2 px. These are not real agricultural corners.
2. **BRP parasite vertices**: the BRP vector source already has collinear vertices along
   straight parcel edges — they survive the raster→vector round-trip and appear as GT corners
   in the middle of edges.

A scan of all training patches (`scripts/inspect_gt_jloc.py`) found:
- Average raw junctions per patch: **5,362**
- After rounding to integer pixel: **4,508 unique pixels** (97.6% of patches have collisions)
- Corners lost to pixel collision: **~873 per patch** on average

### QGIS comparison

To verify the GT quality, the BRP vector corners were rasterized directly from QGIS:

1. Load BRP GeoPackage
2. Vector → Geometry Tools → Convert Geometry Type → Points (extracts all vertices)
3. Raster → Conversion → Rasterize (burn=1, same pixel grid as `NL_BRP2020_10m_labels.tif`)

Comparison on patch `NL_train_z1_r000000_c002460`:
- The QGIS rasterization (red) shows dense clusters in the interior of parcels — these are
  the BRP parasite vertices on straight edges, not real corners.
- The code GT (white) differs from the QGIS GT in distribution, confirming that the
  raster→vector round-trip in step5 introduces additional spurious corners beyond the BRP
  artefacts already present in the vector.

The script `scripts/export_gt_geotiff.py` exports the code GT as georeferenced GeoTIFFs
(EPSG:28992) for QGIS overlay with the BRP vector layer.

### Why this keeps APpoly near 0

APpoly requires matching predicted polygon vertex sequences to GT polygon vertices.
Even if the network learns to predict where the GT says corners are, those GT corners do not
correspond to real parcel geometry — so generated polygons can never match GT polygons at
the vertex level, and APpoly stays near 0.

A fix would require building GT junctions directly from the BRP vector geometry (no
raster→vector round-trip) and filtering collinear vertices with a minimum angle threshold.
This is left as future work.

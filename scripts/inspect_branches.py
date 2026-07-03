#!/usr/bin/env python3
"""
inspect_branches.py — visualize each PLR-Net branch separately.

For each selected val image, saves a 6-panel figure:
  [0] RGB image
  [1] GT mask
  [2] Remask pred  (region branch)   ← does it separate adjacent parcels?
  [3] AFM pred magnitude             (line branch)    ← are boundaries visible?
  [4] jloc pred heatmap              (point branch)   ← where are predicted corners?
  [5] GT jloc                                         ← where should corners be?

Usage (from PLR-Net/ directory):
  /mnt/DATA/IMANE/ai4sf/bin/python scripts/inspect_branches.py \
      --config    config-files/PLR-Net.yaml \
      --checkpoint /home/imane/DATA/PLR-Net_output/PLR-Net/nl_brp/2026-06-29_10-09-57/checkpoints/best_val_loss.pth \
      --output    /home/imane/DATA/PLR-Net_output/inspect_branches
"""

import argparse
import os
import sys

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from pycocotools.coco import COCO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PLRNet.config import cfg
from PLRNet.dataset import build_test_dataset
from PLRNet.detector import BuildingDetector
from PLRNet.encoder import Encoder
from PLRNet.utils.polygon import get_pred_junctions


# ------------------------------------------------------------------ #
#  Helpers
# ------------------------------------------------------------------ #
def tensor_to_rgb(image_tensor, mean, std):
    """Normalised (C,H,W) tensor → uint8 (H,W,3)."""
    img = image_tensor.cpu().numpy().transpose(1, 2, 0)
    img = img * np.array(std) + np.array(mean)
    for c in range(img.shape[2]):
        p2, p98 = np.percentile(img[:, :, c], (2, 98))
        if p98 > p2:
            img[:, :, c] = (img[:, :, c] - p2) / (p98 - p2)
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def build_gt_jloc(ann, H=256, W=256):
    """Build GT jloc map (H,W) from annotation dict — same logic as encoder."""
    junctions = ann.get('junctions', None)
    junc_tag  = ann.get('juncs_tag',  None)
    jmap = np.zeros((H, W), dtype=np.float32)
    if junctions is not None and len(junctions) > 0:
        j  = np.array(junctions)
        jt = np.array(junc_tag)
        xi = np.clip(j[:, 0].astype(int), 0, W - 1)
        yi = np.clip(j[:, 1].astype(int), 0, H - 1)
        jmap[yi, xi] = jt.astype(np.float32)
    return jmap


def run_branches(model, image_tensor, device):
    """
    Run the network and return each branch output separately.
    We hook into the model internals directly (no generate_polygon).
    Returns dict with keys: remask, mask, afm_mag, jloc_concave, jloc_convex, jloc_sum, juncs
    """
    with torch.no_grad():
        images = image_tensor.to(device)
        outputs, features = model.backbone(images)

        mask_feature = model.mask_head(features)
        jloc_feature = model.jloc_head(features)
        afm_feature  = model.afm_head(features)

        mask_att = model.a2m_att(mask_feature, mask_feature + afm_feature)
        jloc_att = model.a2j_att(jloc_feature, jloc_feature + afm_feature)

        mask_pred   = model.mask_predictor(mask_att).softmax(1)[:, 1]   # (B,H,W)
        jloc_pred   = model.jloc_predictor(jloc_att).softmax(1)         # (B,3,H,W)
        afm_pred    = model.afm_predictor(afm_feature)                   # (B,2,H,W)
        afm_conv    = model.refuse_conv(afm_pred)
        remask_pred = model.final_conv(features + afm_conv).softmax(1)[:, 1]  # (B,H,W)
        joff_pred   = outputs[:, :].sigmoid() - 0.5                      # (B,2,H,W)

        b = 0
        jloc_concave = jloc_pred[b:b+1, 1:2]
        jloc_convex  = jloc_pred[b:b+1, 2:3]
        juncs = get_pred_junctions(jloc_concave[0], jloc_convex[0], joff_pred[b])

        # AFM magnitude = sqrt(ch0^2 + ch1^2)
        afm_np = afm_pred[b].cpu().numpy()
        afm_mag = np.sqrt(afm_np[0]**2 + afm_np[1]**2)

        return {
            'mask'         : mask_pred[b].cpu().numpy(),
            'remask'       : remask_pred[b].cpu().numpy(),
            'afm_mag'      : afm_mag,
            'jloc_concave' : jloc_concave[0, 0].cpu().numpy(),
            'jloc_convex'  : jloc_convex[0, 0].cpu().numpy(),
            'jloc_sum'     : (jloc_concave[0, 0] + jloc_convex[0, 0]).cpu().numpy(),
            'juncs'        : juncs,
        }


def save_figure(rgb, gt_mask, gt_jloc, branches, img_name, out_dir, threshold=0.5):
    """Save 6-panel figure for one image."""
    remask   = branches['remask']
    afm_mag  = branches['afm_mag']
    jloc_sum = branches['jloc_sum']
    juncs    = branches['juncs']
    mask_pred = branches['mask']

    fig, axes = plt.subplots(2, 4, figsize=(24, 12), dpi=120)
    fig.patch.set_facecolor('#111111')

    panels = [
        # row 0
        (axes[0, 0], rgb,                       'RGB image',              'none',       None),
        (axes[0, 1], gt_mask,                   'GT mask',                'gray',       None),
        (axes[0, 2], (remask > threshold).astype(float), f'Remask pred (>{threshold})', 'gray', None),
        (axes[0, 3], remask,                    'Remask pred (raw prob)', 'RdYlGn',     None),
        # row 1
        (axes[1, 0], mask_pred,                 'Mask pred (raw prob)',   'RdYlGn',     None),
        (axes[1, 1], afm_mag,                   'AFM magnitude',          'hot',        None),
        (axes[1, 2], jloc_sum,                  'jloc pred (concave+convex)', 'hot',   juncs),
        (axes[1, 3], gt_jloc,                   'GT jloc (0=bg,1=conc,2=conv)', 'hot', None),
    ]

    for ax, data, title, cmap, junc_overlay in panels:
        ax.axis('off')
        ax.set_title(title, color='white', fontsize=9, pad=3)
        if cmap == 'none':
            ax.imshow(data)
        else:
            ax.imshow(data, cmap=cmap, interpolation='nearest')
        if junc_overlay is not None and len(junc_overlay) > 0:
            ax.scatter(junc_overlay[:, 0], junc_overlay[:, 1],
                       s=4, c='cyan', marker='+', linewidths=0.6, zorder=5)

    # stats overlay on remask binary panel
    n_blobs_pred = _count_blobs(remask, threshold)
    n_blobs_gt   = _count_blobs(gt_mask, 0.5)
    axes[0, 2].set_title(
        f'Remask pred (>{threshold})  blobs={n_blobs_pred} (GT={n_blobs_gt})',
        color='white', fontsize=8, pad=3
    )
    axes[1, 2].set_title(
        f'jloc pred — {len(juncs)} junctions detected',
        color='white', fontsize=8, pad=3
    )

    plt.tight_layout(pad=0.5)
    out_path = os.path.join(out_dir, f'{img_name}.png')
    plt.savefig(out_path, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path, n_blobs_pred, n_blobs_gt


def _count_blobs(mask, threshold):
    from skimage.measure import label
    binary = (mask > threshold).astype(np.uint8)
    return label(binary).max()


# ------------------------------------------------------------------ #
#  Main
# ------------------------------------------------------------------ #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',     required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output',     default='/home/imane/DATA/PLR-Net_output/inspect_branches')
    parser.add_argument('--n-images',   type=int, default=6,
                        help='Number of val images to inspect (default 6)')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    cfg.merge_from_file(args.config)
    cfg.freeze()
    device = cfg.MODEL.DEVICE
    mean   = list(cfg.DATASETS.IMAGE.PIXEL_MEAN)
    std    = list(cfg.DATASETS.IMAGE.PIXEL_STD)

    # load model
    model = BuildingDetector(cfg, test=True).to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    state = ckpt.get('model', ckpt)
    state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval()
    epoch = ckpt.get('epoch', '?')
    print(f'Checkpoint loaded — epoch {epoch}')

    # load val dataset
    val_loader, val_ann_file = build_test_dataset(cfg)
    val_dataset = val_loader.dataset   # unwrap DataLoader → Dataset
    coco = val_dataset.coco
    img_ids = coco.getImgIds()

    # pick images: densest + sparsest + middle density
    counts   = {i: len(coco.getAnnIds(imgIds=[i])) for i in img_ids}
    sorted_ids = sorted(img_ids, key=counts.get, reverse=True)
    # spread across dense / medium / sparse
    step     = max(1, len(sorted_ids) // args.n_images)
    selected = [sorted_ids[i * step] for i in range(args.n_images)]

    encoder = Encoder(cfg)

    print(f'\nInspecting {len(selected)} images → {args.output}\n')
    print(f'{"image":<45} {"GT blobs":>9} {"Pred blobs":>11} {"n_juncs":>8} {"fusion?":>8}')
    print('-' * 85)

    for img_id in selected:
        img_info  = coco.loadImgs(ids=[img_id])[0]
        img_name  = os.path.splitext(os.path.basename(img_info['file_name']))[0]
        img_path  = os.path.join(val_dataset.root, img_info['file_name'])
        H, W      = img_info['height'], img_info['width']

        # load image
        with rasterio.open(img_path) as src:
            arr = src.read([1, 2, 3]).astype(np.float32) / 10000.0
        np.nan_to_num(arr, nan=0.0, copy=False)
        img_norm   = (arr.transpose(1, 2, 0) - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
        img_tensor = torch.from_numpy(img_norm.transpose(2, 0, 1).astype(np.float32)).unsqueeze(0)
        rgb        = tensor_to_rgb(img_tensor[0], mean, std)

        # GT mask
        ann_ids  = coco.getAnnIds(imgIds=[img_id])
        anns     = coco.loadAnns(ids=ann_ids)
        gt_mask  = np.zeros((H, W), dtype=np.float32)
        for ann in anns:
            gt_mask = np.clip(gt_mask + coco.annToMask(ann), 0, 1)

        # GT jloc — rebuild from dataset annotation
        gt_jloc = np.zeros((H, W), dtype=np.float32)
        if img_id in val_dataset.ids:
            ds_idx  = val_dataset.ids.index(img_id)
            _, ann_dict = val_dataset[ds_idx]
            gt_jloc = build_gt_jloc(ann_dict, H, W)

        # run branches
        branches = run_branches(model, img_tensor, device)

        # save figure
        out_path, n_pred, n_gt = save_figure(
            rgb, gt_mask, gt_jloc, branches, img_name, args.output
        )

        fusion = 'YES !' if n_pred < n_gt * 0.5 else ('maybe' if n_pred < n_gt * 0.8 else 'no')
        print(f'{img_name:<45} {n_gt:>9} {n_pred:>11} {len(branches["juncs"]):>8} {fusion:>8}')

    print(f'\nDone. Figures saved to: {args.output}')
    print('\nWhat to look for:')
    print('  Remask pred binarized : if n_pred_blobs << n_gt_blobs → parcels are fused (region branch fails)')
    print('  AFM magnitude         : should show HIGH values at parcel boundaries')
    print('  jloc pred vs GT       : do predicted junctions match GT corner locations?')


if __name__ == '__main__':
    main()

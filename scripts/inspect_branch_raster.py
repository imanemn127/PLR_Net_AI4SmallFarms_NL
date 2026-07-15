#!/usr/bin/env python3
"""
inspect_branch_raster.py — visualize a branch trained on GT rasters
(region/line/point/point_sce/all_raster), no COCO involved.

For an isolated branch, saves a 3-panel figure:
  [0] RGB image
  [1] GT raster (gt_region / gt_lines / gt_nodes, matching the checkpoint's branch)
  [2] Predicted probability map (sigmoid of the branch head logit)

For all_raster (mask+line+point trained together), saves a 7-panel figure:
  [0] RGB image
  [1..3] GT region / lines / nodes
  [4..6] Pred region / line / point

Usage (from PLR-Net/ directory):
  /mnt/DATA/IMANE/ai4sf/bin/python scripts/inspect_branch_raster.py \
      --config     config-files/PLR-Net_branch_point.yaml \
      --checkpoint /mnt/DATA/IMANE/PLR-Net_output/PLR-Net/nl_brp_raster/point/2026-07-09_.../checkpoints/best_val_loss.pth \
      --output     /home/imane/DATA/PLR-Net_output/inspect_branch_raster
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PLRNet.config import cfg
from PLRNet.detector import BuildingDetector, afm_to_pixel_offset

GT_SUBDIR_BY_BRANCH = {
    'region':    'gt_region',
    'line':      'gt_lines',
    'point':     'gt_nodes',
    'point_sce': 'gt_nodes',  # same target as "point", predicted with frozen-line SCE guidance
}

# ADDED: for all_raster, one (name, GT subdir) pair per sub-branch
ALL_RASTER_SUBBRANCHES = [
    ('region', 'gt_region'),
    ('line',   'gt_lines'),
    ('point',  'gt_nodes'),
]


def tensor_to_rgb(image_tensor, mean, std):
    img = image_tensor.cpu().numpy().transpose(1, 2, 0)
    img = img * np.array(std) + np.array(mean)
    for c in range(img.shape[2]):
        p2, p98 = np.percentile(img[:, :, c], (2, 98))
        if p98 > p2:
            img[:, :, c] = (img[:, :, c] - p2) / (p98 - p2)
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def load_patch(stem, patches_root, mean, std):
    img_path = os.path.join(patches_root, 'images', stem + '.tif')
    with rasterio.open(img_path) as src:
        arr = src.read([1, 2, 3]).astype(np.float32) / 10000.0
    np.nan_to_num(arr, nan=0.0, copy=False)
    img_norm = (arr.transpose(1, 2, 0) - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    img_tensor = torch.from_numpy(img_norm.transpose(2, 0, 1).astype(np.float32)).unsqueeze(0)
    rgb = tensor_to_rgb(img_tensor[0], mean, std)
    return img_tensor, rgb


def load_gt(stem, patches_root, gt_subdir):
    gt_path = os.path.join(patches_root, gt_subdir, stem + '.tif')
    with rasterio.open(gt_path) as src:
        return src.read(1).astype(np.float32)


def _region_pred(model, mask_feature):
    mask_pred = model.mask_predictor(mask_feature)
    return mask_pred[0, 1].sigmoid().cpu().numpy()


def _point_pred(model, jloc_feature):
    jloc_pred = model.jloc_predictor(jloc_feature)
    return jloc_pred[0, 0].sigmoid().cpu().numpy()


def _point_sce_pred(model, jloc_feature, afm_feature):
    # same SCE path as detector.py forward_train for this mode:
    # jloc guided by the (frozen or jointly-trained) afm_feature
    jloc_att_feature = model.a2j_att(jloc_feature, jloc_feature + afm_feature)
    jloc_pred = model.jloc_predictor(jloc_att_feature)
    return jloc_pred[0, 0].sigmoid().cpu().numpy()


def _line_pred(model, afm_feature):
    # afm_predictor output is log-encoded by afm_op's convention, not a
    # raw pixel offset (see afm_to_pixel_offset in detector.py) — must
    # be decoded before the norm means anything in pixels
    afm_pred = model.afm_predictor(afm_feature)
    H, W = afm_pred.shape[-2:]
    ax, ay = afm_to_pixel_offset(afm_pred, H, W)
    afm_norm_px = torch.sqrt(ax ** 2 + ay ** 2 + 1e-6)
    return (1.0 / (1.0 + afm_norm_px))[0].cpu().numpy()


@torch.no_grad()
def run_branch(model, image_tensor, device, active_branch):
    """Isolated branches: returns a single (H,W) prediction array.
    all_raster: returns a dict {'region': ..., 'line': ..., 'point': ...}."""
    images = image_tensor.to(device)
    outputs, features = model.backbone(images)

    mask_feature = model.mask_head(features)
    jloc_feature = model.jloc_head(features)
    afm_feature  = model.afm_head(features)

    if active_branch == 'region':
        return _region_pred(model, mask_feature)
    if active_branch == 'point':
        return _point_pred(model, jloc_feature)
    if active_branch == 'point_sce':
        return _point_sce_pred(model, jloc_feature, afm_feature)
    if active_branch == 'line':
        return _line_pred(model, afm_feature)
    if active_branch == 'all_raster':
        # SCE active for both mask and jloc, as in detector.py forward_train
        mask_att_feature = model.a2m_att(mask_feature, mask_feature + afm_feature)
        return {
            'region': _region_pred(model, mask_att_feature),
            'line':   _line_pred(model, afm_feature),
            'point':  _point_sce_pred(model, jloc_feature, afm_feature),
        }
    raise ValueError(f"unsupported active_branch {active_branch!r}")


def save_figure(rgb, gt, pred, img_name, branch, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=120)
    fig.patch.set_facecolor('#111111')

    panels = [
        (axes[0], rgb,  'RGB image',        'none'),
        (axes[1], gt,   f'GT ({branch})',   'gray'),
        (axes[2], pred, f'Pred ({branch})', 'hot'),
    ]
    for ax, data, title, cmap in panels:
        ax.axis('off')
        ax.set_title(title, color='white', fontsize=10, pad=4)
        if cmap == 'none':
            ax.imshow(data)
        else:
            ax.imshow(data, cmap=cmap, vmin=0, vmax=1, interpolation='nearest')

    plt.tight_layout(pad=0.5)
    out_path = os.path.join(out_dir, f'{img_name}.png')
    plt.savefig(out_path, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


# ADDED: all_raster figure — RGB + 3 GT panels + 3 Pred panels, one row each
def save_figure_all_raster(rgb, gt_by_branch, pred_by_branch, img_name, out_dir):
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), dpi=120)
    fig.patch.set_facecolor('#111111')

    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title('RGB image', color='white', fontsize=10, pad=4)
    axes[1, 0].axis('off')

    for i, (name, _) in enumerate(ALL_RASTER_SUBBRANCHES):
        axes[0, i + 1].imshow(gt_by_branch[name], cmap='gray', vmin=0, vmax=1, interpolation='nearest')
        axes[0, i + 1].set_title(f'GT ({name})', color='white', fontsize=10, pad=4)
        axes[1, i + 1].imshow(pred_by_branch[name], cmap='hot', vmin=0, vmax=1, interpolation='nearest')
        axes[1, i + 1].set_title(f'Pred ({name})', color='white', fontsize=10, pad=4)

    for ax in axes.flat:
        ax.axis('off')

    plt.tight_layout(pad=0.5)
    out_path = os.path.join(out_dir, f'{img_name}.png')
    plt.savefig(out_path, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',     required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output',     default='/home/imane/DATA/PLR-Net_output/inspect_branch_raster')
    parser.add_argument('--n-images',   type=int, default=6)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    cfg.merge_from_file(args.config)
    cfg.freeze()
    device        = cfg.MODEL.DEVICE
    active_branch = cfg.MODEL.ACTIVE_BRANCH
    mean = list(cfg.DATASETS.IMAGE.PIXEL_MEAN)
    std  = list(cfg.DATASETS.IMAGE.PIXEL_STD)
    assert active_branch in GT_SUBDIR_BY_BRANCH or active_branch == 'all_raster', \
        f"inspect_branch_raster.py needs ACTIVE_BRANCH in (region,line,point,point_sce,all_raster), got {active_branch!r}"

    model = BuildingDetector(cfg, test=True).to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    state = ckpt.get('model', ckpt)
    state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval()
    epoch = ckpt.get('epoch', '?')
    print(f'Checkpoint loaded — epoch {epoch}, branch={active_branch}')

    data_nl_patches = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data_nl', 'patches')
    patches_root = os.path.join(data_nl_patches, 'train')  # train/val share this folder
    val_stems_file = os.path.join(data_nl_patches, 'val.txt')

    with open(val_stems_file) as f:
        val_stems = [line.strip() for line in f if line.strip()]

    step = max(1, len(val_stems) // args.n_images)
    selected = val_stems[::step][:args.n_images]

    print(f'\nInspecting {len(selected)} images -> {args.output}\n')

    if active_branch == 'all_raster':
        for stem in selected:
            img_tensor, rgb = load_patch(stem, patches_root, mean, std)
            gt_by_branch   = {name: load_gt(stem, patches_root, subdir)
                              for name, subdir in ALL_RASTER_SUBBRANCHES}
            pred_by_branch = run_branch(model, img_tensor, device, active_branch)

            out_path = save_figure_all_raster(rgb, gt_by_branch, pred_by_branch, stem, args.output)
            print(f'  {stem} -> {out_path}')
    else:
        gt_subdir = GT_SUBDIR_BY_BRANCH[active_branch]
        for stem in selected:
            img_tensor, rgb = load_patch(stem, patches_root, mean, std)
            gt   = load_gt(stem, patches_root, gt_subdir)
            pred = run_branch(model, img_tensor, device, active_branch)

            out_path = save_figure(rgb, gt, pred, stem, active_branch, args.output)
            print(f'  {stem} -> {out_path}')

    print(f'\nDone. Figures saved to: {args.output}')


if __name__ == '__main__':
    main()

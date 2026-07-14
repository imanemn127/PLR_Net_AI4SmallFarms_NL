import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
import time
import argparse
import logging
import random
import numpy as np
import datetime

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import rasterio

from PLRNet.config import cfg
from PLRNet.detector import BuildingDetector
from PLRNet.dataset import build_train_dataset, build_test_dataset
from PLRNet.dataset.build import build_train_dataset_raster
from PLRNet.utils.comm import to_single_device
from PLRNet.solver import make_lr_scheduler, make_optimizer
from PLRNet.utils.logger import setup_logger
from PLRNet.utils.miscellaneous import save_config
from PLRNet.utils.metric_logger import MetricLogger
from PLRNet.utils.metrics.cIoU import calc_IoU
from PLRNet.utils.polygon import get_pred_junctions
from scipy.spatial.distance import cdist

import torch
torch.multiprocessing.set_sharing_strategy('file_system')

RECALL_THRESHOLDS = [3, 5, 8]  # pixels


def compute_junction_recall(gt_polys, pred_juncs, thresholds=RECALL_THRESHOLDS):
    """Fraction of GT polygon corners matched within T pixels by pred_juncs."""
    all_gt = []
    for poly in gt_polys:
        pts = np.array(poly).reshape(-1, 2)
        if len(pts) > 1 and np.allclose(pts[0], pts[-1], atol=0.5):
            pts = pts[:-1]
        all_gt.append(pts)
    if not all_gt:
        return {t: float('nan') for t in thresholds}
    gt_corners = np.vstack(all_gt)
    if len(pred_juncs) == 0:
        return {t: 0.0 for t in thresholds}
    min_dists = cdist(gt_corners, pred_juncs).min(axis=1)
    return {t: float((min_dists <= t).mean()) for t in thresholds}

# ------------------------------------------------------------------ #
#  Constants
# ------------------------------------------------------------------ #
VAL_EVERY = 5   # run validation every N epochs
N_VIZ     = 2   # images to visualize per val/train run


# ------------------------------------------------------------------ #
#  Helpers
# ------------------------------------------------------------------ #
class LossReducer(object):
    def __init__(self, cfg):
        self.loss_weights = dict(cfg.MODEL.LOSS_WEIGHTS)

    def __call__(self, loss_dict):
        return sum(self.loss_weights[k] * loss_dict[k]
                   for k in self.loss_weights)


def parse_args():
    parser = argparse.ArgumentParser(description='Training PLR-Net')
    parser.add_argument("--config-file", metavar="FILE", type=str, default=None)
    parser.add_argument("--clean", default=False, action='store_true')
    parser.add_argument("--val-every", default=VAL_EVERY, type=int,
                        help="Run validation every N epochs (default: 5)")
    parser.add_argument("--seed", default=2, type=int)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    return parser.parse_args()


def set_random_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def init_metrics_csv(csv_path, loss_names):
    fieldnames = (
        ['epoch', 'train_loss', 'train_mask_iou']
        + ['w_' + k for k in loss_names]
        + ['val_loss']
        + ['val_w_' + k for k in loss_names]
        + ['val_mask_iou']
        + [f'val_junc_recall@{t}px' for t in RECALL_THRESHOLDS]
    )
    with open(csv_path, 'w', newline='') as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()
    return fieldnames


def append_metrics_csv(csv_path, fieldnames, row):
    with open(csv_path, 'a', newline='') as f:
        csv.DictWriter(f, fieldnames=fieldnames).writerow(row)


# ------------------------------------------------------------------ #
#  Image utilities
# ------------------------------------------------------------------ #
def tensor_to_display(image_tensor, mean, std):
    """Normalised CxHxW tensor → display-ready uint8 HxWx3 array."""
    img = image_tensor.cpu().numpy().transpose(1, 2, 0)
    img = img * np.array(std) + np.array(mean)
    for c in range(img.shape[2]):
        p2, p98 = np.percentile(img[:, :, c], (2, 98))
        if p98 > p2:
            img[:, :, c] = (img[:, :, c] - p2) / (p98 - p2)
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def draw_polygons(ax, polys, color, label):
    """Draw a list of Nx2 polygon arrays on a matplotlib axis."""
    for poly in polys:
        if len(poly) < 3:
            continue
        pts = np.array(poly)
        closed = np.vstack([pts, pts[0]])
        ax.plot(closed[:, 0], closed[:, 1], '-', color=color, linewidth=1.2)
    return mpatches.Patch(color=color, label=label)


# ------------------------------------------------------------------ #
#  visualization 
# ------------------------------------------------------------------ #
def _render_viz(img_disp, gt_polys, pred_polys, epoch, img_name, viz_dir, split):
    """Save one GT | Pred side-by-side figure.
    Filename : {epoch}_{img_name}.png  (no zero-padding)
    Titles   : GT_{split}_{epoch}_{img_name}  /  PRED_{split}_{epoch}_{img_name}
    """
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=150)
    fig.patch.set_facecolor('black')
    for ax in axes:
        ax.imshow(img_disp)
        ax.axis('off')
    patches = [
        draw_polygons(axes[0], gt_polys,   '#00ff00', 'GT'),
        draw_polygons(axes[1], pred_polys, '#ff6600', 'Pred'),
    ]
    axes[0].set_title(f"GT_{split}_{epoch}_{img_name}", color='white', fontsize=7)
    axes[1].set_title(f"PRED_{split}_{epoch}_{img_name}", color='white', fontsize=7)
    axes[1].legend(handles=patches, loc='upper right', fontsize=6, framealpha=0.5)
    plt.tight_layout(pad=0.3)
    plt.savefig(os.path.join(viz_dir, f"{epoch}_{img_name}.png"),
                bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)


def _pick_dense_sparse_ids(coco_obj, min_anns=10):
    """Return (dense_img_id, sparse_img_id) using only the COCO annotation index.
    Both patches are guaranteed to have >= min_anns annotations (real agricultural content).
    """
    img_ids = sorted(coco_obj.getImgIds())
    counts  = {iid: len(coco_obj.getAnnIds(imgIds=[iid])) for iid in img_ids}
    # keep only patches with enough parcels to be visually meaningful
    valid   = [iid for iid in img_ids if counts[iid] >= min_anns]
    if not valid:
        valid = [iid for iid in img_ids if counts[iid] > 0]
    dense_id  = max(valid, key=counts.get)
    sparse_id = min(valid, key=counts.get)
    return dense_id, sparse_id


def select_viz_indices_val(val_loader):
    """
    Pick dense + sparse viz samples using COCO index only (no full scan).
    Returns list of (images_tensor, b, annotation).
    """
    from torch.utils.data import DataLoader as _DL

    dataset  = val_loader.dataset
    coco_obj = dataset.coco
    dense_id, sparse_id = _pick_dense_sparse_ids(coco_obj)

    id_to_idx = {iid: i for i, iid in enumerate(dataset.ids)}
    selected  = []
    for target_id in [dense_id, sparse_id]:
        idx          = id_to_idx[target_id]
        image, ann   = dataset[idx]           # (tensor, dict)
        images       = image.unsqueeze(0)
        selected.append((images, 0, ann))
    return selected


def select_viz_indices_train(train_dataset_obj, transform):
    """
    Pick dense + sparse viz samples using COCO index only (no full scan).
    Returns list of (images_tensor, b=0, annotation).
    """
    from torch.utils.data import DataLoader as _DL
    from PLRNet.dataset.train_dataset import collate_fn as _train_collate

    coco_obj = train_dataset_obj.coco
    img_ids  = train_dataset_obj.images
    dense_id, sparse_id = _pick_dense_sparse_ids(coco_obj)

    id_to_idx = {iid: i for i, iid in enumerate(img_ids)}
    selected  = []
    for target_id in [dense_id, sparse_id]:
        idx             = id_to_idx[target_id]
        images, anns    = next(iter(
            _DL(train_dataset_obj, batch_size=1, shuffle=False,
                collate_fn=_train_collate, num_workers=0,
                sampler=torch.utils.data.SubsetRandomSampler([idx]))
        ))
        ann             = anns[0]
        ann['filename'] = coco_obj.loadImgs(ids=[target_id])[0]['file_name']
        selected.append((images, 0, ann))
    return selected


# ------------------------------------------------------------------ #
#  Validation visualization  (GT from COCO object)
# ------------------------------------------------------------------ #
@torch.no_grad()
def visualize_val(model, val_viz_entries, coco_obj, epoch, output_dir, mean, std):
    """
    Save the 2 pre-selected val images (dense + sparse GT).
    val_viz_entries is computed once before training by select_viz_indices_val().
    Output: visualizations/val/{epoch:03d}_{full_patch_name}.png
    """
    model.eval()
    device   = next(model.parameters()).device
    viz_dir  = os.path.join(output_dir, 'visualizations', 'val')
    os.makedirs(viz_dir, exist_ok=True)

    for images, b, ann in val_viz_entries:
        img_id   = ann.get('img_id', None)
        img_name = os.path.splitext(os.path.basename(ann.get('filename', 'val')))[0]
        img_disp = tensor_to_display(images[b], mean, std)
        output, _ = model(images.to(device))

        gt_polys = []
        if img_id is not None:
            gt_polys = [np.array(seg).reshape(-1, 2)
                        for a in coco_obj.loadAnns(coco_obj.getAnnIds(imgIds=[img_id]))
                        for seg in a['segmentation']]
        pred_polys = output['polys_pred'][b] if output['polys_pred'] else []
        _render_viz(img_disp, gt_polys, pred_polys, epoch, img_name, viz_dir, split='val')


# ------------------------------------------------------------------ #
#  Train visualization  (GT from mask via cv2.findContours)
# ------------------------------------------------------------------ #
@torch.no_grad()
def visualize_train(model, train_viz_entries, epoch, output_dir, mean, std):
    """
    Save the 2 pre-selected train images (dense + sparse GT).
    train_viz_entries is computed once before training by select_viz_indices_train().
    Output: visualizations/train/{epoch:03d}_{full_patch_name}.png
    """
    model.eval()
    device  = next(model.parameters()).device
    viz_dir = os.path.join(output_dir, 'visualizations', 'train')
    os.makedirs(viz_dir, exist_ok=True)

    for images, b, ann in train_viz_entries:
        img_name = os.path.splitext(
            os.path.basename(ann.get('filename', 'train')))[0]
        img_disp = tensor_to_display(images[b], mean, std)
        output, _ = model(images.to(device))

        mask = ann.get('mask', None)
        gt_polys = []
        if mask is not None:
            m = mask.cpu().numpy() if torch.is_tensor(mask) else np.array(mask)
            contours, _ = cv2.findContours(
                (m * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            gt_polys = [c.reshape(-1, 2) for c in contours if len(c) >= 3]
        pred_polys = output['polys_pred'][b] if output['polys_pred'] else []
        _render_viz(img_disp, gt_polys, pred_polys, epoch, img_name, viz_dir, split='train')

    model.train()


# ------------------------------------------------------------------ #
#  Validation loss pass
# ------------------------------------------------------------------ #
@torch.no_grad()
def validate(model, val_loader, loss_reducer, device, loss_names):
    model.eval()
    sums    = {k: 0.0 for k in loss_names}
    total   = 0.0
    iou_sum = 0.0
    n_loss  = 0
    n_iou   = 0

    # junction recall: we measure this directly from the raw heatmap,
    # no polygon post-processing needed
    recall_sums = {t: 0.0 for t in RECALL_THRESHOLDS}
    n_recall    = 0

    for images, annotations in val_loader:
        images      = images.to(device)
        annotations = to_single_device(annotations, device)

        loss_dict, extras = model.forward_train(images, annotations)

        weighted = loss_reducer(loss_dict)
        for k in loss_names:
            sums[k] += loss_dict[k].item()
        total  += weighted.item()
        n_loss += images.size(0)

        remask  = extras['remask_pred'].sigmoid()
        mask_gt = torch.stack([a['mask'].squeeze() for a in annotations]).to(device)

        for b in range(remask.size(0)):
            pred_bin = (remask[b] > 0.5).cpu().numpy()
            gt_bin   = (mask_gt[b] > 0.5).cpu().numpy()
            if gt_bin.sum() == 0:
                continue
            iou_sum += calc_IoU(pred_bin, gt_bin)
            n_iou   += 1

        # get jloc and joff from the network heads (no polygon building)
        jloc_pred_batch, joff_pred_batch = _get_jloc_joff(model, images)

        for b in range(images.size(0)):
            ann     = annotations[b]
            gt_juncs = ann.get('junctions', None)
            if gt_juncs is None or len(gt_juncs) == 0:
                continue
            gt_pts = gt_juncs.cpu().numpy()  # (N, 2) pixel coords

            # === CHANGED: binary heatmap (1 channel), same map for concave and convex
            jloc_concave = jloc_pred_batch[b:b+1, 0:1]
            jloc_convex  = jloc_pred_batch[b:b+1, 0:1]
            joff         = joff_pred_batch[b]
            pred_juncs   = get_pred_junctions(jloc_concave[0], jloc_convex[0], joff)

            if len(pred_juncs) == 0:
                for t in RECALL_THRESHOLDS:
                    recall_sums[t] += 0.0
            else:
                min_dists = cdist(gt_pts, pred_juncs).min(axis=1)
                for t in RECALL_THRESHOLDS:
                    recall_sums[t] += float((min_dists <= t).mean())
            n_recall += 1

    avg        = {k: sums[k] / max(n_loss, 1) for k in loss_names}
    avg_total  = total   / max(n_loss, 1)
    avg_iou    = iou_sum / max(n_iou,  1)
    avg_recall = {t: recall_sums[t] / max(n_recall, 1) for t in RECALL_THRESHOLDS}
    return avg_total, avg, avg_iou, avg_recall


def _get_jloc_joff(model, images):
    """Run only the junction heads — skip mask branch and polygon generation."""
    outputs, features = model.backbone(images)
    jloc_feature     = model.jloc_head(features)
    afm_feature      = model.afm_head(features)
    jloc_att_feature = model.a2j_att(jloc_feature, jloc_feature + afm_feature)
    # === CHANGED: sigmoid on 1-channel binary heatmap instead of softmax over 3 classes
    jloc_pred        = model.jloc_predictor(jloc_att_feature).sigmoid()   # (B, 1, H, W)
    joff_pred        = outputs[:, :].sigmoid() - 0.5                       # (B, 2, H, W)
    return jloc_pred, joff_pred


# ------------------------------------------------------------------ #
#  Train loop
# ------------------------------------------------------------------ #
def train(cfg, output_dir, val_every):
    logger     = logging.getLogger("training")
    device     = cfg.MODEL.DEVICE
    mean       = list(cfg.DATASETS.IMAGE.PIXEL_MEAN)
    std        = list(cfg.DATASETS.IMAGE.PIXEL_STD)

    model = BuildingDetector(cfg).to(device)

    train_dataset            = build_train_dataset(cfg)
    val_dataset, val_ann_file = build_test_dataset(cfg)

    optimizer    = make_optimizer(cfg, model)
    scheduler    = make_lr_scheduler(cfg, optimizer)
    loss_reducer = LossReducer(cfg)
    loss_weights = dict(cfg.MODEL.LOSS_WEIGHTS)
    loss_names   = list(loss_weights.keys())

    max_epoch  = cfg.SOLVER.MAX_EPOCH
    epoch_size = len(train_dataset)

    checkpoints_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(checkpoints_dir, exist_ok=True)

    best_val_loss = float('inf')

    def save_checkpoint(name, current_epoch):
        state = {
            'epoch'     : current_epoch,
            'model'     : model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'scheduler' : scheduler.state_dict(),
        }
        path = os.path.join(checkpoints_dir, name)
        torch.save(state, path)
        logger.info(f"Checkpoint saved → checkpoints/{name}")

    csv_path   = os.path.join(output_dir, 'metrics.csv')
    fieldnames = init_metrics_csv(csv_path, loss_names)

    # Pre-select visualization samples once before training starts
    # val: scan val_loader (no shuffle) → dense + sparse by COCO polygon count
    # train: scan raw dataset with shuffle=False → dense + sparse by mask contour count
    val_coco_obj      = val_dataset.dataset.coco
    val_viz_entries   = select_viz_indices_val(val_dataset)
    train_viz_entries = select_viz_indices_train(train_dataset.dataset, None)
    logger.info(f"Viz samples selected — val: {len(val_viz_entries)}, "
                f"train: {len(train_viz_entries)}")

    start_time = time.time()
    end        = time.time()

    for epoch in range(1, max_epoch + 1):
        meters = MetricLogger(" ")
        model.train()

        epoch_loss_sums  = {k: 0.0 for k in loss_names}
        epoch_total      = 0.0
        epoch_iou_sum    = 0.0
        epoch_n_images   = 0
        n_batches        = 0

        for it, (images, annotations) in enumerate(train_dataset):
            data_time   = time.time() - end
            images      = images.to(device)
            annotations = to_single_device(annotations, device)

            loss_dict, extras = model(images, annotations)
            total_loss        = loss_reducer(loss_dict)

            with torch.no_grad():
                loss_dict_red = {k: v.item() for k, v in loss_dict.items()}
                loss_red      = total_loss.item()
                meters.update(loss=loss_red, **loss_dict_red)
                for k in loss_names:
                    epoch_loss_sums[k] += loss_dict_red.get(k, 0.0)
                epoch_total += loss_red
                n_batches   += 1

                # accumulate train mask IoU using the remask logit already computed
                remask   = extras['remask_pred'].sigmoid()
                mask_gt  = torch.stack([a['mask'].squeeze() for a in annotations]).to(device)
                for b in range(remask.size(0)):
                    pred_bin = (remask[b] > 0.5).cpu().numpy()
                    gt_bin   = (mask_gt[b] > 0.5).cpu().numpy()
                    epoch_iou_sum  += calc_IoU(pred_bin, gt_bin)
                    epoch_n_images += 1

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            batch_time = time.time() - end
            end = time.time()
            meters.update(time=batch_time, data=data_time)

            if it % 20 == 0 or it + 1 == epoch_size:
                eta_seconds = meters.time.global_avg * (
                    epoch_size * (max_epoch - epoch + 1) - it + 1)
                logger.info(
                    meters.delimiter.join([
                        "eta: {eta}", "epoch: {epoch}", "iter: {iter}",
                        "{meters}", "lr: {lr:.6f}", "max mem: {memory:.0f}\n",
                    ]).format(
                        eta=str(datetime.timedelta(seconds=int(eta_seconds))),
                        epoch=epoch, iter=it, meters=str(meters),
                        lr=optimizer.param_groups[0]["lr"],
                        memory=torch.cuda.max_memory_allocated() / 1024**2,
                    )
                )

        scheduler.step()

        avg_losses     = {k: epoch_loss_sums[k] / max(n_batches, 1) for k in loss_names}
        avg_total      = epoch_total / max(n_batches, 1)
        avg_train_iou  = epoch_iou_sum / max(epoch_n_images, 1)
        current_lr     = optimizer.param_groups[0]["lr"]

        # always overwrite latest
        save_checkpoint('latest.pth', epoch)

        # --- validation + visualizations every val_every epochs ---
        val_total       = float('nan')
        val_losses      = {k: float('nan') for k in loss_names}
        val_iou         = float('nan')
        val_junc_recall = {t: float('nan') for t in RECALL_THRESHOLDS}

        if epoch % val_every == 0:
            logger.info(f"=== Validation at epoch {epoch} ===")
            val_total, val_losses, val_iou, val_junc_recall = validate(
                model, val_dataset, loss_reducer, device, loss_names)
            logger.info(
                "Val total_loss: {:.4f}  |  {}".format(
                    val_total,
                    "  ".join(f"{k}: {v:.4f}" for k, v in val_losses.items())
                )
            )
            logger.info(f"Val mask_iou: {val_iou:.4f}")
            # junction recall: fraction of GT corners matched within T pixels
            # measured on raw jloc heatmap, no polygon post-processing
            recall_str = "  ".join(
                f"R@{t}px={val_junc_recall[t]:.4f}" for t in RECALL_THRESHOLDS
            )
            logger.info(f"Val junction_recall — {recall_str}")
            # checkpoint at this val epoch
            save_checkpoint(f'epoch_{epoch}.pth', epoch)

            # best val loss
            if val_total < best_val_loss:
                best_val_loss = val_total
                save_checkpoint('best_val_loss.pth', epoch)
                logger.info(f"New best val loss: {best_val_loss:.4f}")

            # visualizations — same 2 images every epoch (dense + sparse GT)
            visualize_val(model, val_viz_entries, val_coco_obj, epoch, output_dir, mean, std)
            visualize_train(model, train_viz_entries, epoch, output_dir, mean, std)

        # --- write metrics.csv ---
        row = {'epoch': epoch, 'train_loss': round(avg_total, 6),
               'train_mask_iou': round(avg_train_iou, 6)}
        for k in loss_names:
            row['w_' + k] = round(avg_losses[k] * loss_weights[k], 6)
        row['val_loss'] = round(val_total, 6) if not np.isnan(val_total) else ''
        for k in loss_names:
            row['val_w_' + k] = (round(val_losses[k] * loss_weights[k], 6)
                                 if not np.isnan(val_losses[k]) else '')
        row['val_mask_iou'] = round(val_iou, 6) if epoch % val_every == 0 else ''
        for t in RECALL_THRESHOLDS:
            v = val_junc_recall[t]
            row[f'val_junc_recall@{t}px'] = round(v, 6) if (epoch % val_every == 0 and not np.isnan(v)) else ''
        append_metrics_csv(csv_path, fieldnames, row)

        logger.info(
            "Epoch {:03d} | train_loss: {:.4f} | val_loss: {} | lr: {:.6f}".format(
                epoch, avg_total,
                f"{val_total:.4f}" if not np.isnan(val_total) else "-",
                current_lr,
            )
        )

    total_time = time.time() - start_time
    logger.info("Total training time: {} ({:.4f} s / epoch)".format(
        str(datetime.timedelta(seconds=int(total_time))),
        total_time / max_epoch,
    ))


# ------------------------------------------------------------------ #
# ADDED: isolated single-branch training on GT rasters (region/line/point)
# instead of COCO. Separate from train() to avoid entangling the COCO-only
# visualization / junction-recall code with the raster path.
# ------------------------------------------------------------------ #
GT_KEY_BY_BRANCH = {
    'region':    'gt_region',
    'line':      'gt_lines',
    'point':     'gt_nodes',
    'point_sce': 'gt_nodes',
}


@torch.no_grad()
def validate_raster(model, val_loader, loss_reducer, device, loss_names, active_branch):
    """Simple deterministic validation for one isolated branch: mean loss
    + a pixel metric (IoU for region, precision/recall for line/point)."""
    model.eval()
    sums   = {k: 0.0 for k in loss_names}
    total  = 0.0
    n_loss = 0

    gt_key   = GT_KEY_BY_BRANCH[active_branch]
    iou_sum  = 0.0
    tp = fp = fn = 0
    n_iou = 0

    for images, annotations in val_loader:
        images      = images.to(device)
        annotations = to_single_device(annotations, device)

        loss_dict, extras = model.forward_train(images, annotations)
        weighted = loss_reducer(loss_dict)
        for k in loss_names:
            sums[k] += loss_dict[k].item()
        total  += weighted.item()
        n_loss += images.size(0)

        gt_batch = torch.stack([a[gt_key].squeeze() for a in annotations]).to(device)
        # ---------------------------------------------------------------
        # ADDED: branch_pred is a raw logit for region/point/point_sce
        # (needs sigmoid), but for line it is already a [0,1] boundary
        # likeness map (1/(1+afm_norm), see detector.py) — applying
        # sigmoid again would flatten it and break precision/recall.
        # ---------------------------------------------------------------
        if active_branch == 'line':
            pred_prob = extras['branch_pred']
        else:
            pred_prob = extras['branch_pred'].sigmoid()

        if active_branch == 'region':
            for b in range(pred_prob.size(0)):
                pred_bin = (pred_prob[b] > 0.5).cpu().numpy()
                gt_bin   = (gt_batch[b] > 0.5).cpu().numpy()
                if gt_bin.sum() == 0:
                    continue
                iou_sum += calc_IoU(pred_bin, gt_bin)
                n_iou   += 1
        else:
            for b in range(pred_prob.size(0)):
                pred_bin = (pred_prob[b] > 0.5).cpu().numpy()
                gt_bin   = (gt_batch[b] > 0.5).cpu().numpy()
                tp += int(np.logical_and(pred_bin, gt_bin).sum())
                fp += int(np.logical_and(pred_bin, ~gt_bin).sum())
                fn += int(np.logical_and(~pred_bin, gt_bin).sum())

    avg       = {k: sums[k] / max(n_loss, 1) for k in loss_names}
    avg_total = total / max(n_loss, 1)

    metrics = {}
    if active_branch == 'region':
        metrics['iou'] = iou_sum / max(n_iou, 1)
    else:
        metrics['precision'] = tp / max(tp + fp, 1)
        metrics['recall']    = tp / max(tp + fn, 1)

    return avg_total, avg, metrics


def train_raster(cfg, output_dir, val_every):
    """Train a single isolated branch (region/line/point/point_sce) on GT
    rasters produced by data_nl/build_gt_rasters_from_brp.py, no COCO involved."""
    logger = logging.getLogger("training")
    device = cfg.MODEL.DEVICE
    active_branch = cfg.MODEL.ACTIVE_BRANCH
    assert active_branch in ('region', 'line', 'point', 'point_sce'), \
        f"train_raster requires ACTIVE_BRANCH in (region,line,point,point_sce), got {active_branch!r}"

    model = BuildingDetector(cfg).to(device)

    # ADDED: point_sce loads the afm_head weights from an already-trained
    # line-branch checkpoint and freezes them, so the point branch gets SCE
    # guidance from a working boundary feature instead of a random one.
    if active_branch == 'point_sce':
        line_ckpt = cfg.MODEL.LINE_CHECKPOINT
        assert line_ckpt, "ACTIVE_BRANCH=point_sce requires MODEL.LINE_CHECKPOINT to be set"
        model.load_and_freeze_afm_head(line_ckpt, device)
        logger.info(f"Loaded and froze afm_head from {line_ckpt}")

    data_nl_patches = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data_nl', 'patches')
    train_root  = os.path.join(data_nl_patches, 'train')
    train_stems = os.path.join(data_nl_patches, 'train.txt')
    val_stems   = os.path.join(data_nl_patches, 'val.txt')

    train_loader = build_train_dataset_raster(cfg, train_root, stems_file=train_stems, is_val=False)
    val_loader   = build_train_dataset_raster(cfg, train_root, stems_file=val_stems,   is_val=True)

    optimizer    = make_optimizer(cfg, model)
    scheduler    = make_lr_scheduler(cfg, optimizer)
    loss_reducer = LossReducer(cfg)
    loss_weights = dict(cfg.MODEL.LOSS_WEIGHTS)
    loss_names   = list(loss_weights.keys())

    max_epoch = cfg.SOLVER.MAX_EPOCH

    checkpoints_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(checkpoints_dir, exist_ok=True)

    best_val_loss = float('inf')

    def save_checkpoint(name, current_epoch):
        state = {
            'epoch'    : current_epoch,
            'model'    : model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }
        path = os.path.join(checkpoints_dir, name)
        torch.save(state, path)
        logger.info(f"Checkpoint saved -> checkpoints/{name}")

    csv_path = os.path.join(output_dir, 'metrics.csv')
    metric_name = 'iou' if active_branch == 'region' else 'precision,recall'
    fieldnames = (['epoch', 'train_loss'] + ['w_' + k for k in loss_names]
                  + ['val_loss'] + ['val_w_' + k for k in loss_names]
                  + [f'val_{m}' for m in (['iou'] if active_branch == 'region'
                                           else ['precision', 'recall'])])
    with open(csv_path, 'w', newline='') as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    start_time = time.time()
    end = time.time()

    for epoch in range(1, max_epoch + 1):
        meters = MetricLogger(" ")
        model.train()

        epoch_loss_sums = {k: 0.0 for k in loss_names}
        epoch_total     = 0.0
        n_batches       = 0

        for it, (images, annotations) in enumerate(train_loader):
            data_time   = time.time() - end
            images      = images.to(device)
            annotations = to_single_device(annotations, device)

            loss_dict, extras = model(images, annotations)
            total_loss        = loss_reducer(loss_dict)

            with torch.no_grad():
                loss_dict_red = {k: v.item() for k, v in loss_dict.items()}
                loss_red      = total_loss.item()
                meters.update(loss=loss_red, **loss_dict_red)
                for k in loss_names:
                    epoch_loss_sums[k] += loss_dict_red.get(k, 0.0)
                epoch_total += loss_red
                n_batches   += 1

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            batch_time = time.time() - end
            end = time.time()
            meters.update(time=batch_time, data=data_time)

            if it % 20 == 0:
                logger.info(
                    meters.delimiter.join([
                        "epoch: {epoch}", "iter: {iter}", "{meters}",
                        "lr: {lr:.6f}",
                    ]).format(
                        epoch=epoch, iter=it, meters=str(meters),
                        lr=optimizer.param_groups[0]["lr"],
                    )
                )

        scheduler.step()

        avg_total  = epoch_total / max(n_batches, 1)
        avg_losses = {k: epoch_loss_sums[k] / max(n_batches, 1) for k in loss_names}

        save_checkpoint('latest.pth', epoch)

        val_total  = float('nan')
        val_losses = {k: float('nan') for k in loss_names}
        val_metrics = {}

        if epoch % val_every == 0:
            logger.info(f"=== Validation at epoch {epoch} (branch={active_branch}) ===")
            val_total, val_losses, val_metrics = validate_raster(
                model, val_loader, loss_reducer, device, loss_names, active_branch)
            logger.info(
                "Val total_loss: {:.4f}  |  {}".format(
                    val_total,
                    "  ".join(f"{k}: {v:.4f}" for k, v in val_losses.items())
                )
            )
            logger.info(f"Val metrics ({active_branch}): " +
                        "  ".join(f"{k}={v:.4f}" for k, v in val_metrics.items()))
            save_checkpoint(f'epoch_{epoch}.pth', epoch)

            if val_total < best_val_loss:
                best_val_loss = val_total
                save_checkpoint('best_val_loss.pth', epoch)
                logger.info(f"New best val loss: {best_val_loss:.4f}")

        row = {'epoch': epoch, 'train_loss': round(avg_total, 6)}
        for k in loss_names:
            row['w_' + k] = round(avg_losses[k] * loss_weights[k], 6)
        row['val_loss'] = round(val_total, 6) if not np.isnan(val_total) else ''
        for k in loss_names:
            row['val_w_' + k] = (round(val_losses[k] * loss_weights[k], 6)
                                 if not np.isnan(val_losses[k]) else '')
        for m in (['iou'] if active_branch == 'region' else ['precision', 'recall']):
            row[f'val_{m}'] = round(val_metrics[m], 6) if val_metrics else ''
        with open(csv_path, 'a', newline='') as f:
            csv.DictWriter(f, fieldnames=fieldnames).writerow(row)

        logger.info(
            "Epoch {:03d} | train_loss: {:.4f} | val_loss: {} | lr: {:.6f}".format(
                epoch, avg_total,
                f"{val_total:.4f}" if not np.isnan(val_total) else "-",
                optimizer.param_groups[0]["lr"],
            )
        )

    total_time = time.time() - start_time
    logger.info("Total training time: {} ({:.4f} s / epoch)".format(
        str(datetime.timedelta(seconds=int(total_time))),
        total_time / max_epoch,
    ))


# ------------------------------------------------------------------ #
#  Entry point
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    args = parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    timestamp  = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(cfg.OUTPUT_DIR, timestamp)

    if os.path.isdir(cfg.OUTPUT_DIR) and args.clean:
        import shutil
        shutil.rmtree(cfg.OUTPUT_DIR)

    os.makedirs(output_dir, exist_ok=True)

    logger = setup_logger('training', output_dir, out_file='train.log')
    logger.info(args)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Validation every {args.val_every} epochs, "
                f"visualizing {N_VIZ} train + {N_VIZ} val images")

    with open(args.config_file, "r") as cf:
        logger.info("\n" + cf.read())
    logger.info("Running with config:\n{}".format(cfg))

    save_config(cfg, os.path.join(output_dir, 'config.yml'))
    set_random_seed(args.seed, True)

    # -------------------------------------------------------------------
    # ADDED: dispatch to the raster-GT isolated-branch trainer when
    # ACTIVE_BRANCH names a single branch. "all" keeps the original
    # COCO-based full-model training path unchanged.
    # -------------------------------------------------------------------
    if cfg.MODEL.ACTIVE_BRANCH == 'all':
        train(cfg, output_dir, val_every=args.val_every)
    else:
        train_raster(cfg, output_dir, val_every=args.val_every)

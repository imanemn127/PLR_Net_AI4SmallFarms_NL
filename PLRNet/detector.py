import cv2
import torch
import torch.nn.functional as F

from math import log
from torch import nn
from PLRNet.backbones import build_backbone
from PLRNet.utils.polygon import generate_polygon
from PLRNet.utils.polygon import get_pred_junctions
from skimage.measure import label, regionprops


# -------------------------------------------------------------------
# ADDED: afm_op stores -sign(a)*log(|a|/size + 1e-6), not a raw pixel
# offset (see csrc/lib/afm_op/cuda/afm.cu). This inverts it back to
# a pixel displacement (ax, ay).
# -------------------------------------------------------------------
def afm_to_pixel_offset(afm_pred, height, width):
    dx_log, dy_log = afm_pred[:, 0], afm_pred[:, 1]
    ax = -torch.sign(dx_log) * width  * torch.exp(-dx_log.abs())
    ay = -torch.sign(dy_log) * height * torch.exp(-dy_log.abs())
    return ax, ay


class RSCSEModule(nn.Module):
    def __init__(self, in_channels=32, reduction=4):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x, x1):
        return x + x * self.cSE(x1) + x * self.sSE(x1)


class RAMAttention(nn.Module):

    def __init__(self, dim_in=2, dim_hid=16, dim_out=32, reduction=4):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(dim_in, dim_hid, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim_hid),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_hid, dim_hid, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim_hid),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_hid, dim_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim_out, dim_out // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(dim_out // reduction, dim_out, bias=False),
            nn.Sigmoid()
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.layer(x)
        b, c, _, _ = x1.size()
        y = self.avg_pool(x1).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x1 * y.expand_as(x1)



class BuildingDetector(nn.Module):
    def __init__(self, cfg, test=False):
        super(BuildingDetector, self).__init__()

        self.backbone = build_backbone(cfg)
        self.backbone_name = cfg.MODEL.NAME

        # === CHANGED: BCE binary loss instead of CrossEntropy 3-class
        # The article uses BCE on a binary heatmap (0=background, 1=any corner).
        # pos_weight upweights corner pixels to handle the ~99.8% background imbalance.
        self.junc_loss = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([cfg.MODEL.JLOC_POS_WEIGHT])
        )

        self.test_inria = 'inria' in cfg.DATASETS.TEST[0]
        # self.test_inria = 'inria' not in cfg.DATASETS.TEST[0]
        if not test:
            from PLRNet.encoder import Encoder
            self.encoder = Encoder(cfg)

        self.pred_height = cfg.DATASETS.TARGET.HEIGHT
        self.pred_width = cfg.DATASETS.TARGET.WIDTH
        self.origin_height = cfg.DATASETS.ORIGIN.HEIGHT
        self.origin_width = cfg.DATASETS.ORIGIN.WIDTH

        dim_in = cfg.MODEL.OUT_FEATURE_CHANNELS
        self.mask_head = self._make_conv(dim_in, dim_in, dim_in)
        self.jloc_head = self._make_conv(dim_in, dim_in, dim_in)
        self.afm_head = self._make_conv(dim_in, dim_in, dim_in)

        # self.a2m_att = ECA(dim_in)
        # self.a2j_att = ECA(dim_in)
        self.a2m_att = RSCSEModule(dim_in, 4)
        self.a2j_att = RSCSEModule(dim_in, 4)

        self.mask_predictor = self._make_predictor(dim_in, 2)
        # === CHANGED: 1 output channel (binary heatmap) instead of 3 (3-class)
        self.jloc_predictor = self._make_predictor(dim_in, 1)
        self.afm_predictor = self._make_predictor(dim_in, 2)

        # === ADDED: dedicated 1-channel binary head for isolated line-branch
        # training against a thin boundary raster (BCE, same idea as jloc).
        # Not used in "all"/vector-AFM mode.
        self.line_predictor = self._make_predictor(dim_in, 1)

        self.refuse_conv = RAMAttention(2, dim_in // 2, dim_in, 4)
        # self.final_conv = self._make_conv(dim_in*2, dim_in, 2)
        self.final_conv = self._make_conv(dim_in, dim_in // 2, 2)

        self.train_step = 0

        # which branch to train in isolation: "all", "region", "line", "point", "point_sce"
        self.active_branch = getattr(cfg.MODEL, 'ACTIVE_BRANCH', 'all')

    # -------------------------------------------------------------------
    # ADDED: for "point_sce" — load afm_head weights from an already
    # trained line-branch checkpoint and freeze them, so the point branch
    # gets SCE guidance from a fixed, working boundary feature instead of
    # a randomly-initialized one.
    # -------------------------------------------------------------------
    def load_and_freeze_afm_head(self, line_checkpoint_path, device):
        ckpt = torch.load(line_checkpoint_path, map_location=device)
        state = ckpt.get('model', ckpt)
        afm_head_state = {
            k[len('afm_head.'):]: v for k, v in state.items() if k.startswith('afm_head.')
        }
        self.afm_head.load_state_dict(afm_head_state)
        for p in self.afm_head.parameters():
            p.requires_grad = False
        self._afm_head_frozen = True
        self.afm_head.eval()

    def train(self, mode=True):
        # keep afm_head in eval() (frozen BatchNorm stats) even when the
        # rest of the model is switched to train() every epoch
        super().train(mode)
        if getattr(self, '_afm_head_frozen', False):
            self.afm_head.eval()
        return self

    def forward(self, images, annotations=None):
        if self.training:
            return self.forward_train(images, annotations)
        return self.forward_test(images, annotations=annotations)

    def forward_train(self, images, annotations):
        device = images.device

        # ---------------------------------------------------------------
        # ADDED: raster GT path. If the dataset already gives per-branch
        # GT rasters (RasterGTDataset), skip the COCO Encoder entirely and
        # build targets straight from the rasters.
        # ---------------------------------------------------------------
        is_raster_mode = 'gt_region' in annotations[0]
        if is_raster_mode:
            targets = self._targets_from_rasters(annotations, device)
        else:
            targets, metas = self.encoder(annotations)
            targets = {k: v.to(device) for k, v in targets.items()}

        outputs, features = self.backbone(images)

        mask_feature = self.mask_head(features)
        jloc_feature = self.jloc_head(features)
        afm_feature  = self.afm_head(features)

        # === ISOLATED BRANCH MODE ===
        # When ACTIVE_BRANCH != "all", SCE cross-attention is disabled so each
        # branch learns only from backbone features with no help from other branches.
        # "all"      → original full model with SCE (default)
        # "region"   → mask head only, no AFM guidance
        # "line"     → afm head only, no cross-branch
        # "point"    → jloc head only, no AFM guidance
        # "point_sce"→ jloc head with SCE guidance from a frozen, pre-trained
        #              afm_head (see train_raster / freeze_afm_head)
        if self.active_branch == 'all' or self.active_branch == 'point_sce':
            mask_att_feature = self.a2m_att(mask_feature, mask_feature + afm_feature)
            jloc_att_feature = self.a2j_att(jloc_feature, jloc_feature + afm_feature)
        else:
            mask_att_feature = mask_feature
            jloc_att_feature = jloc_feature

        mask_pred   = self.mask_predictor(mask_att_feature)  # (B,2,H,W) logits
        jloc_pred   = self.jloc_predictor(jloc_att_feature) # (B,1,H,W) logits
        afm_pred    = self.afm_predictor(afm_feature)        # (B,2,H,W)
        afm_conv    = self.refuse_conv(afm_pred)
        remask_pred = self.final_conv(features + afm_conv)   # (B,2,H,W) logits
        joff_pred   = outputs[:, :].sigmoid() - 0.5         # (B,2,H,W) in [-0.5, 0.5]

        # targets — enforce dtypes expected by each loss
        jloc_gt  = targets['jloc'].squeeze(1).float()        # (B,H,W) float32 binary {0,1}
        joff_gt  = targets['joff']                           # (B,2,H,W) float32
        mask_gt  = targets['mask'].squeeze(1).float()        # (B,H,W) float32 in [0,1]
        afmap_gt = targets['afmap']                          # (B,2,H,W) float32

        zero = torch.tensor(0.0, device=device)

        # point branch losses
        if self.active_branch in ('point', 'point_sce', 'all'):
            self.junc_loss.pos_weight = self.junc_loss.pos_weight.to(device)
            loss_jloc = self.junc_loss(jloc_pred.squeeze(1), jloc_gt)
            junc_mask = jloc_gt.unsqueeze(1)                    # (B,1,H,W) already binary float
            loss_joff = F.l1_loss(joff_pred * junc_mask,
                                  joff_gt  * junc_mask,
                                  reduction='sum') / (junc_mask.sum() + 1e-6)
        else:
            loss_jloc = zero
            loss_joff = zero

        # region branch losses
        # ADDED: remask_pred is computed via afm_predictor -> refuse_conv ->
        # final_conv, so it is never independent of the line branch weights.
        # In isolated "region" mode only loss_mask (mask_predictor, a fully
        # dedicated head) is used; loss_remask is skipped so no gradient
        # leaks into afm_predictor/refuse_conv through the region branch.
        if self.active_branch == 'all':
            loss_mask   = F.binary_cross_entropy_with_logits(mask_pred[:, 1], mask_gt)
            loss_remask = F.binary_cross_entropy_with_logits(remask_pred[:, 1], mask_gt)
        elif self.active_branch == 'region':
            loss_mask   = F.binary_cross_entropy_with_logits(mask_pred[:, 1], mask_gt)
            loss_remask = zero
        else:
            loss_mask   = zero
            loss_remask = zero

        # line branch loss
        # ADDED: isolated raster mode uses afm_predictor (2 channels, dx/dy
        # to nearest contour pixel) trained with MAE against the distance
        # transform of gt_lines. Smoother gradient than a direct binary
        # target, same head/loss shape as "all" mode.
        if self.active_branch in ('line', 'all'):
            loss_afm = F.l1_loss(afm_pred, afmap_gt)
        else:
            loss_afm = zero

        loss_dict = {
            'loss_jloc'  : loss_jloc,
            'loss_joff'  : loss_joff,
            'loss_mask'  : loss_mask,
            'loss_afm'   : loss_afm,
            'loss_remask': loss_remask,
        }

        extras = {'remask_pred': remask_pred[:, 1].detach()}  # expose refined mask logit for validation
        # -----------------------------------------------------------------
        # ADDED: expose a single (B,H,W) map for raster-mode validation.
        # For the line branch, afm_pred must first be decoded back to a
        # pixel offset (afm_to_pixel_offset) before its norm means anything
        # in pixels; 1/(1+norm_px) then gives a [0,1] boundary-likeness map.
        # -----------------------------------------------------------------
        if is_raster_mode:
            if self.active_branch in ('point', 'point_sce'):
                extras['branch_pred'] = jloc_pred.squeeze(1).detach()
            elif self.active_branch == 'line':
                ax, ay = afm_to_pixel_offset(afm_pred, self.pred_height, self.pred_width)
                afm_norm_px = torch.sqrt(ax ** 2 + ay ** 2 + 1e-6)
                extras['branch_pred'] = (1.0 / (1.0 + afm_norm_px)).detach()
            elif self.active_branch == 'region':
                # use mask_pred (isolated head), not remask_pred (goes through afm_predictor)
                extras['branch_pred'] = mask_pred[:, 1].detach()

        return loss_dict, extras

    # -------------------------------------------------------------------
    # ADDED: builds the targets dict from GT rasters (RasterGTDataset)
    # instead of COCO polygons, so isolated branches skip Encoder entirely.
    # -------------------------------------------------------------------
    def _targets_from_rasters(self, annotations, device):
        mask_gt  = torch.stack([torch.as_tensor(a['gt_region']) for a in annotations]).float()
        lines_gt = torch.stack([torch.as_tensor(a['gt_lines'])  for a in annotations]).float()
        jloc_gt  = torch.stack([torch.as_tensor(a['gt_nodes'])  for a in annotations]).float()

        b, h, w = mask_gt.shape
        joff_gt = torch.zeros((b, 2, h, w), dtype=torch.float32)

        # -----------------------------------------------------------------
        # ADDED: gt_afm is precomputed by build_gt_rasters_from_brp.py
        # using the real afm_op CUDA operator on vector BRP edges (not a
        # raster approximation).
        # -----------------------------------------------------------------
        if 'gt_afm' in annotations[0]:
            afmap_gt = torch.stack([torch.as_tensor(a['gt_afm']) for a in annotations]).float()
        else:
            afmap_gt = torch.zeros((b, 2, h, w), dtype=torch.float32)

        targets = {
            'jloc':     jloc_gt.unsqueeze(1),
            'joff':     joff_gt,
            'mask':     mask_gt.unsqueeze(1),
            'afmap':    afmap_gt,
            'lines_gt': lines_gt,
        }
        return {k: v.to(device) for k, v in targets.items()}

    def forward_test(self, images, annotations=None):
        device = images.device
        outputs, features = self.backbone(images)

        mask_feature = self.mask_head(features)
        jloc_feature = self.jloc_head(features)
        afm_feature = self.afm_head(features)

        # mask_att_feature = self.a2m_att(afm_feature, mask_feature)
        # jloc_att_feature = self.a2j_att(afm_feature, jloc_feature)
        mask_att_feature = self.a2m_att(mask_feature, mask_feature + afm_feature)
        jloc_att_feature = self.a2j_att(jloc_feature, jloc_feature + afm_feature)

        # mask_pred = self.mask_predictor(mask_feature + mask_att_feature)
        # jloc_pred = self.jloc_predictor(jloc_feature + jloc_att_feature)
        mask_pred = self.mask_predictor(mask_att_feature)
        jloc_pred = self.jloc_predictor(jloc_att_feature)
        afm_pred = self.afm_predictor(afm_feature)

        afm_conv = self.refuse_conv(afm_pred)
        # remask_pred = self.final_conv(torch.cat((features, afm_conv), dim=1))
        remask_pred = self.final_conv(features + afm_conv)

        joff_pred = outputs[:, :].sigmoid() - 0.5

        mask_pred = mask_pred.softmax(1)[:, 1:]

        # mask_pred = mask_pred.softmax(1)

        # === CHANGED: 1-channel binary heatmap, sigmoid instead of softmax over 3 classes
        # Both concave and convex pred point to the same sigmoid map (no class distinction in loss)
        jloc_prob = jloc_pred.sigmoid()          # (B,1,H,W)
        jloc_convex_pred  = jloc_prob            # (B,1,H,W)
        jloc_concave_pred = jloc_prob            # (B,1,H,W)

        # remask_pred = mask_pred
        remask_pred = remask_pred.softmax(1)[:, 1:]

        scale_y = self.origin_height / self.pred_height
        scale_x = self.origin_width / self.pred_width

        batch_polygons = []
        batch_masks = []
        batch_scores = []
        batch_juncs = []

        for b in range(remask_pred.size(0)):

            mask_pred_per_im = cv2.resize(remask_pred[b][0].cpu().numpy(), (self.origin_width, self.origin_height))

            juncs_pred = get_pred_junctions(jloc_concave_pred[b], jloc_convex_pred[b], joff_pred[b])

            juncs_pred[:, 0] *= scale_x
            juncs_pred[:, 1] *= scale_y

            if not self.test_inria:
                polys, scores = [], []


                props = regionprops(label(mask_pred_per_im > 0.5))
                for prop in props:


                    poly, juncs_sa, edges_sa, score, juncs_index = generate_polygon(prop, mask_pred_per_im, \
                                                                                    juncs_pred, 0, self.test_inria)
                    if juncs_sa.shape[0] == 0:
                        continue

                    polys.append(poly) 
                    scores.append(score)  
                batch_scores.append(scores)  
                batch_polygons.append(polys)

            batch_masks.append(mask_pred_per_im)  
            batch_juncs.append(juncs_pred)

        extra_info = {}

        output = {
            'polys_pred': batch_polygons,
            'mask_pred': batch_masks,
            'scores': batch_scores,
            'juncs_pred': batch_juncs
        }
        return output, extra_info

        # return output, mask

    def _make_conv(self, dim_in, dim_hid, dim_out):
        layer = nn.Sequential(
            nn.Conv2d(dim_in, dim_hid, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim_hid),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_hid, dim_hid, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim_hid),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_hid, dim_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),  # added: reduce overfitting on small dataset
        )
        return layer


    def _make_predictor(self, dim_in, dim_out):
        m = int(dim_in / 4)
        layer = nn.Sequential(
            nn.Conv2d(dim_in, m, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(m, dim_out, kernel_size=1),
        )
        return layer



# MODIFIED (AI4SmallFarms adaptation): orphan block from original repo commented
# out — incorrect indentation caused an IndentationError at import time.
#    model = BuildingDetector(cfg, test=True)
#    if pretrained:
#        url = PRETRAINED[dataset]
#        state_dict = torch.hub.load_state_dict_from_url(url, map_location=device, progress=True)
#        state_dict = {k[7:]: v for k, v in state_dict['model'].items() if k[0:7] == 'module.'}
#        model.load_state_dict(state_dict)
#        model = model.eval()
#        return model
#    return model

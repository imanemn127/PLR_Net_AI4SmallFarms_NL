import cv2
import os
import random
import os.path as osp
import numpy as np

from skimage import io
import rasterio
from pycocotools.coco import COCO
from shapely.geometry import Polygon
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate


def affine_transform(pt, t):

    new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]

class TrainDataset(Dataset):
    def __init__(self, root, ann_file, transform=None, rotate_f=None):
        self.root = root

        self.coco = COCO(ann_file)
        images_id = self.coco.getImgIds()
        #[0,1,2,3,.......]
        self.images=images_id.copy()
        self.num_samples = len(self.images)

        self.transform = transform
        self.rotate_f = rotate_f

    def __getitem__(self, idx_):
        
        
        img_id = self.images[idx_]
        img_info = self.coco.loadImgs(ids=[img_id])[0]
        file_name = img_info['file_name']
        width = img_info['width']
        height = img_info['height']


        ann_ids = self.coco.getAnnIds(imgIds=[img_id])

        ann_coco = self.coco.loadAnns(ids=ann_ids)
        
        ann = {
            'junctions': [],
            'juncs_index': [],
            'juncs_tag': [],
            'edges_positive': [],
            'bbox': [],
            'width': width,
            'height': height,
        }

        pid = 0
        instance_id = 0
        seg_mask = np.zeros([width, height])
 
        for ann_per_ins in ann_coco:
            juncs, tags = [], []

            segmentations = ann_per_ins['segmentation']

            for i, segm in enumerate(segmentations):

                segm = np.array(segm).reshape(-1, 2) 

                segm[:, 0] = np.clip(segm[:, 0], 0, width - 1e-4)
                segm[:, 1] = np.clip(segm[:, 1], 0, height - 1e-4)

                points = segm[:-1]

                junc_tags = np.ones(points.shape[0])
                if i == 0:

                    poly = Polygon(points)
                    if poly.area > 0:

                        convex_point = np.array(poly.convex_hull.exterior.coords)
                        convex_index = [(p == convex_point).all(1).any() for p in points]
                        juncs.extend(points.tolist())
                        junc_tags[convex_index] = 2    
                        tags.extend(junc_tags.tolist())
                        ann['bbox'].append(list(poly.bounds))
                        seg_mask += self.coco.annToMask(ann_per_ins)
                else:
                    juncs.extend(points.tolist())
                    tags.extend(junc_tags.tolist())
                    interior_contour = segm.reshape(-1, 1, 2)
                    cv2.drawContours(seg_mask, [np.int0(interior_contour)], -1, color=0, thickness=-1)
            idxs = np.arange(len(juncs))
            '''
            >> x = np.arange(10) 
            array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
            >> np.roll(x, 2)  
            array([8, 9, 0, 1, 2, 3, 4, 5, 6, 7])
            '''

            edges = np.stack((idxs, np.roll(idxs, 1))).transpose(1,0) + pid
            ann['juncs_index'].extend([instance_id] * len(juncs))
            ann['junctions'].extend(juncs)
            ann['juncs_tag'].extend(tags)
            ann['edges_positive'].extend(edges.tolist())
            if len(juncs) > 0:
                instance_id += 1
                pid += len(juncs)
        seg_mask = np.clip(seg_mask, 0, 1)


        img_path = osp.join(self.root, file_name)
        if img_path.lower().endswith(('.tif', '.tiff')):
            with rasterio.open(img_path) as src:
                # bands are 1-indexed in rasterio; read first 3 (R,G,B)
                image = src.read([1, 2, 3]).transpose(1, 2, 0).astype(np.float32) / 10000.0
            np.nan_to_num(image, nan=0.0, copy=False)  # replace mosaic NoData NaNs with 0
        else:
            image = io.imread(img_path).astype(np.float32)[:, :, :3]
            if image.max() > 1.0:
                image /= 255.0
        for key, _type in (['junctions', np.float32],
                           ['edges_positive', np.int64],
                           ['juncs_tag', np.int64],
                           ['juncs_index', np.int64],
                           ['bbox', np.float32],
                           ):

            ann[key] = np.array(ann[key], dtype=_type)


        if self.rotate_f:
            reminder = random.randint(0, 5)
        else:
            reminder = random.randint(0, 3)
        ann['reminder'] = reminder

        if len(ann['junctions']) > 0:
            if reminder == 1:  

                image = image[:, ::-1, :]

                ann['junctions'][:, 0] = width - ann['junctions'][:, 0]

                ann['bbox'] = ann['bbox'][:, [2, 1, 0, 3]]
                ann['bbox'][:, 0] = width - ann['bbox'][:, 0]
                ann['bbox'][:, 2] = width - ann['bbox'][:, 2]

                seg_mask = np.fliplr(seg_mask)

            elif reminder == 2: 
                image = image[::-1, :, :]
                ann['junctions'][:, 1] = height - ann['junctions'][:, 1]
                ann['bbox'] = ann['bbox'][:, [0, 3, 2, 1]]
                ann['bbox'][:, 1] = height - ann['bbox'][:, 1]
                ann['bbox'][:, 3] = height - ann['bbox'][:, 3]
                seg_mask = np.flipud(seg_mask)

            elif reminder == 3: # horizontal and vertical flip
                image = image[::-1, ::-1, :]
                seg_mask = np.fliplr(seg_mask)
                seg_mask = np.flipud(seg_mask)
                ann['junctions'][:, 0] = width - ann['junctions'][:, 0]
                ann['junctions'][:, 1] = height - ann['junctions'][:, 1]
                ann['bbox'] = ann['bbox'][:, [2, 3, 0, 1]]
                ann['bbox'][:, 0] = width - ann['bbox'][:, 0]
                ann['bbox'][:, 2] = width - ann['bbox'][:, 2]
                ann['bbox'][:, 1] = height - ann['bbox'][:, 1]
                ann['bbox'][:, 3] = height - ann['bbox'][:, 3]
            elif reminder == 4: # rotate 90 degree

                rot_matrix = cv2.getRotationMatrix2D((int(width/2), (height/2)), 90, 1)

                image = cv2.warpAffine(image, rot_matrix, (width, height))

                seg_mask = cv2.warpAffine(seg_mask, rot_matrix, (width, height))

                ann['junctions'] = np.asarray([affine_transform(p, rot_matrix) for p in ann['junctions']], dtype=np.float32)
                ann['bbox'] = np.asarray([affine_transform(p, rot_matrix) for p in ann['bbox']], dtype=np.float32)
            elif reminder == 5: # rotate 270 degree
                rot_matrix = cv2.getRotationMatrix2D((int(width / 2), (height / 2)), 270, 1)
                image = cv2.warpAffine(image, rot_matrix, (width, height))
                seg_mask = cv2.warpAffine(seg_mask, rot_matrix, (width, height))
                ann['junctions'] = np.asarray([affine_transform(p, rot_matrix) for p in ann['junctions']], dtype=np.float32)
                ann['bbox'] = np.asarray([affine_transform(p, rot_matrix) for p in ann['bbox']], dtype=np.float32)
            else:
                pass
            ann['mask'] = seg_mask
        else:
            ann['mask'] = np.zeros((height, width), dtype=np.float64)

            ann['junctions'] = np.asarray([[0, 0]])
            ann['bbox'] = np.asarray([[0,0,0,0]])
            ann['juncs_tag'] = np.asarray([0])
            ann['juncs_index'] = np.asarray([0])

        if self.transform is not None:
            return self.transform(image, ann)
        return image, ann

    def __len__(self):
        return self.num_samples


def collate_fn(batch):

    return (default_collate([b[0] for b in batch]),
            [b[1] for b in batch])


# ---------------------------------------------------------------------------
# ADDED: raster-based GT dataset for isolated single-branch training
# (region / line / point), no COCO json involved.
# ---------------------------------------------------------------------------

class RasterGTDataset(Dataset):
    """
    Loads (Sentinel-2 patch, GT raster) pairs from separate folders instead
    of a COCO json. Used for isolated single-branch training with GT rasters
    produced by build_gt_rasters_from_brp.py.

    Expected layout:
      root/images/<stem>.tif
      root/gt_region/<stem>.tif   binary parcel mask {0,1}   (mask head)
      root/gt_lines/<stem>.tif    1px boundary map {0,1}     (afm head)
      root/gt_nodes/<stem>.tif    filtered corners {0,1}     (jloc head)

    Only the raster required by active_branch is actually read from disk;
    the other two are returned as zero arrays so the collate/model code
    stays shape-consistent regardless of which branch is being trained.
    """

    GT_SUBDIRS = {
        'region':    'gt_region',
        'line':      'gt_lines',
        'point':     'gt_nodes',
        'point_sce': 'gt_nodes',  # same target as "point", just with frozen-line SCE guidance
    }

    def __init__(self, root, active_branch, transform=None, rotate_f=None,
                 stems_file=None, augment=True):
        assert active_branch in ('region', 'line', 'point', 'point_sce'), \
            f"RasterGTDataset only supports single-branch training, got {active_branch!r}"
        self.root = root
        self.active_branch = active_branch
        self.transform = transform
        self.rotate_f = rotate_f
        # augment=False forces no flip/rotate at all (deterministic val/test)
        self.augment = augment

        if stems_file is not None:
            # train/val share the same patches/train/ folder; the actual
            # split is a list of stems (see data_nl/patches/train.txt, val.txt)
            with open(stems_file) as f:
                self.stems = sorted(line.strip() for line in f if line.strip())
        else:
            img_dir = osp.join(root, 'images')
            self.stems = sorted(
                osp.splitext(f)[0] for f in os.listdir(img_dir)
                if f.lower().endswith(('.tif', '.tiff'))
            )
        self.num_samples = len(self.stems)

    def _read_tif(self, path):
        with rasterio.open(path) as src:
            return src.read(1).astype(np.float32)

    # -------------------------------------------------------------------
    # ADDED: reads a multi-band raster (used for gt_afm, 2 bands dx/dy)
    # -------------------------------------------------------------------
    def _read_tif_multiband(self, path):
        with rasterio.open(path) as src:
            return src.read().astype(np.float32)  # (bands, H, W)

    def __getitem__(self, idx_):
        stem = self.stems[idx_]

        img_path = osp.join(self.root, 'images', stem + '.tif')
        with rasterio.open(img_path) as src:
            image = src.read([1, 2, 3]).transpose(1, 2, 0).astype(np.float32) / 10000.0
        np.nan_to_num(image, nan=0.0, copy=False)
        height, width = image.shape[0], image.shape[1]

        gt_subdir = self.GT_SUBDIRS[self.active_branch]
        gt_path = osp.join(self.root, gt_subdir, stem + '.tif')
        gt_raster = self._read_tif(gt_path)
        gt_raster = np.clip(gt_raster, 0.0, 1.0)

        ann = {
            'width': width,
            'height': height,
            'filename': stem + '.tif',
            'active_branch': self.active_branch,
            'gt_region': np.zeros((height, width), dtype=np.float32),
            'gt_lines':  np.zeros((height, width), dtype=np.float32),
            'gt_nodes':  np.zeros((height, width), dtype=np.float32),
        }
        ann['gt_' + self.GT_SUBDIRS[self.active_branch][3:]] = gt_raster

        # -------------------------------------------------------------------
        # ADDED: line branch needs the 2-band (dx,dy) AFM target,
        # precomputed by build_gt_rasters_from_brp.py via afm_op
        # -------------------------------------------------------------------
        if self.active_branch == 'line':
            afm_path = osp.join(self.root, 'gt_afm', stem + '.tif')
            ann['gt_afm'] = self._read_tif_multiband(afm_path)

        # ADDED: augment=False forces reminder=0 (no flip/rotate at all),
        # needed for deterministic validation — the rotate_f branch alone
        # still applied a random flip even when rotate_f=False.
        if not self.augment:
            reminder = 0
        elif self.rotate_f:
            reminder = random.randint(0, 5)
        else:
            reminder = random.randint(0, 3)
        ann['reminder'] = reminder

        # flip/copy: fliplr/flipud return negative-stride views, which
        # torch.from_numpy (called later in ToTensor) cannot handle
        gt_key = 'gt_' + gt_subdir[3:]
        if reminder == 1:
            image = image[:, ::-1, :].copy()
            ann[gt_key] = np.fliplr(ann[gt_key]).copy()
        elif reminder == 2:
            image = image[::-1, :, :].copy()
            ann[gt_key] = np.flipud(ann[gt_key]).copy()
        elif reminder == 3:
            image = image[::-1, ::-1, :].copy()
            ann[gt_key] = np.fliplr(np.flipud(ann[gt_key])).copy()
        elif reminder == 4:
            rot_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 90, 1)
            image = cv2.warpAffine(image, rot_matrix, (width, height))
            ann[gt_key] = cv2.warpAffine(ann[gt_key], rot_matrix, (width, height))
        elif reminder == 5:
            rot_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 270, 1)
            image = cv2.warpAffine(image, rot_matrix, (width, height))
            ann[gt_key] = cv2.warpAffine(ann[gt_key], rot_matrix, (width, height))

        # -------------------------------------------------------------------
        # ADDED: gt_afm holds (dx,dy) vectors, not a plain raster — flipping
        # or rotating the image also needs to rotate the vector components
        # themselves, not just move the pixels around.
        # -------------------------------------------------------------------
        if 'gt_afm' in ann:
            afm = ann['gt_afm']  # (2, H, W): afm[0]=dx, afm[1]=dy
            dx, dy = afm[0], afm[1]
            if reminder == 1:
                dx, dy = np.fliplr(-dx), np.fliplr(dy)
            elif reminder == 2:
                dx, dy = np.flipud(dx), np.flipud(-dy)
            elif reminder == 3:
                dx, dy = np.fliplr(np.flipud(-dx)), np.fliplr(np.flipud(-dy))
            elif reminder == 4:
                rot_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 90, 1)
                dx, dy = cv2.warpAffine(dy, rot_matrix, (width, height)), \
                         cv2.warpAffine(-dx, rot_matrix, (width, height))
            elif reminder == 5:
                rot_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 270, 1)
                dx, dy = cv2.warpAffine(-dy, rot_matrix, (width, height)), \
                         cv2.warpAffine(dx, rot_matrix, (width, height))
            ann['gt_afm'] = np.stack([dx, dy], axis=0).copy().astype(np.float32)

        if self.transform is not None:
            return self.transform(image, ann)
        return image, ann

    def __len__(self):
        return self.num_samples


def collate_fn_raster(batch):
    return (default_collate([b[0] for b in batch]),
            [b[1] for b in batch])

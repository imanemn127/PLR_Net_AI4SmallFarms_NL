from yacs.config import CfgNode as CN

MODELS = CN()

MODELS.NAME = "PLRNet"
MODELS.DEVICE = "cuda"
MODELS.HEAD_SIZE  = [[2]]
MODELS.OUT_FEATURE_CHANNELS = 256
MODELS.LOSS_WEIGHTS = CN(new_allowed=True)
# === CHANGED: pos_weight for BCEWithLogitsLoss on binary jloc heatmap
# corner pixels are ~0.5% of all pixels → pos_weight balances the imbalance
MODELS.JLOC_POS_WEIGHT = 50.0

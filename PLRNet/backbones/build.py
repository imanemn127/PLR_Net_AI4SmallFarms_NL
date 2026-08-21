from .registry import MODELS
from .bsinet_v2 import BsiNet_2 as BsiNet_2
from .swin_backbone import SwinBackbone
from .multi_task_head import MultitaskHead
import os

@MODELS.register("BsiNet_2")
def build_BsiNet_2(cfg):
    head_size = cfg.MODEL.HEAD_SIZE
    #2
    num_class = sum(sum(head_size, []))
    model = BsiNet_2(cfg,
                      head=lambda c_in, c_out: MultitaskHead(c_in, c_out, head_size=head_size),
                      num_class = num_class)
    print('INFO:build BsiNetv2 backbone')
    return model

@MODELS.register("Swin_ViT")
def build_Swin_ViT(cfg):
    head_size = cfg.MODEL.HEAD_SIZE
    num_class = sum(sum(head_size, []))
    model = SwinBackbone(cfg,
                          head=lambda c_in, c_out: MultitaskHead(c_in, c_out, head_size=head_size),
                          num_class=num_class)
    print('INFO:build Swin_ViT backbone')
    return model

def build_backbone(cfg):
    assert cfg.MODEL.NAME in MODELS,  \
        "cfg.MODELS.NAME: {} is not registered in registry".format(cfg.MODELS.NAME)

    return MODELS[cfg.MODEL.NAME](cfg)

"""
Swin Transformer encoder + conv decoder, same role as BsiNet_2 but with a
pretrained transformer encoder instead of a from-scratch CNN encoder.

Same interface as BsiNet_2: forward(x) -> (out, features), features is
(B, 64, H, W) at full input resolution, so detector.py, the loss heads and
the post-processing pipeline don't need any change.
"""

import timm
import torch
from torch import nn

from .bsinet_v2 import CoordAtt, Conv3BN


class DecoderBlock(nn.Module):
    """Upsample low-res features, concat with an encoder skip, 2x Conv3BN."""

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv1 = Conv3BN(in_ch + skip_ch, out_ch)
        self.conv2 = Conv3BN(out_ch, out_ch)

    def forward(self, x, skip):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SwinBackbone(nn.Module):
    def __init__(self, config, head, num_class, swin_name='swin_tiny_patch4_window7_224',
                 pretrained=True):
        super().__init__()
        img_size = config.DATASETS.IMAGE.HEIGHT

        self.encoder = timm.create_model(
            swin_name, pretrained=pretrained, img_size=img_size, features_only=True)
        # feature_info: reduction x4, x8, x16, x32 -> channel counts c1..c4
        c1, c2, c3, c4 = [f['num_chs'] for f in self.encoder.feature_info.info]

        # half-res skip (128x128) for the last decoder block
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.dec4 = DecoderBlock(c4, c3, 384)   # 8x8   -> 16x16
        self.dec3 = DecoderBlock(384, c2, 192)  # 16x16 -> 32x32
        self.dec2 = DecoderBlock(192, c1, 96)   # 32x32 -> 64x64
        self.dec1 = DecoderBlock(96, 32, 64)    # 64x64 -> 128x128
        self.up_full = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # 128x128 -> 256x256
        self.final_conv = Conv3BN(64, 64)

        self.CA = CoordAtt(64, 64, 4)
        self.head = head(64, num_class)

    def forward(self, x):
        stem_feat = self.stem(x)  # (B,32,H/2,W/2)

        feats = self.encoder(x)  # NHWC, 4 levels, reduction x4/x8/x16/x32
        f1, f2, f3, f4 = [f.permute(0, 3, 1, 2).contiguous() for f in feats]

        d = self.dec4(f4, f3)
        d = self.dec3(d, f2)
        d = self.dec2(d, f1)
        d = self.dec1(d, stem_feat)
        d = self.up_full(d)
        d = self.final_conv(d)

        x_out = self.CA(d)
        out = self.head(x_out)
        return out, x_out
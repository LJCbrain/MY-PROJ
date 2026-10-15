"""Large Selective Kernel attention with a small residual for pretrained heads.

The core follows the official ICCV 2023 LSKNet implementation (CC BY-NC 4.0):
https://github.com/zcablii/LSKNet/blob/main/mmrotate/models/backbones/lsknet.py
The outer residual below is an adaptation for YOLO-World, not a full LSKNet block.
"""

import math

import torch
from torch import Tensor, nn


class LSKBlock(nn.Module):
    """Select spatial context from 5x5 and decomposed 23x23 receptive fields."""

    def __init__(self, channels: int):
        super().__init__()
        if not isinstance(channels, int) or channels < 2:
            raise ValueError('LSK requires an integer channel count of at least 2.')
        self.conv0 = nn.Conv2d(channels, channels, 5, padding=2, groups=channels)
        self.conv_spatial = nn.Conv2d(
            channels, channels, 7, padding=9, groups=channels, dilation=3)
        self.conv1 = nn.Conv2d(channels, channels // 2, 1)
        self.conv2 = nn.Conv2d(channels, channels // 2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(channels // 2, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        local = self.conv0(x)
        context = self.conv_spatial(local)
        local = self.conv1(local)
        context = self.conv2(context)
        combined = torch.cat((local, context), dim=1)
        descriptors = torch.cat((
            combined.mean(dim=1, keepdim=True),
            combined.max(dim=1, keepdim=True)[0]), dim=1)
        selection = self.conv_squeeze(descriptors).sigmoid()
        attention = self.conv(
            local * selection[:, :1] + context * selection[:, 1:])
        return x * attention


class ResidualLSKBlock(nn.Module):
    """Keep the original feature and learn a per-channel LSK residual."""

    def __init__(self, channels: int, init_scale: float = 1e-2):
        super().__init__()
        if not math.isfinite(init_scale):
            raise ValueError('LSK init_scale must be finite.')
        self.lsk = LSKBlock(channels)
        self.residual_scale = nn.Parameter(torch.full((channels,), float(init_scale)))

    def forward(self, x: Tensor) -> Tensor:
        scale = self.residual_scale.view(1, -1, 1, 1).to(dtype=x.dtype)
        return x + scale * self.lsk(x)

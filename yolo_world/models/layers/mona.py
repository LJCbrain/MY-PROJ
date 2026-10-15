"""Mona adapted to BCHW features, with an identity start for continued training.

The normalization, bottleneck and parallel 3/5/7 depthwise filters follow
``D:/模块/Mona(CV,CVPR2025).py``. Zeroing the output projection is the local
initialization change used when adding an adapter to a trained detector.
"""

import torch
from torch import nn
import torch.nn.functional as F
from mmyolo.registry import MODELS


class MonaOp(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.conv2 = nn.Conv2d(channels, channels, 5, padding=2, groups=channels)
        self.conv3 = nn.Conv2d(channels, channels, 7, padding=3, groups=channels)
        self.projector = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        x = x + (self.conv1(x) + self.conv2(x) + self.conv3(x)) / 3.0
        return x + self.projector(x)


@MODELS.register_module()
class Mona2d(nn.Module):
    """Residual Mona on a feature map; it does not consume text features."""

    def __init__(self, in_channels, bottleneck_dim=64, dropout=0.1,
                 zero_init=True):
        super().__init__()
        if in_channels <= 0 or bottleneck_dim <= 0:
            raise ValueError('Mona channel dimensions must be positive.')
        if not 0 <= dropout < 1:
            raise ValueError('Mona dropout must be in [0, 1).')
        self.in_channels = in_channels
        self.norm = nn.LayerNorm(in_channels)
        self.gamma = nn.Parameter(torch.full((in_channels,), 1e-6))
        self.gammax = nn.Parameter(torch.ones(in_channels))
        self.project1 = nn.Linear(in_channels, bottleneck_dim)
        self.adapter_conv = MonaOp(bottleneck_dim)
        self.dropout = nn.Dropout(dropout)
        self.project2 = nn.Linear(bottleneck_dim, in_channels)
        if zero_init:
            nn.init.zeros_(self.project2.weight)
            nn.init.zeros_(self.project2.bias)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError('Mona expects BCHW features with its configured channels.')
        batch, channels, height, width = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        features = self.norm(tokens) * self.gamma + tokens * self.gammax
        features = self.project1(features)
        features = features.transpose(1, 2).reshape(batch, -1, height, width)
        features = self.adapter_conv(features).flatten(2).transpose(1, 2)
        update = self.project2(self.dropout(F.gelu(features)))
        return x + update.transpose(1, 2).reshape(batch, channels, height, width)

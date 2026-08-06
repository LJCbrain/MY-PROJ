# Copyright (c) Tencent Inc. All rights reserved.
import math
from typing import List, Tuple

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule, Linear
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmengine.model import BaseModule
from mmyolo.registry import MODELS
from mmyolo.models.layers import CSPLayerWithTwoConv


@MODELS.register_module()
class MaxSigmoidAttnBlock(BaseModule):
    """Max Sigmoid attention block."""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 guide_channels: int,
                 embed_channels: int,
                 kernel_size: int = 3,
                 padding: int = 1,
                 num_heads: int = 1,
                 use_depthwise: bool = False,
                 with_scale: bool = False,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(type='BN',
                                             momentum=0.03,
                                             eps=0.001),
                 init_cfg: OptMultiConfig = None,
                 use_einsum: bool = True) -> None:
        super().__init__(init_cfg=init_cfg)
        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule

        assert (out_channels % num_heads == 0 and
                embed_channels % num_heads == 0), \
            'out_channels and embed_channels should be divisible by num_heads.'
        self.num_heads = num_heads
        self.head_channels = embed_channels // num_heads
        self.use_einsum = use_einsum

        self.embed_conv = ConvModule(
            in_channels,
            embed_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None) if embed_channels != in_channels else None
        self.guide_fc = Linear(guide_channels, embed_channels)
        self.bias = nn.Parameter(torch.zeros(num_heads))
        if with_scale:
            self.scale = nn.Parameter(torch.ones(1, num_heads, 1, 1))
        else:
            self.scale = 1.0

        self.project_conv = conv(in_channels,
                                 out_channels,
                                 kernel_size,
                                 stride=1,
                                 padding=padding,
                                 conv_cfg=conv_cfg,
                                 norm_cfg=norm_cfg,
                                 act_cfg=None)

    def forward(self, x: Tensor, guide: Tensor) -> Tensor:
        """Forward process."""
        B, _, H, W = x.shape

        guide = self.guide_fc(guide)
        guide = guide.reshape(B, -1, self.num_heads, self.head_channels)
        embed = self.embed_conv(x) if self.embed_conv is not None else x
        embed = embed.reshape(B, self.num_heads, self.head_channels, H, W)

        if self.use_einsum:
            attn_weight = torch.einsum('bmchw,bnmc->bmhwn', embed, guide)
        else:
            batch, m, channel, height, width = embed.shape
            _, n, _, _ = guide.shape
            embed = embed.permute(0, 1, 3, 4, 2)
            embed = embed.reshape(batch, m, -1, channel)
            guide = guide.permute(0, 2, 3, 1)
            attn_weight = torch.matmul(embed, guide)
            attn_weight = attn_weight.reshape(batch, m, height, width, n)

        attn_weight = attn_weight.max(dim=-1)[0]
        attn_weight = attn_weight / (self.head_channels**0.5)
        attn_weight = attn_weight + self.bias[None, :, None, None]
        attn_weight = attn_weight.sigmoid() * self.scale

        x = self.project_conv(x)
        x = x.reshape(B, self.num_heads, -1, H, W)
        x = x * attn_weight.unsqueeze(2)
        x = x.reshape(B, -1, H, W)
        return x


@MODELS.register_module()
class RepMatrixMaxSigmoidAttnBlock(BaseModule):
    """Max Sigmoid attention block."""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 embed_channels: int,
                 guide_channels: int,
                 kernel_size: int = 3,
                 padding: int = 1,
                 num_heads: int = 1,
                 use_depthwise: bool = False,
                 with_scale: bool = False,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(type='BN',
                                             momentum=0.03,
                                             eps=0.001),
                 init_cfg: OptMultiConfig = None,
                 use_einsum: bool = True) -> None:
        super().__init__(init_cfg=init_cfg)
        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule

        assert (out_channels % num_heads == 0 and
                embed_channels % num_heads == 0), \
            'out_channels and embed_channels should be divisible by num_heads.'
        self.num_heads = num_heads
        self.head_channels = out_channels // num_heads
        self.use_einsum = use_einsum

        self.embed_conv = ConvModule(
            in_channels,
            embed_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None) if embed_channels != in_channels else None
        self.bias = nn.Parameter(torch.zeros(num_heads))
        self.guide_weight = nn.Parameter(
            torch.zeros(guide_channels, embed_channels // num_heads,
                        num_heads))
        self.project_conv = conv(in_channels,
                                 out_channels,
                                 kernel_size,
                                 stride=1,
                                 padding=padding,
                                 conv_cfg=conv_cfg,
                                 norm_cfg=norm_cfg,
                                 act_cfg=None)

    def forward(self, x: Tensor, txt_feats: Tensor = None) -> Tensor:
        """Forward process."""
        B, _, H, W = x.shape

        embed = self.embed_conv(x) if self.embed_conv is not None else x
        embed = embed.reshape(B, self.num_heads, self.head_channels, H, W)

        batch, m, channel, height, width = embed.shape
        _, n, _, _ = self.guide_weight.shape
        # can be formulated to split conv
        embed = embed.permute(0, 1, 3, 4, 2)
        embed = embed.reshape(batch, m, -1, channel)
        attn_weight = torch.matmul(embed, self.guide_weight)
        attn_weight = attn_weight.reshape(batch, m, height, width, n)

        attn_weight = attn_weight.max(dim=-1)[0]
        attn_weight = attn_weight / (self.head_channels**0.5)
        attn_weight = attn_weight + self.bias[None, :, None, None]
        attn_weight = attn_weight.sigmoid()

        x = self.project_conv(x)
        x = x.reshape(B, self.num_heads, -1, H, W)
        x = x * attn_weight.unsqueeze(2)
        x = x.reshape(B, -1, H, W)
        return x


@MODELS.register_module()
class RepConvMaxSigmoidAttnBlock(BaseModule):
    """Max Sigmoid attention block."""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 embed_channels: int,
                 guide_channels: int,
                 kernel_size: int = 3,
                 padding: int = 1,
                 num_heads: int = 1,
                 use_depthwise: bool = False,
                 with_scale: bool = False,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(type='BN',
                                             momentum=0.03,
                                             eps=0.001),
                 init_cfg: OptMultiConfig = None,
                 use_einsum: bool = True) -> None:
        super().__init__(init_cfg=init_cfg)
        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule

        assert (out_channels % num_heads == 0 and
                embed_channels % num_heads == 0), \
            'out_channels and embed_channels should be divisible by num_heads.'
        self.num_heads = num_heads
        self.head_channels = out_channels // num_heads
        self.use_einsum = use_einsum

        self.embed_conv = ConvModule(
            in_channels,
            embed_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None) if embed_channels != in_channels else None
        self.bias = nn.Parameter(torch.zeros(num_heads))
        self.num_heads = num_heads
        self.split_channels = embed_channels // num_heads
        self.guide_convs = nn.ModuleList(
            nn.Conv2d(self.split_channels, guide_channels, 1, bias=False)
            for _ in range(num_heads))
        self.project_conv = conv(in_channels,
                                 out_channels,
                                 kernel_size,
                                 stride=1,
                                 padding=padding,
                                 conv_cfg=conv_cfg,
                                 norm_cfg=norm_cfg,
                                 act_cfg=None)

    def forward(self, x: Tensor, txt_feats: Tensor = None) -> Tensor:
        """Forward process."""
        B, C, H, W = x.shape

        embed = self.embed_conv(x) if self.embed_conv is not None else x
        embed = list(embed.split(self.split_channels, 1))
        # Bx(MxN)xHxW (H*c=C, H: heads)
        attn_weight = torch.cat(
            [conv(x) for conv, x in zip(self.guide_convs, embed)], dim=1)
        # BxMxNxHxW
        attn_weight = attn_weight.view(B, self.num_heads, -1, H, W)
        # attn_weight = torch.stack(
        #     [conv(x) for conv, x in zip(self.guide_convs, embed)])
        # BxMxNxHxW -> BxMxHxW
        attn_weight = attn_weight.max(dim=2)[0] / (self.head_channels**0.5)
        attn_weight = (attn_weight + self.bias.view(1, -1, 1, 1)).sigmoid()
        # .transpose(0, 1)
        # BxMx1xHxW
        attn_weight = attn_weight[:, :, None]
        x = self.project_conv(x)
        # BxHxCxHxW
        x = x.view(B, self.num_heads, -1, H, W)
        x = x * attn_weight
        x = x.view(B, -1, H, W)
        return x


@MODELS.register_module()
class MaxSigmoidCSPLayerWithTwoConv(CSPLayerWithTwoConv):
    """Sigmoid-attention based CSP layer with two convolution layers."""

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            guide_channels: int,
            embed_channels: int,
            num_heads: int = 1,
            expand_ratio: float = 0.5,
            num_blocks: int = 1,
            with_scale: bool = False,
            add_identity: bool = True,  # shortcut
            conv_cfg: OptConfigType = None,
            norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: ConfigType = dict(type='SiLU', inplace=True),
            init_cfg: OptMultiConfig = None,
            use_einsum: bool = True) -> None:
        super().__init__(in_channels=in_channels,
                         out_channels=out_channels,
                         expand_ratio=expand_ratio,
                         num_blocks=num_blocks,
                         add_identity=add_identity,
                         conv_cfg=conv_cfg,
                         norm_cfg=norm_cfg,
                         act_cfg=act_cfg,
                         init_cfg=init_cfg)

        self.final_conv = ConvModule((3 + num_blocks) * self.mid_channels,
                                     out_channels,
                                     1,
                                     conv_cfg=conv_cfg,
                                     norm_cfg=norm_cfg,
                                     act_cfg=act_cfg)

        self.attn_block = MaxSigmoidAttnBlock(self.mid_channels,
                                              self.mid_channels,
                                              guide_channels=guide_channels,
                                              embed_channels=embed_channels,
                                              num_heads=num_heads,
                                              with_scale=with_scale,
                                              conv_cfg=conv_cfg,
                                              norm_cfg=norm_cfg,
                                              use_einsum=use_einsum)

    def forward(self, x: Tensor, guide: Tensor) -> Tensor:
        """Forward process."""
        x_main = self.main_conv(x)
        x_main = list(x_main.split((self.mid_channels, self.mid_channels), 1))
        x_main.extend(blocks(x_main[-1]) for blocks in self.blocks)
        x_main.append(self.attn_block(x_main[-1], guide))
        return self.final_conv(torch.cat(x_main, 1))


@MODELS.register_module()
class GuideAgnosticCSPLayerWithTwoConv(CSPLayerWithTwoConv):
    """CSP layer that keeps the YOLO-World neck call signature.

    YOLOWorldPAFPN calls each CSP layer as ``layer(x, txt_feats)``. This
    class intentionally ignores the guide so the neck no longer applies
    text-image sigmoid attention before foreground recalibration.
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            guide_channels: int = None,
            embed_channels: int = None,
            num_heads: int = 1,
            expand_ratio: float = 0.5,
            num_blocks: int = 1,
            with_scale: bool = False,
            add_identity: bool = True,
            conv_cfg: OptConfigType = None,
            norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: ConfigType = dict(type='SiLU', inplace=True),
            init_cfg: OptMultiConfig = None,
            use_einsum: bool = True) -> None:
        super().__init__(in_channels=in_channels,
                         out_channels=out_channels,
                         expand_ratio=expand_ratio,
                         num_blocks=num_blocks,
                         add_identity=add_identity,
                         conv_cfg=conv_cfg,
                         norm_cfg=norm_cfg,
                         act_cfg=act_cfg,
                         init_cfg=init_cfg)

    def forward(self, x: Tensor, guide: Tensor = None) -> Tensor:
        return super().forward(x)


@MODELS.register_module()
class RepMaxSigmoidCSPLayerWithTwoConv(CSPLayerWithTwoConv):
    """Sigmoid-attention based CSP layer with two convolution layers."""

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            guide_channels: int,
            embed_channels: int,
            num_heads: int = 1,
            expand_ratio: float = 0.5,
            num_blocks: int = 1,
            with_scale: bool = False,
            add_identity: bool = True,  # shortcut
            conv_cfg: OptConfigType = None,
            norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: ConfigType = dict(type='SiLU', inplace=True),
            init_cfg: OptMultiConfig = None,
            use_einsum: bool = True) -> None:
        super().__init__(in_channels=in_channels,
                         out_channels=out_channels,
                         expand_ratio=expand_ratio,
                         num_blocks=num_blocks,
                         add_identity=add_identity,
                         conv_cfg=conv_cfg,
                         norm_cfg=norm_cfg,
                         act_cfg=act_cfg,
                         init_cfg=init_cfg)

        self.final_conv = ConvModule((3 + num_blocks) * self.mid_channels,
                                     out_channels,
                                     1,
                                     conv_cfg=conv_cfg,
                                     norm_cfg=norm_cfg,
                                     act_cfg=act_cfg)

        self.attn_block = RepMatrixMaxSigmoidAttnBlock(
            self.mid_channels,
            self.mid_channels,
            embed_channels=embed_channels,
            guide_channels=guide_channels,
            num_heads=num_heads,
            with_scale=with_scale,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            use_einsum=use_einsum)

    def forward(self, x: Tensor, guide: Tensor) -> Tensor:
        """Forward process."""
        x_main = self.main_conv(x)
        x_main = list(x_main.split((self.mid_channels, self.mid_channels), 1))
        x_main.extend(blocks(x_main[-1]) for blocks in self.blocks)
        x_main.append(self.attn_block(x_main[-1], guide))
        return self.final_conv(torch.cat(x_main, 1))


@MODELS.register_module()
class RepConvMaxSigmoidCSPLayerWithTwoConv(CSPLayerWithTwoConv):
    """Sigmoid-attention based CSP layer with two convolution layers."""

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            guide_channels: int,
            embed_channels: int,
            num_heads: int = 1,
            expand_ratio: float = 0.5,
            num_blocks: int = 1,
            with_scale: bool = False,
            add_identity: bool = True,  # shortcut
            conv_cfg: OptConfigType = None,
            norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: ConfigType = dict(type='SiLU', inplace=True),
            init_cfg: OptMultiConfig = None,
            use_einsum: bool = True) -> None:
        super().__init__(in_channels=in_channels,
                         out_channels=out_channels,
                         expand_ratio=expand_ratio,
                         num_blocks=num_blocks,
                         add_identity=add_identity,
                         conv_cfg=conv_cfg,
                         norm_cfg=norm_cfg,
                         act_cfg=act_cfg,
                         init_cfg=init_cfg)

        self.final_conv = ConvModule((3 + num_blocks) * self.mid_channels,
                                     out_channels,
                                     1,
                                     conv_cfg=conv_cfg,
                                     norm_cfg=norm_cfg,
                                     act_cfg=act_cfg)

        self.attn_block = RepConvMaxSigmoidAttnBlock(
            self.mid_channels,
            self.mid_channels,
            embed_channels=embed_channels,
            guide_channels=guide_channels,
            num_heads=num_heads,
            with_scale=with_scale,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            use_einsum=use_einsum)

    def forward(self, x: Tensor, guide: Tensor) -> Tensor:
        """Forward process."""
        x_main = self.main_conv(x)
        x_main = list(x_main.split((self.mid_channels, self.mid_channels), 1))
        x_main.extend(blocks(x_main[-1]) for blocks in self.blocks)
        x_main.append(self.attn_block(x_main[-1], guide))
        return self.final_conv(torch.cat(x_main, 1))


@MODELS.register_module()
class ImagePoolingAttentionModule(nn.Module):

    def __init__(self,
                 image_channels: List[int],
                 text_channels: int,
                 embed_channels: int,
                 with_scale: bool = False,
                 num_feats: int = 3,
                 num_heads: int = 8,
                 pool_size: int = 3,
                 use_einsum: bool = True):
        super().__init__()

        self.text_channels = text_channels
        self.embed_channels = embed_channels
        self.num_heads = num_heads
        self.num_feats = num_feats
        self.head_channels = embed_channels // num_heads
        self.pool_size = pool_size
        self.use_einsum = use_einsum
        if with_scale:
            self.scale = nn.Parameter(torch.tensor([0.]), requires_grad=True)
        else:
            self.scale = 1.0
        self.projections = nn.ModuleList([
            ConvModule(in_channels, embed_channels, 1, act_cfg=None)
            for in_channels in image_channels
        ])
        self.query = nn.Sequential(nn.LayerNorm(text_channels),
                                   Linear(text_channels, embed_channels))
        self.key = nn.Sequential(nn.LayerNorm(embed_channels),
                                 Linear(embed_channels, embed_channels))
        self.value = nn.Sequential(nn.LayerNorm(embed_channels),
                                   Linear(embed_channels, embed_channels))
        self.proj = Linear(embed_channels, text_channels)

        self.image_pools = nn.ModuleList([
            nn.AdaptiveMaxPool2d((pool_size, pool_size))
            for _ in range(num_feats)
        ])

    def forward(self, text_features, image_features):
        B = image_features[0].shape[0]
        assert len(image_features) == self.num_feats
        num_patches = self.pool_size**2
        mlvl_image_features = [
            pool(proj(x)).view(B, -1, num_patches)
            for (x, proj, pool
                 ) in zip(image_features, self.projections, self.image_pools)
        ]
        mlvl_image_features = torch.cat(mlvl_image_features,
                                        dim=-1).transpose(1, 2)
        q = self.query(text_features)
        k = self.key(mlvl_image_features)
        v = self.value(mlvl_image_features)

        q = q.reshape(B, -1, self.num_heads, self.head_channels)
        k = k.reshape(B, -1, self.num_heads, self.head_channels)
        v = v.reshape(B, -1, self.num_heads, self.head_channels)
        if self.use_einsum:
            attn_weight = torch.einsum('bnmc,bkmc->bmnk', q, k)
        else:
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 3, 1)
            attn_weight = torch.matmul(q, k)

        attn_weight = attn_weight / (self.head_channels**0.5)
        attn_weight = F.softmax(attn_weight, dim=-1)
        if self.use_einsum:
            x = torch.einsum('bmnk,bkmc->bnmc', attn_weight, v)
        else:
            v = v.permute(0, 2, 1, 3)
            x = torch.matmul(attn_weight, v)
            x = x.permute(0, 2, 1, 3)
        x = self.proj(x.reshape(B, -1, self.embed_channels))
        return x * self.scale + text_features


@MODELS.register_module()
class VanillaSigmoidBlock(BaseModule):
    """Sigmoid attention block."""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 guide_channels: int,
                 embed_channels: int,
                 kernel_size: int = 3,
                 padding: int = 1,
                 num_heads: int = 1,
                 use_depthwise: bool = False,
                 with_scale: bool = False,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(type='BN',
                                             momentum=0.03,
                                             eps=0.001),
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule

        assert (out_channels % num_heads == 0 and
                embed_channels % num_heads == 0), \
            'out_channels and embed_channels should be divisible by num_heads.'
        self.num_heads = num_heads
        self.head_channels = out_channels // num_heads

        self.project_conv = conv(in_channels,
                                 out_channels,
                                 kernel_size,
                                 stride=1,
                                 padding=padding,
                                 conv_cfg=conv_cfg,
                                 norm_cfg=norm_cfg,
                                 act_cfg=None)

    def forward(self, x: Tensor, guide: Tensor) -> Tensor:
        """Forward process."""
        x = self.project_conv(x)
        # remove sigmoid
        # x = x * x.sigmoid()
        return x


@MODELS.register_module()
class EfficientCSPLayerWithTwoConv(CSPLayerWithTwoConv):
    """Sigmoid-attention based CSP layer with two convolution layers."""

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            guide_channels: int,
            embed_channels: int,
            num_heads: int = 1,
            expand_ratio: float = 0.5,
            num_blocks: int = 1,
            with_scale: bool = False,
            add_identity: bool = True,  # shortcut
            conv_cfg: OptConfigType = None,
            norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: ConfigType = dict(type='SiLU', inplace=True),
            init_cfg: OptMultiConfig = None) -> None:
        super().__init__(in_channels=in_channels,
                         out_channels=out_channels,
                         expand_ratio=expand_ratio,
                         num_blocks=num_blocks,
                         add_identity=add_identity,
                         conv_cfg=conv_cfg,
                         norm_cfg=norm_cfg,
                         act_cfg=act_cfg,
                         init_cfg=init_cfg)

        self.final_conv = ConvModule((3 + num_blocks) * self.mid_channels,
                                     out_channels,
                                     1,
                                     conv_cfg=conv_cfg,
                                     norm_cfg=norm_cfg,
                                     act_cfg=act_cfg)

        self.attn_block = VanillaSigmoidBlock(self.mid_channels,
                                              self.mid_channels,
                                              guide_channels=guide_channels,
                                              embed_channels=embed_channels,
                                              num_heads=num_heads,
                                              with_scale=with_scale,
                                              conv_cfg=conv_cfg,
                                              norm_cfg=norm_cfg)

    def forward(self, x: Tensor, guide: Tensor) -> Tensor:
        """Forward process."""
        x_main = self.main_conv(x)
        x_main = list(x_main.split((self.mid_channels, self.mid_channels), 1))
        x_main.extend(blocks(x_main[-1]) for blocks in self.blocks)
        x_main.append(self.attn_block(x_main[-1], guide))
        return self.final_conv(torch.cat(x_main, 1))


@MODELS.register_module()
class ClassAgnosticForegroundGate(BaseModule):
    """Class-agnostic quality recalibration for multi-level features.

    The historical class name is retained for checkpoint and configuration
    compatibility. The foreground mask, quality map, and dynamic gamma form
    a visual reliability map that softly recalibrates each feature level.
    """

    def __init__(self,
                 in_channels: List[int],
                 hidden_ratio: float = 0.25,
                 min_hidden: int = 16,
                 init_gamma: float = 0.0,
                 max_gamma: float = 0.15,
                 gamma_init_bias: float = -0.3,
                 gamma_hidden_ratio: float = 0.25,
                 gamma_warmup_iters: int = 0,
                 return_parts: bool = True,
                 norm_cfg: ConfigType = dict(type='BN',
                                             momentum=0.03,
                                             eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.max_gamma = max_gamma
        self.gamma_warmup_iters = int(gamma_warmup_iters)
        self.return_parts = return_parts
        self.mask_convs = nn.ModuleList()
        self.quality_convs = nn.ModuleList()
        self.gamma_mlps = nn.ModuleList()
        self.register_buffer(
            '_gamma_step', torch.zeros((), dtype=torch.long), persistent=False)

        for c in in_channels:
            hidden = max(min_hidden, int(c * hidden_ratio))
            gamma_hidden = max(min_hidden, int(c * gamma_hidden_ratio))
            self.mask_convs.append(
                nn.Sequential(
                    ConvModule(c,
                               hidden,
                               3,
                               padding=1,
                               norm_cfg=norm_cfg,
                               act_cfg=act_cfg),
                    nn.Conv2d(hidden, 1, 1)))
            self.quality_convs.append(
                nn.Sequential(
                    ConvModule(2 * c + 1,
                               hidden,
                               3,
                               padding=1,
                               norm_cfg=norm_cfg,
                               act_cfg=act_cfg),
                    nn.Conv2d(hidden, 1, 1)))
            gamma_mlp = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(c, gamma_hidden),
                nn.SiLU(inplace=True),
                nn.Linear(gamma_hidden, 1))
            nn.init.constant_(gamma_mlp[-1].weight, 0.0)
            if gamma_init_bias is None:
                gamma_ratio = min(max(init_gamma / max(max_gamma, 1e-6),
                                      1e-6), 1 - 1e-6)
                gamma_init_bias = torch.logit(
                    torch.tensor(gamma_ratio)).item()
            nn.init.constant_(gamma_mlp[-1].bias, float(gamma_init_bias))
            self.gamma_mlps.append(gamma_mlp)

    def _gamma_warmup_scale(self) -> float:
        if self.gamma_warmup_iters <= 0:
            return 1.0
        if self.training:
            self._gamma_step.add_(1)
        return min(
            float(self._gamma_step.item()) / float(self.gamma_warmup_iters),
            1.0)

    def forward(self,
                feats,
                txt_feats=None,
                txt_masks=None,
                return_parts: bool = None):
        """Forward multi-level features without producing class logits."""
        assert len(feats) == len(self.mask_convs)
        if return_parts is None:
            return_parts = self.return_parts
        out_feats = []
        masks = []
        qualities = []
        recalib_maps = []
        gammas = []
        gamma_scale = self._gamma_warmup_scale()

        for feat, mask_conv, quality_conv, gamma_mlp in zip(
                feats, self.mask_convs, self.quality_convs,
                self.gamma_mlps):
            mask = torch.sigmoid(mask_conv(feat))
            masked_feat = feat * mask
            quality_input = torch.cat((feat, masked_feat, mask), dim=1)
            quality = torch.sigmoid(quality_conv(quality_input))
            recalib = mask * quality
            gamma = self.max_gamma * torch.sigmoid(gamma_mlp(feat))
            gamma = gamma.view(-1, 1, 1, 1) * gamma_scale
            out = feat * (1.0 + gamma * recalib)

            out_feats.append(out)
            masks.append(mask)
            qualities.append(quality)
            recalib_maps.append(recalib)
            gammas.append(gamma)

        out_feats = tuple(out_feats)
        if return_parts:
            return out_feats, dict(masks=masks,
                                   qualities=qualities,
                                   recalib_maps=recalib_maps,
                                   reliability_maps=recalib_maps,
                                   gammas=gammas)
        return out_feats


@MODELS.register_module()
class RegionalBackgroundPriorGate(BaseModule):
    """Regional background prior for class-agnostic object routing.

    Each feature level is split into a small regular grid.  A robust regional
    background token is estimated from the most background-like locations and
    refined by attention that is restricted to the same region.  The inverted
    cosine similarity to that token is the base foreground prior.  CAQR fusion
    and direct feature amplification remain optional for backward
    compatibility; disabling both turns this module into a pure routing prior.
    """

    def __init__(self,
                 in_channels: List[int],
                 region_grid_size: int = 3,
                 background_keep_ratio: float = 0.75,
                 attention_heads: List[int] = (4, 8, 8),
                 hidden_ratio: float = 0.25,
                 min_hidden: int = 16,
                 caqr_init_prob: float = 0.01,
                 caqr_beta_max: float = 0.3,
                 caqr_beta_warmup_iters: int = 1000,
                 init_gamma: float = 0.0,
                 max_gamma: float = 0.15,
                 gamma_init_bias: float = -0.3,
                 gamma_hidden_ratio: float = 0.25,
                 gamma_warmup_iters: int = 1000,
                 fusion_weights: Tuple[float, float] = (0.5, 0.25),
                 use_caqr: bool = True,
                 direct_feature_boost: bool = True,
                 return_parts: bool = True,
                 norm_cfg: ConfigType = dict(type='BN',
                                             momentum=0.03,
                                             eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        if len(in_channels) != len(attention_heads):
            raise ValueError('attention_heads must match in_channels.')
        if not 0 < background_keep_ratio <= 1:
            raise ValueError('background_keep_ratio must be in (0, 1].')
        if not 0 <= caqr_beta_max <= 1:
            raise ValueError('caqr_beta_max must be in [0, 1].')
        if len(in_channels) != 3:
            raise ValueError('Scheme A expects P3/P4/P5 feature levels.')

        self.region_grid_size = int(region_grid_size)
        self.background_keep_ratio = float(background_keep_ratio)
        self.caqr_beta_max = float(caqr_beta_max)
        self.caqr_beta_warmup_iters = int(caqr_beta_warmup_iters)
        self.max_gamma = float(max_gamma)
        self.gamma_warmup_iters = int(gamma_warmup_iters)
        self.fusion_weights = tuple(float(v) for v in fusion_weights)
        self.use_caqr = bool(use_caqr)
        self.direct_feature_boost = bool(direct_feature_boost)
        self.return_parts = bool(return_parts)

        self.region_attentions = nn.ModuleList()
        self.region_norms = nn.ModuleList()
        self.caqr_mask_convs = nn.ModuleList()
        self.gamma_mlps = nn.ModuleList()
        # This state must survive checkpoints so resumed training and inference
        # do not silently restart the residual warm-up.
        self.register_buffer(
            '_gamma_step', torch.zeros((), dtype=torch.long), persistent=True)
        self.register_buffer(
            '_caqr_beta_step', torch.zeros((), dtype=torch.long),
            persistent=True)

        caqr_bias = torch.logit(
            torch.tensor(float(caqr_init_prob)).clamp(1e-6, 1 - 1e-6))
        for channels, num_heads in zip(in_channels, attention_heads):
            if channels % int(num_heads) != 0:
                raise ValueError(
                    f'{channels} channels are not divisible by {num_heads} heads.')
            attention = nn.MultiheadAttention(
                channels, int(num_heads), dropout=0.0, batch_first=True)
            # Start from the robust regional mean. Attention is introduced
            # gradually as its zero-initialized output projection learns.
            nn.init.constant_(attention.out_proj.weight, 0.0)
            nn.init.constant_(attention.out_proj.bias, 0.0)
            self.region_attentions.append(attention)
            self.region_norms.append(nn.LayerNorm(channels))

            hidden = max(min_hidden, int(channels * hidden_ratio))
            caqr_mask_conv = nn.Sequential(
                ConvModule(channels,
                           hidden,
                           3,
                           padding=1,
                           norm_cfg=norm_cfg,
                           act_cfg=act_cfg),
                nn.Conv2d(hidden, 1, 1))
            # With beta=0 at the start of training, this branch is initially
            # inert. Its low prior prevents an abrupt background boost later.
            nn.init.constant_(caqr_mask_conv[-1].weight, 0.0)
            nn.init.constant_(caqr_mask_conv[-1].bias, float(caqr_bias))
            if not self.use_caqr:
                caqr_mask_conv.requires_grad_(False)
            self.caqr_mask_convs.append(caqr_mask_conv)

            gamma_hidden = max(min_hidden, int(channels * gamma_hidden_ratio))
            gamma_mlp = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(channels, gamma_hidden),
                nn.SiLU(inplace=True),
                nn.Linear(gamma_hidden, 1))
            nn.init.constant_(gamma_mlp[-1].weight, 0.0)
            if gamma_init_bias is None:
                ratio = min(max(init_gamma / max(max_gamma, 1e-6), 1e-6),
                            1 - 1e-6)
                gamma_init_bias = torch.logit(torch.tensor(ratio)).item()
            nn.init.constant_(gamma_mlp[-1].bias, float(gamma_init_bias))
            if not self.direct_feature_boost:
                gamma_mlp.requires_grad_(False)
            self.gamma_mlps.append(gamma_mlp)

    def _gamma_warmup_scale(self) -> float:
        if self.gamma_warmup_iters <= 0:
            return 1.0
        if self.training:
            self._gamma_step.add_(1)
        return min(
            float(self._gamma_step.item()) / float(self.gamma_warmup_iters),
            1.0)

    def _caqr_beta(self) -> float:
        """Return the CAQR contribution after a zero-to-beta warm-up."""
        if self.caqr_beta_warmup_iters <= 0:
            return self.caqr_beta_max
        scale = min(
            float(self._caqr_beta_step.item()) /
            float(self.caqr_beta_warmup_iters), 1.0)
        if self.training:
            self._caqr_beta_step.add_(1)
        return self.caqr_beta_max * scale

    def _regional_foreground_prior(self, feat: Tensor,
                                   attention: nn.MultiheadAttention,
                                   token_norm: nn.LayerNorm) -> Tensor:
        batch_size, channels, height, width = feat.shape
        prior = feat.new_zeros((batch_size, 1, height, width))
        y_edges = torch.linspace(
            0, height, self.region_grid_size + 1,
            device=feat.device).round().long()
        x_edges = torch.linspace(
            0, width, self.region_grid_size + 1,
            device=feat.device).round().long()

        for row in range(self.region_grid_size):
            y1, y2 = int(y_edges[row]), int(y_edges[row + 1])
            for col in range(self.region_grid_size):
                x1, x2 = int(x_edges[col]), int(x_edges[col + 1])
                region = feat[:, :, y1:y2, x1:x2]
                tokens = region.flatten(2).transpose(1, 2)
                if tokens.shape[1] == 0:
                    continue

                initial_token = tokens.mean(dim=1, keepdim=True)
                similarities = F.cosine_similarity(
                    tokens, initial_token, dim=-1)
                keep_count = max(
                    1,
                    int(math.ceil(tokens.shape[1] *
                                  self.background_keep_ratio)))
                keep_indices = similarities.topk(
                    keep_count, dim=1, largest=True).indices
                gather_indices = keep_indices.unsqueeze(-1).expand(
                    -1, -1, channels)
                background_tokens = torch.gather(
                    tokens, 1, gather_indices).mean(dim=1, keepdim=True)
                key_padding_mask = torch.ones(
                    (batch_size, tokens.shape[1]),
                    dtype=torch.bool,
                    device=feat.device)
                key_padding_mask.scatter_(1, keep_indices, False)

                refined, _ = attention(
                    background_tokens,
                    tokens,
                    tokens,
                    key_padding_mask=key_padding_mask,
                    need_weights=False)
                background_tokens = token_norm(background_tokens + refined)
                cosine = F.cosine_similarity(
                    tokens, background_tokens, dim=-1)
                region_prior = (1.0 - cosine) * 0.5
                region_prior = region_prior.to(feat.dtype).reshape(
                    batch_size, 1, y2 - y1, x2 - x1)
                prior[:, :, y1:y2, x1:x2] = region_prior
        return prior.clamp_(0.0, 1.0)

    def _fuse_reliability_maps(self, maps: List[Tensor]) -> Tensor:
        target_size = maps[0].shape[-2:]
        p4 = F.interpolate(
            maps[1], target_size, mode='bilinear', align_corners=False)
        p5 = F.interpolate(
            maps[2], target_size, mode='bilinear', align_corners=False)
        w4, w5 = self.fusion_weights
        fused = 1.0 - ((1.0 - maps[0]) * (1.0 - w4 * p4) *
                       (1.0 - w5 * p5))
        return fused.clamp_(0.0, 1.0)

    def forward(self,
                feats,
                txt_feats=None,
                txt_masks=None,
                return_parts: bool = None):
        del txt_feats, txt_masks
        if return_parts is None:
            return_parts = self.return_parts
        if len(feats) != len(self.region_attentions):
            raise ValueError('Feature levels do not match configured channels.')

        gamma_scale = (
            self._gamma_warmup_scale() if self.direct_feature_boost else 0.0)
        caqr_beta = self._caqr_beta() if self.use_caqr else 0.0
        out_feats, masks = [], []
        regional_priors, caqr_masks = [], []
        recalib_maps, gammas = [], []
        for feat, attention, token_norm, caqr_mask_conv, gamma_mlp in zip(
                feats, self.region_attentions, self.region_norms,
                self.caqr_mask_convs, self.gamma_mlps):
            regional_prior = self._regional_foreground_prior(
                feat, attention, token_norm)
            if self.use_caqr:
                caqr_mask = torch.sigmoid(caqr_mask_conv(feat))
                # P_safe = 1 - (1 - P_region) * (1 - beta * C).
                foreground_prior = 1.0 - (
                    (1.0 - regional_prior) * (1.0 - caqr_beta * caqr_mask))
                foreground_prior = foreground_prior.clamp_(0.0, 1.0)
            else:
                caqr_mask = torch.zeros_like(regional_prior)
                foreground_prior = regional_prior
            reliability = foreground_prior
            if self.direct_feature_boost:
                gamma = self.max_gamma * torch.sigmoid(gamma_mlp(feat))
                gamma = gamma.view(-1, 1, 1, 1) * gamma_scale
                out_feats.append(feat * (1.0 + gamma * reliability))
            else:
                gamma = feat.new_zeros((feat.shape[0], 1, 1, 1))
                out_feats.append(feat)
            masks.append(foreground_prior)
            regional_priors.append(regional_prior)
            caqr_masks.append(caqr_mask)
            recalib_maps.append(reliability)
            gammas.append(gamma)

        out_feats = tuple(out_feats)
        if not return_parts:
            return out_feats
        fused = self._fuse_reliability_maps(recalib_maps)
        return out_feats, dict(
            masks=masks,
            regional_priors=regional_priors,
            caqr_masks=caqr_masks,
            caqr_beta=caqr_beta,
            recalib_maps=recalib_maps,
            reliability_maps=recalib_maps,
            fused_reliability=fused,
            gammas=gammas)


@MODELS.register_module()
class RegionalTextResidual(BaseModule):
    """Inject a small text residual only at reliable P3 locations.

    The original feature is always preserved.  A class-agnostic Regional
    reliability map and a text-similarity gate jointly route the residual.
    Reliability is detached by default so detection classification gradients
    cannot turn the Regional branch into a base-class gate.
    """

    def __init__(self,
                 in_channels: int,
                 text_channels: int,
                 embed_channels: int = 64,
                 max_residual_scale: float = 0.1,
                 warmup_iters: int = 1000,
                 logit_scale: float = 10.0,
                 detach_reliability: bool = True,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        if embed_channels <= 0:
            raise ValueError('embed_channels must be positive.')
        if max_residual_scale < 0:
            raise ValueError('max_residual_scale must be non-negative.')

        self.max_residual_scale = float(max_residual_scale)
        self.warmup_iters = int(warmup_iters)
        self.logit_scale = float(logit_scale)
        self.detach_reliability = bool(detach_reliability)

        self.query_proj = nn.Conv2d(
            in_channels, embed_channels, kernel_size=1, bias=False)
        self.key_proj = nn.Linear(text_channels, embed_channels, bias=False)
        self.value_proj = nn.Linear(
            text_channels, embed_channels, bias=False)
        self.out_proj = nn.Conv2d(
            embed_channels, in_channels, kernel_size=1, bias=False)

        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.xavier_uniform_(self.key_proj.weight)
        nn.init.xavier_uniform_(self.value_proj.weight)
        # Identity initialization protects the original YOLO-World path.
        nn.init.constant_(self.out_proj.weight, 0.0)
        self.register_buffer(
            '_warmup_step', torch.zeros((), dtype=torch.long),
            persistent=True)

    def _residual_scale(self) -> float:
        if self.warmup_iters <= 0:
            return self.max_residual_scale
        if self.training:
            self._warmup_step.add_(1)
        ratio = min(
            float(self._warmup_step.item()) / float(self.warmup_iters), 1.0)
        return self.max_residual_scale * ratio

    def forward(self,
                feat: Tensor,
                txt_feats: Tensor,
                reliability: Tensor,
                txt_masks: Tensor = None) -> Tensor:
        if txt_feats is None or reliability is None:
            return feat
        if txt_feats.dim() == 2:
            txt_feats = txt_feats.unsqueeze(0)
        if txt_feats.shape[0] == 1 and feat.shape[0] > 1:
            txt_feats = txt_feats.expand(feat.shape[0], -1, -1)
        if txt_feats.shape[0] != feat.shape[0]:
            raise ValueError('Text and image batch sizes must match.')

        batch_size, _, height, width = feat.shape
        queries = self.query_proj(feat).flatten(2).transpose(1, 2)
        keys = self.key_proj(txt_feats)
        queries = F.normalize(queries, dim=-1)
        keys = F.normalize(keys, dim=-1)
        logits = torch.einsum('bnd,bcd->bnc', queries, keys)
        logits = logits * self.logit_scale

        valid_mask = None
        if txt_masks is not None:
            valid_mask = txt_masks.to(device=logits.device, dtype=torch.bool)
            if valid_mask.dim() == 1:
                valid_mask = valid_mask.unsqueeze(0)
            if valid_mask.shape[0] == 1 and batch_size > 1:
                valid_mask = valid_mask.expand(batch_size, -1)
            if valid_mask.shape != logits.shape[:1] + logits.shape[2:]:
                raise ValueError('Text mask shape does not match text features.')
            logits = logits.masked_fill(
                ~valid_mask[:, None, :], torch.finfo(logits.dtype).min)

        attention = logits.softmax(dim=-1)
        values = self.value_proj(txt_feats)
        context = torch.einsum('bnc,bcd->bnd', attention, values)
        context = context.transpose(1, 2).reshape(
            batch_size, -1, height, width)
        delta = self.out_proj(context)

        text_gate = torch.sigmoid(logits.max(dim=-1).values)
        text_gate = text_gate.reshape(batch_size, 1, height, width)
        if reliability.shape[-2:] != (height, width):
            reliability = F.interpolate(
                reliability,
                size=(height, width),
                mode='bilinear',
                align_corners=False)
        if self.detach_reliability:
            reliability = reliability.detach()
        route = reliability.to(dtype=feat.dtype).clamp(0.0, 1.0) * text_gate
        return feat + self._residual_scale() * route * delta


@MODELS.register_module()
class WeakSupportRetriever(BaseModule):
    """Reliability-guided same-image support retrieval for weak P3 features.

    This module keeps the open-vocabulary head unchanged. It uses the
    class-agnostic reliability map produced by the foreground/background gate
    to build a compact same-image memory, retrieves visual support for every
    P3 location, and applies a zero-initialized residual refinement to P3 only.
    """

    def __init__(self,
                 in_channels: List[int],
                 embed_channels: int = 64,
                 num_reliable_tokens: int = 128,
                 num_detail_tokens: int = 64,
                 grid_size: int = 8,
                 reliability_nms_kernel: int = 7,
                 detail_kernel: int = 5,
                 exclusion_radius: int = 1,
                 reliability_log_weight: float = 0.5,
                 support_dropout: float = 0.2,
                 residual_scale: float = 0.1,
                 detach_reliability: bool = True,
                 enhance_level: int = 0,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        assert len(in_channels) == 3, 'WSR expects P3/P4/P5 features.'
        assert enhance_level == 0, 'WSR v1 only refines P3.'
        self.embed_channels = embed_channels
        self.num_reliable_tokens = int(num_reliable_tokens)
        self.num_detail_tokens = int(num_detail_tokens)
        self.grid_size = int(grid_size)
        self.num_grid_tokens = self.grid_size * self.grid_size
        self.num_support_tokens = (
            self.num_reliable_tokens + self.num_detail_tokens +
            self.num_grid_tokens)
        self.reliability_nms_kernel = int(reliability_nms_kernel)
        self.detail_kernel = int(detail_kernel)
        self.exclusion_radius = int(exclusion_radius)
        self.reliability_log_weight = float(reliability_log_weight)
        self.support_dropout = float(support_dropout)
        self.residual_scale = float(residual_scale)
        self.detach_reliability = detach_reliability

        c3, c4, c5 = in_channels
        self.q_convs = nn.ModuleList([
            nn.Conv2d(c3, embed_channels, 1),
            nn.Conv2d(c4, embed_channels, 1),
            nn.Conv2d(c5, embed_channels, 1),
        ])
        self.k_convs = nn.ModuleList([
            nn.Conv2d(c3, embed_channels, 1),
            nn.Conv2d(c4, embed_channels, 1),
            nn.Conv2d(c5, embed_channels, 1),
        ])
        self.v_conv = nn.Conv2d(c3, embed_channels, 1)
        self.q_norm = nn.LayerNorm(embed_channels)
        self.k_norm = nn.LayerNorm(embed_channels)
        self.gate_mlp = nn.Sequential(
            nn.Linear(embed_channels * 2 + 2, embed_channels),
            nn.SiLU(inplace=True),
            nn.Linear(embed_channels, 1),
        )
        self.out_proj = nn.Linear(embed_channels, c3)
        nn.init.constant_(self.out_proj.weight, 0.0)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def _project_multiscale(self, feats: Tuple[Tensor, Tensor, Tensor],
                            convs: nn.ModuleList) -> Tensor:
        p3, p4, p5 = feats
        size = p3.shape[-2:]
        out = convs[0](p3)
        out = out + F.interpolate(
            convs[1](p4), size=size, mode='bilinear', align_corners=False)
        out = out + F.interpolate(
            convs[2](p5), size=size, mode='bilinear', align_corners=False)
        return out

    def _local_max_topk(self, scores: Tensor, k: int,
                        nms_kernel: int) -> Tensor:
        b, _, h, w = scores.shape
        if k <= 0:
            return scores.new_zeros((b, 0), dtype=torch.long)
        flat_size = h * w
        topk = min(k, flat_size)
        if nms_kernel > 1:
            pooled = F.max_pool2d(
                scores, nms_kernel, stride=1, padding=nms_kernel // 2)
            keep = scores >= pooled
            scores = scores.masked_fill(~keep, -torch.inf)
        return scores.flatten(1).topk(topk, dim=1).indices

    def _grid_indices(self, h: int, w: int, device: torch.device) -> Tensor:
        y = ((torch.arange(self.grid_size, device=device, dtype=torch.float32) +
              0.5) * h / self.grid_size).long().clamp_(0, h - 1)
        x = ((torch.arange(self.grid_size, device=device, dtype=torch.float32) +
              0.5) * w / self.grid_size).long().clamp_(0, w - 1)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        return (yy.reshape(-1) * w + xx.reshape(-1)).long()

    def _select_support_indices(self, p3: Tensor,
                                reliability: Tensor) -> Tensor:
        b, _, h, w = p3.shape
        target = self.num_support_tokens

        reliable_idx = self._local_max_topk(
            reliability, self.num_reliable_tokens,
            self.reliability_nms_kernel)

        detail = p3 - F.avg_pool2d(
            p3,
            kernel_size=self.detail_kernel,
            stride=1,
            padding=self.detail_kernel // 2)
        detail_scores = detail.square().mean(dim=1, keepdim=True).sqrt()
        detail_idx = self._local_max_topk(
            detail_scores, self.num_detail_tokens, self.reliability_nms_kernel)

        grid_idx = self._grid_indices(h, w, p3.device)
        fallback_k = min(max(target, 1), h * w)
        fallback_idx = reliability.flatten(1).topk(fallback_k, dim=1).indices

        candidates = torch.cat([
            reliable_idx,
            detail_idx,
            grid_idx.view(1, -1).expand(b, -1),
            fallback_idx,
        ],
                               dim=1)
        num_candidates = candidates.shape[1]
        order = torch.arange(num_candidates, device=p3.device)
        same = candidates.unsqueeze(2) == candidates.unsqueeze(1)
        first_order = torch.where(same,
                                  order.view(1, 1, num_candidates),
                                  num_candidates).min(dim=2).values
        keep = first_order == order.view(1, num_candidates)
        ordered_keep = torch.where(keep, order.view(1, num_candidates),
                                   num_candidates)
        select_k = min(target, num_candidates)
        select_pos = ordered_keep.topk(
            select_k, dim=1, largest=False).indices
        selected = torch.gather(candidates, 1, select_pos)
        if select_k < target:
            pad = selected[:, :1].expand(-1, target - select_k)
            selected = torch.cat([selected, pad], dim=1)
        return selected[:, :target]

    def _gather_tokens(self, token_map: Tensor, indices: Tensor) -> Tensor:
        b, c, h, w = token_map.shape
        flat = token_map.flatten(2).transpose(1, 2)
        gather_idx = indices.unsqueeze(-1).expand(-1, -1, c)
        return torch.gather(flat, 1, gather_idx)

    def forward(self, p3: Tensor, p4: Tensor, p5: Tensor,
                reliability: Tensor) -> Tensor:
        """Refine P3 with sparse same-image support tokens."""
        if self.detach_reliability:
            reliability = reliability.detach()
        reliability = reliability.clamp(min=1e-6, max=1.0)

        b, c, h, w = p3.shape
        support_idx = self._select_support_indices(p3, reliability)
        support_scores = torch.gather(
            reliability.flatten(1), 1, support_idx).clamp(min=1e-6)

        feats = (p3, p4, p5)
        q_map = self._project_multiscale(feats, self.q_convs)
        k_map = self._project_multiscale(feats, self.k_convs)
        v_map = self.v_conv(p3)

        q = q_map.flatten(2).transpose(1, 2)
        k = k_map.flatten(2).transpose(1, 2)
        q = self.q_norm(q)
        k = self.k_norm(k)
        v = v_map.flatten(2).transpose(1, 2)

        support_k = torch.gather(
            k, 1,
            support_idx.unsqueeze(-1).expand(-1, -1, self.embed_channels))
        support_v = torch.gather(
            v, 1,
            support_idx.unsqueeze(-1).expand(-1, -1, self.embed_channels))

        logits = torch.bmm(q, support_k.transpose(1, 2))
        logits = logits * (self.embed_channels**-0.5)
        logits = logits + (
            self.reliability_log_weight *
            support_scores.log().unsqueeze(1))

        if self.exclusion_radius >= 0:
            y = torch.arange(h, device=p3.device).repeat_interleave(w)
            x = torch.arange(w, device=p3.device).repeat(h)
            support_y = support_idx // w
            support_x = support_idx % w
            near = (
                (y.view(1, -1, 1) - support_y.view(b, 1, -1)).abs() <=
                self.exclusion_radius) & (
                    (x.view(1, -1, 1) - support_x.view(b, 1, -1)).abs() <=
                    self.exclusion_radius)
            logits = logits.masked_fill(near, -1e4)

        if self.training and self.support_dropout > 0:
            keep = torch.rand(
                b,
                self.num_support_tokens,
                device=p3.device) >= self.support_dropout
            empty = ~keep.any(dim=1)
            if empty.any():
                keep[empty, 0] = True
            logits = logits.masked_fill(~keep.unsqueeze(1), -1e4)

        attn = logits.softmax(dim=-1)
        retrieved = torch.bmm(attn, support_v)
        peak = attn.max(dim=-1, keepdim=True).values
        entropy = -(attn * (attn + 1e-6).log()).sum(dim=-1, keepdim=True)
        entropy = entropy / math.log(float(self.num_support_tokens))
        gate = torch.sigmoid(
            self.gate_mlp(torch.cat([q, retrieved, peak, entropy], dim=-1)))

        update = self.out_proj(retrieved) * gate
        update = update.transpose(1, 2).reshape(b, c, h, w)
        return p3 + self.residual_scale * update


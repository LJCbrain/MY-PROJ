# Copyright (c) Tencent Inc. All rights reserved.
from typing import List

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


@MODELS.register_module()  # 新增代码：注册类别无关前景门控模块
class ClassAgnosticForegroundGate(BaseModule):  # 新增代码：neck 内使用的前景感知特征门控
    """Class-agnostic foreground gate for multi-level neck features."""  # 新增代码：说明该模块不做分类

    def __init__(self,  # 新增代码：初始化多尺度 mask 分支
                 in_channels: List[int],  # 新增代码：每个输出尺度的通道数
                 hidden_ratio: float = 0.25,  # 新增代码：mask 分支隐藏通道比例
                 min_hidden: int = 16,  # 新增代码：mask 分支最小隐藏通道数
                 init_gamma: float = 0.01,  # 新增代码：残差门控初始强度
                 norm_cfg: ConfigType = dict(type='BN',  # 新增代码：BN 类型
                                             momentum=0.03,  # 新增代码：BN momentum
                                             eps=0.001),  # 新增代码：沿用项目 BN 配置
                 act_cfg: ConfigType = dict(type='SiLU',  # 新增代码：SiLU 激活
                                            inplace=True),  # 新增代码：沿用项目激活配置
                 init_cfg: OptMultiConfig = None) -> None:  # 新增代码：兼容 BaseModule 初始化
        super().__init__(init_cfg=init_cfg)  # 新增代码：初始化 BaseModule
        self.mask_convs = nn.ModuleList()  # 新增代码：保存各尺度 mask 预测分支
        self.gammas = nn.ParameterList()  # 新增代码：保存各尺度可学习残差门控强度

        for c in in_channels:  # 新增代码：为每个尺度构建独立类别无关 mask 分支
            hidden = max(min_hidden, int(c * hidden_ratio))  # 新增代码：计算隐藏层通道数
            self.mask_convs.append(  # 新增代码：添加当前尺度的 mask 分支
                nn.Sequential(  # 新增代码：局部 objectness 预测结构
                    ConvModule(c,  # 新增代码：输入通道数
                               hidden,  # 新增代码：隐藏通道数
                               3,  # 新增代码：3x3 卷积核
                               padding=1,  # 新增代码：保持空间尺寸
                               norm_cfg=norm_cfg,  # 新增代码：归一化配置
                               act_cfg=act_cfg),  # 新增代码：3x3 局部特征提取
                    nn.Conv2d(hidden, 1, 1)))  # 新增代码：输出 [B,1,H,W] 类别无关 logit
            self.gammas.append(  # 新增代码：添加当前尺度的可学习残差强度
                nn.Parameter(torch.tensor(float(init_gamma))))  # 新增代码：初始化 gamma

    def forward(self,  # 新增代码：执行类别无关前景门控
                feats,  # 新增代码：输入 tuple/list 多尺度特征
                txt_feats=None,  # 新增代码：保留文本接口，第一阶段不使用
                txt_masks=None,  # 新增代码：保留文本 mask 接口，第一阶段不使用
                return_parts: bool = False):  # 新增代码：是否返回 mask/fg/bg 调试信息
        """Forward multi-level features without producing class logits."""  # 新增代码：强调不做分类
        assert len(feats) == len(self.mask_convs)  # 新增代码：确保尺度数量匹配
        out_feats = []  # 新增代码：保存门控后的特征
        masks = []  # 新增代码：保存类别无关前景 mask
        fg_feats = []  # 新增代码：保存前景分量，供后续可视化或辅助 loss 使用
        bg_feats = []  # 新增代码：保存背景分量，供后续可视化或辅助 loss 使用

        for feat, mask_conv, gamma in zip(feats, self.mask_convs, self.gammas):  # 新增代码：逐尺度处理
            logit = mask_conv(feat)  # 新增代码：预测 [B,1,H,W] 类别无关前景 logit
            mask = torch.sigmoid(logit)  # 新增代码：得到类别无关前景 mask
            fg = feat * mask  # 新增代码：提取前景增强分量
            bg = feat * (1.0 - mask)  # 新增代码：提取背景分量
            out = feat + gamma * fg  # 新增代码：残差门控，保持 head 输入通道不变

            out_feats.append(out)  # 新增代码：收集当前尺度输出
            masks.append(mask)  # 新增代码：收集当前尺度 mask
            fg_feats.append(fg)  # 新增代码：收集当前尺度前景特征
            bg_feats.append(bg)  # 新增代码：收集当前尺度背景特征

        out_feats = tuple(out_feats)  # 新增代码：保持 neck 输出为 tuple[Tensor]
        if return_parts:  # 新增代码：可选返回调试信息，不传给 head
            return out_feats, dict(masks=masks,  # 新增代码：返回 mask 列表
                                   fg_feats=fg_feats,  # 新增代码：返回前景特征列表
                                   bg_feats=bg_feats)  # 新增代码：返回背景特征列表
        return out_feats  # 新增代码：默认只返回 tuple[Tensor]


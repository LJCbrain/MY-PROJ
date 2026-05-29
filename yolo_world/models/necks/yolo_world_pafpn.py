# Copyright (c) Tencent Inc. All rights reserved.
import copy
from typing import List, Union

import torch
import torch.nn as nn
from torch import Tensor
from mmdet.utils import ConfigType, OptMultiConfig

from mmyolo.registry import MODELS
from mmyolo.models.utils import make_divisible, make_round
from mmyolo.models.necks.yolov8_pafpn import YOLOv8PAFPN


@MODELS.register_module()
class YOLOWorldPAFPN(YOLOv8PAFPN):
    """Path Aggregation Network used in YOLO World
    Following YOLOv8 PAFPN, including text to image fusion
    """
    def __init__(self,
                 in_channels: List[int],
                 out_channels: Union[List[int], int],
                 guide_channels: int,
                 embed_channels: List[int],
                 num_heads: List[int],
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 num_csp_blocks: int = 3,
                 freeze_all: bool = False,
                 block_cfg: ConfigType = dict(type='CSPLayerWithTwoConv'),
                 norm_cfg: ConfigType = dict(type='BN',
                                             momentum=0.03,
                                             eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 fgbg_cfg: ConfigType = None,  # 新增代码：可选类别无关前景门控配置
                 init_cfg: OptMultiConfig = None) -> None:
        self.guide_channels = guide_channels
        self.embed_channels = embed_channels
        self.num_heads = num_heads
        self.block_cfg = block_cfg
        super().__init__(in_channels=in_channels,
                         out_channels=out_channels,
                         deepen_factor=deepen_factor,
                         widen_factor=widen_factor,
                         num_csp_blocks=num_csp_blocks,
                         freeze_all=freeze_all,
                         norm_cfg=norm_cfg,
                         act_cfg=act_cfg,
                         init_cfg=init_cfg)
        self.fgbg_gate = (  # 新增代码：构建 neck 内部前景门控模块
            MODELS.build(fgbg_cfg) if fgbg_cfg is not None else None)  # 新增代码：无配置时保持原 neck 行为
        self._fgbg_parts = None  # 新增代码：缓存 mask/fg/bg，供后续调试或辅助 loss 使用

    def build_top_down_layer(self, idx: int) -> nn.Module:
        """build top down layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The top down layer.
        """
        block_cfg = copy.deepcopy(self.block_cfg)
        block_cfg.update(
            dict(in_channels=make_divisible(
                (self.in_channels[idx - 1] + self.in_channels[idx]),
                self.widen_factor),
                 out_channels=make_divisible(self.out_channels[idx - 1],
                                             self.widen_factor),
                 guide_channels=self.guide_channels,
                 embed_channels=make_round(self.embed_channels[idx - 1],
                                           self.widen_factor),
                 num_heads=make_round(self.num_heads[idx - 1],
                                      self.widen_factor),
                 num_blocks=make_round(self.num_csp_blocks,
                                       self.deepen_factor),
                 add_identity=False,
                 norm_cfg=self.norm_cfg,
                 act_cfg=self.act_cfg))
        return MODELS.build(block_cfg)

    def build_bottom_up_layer(self, idx: int) -> nn.Module:
        """build bottom up layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The bottom up layer.
        """
        block_cfg = copy.deepcopy(self.block_cfg)
        block_cfg.update(
            dict(in_channels=make_divisible(
                (self.out_channels[idx] + self.out_channels[idx + 1]),
                self.widen_factor),
                 out_channels=make_divisible(self.out_channels[idx + 1],
                                             self.widen_factor),
                 guide_channels=self.guide_channels,
                 embed_channels=make_round(self.embed_channels[idx + 1],
                                           self.widen_factor),
                 num_heads=make_round(self.num_heads[idx + 1],
                                      self.widen_factor),
                 num_blocks=make_round(self.num_csp_blocks,
                                       self.deepen_factor),
                 add_identity=False,
                 norm_cfg=self.norm_cfg,
                 act_cfg=self.act_cfg))
        return MODELS.build(block_cfg)

    def forward(self, img_feats: List[Tensor], txt_feats: Tensor = None) -> tuple:
        """Forward function.
        including multi-level image features, text features: BxLxD
        """
        assert len(img_feats) == len(self.in_channels)
        # reduce layers
        reduce_outs = []
        for idx in range(len(self.in_channels)):
            reduce_outs.append(self.reduce_layers[idx](img_feats[idx]))

        # top-down path
        inner_outs = [reduce_outs[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = reduce_outs[idx - 1]
            upsample_feat = self.upsample_layers[len(self.in_channels) - 1 -
                                                 idx](feat_high)
            if self.upsample_feats_cat_first:
                top_down_layer_inputs = torch.cat([upsample_feat, feat_low], 1)
            else:
                top_down_layer_inputs = torch.cat([feat_low, upsample_feat], 1)
            inner_out = self.top_down_layers[len(self.in_channels) - 1 - idx](
                top_down_layer_inputs, txt_feats)
            inner_outs.insert(0, inner_out)

        # bottom-up path
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            downsample_feat = self.downsample_layers[idx](feat_low)
            out = self.bottom_up_layers[idx](torch.cat(
                [downsample_feat, feat_high], 1), txt_feats)
            outs.append(out)

        # out_layers
        results = []
        for idx in range(len(self.in_channels)):
            results.append(self.out_layers[idx](outs[idx]))

        results = tuple(results)  # 新增代码：保持 neck 输出为 tuple[Tensor]
        if self.fgbg_gate is not None:  # 新增代码：可选执行类别无关前景门控
            results, self._fgbg_parts = self.fgbg_gate(  # 新增代码：缓存 parts，但只返回特征给 head
                results,  # 新增代码：传入 tuple[Tensor] 多尺度特征
                txt_feats=txt_feats,  # 新增代码：保留文本接口，第一阶段模块内部不使用
                txt_masks=None,  # 新增代码：第一阶段不使用文本 mask
                return_parts=True)  # 新增代码：返回调试 parts 但不传给 head

        return results  # 新增代码：最终仍只返回 tuple[Tensor]


@MODELS.register_module()
class YOLOWorldDualPAFPN(YOLOWorldPAFPN):
    """Path Aggregation Network used in YOLO World v8."""
    def __init__(self,
                 in_channels: List[int],
                 out_channels: Union[List[int], int],
                 guide_channels: int,
                 embed_channels: List[int],
                 num_heads: List[int],
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 num_csp_blocks: int = 3,
                 freeze_all: bool = False,
                 text_enhancder: ConfigType = dict(
                     type='ImagePoolingAttentionModule',
                     embed_channels=256,
                     num_heads=8,
                     pool_size=3),
                 block_cfg: ConfigType = dict(type='CSPLayerWithTwoConv'),
                 norm_cfg: ConfigType = dict(type='BN',
                                             momentum=0.03,
                                             eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 fgbg_cfg: ConfigType = None,  # 新增代码：兼容类别无关前景门控配置
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(in_channels=in_channels,
                         out_channels=out_channels,
                         guide_channels=guide_channels,
                         embed_channels=embed_channels,
                         num_heads=num_heads,
                         deepen_factor=deepen_factor,
                         widen_factor=widen_factor,
                         num_csp_blocks=num_csp_blocks,
                         freeze_all=freeze_all,
                         block_cfg=block_cfg,
                         norm_cfg=norm_cfg,
                         act_cfg=act_cfg,
                         fgbg_cfg=fgbg_cfg,  # 新增代码：传递前景门控配置给父类
                         init_cfg=init_cfg)

        text_enhancder.update(
            dict(
                image_channels=[int(x * widen_factor) for x in out_channels],
                text_channels=guide_channels,
                num_feats=len(out_channels),
            ))
        print(text_enhancder)
        self.text_enhancer = MODELS.build(text_enhancder)

    def forward(self, img_feats: List[Tensor], txt_feats: Tensor) -> tuple:
        """Forward function."""
        assert len(img_feats) == len(self.in_channels)
        # reduce layers
        reduce_outs = []
        for idx in range(len(self.in_channels)):
            reduce_outs.append(self.reduce_layers[idx](img_feats[idx]))

        # top-down path
        inner_outs = [reduce_outs[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = reduce_outs[idx - 1]
            upsample_feat = self.upsample_layers[len(self.in_channels) - 1 -
                                                 idx](feat_high)
            if self.upsample_feats_cat_first:
                top_down_layer_inputs = torch.cat([upsample_feat, feat_low], 1)
            else:
                top_down_layer_inputs = torch.cat([feat_low, upsample_feat], 1)
            inner_out = self.top_down_layers[len(self.in_channels) - 1 - idx](
                top_down_layer_inputs, txt_feats)
            inner_outs.insert(0, inner_out)

        txt_feats = self.text_enhancer(txt_feats, inner_outs)
        # bottom-up path
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            downsample_feat = self.downsample_layers[idx](feat_low)
            out = self.bottom_up_layers[idx](torch.cat(
                [downsample_feat, feat_high], 1), txt_feats)
            outs.append(out)

        # out_layers
        results = []
        for idx in range(len(self.in_channels)):
            results.append(self.out_layers[idx](outs[idx]))

        results = tuple(results)  # 新增代码：保持 neck 输出为 tuple[Tensor]
        if self.fgbg_gate is not None:  # 新增代码：DualPAFPN 也应用类别无关前景门控
            results, self._fgbg_parts = self.fgbg_gate(  # 新增代码：缓存 parts，但只返回特征给 head
                results,  # 新增代码：传入 tuple[Tensor] 多尺度特征
                txt_feats=txt_feats,  # 新增代码：保留文本接口，第一阶段模块内部不使用
                txt_masks=None,  # 新增代码：第一阶段不使用文本 mask
                return_parts=True)  # 新增代码：返回调试 parts 但不传给 head

        return results  # 新增代码：最终仍只返回 tuple[Tensor]



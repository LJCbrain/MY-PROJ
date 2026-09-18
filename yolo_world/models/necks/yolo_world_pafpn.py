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
                 prompt_bank_compatible: bool = False,
                 init_cfg: OptMultiConfig = None) -> None:
        self.guide_channels = guide_channels
        self.embed_channels = embed_channels
        self.num_heads = num_heads
        self.block_cfg = block_cfg
        self.prompt_bank_compatible = bool(prompt_bank_compatible)
        super().__init__(in_channels=in_channels,
                         out_channels=out_channels,
                         deepen_factor=deepen_factor,
                         widen_factor=widen_factor,
                         num_csp_blocks=num_csp_blocks,
                         freeze_all=freeze_all,
                         norm_cfg=norm_cfg,
                         act_cfg=act_cfg,
                         init_cfg=init_cfg)
        self.fgbg_gate = (
            MODELS.build(fgbg_cfg) if fgbg_cfg is not None else None)
        self._fgbg_parts = None

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

    def _get_bottom_up_text(self, txt_feats: Tensor,
                            inner_outs: List[Tensor]) -> Tensor:
        """Return the text features used to guide the bottom-up path.

        Subclasses can override this hook to condition the neck guidance on
        the current top-down visual features without changing the text
        features that are returned to, or independently adapted for, the
        detection head.
        """
        return txt_feats

    def forward(self,
                img_feats: List[Tensor],
                txt_feats: Tensor = None,
                txt_masks: Tensor = None) -> tuple:
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

        # Keep neck-side text conditioning local to the bottom-up path.  In
        # particular, GuardPrompt still starts from the original ``txt_feats``
        # after the neck and therefore runs in parallel with this branch.
        bottom_up_txt_feats = self._get_bottom_up_text(txt_feats, inner_outs)

        # bottom-up path
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            downsample_feat = self.downsample_layers[idx](feat_low)
            out = self.bottom_up_layers[idx](torch.cat(
                [downsample_feat, feat_high], 1), bottom_up_txt_feats)
            outs.append(out)

        # out_layers
        results = []
        for idx in range(len(self.in_channels)):
            results.append(self.out_layers[idx](outs[idx]))

        results = tuple(results)
        self._fgbg_parts = None
        if self.fgbg_gate is not None:
            results, self._fgbg_parts = self.fgbg_gate(
                results,
                txt_feats=txt_feats,
                txt_masks=txt_masks,
                return_parts=True)

        return results

    def get_spatial_prompt_routing_context(self):
        """Return detached foreground gates and reliability ranking maps."""
        if not self._fgbg_parts:
            raise RuntimeError(
                'CSPR context is unavailable before the reliability gate '
                'runs.')
        foreground_maps = self._fgbg_parts.get('masks', None)
        reliability_maps = self._fgbg_parts.get('reliability_maps', None)
        if not foreground_maps or not reliability_maps:
            raise RuntimeError(
                'CSPR requires foreground and reliability maps from the neck.')
        if len(foreground_maps) != len(reliability_maps):
            raise RuntimeError(
                'CSPR foreground and reliability levels do not match.')
        return dict(
            foreground_maps=tuple(
                route.detach() for route in foreground_maps),
            reliability_maps=tuple(
                route.detach() for route in reliability_maps))


@MODELS.register_module()
class YOLOWorldWSRPAFPN(YOLOWorldPAFPN):
    """YOLO-World PAFPN with reliability-guided weak support retrieval.

    An optional I-Pooling branch conditions the text guidance used only by the
    bottom-up PAFPN.  GuardPrompt remains an independent post-neck text branch
    for the detection head.

    The foreground/background gate is applied in ``YOLOWorldPAFPN.forward``.
    This subclass keeps its cached parts and uses the fused multi-scale
    reliability map to refine P3 with same-image visual support. P4/P5 and the
    detection head stay unchanged.
    """

    def __init__(self,
                 *args,
                 wsr_cfg: ConfigType = None,
                 ipooling_cfg: ConfigType = None,
                 text_residual_cfg: ConfigType = None,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.wsr = MODELS.build(wsr_cfg) if wsr_cfg is not None else None
        if ipooling_cfg is not None:
            ipooling_cfg = copy.deepcopy(ipooling_cfg)
            ipooling_cfg.setdefault(
                'image_channels',
                [int(x * self.widen_factor) for x in self.out_channels])
            ipooling_cfg.setdefault('text_channels', self.guide_channels)
            ipooling_cfg.setdefault('num_feats', len(self.out_channels))
            # Keep the original YOLO-World attribute name so pretrained
            # ``neck.text_enhancer.*`` I-Pooling weights remain loadable.
            self.text_enhancer = MODELS.build(ipooling_cfg)
        else:
            self.text_enhancer = None
        self.text_residual = (
            MODELS.build(text_residual_cfg)
            if text_residual_cfg is not None else None)
        self._prompt_bank_context = None

    def _get_bottom_up_text(self, txt_feats: Tensor,
                            inner_outs: List[Tensor]) -> Tensor:
        """Apply neck-local I-Pooling without feeding it into GuardPrompt."""
        if self.text_enhancer is None or txt_feats is None:
            return txt_feats
        return self.text_enhancer(txt_feats, inner_outs)

    def forward(self,
                img_feats: List[Tensor],
                txt_feats: Tensor = None,
                txt_masks: Tensor = None) -> tuple:
        self._prompt_bank_context = None
        results = super().forward(img_feats, txt_feats, txt_masks=txt_masks)
        if len(results) < 3:
            return results
        if not self._fgbg_parts or 'reliability_maps' not in self._fgbg_parts:
            # A prompt-bank ablation may intentionally disable every visual
            # prior.  In that case expose the ordinary neck features and an
            # explicit ``None`` reliability input instead of making prompted
            # text encoding depend on a CAQR/Regional gate.
            self._prompt_bank_context = (
                tuple(feat.detach() for feat in results), None)
            return results
        reliability_maps = self._fgbg_parts['reliability_maps']
        if not reliability_maps:
            self._prompt_bank_context = (
                tuple(feat.detach() for feat in results), None)
            return results

        fused_reliability = self._fgbg_parts.get(
            'fused_reliability', reliability_maps[0])
        p3 = results[0]
        if self.wsr is not None:
            p3 = self.wsr(p3, results[1], results[2],
                          fused_reliability.detach())
        if self.text_residual is not None:
            p3 = self.text_residual(
                p3,
                txt_feats,
                fused_reliability,
                txt_masks=txt_masks)
        results = (p3, *results[1:])
        # GuardPrompt consumes the current multi-scale visual features and
        # reliability maps, but it must not send its text-side gradients back
        # into C2F/Regional/WSR.  The detection head keeps using ``results``
        # directly, so its normal visual supervision is unaffected.
        self._prompt_bank_context = (
            tuple(feat.detach() for feat in results),
            tuple(route.detach() for route in reliability_maps))
        return results

    def get_prompt_bank_context(self):
        if self._prompt_bank_context is None:
            raise RuntimeError(
                'Prompt-bank context is unavailable. Run the neck with '
                'image features before encoding prompted text.')
        return self._prompt_bank_context

    def get_proposal_prior_maps(self):
        """Return the per-level class-agnostic priors from the last forward."""
        if not self._fgbg_parts:
            return None
        return tuple(self._fgbg_parts.get('reliability_maps', ()))


class _ParallelVisualResidualFusion(nn.Module):
    """Fuse a bounded text-free visual residual into text-aware features."""

    def __init__(self,
                 visual_channels: List[int],
                 output_channels: List[int],
                 max_scale: float = 0.1,
                 init_scale: float = 0.05,
                 warmup_iters: int = 1000,
                 detach_prior: bool = True) -> None:
        super().__init__()
        if len(visual_channels) != len(output_channels):
            raise ValueError(
                'visual_channels and output_channels must have equal length.')
        if max_scale <= 0:
            raise ValueError('max_scale must be positive.')
        if not 0 <= init_scale <= max_scale:
            raise ValueError('init_scale must be in [0, max_scale].')
        if warmup_iters < 0:
            raise ValueError('warmup_iters must be non-negative.')

        self.max_scale = float(max_scale)
        self.warmup_iters = int(warmup_iters)
        self.detach_prior = bool(detach_prior)
        self.projections = nn.ModuleList()
        for in_channels, out_channels in zip(
                visual_channels, output_channels):
            projection = nn.Sequential(
                nn.Conv2d(
                    int(in_channels),
                    int(out_channels),
                    kernel_size=3,
                    padding=1,
                    bias=False),
                nn.BatchNorm2d(int(out_channels), eps=0.001, momentum=0.03),
                nn.SiLU(inplace=True),
                nn.Conv2d(
                    int(out_channels),
                    int(out_channels),
                    kernel_size=1,
                    bias=False))
            # Preserve the pretrained YOLO-World path at initialization.  The
            # final projection still receives gradients through the bounded
            # non-zero scale and learns the visual residual gradually.
            nn.init.constant_(projection[-1].weight, 0.0)
            self.projections.append(projection)

        ratio = min(max(init_scale / self.max_scale, 1e-4), 1 - 1e-4)
        scale_logits = torch.full(
            (len(visual_channels), ), float(ratio), dtype=torch.float32)
        self.scale_logits = nn.Parameter(torch.logit(scale_logits))
        self.register_buffer(
            '_fusion_step', torch.zeros((), dtype=torch.long), persistent=True)
        self.last_scales = torch.zeros(len(visual_channels))

    def _warmup_scale(self) -> float:
        if self.warmup_iters <= 0:
            return 1.0
        if self.training:
            self._fusion_step.add_(1)
        return min(
            float(self._fusion_step.item()) / float(self.warmup_iters), 1.0)

    def forward(self, text_features, visual_features, prior_maps):
        if not (len(text_features) == len(visual_features) ==
                len(prior_maps) == len(self.projections)):
            raise ValueError('Parallel visual fusion levels do not match.')
        warmup = self._warmup_scale()
        scales = (
            self.max_scale * torch.sigmoid(self.scale_logits) * warmup)
        outputs = []
        for level, (text_feature, visual_feature, prior, projection) in enumerate(
                zip(text_features, visual_features, prior_maps,
                    self.projections)):
            residual = projection(visual_feature)
            if residual.shape[-2:] != text_feature.shape[-2:]:
                residual = torch.nn.functional.interpolate(
                    residual,
                    size=text_feature.shape[-2:],
                    mode='bilinear',
                    align_corners=False)
            if residual.shape[1] != text_feature.shape[1]:
                raise ValueError(
                    'Projected visual and text-aware channels must match.')
            if prior.shape[-2:] != text_feature.shape[-2:]:
                prior = torch.nn.functional.interpolate(
                    prior.float(),
                    size=text_feature.shape[-2:],
                    mode='bilinear',
                    align_corners=False)
            if self.detach_prior:
                prior = prior.detach()
            prior = prior.to(device=residual.device, dtype=residual.dtype)
            scale = scales[level].to(
                device=residual.device, dtype=residual.dtype)
            outputs.append(text_feature + scale * prior * residual)
        self.last_scales = scales.detach()
        return tuple(outputs)


@MODELS.register_module()
class ParallelVisualPriorYOLOWorldWSRPAFPN(YOLOWorldWSRPAFPN):
    """YOLO-World neck with an independent pre-text visual-prior branch.

    The class-agnostic prior is computed from reduced backbone features before
    any text-guided CSP layer.  The original text-aware PAFPN is preserved, and
    a zero-initialized, bounded visual residual is fused after its output.  WSR
    then refines P3, while GuardPrompt consumes detached post-WSR features and
    the same detached per-level prior maps.
    """

    def __init__(self,
                 *args,
                 visual_prior_cfg: ConfigType,
                 visual_residual_cfg: ConfigType,
                 **kwargs) -> None:
        if visual_prior_cfg is None:
            raise ValueError('visual_prior_cfg is required.')
        if visual_residual_cfg is None:
            raise ValueError('visual_residual_cfg is required.')
        if kwargs.pop('fgbg_cfg', None) is not None:
            raise ValueError(
                'Use visual_prior_cfg instead of the post-PAFPN fgbg_cfg.')
        super().__init__(*args, fgbg_cfg=None, **kwargs)
        self.visual_prior_gate = MODELS.build(
            copy.deepcopy(visual_prior_cfg))
        if getattr(self.visual_prior_gate, 'direct_feature_boost', False):
            raise ValueError(
                'The parallel prior must set direct_feature_boost=False; '
                'visual fusion is handled by the isolated residual branch.')
        self.visual_residual_fusion = _ParallelVisualResidualFusion(
            **copy.deepcopy(visual_residual_cfg))
        self._parallel_visual_scales = None

    def forward(self,
                img_feats: List[Tensor],
                txt_feats: Tensor = None,
                txt_masks: Tensor = None) -> tuple:
        """Run visual prior and text-aware PAFPN in parallel."""
        self._prompt_bank_context = None
        self._parallel_visual_scales = None
        self._fgbg_parts = None
        assert len(img_feats) == len(self.in_channels)

        # This is the sole shared prefix.  The prior is produced here, before
        # any top-down or bottom-up text-guided CSP operation.
        reduce_outs = [
            self.reduce_layers[idx](img_feats[idx])
            for idx in range(len(self.in_channels))
        ]
        _, prior_parts = self.visual_prior_gate(
            tuple(reduce_outs), return_parts=True)
        if not prior_parts or not prior_parts.get('reliability_maps', None):
            raise RuntimeError(
                'The parallel visual prior did not return reliability maps.')
        self._fgbg_parts = prior_parts

        # Original YOLO-World text-aware top-down path.
        inner_outs = [reduce_outs[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = reduce_outs[idx - 1]
            upsample_feat = self.upsample_layers[
                len(self.in_channels) - 1 - idx](feat_high)
            if self.upsample_feats_cat_first:
                layer_inputs = torch.cat([upsample_feat, feat_low], 1)
            else:
                layer_inputs = torch.cat([feat_low, upsample_feat], 1)
            inner_out = self.top_down_layers[
                len(self.in_channels) - 1 - idx](layer_inputs, txt_feats)
            inner_outs.insert(0, inner_out)

        bottom_up_txt_feats = self._get_bottom_up_text(txt_feats, inner_outs)
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            downsample_feat = self.downsample_layers[idx](feat_low)
            out = self.bottom_up_layers[idx](
                torch.cat([downsample_feat, feat_high], 1),
                bottom_up_txt_feats)
            outs.append(out)

        text_features = tuple(
            self.out_layers[idx](outs[idx])
            for idx in range(len(self.in_channels)))
        reliability_maps = prior_parts['reliability_maps']
        results = self.visual_residual_fusion(
            text_features, tuple(reduce_outs), reliability_maps)
        self._parallel_visual_scales = (
            self.visual_residual_fusion.last_scales)

        fused_reliability = prior_parts.get(
            'fused_reliability', reliability_maps[0])
        p3 = results[0]
        if self.wsr is not None:
            p3 = self.wsr(
                p3, results[1], results[2], fused_reliability.detach())
        if self.text_residual is not None:
            p3 = self.text_residual(
                p3,
                txt_feats,
                fused_reliability,
                txt_masks=txt_masks)
        results = (p3, *results[1:])

        # Prompt losses cannot rewrite the visual prior or visual features.
        self._prompt_bank_context = (
            tuple(feat.detach() for feat in results),
            tuple(route.detach() for route in reliability_maps))
        return results


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



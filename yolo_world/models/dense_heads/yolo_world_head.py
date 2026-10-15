# Copyright (c) Tencent Inc. All rights reserved.
import math
import copy
from typing import List, Optional, Tuple, Union, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmcv.cnn import ConvModule
from mmcv.ops import nms
from mmengine.config import ConfigDict
from mmengine.model import BaseModule
from torch import Tensor
from torch.nn.modules.batchnorm import _BatchNorm

from mmengine.dist import get_dist_info
from mmengine.structures import InstanceData
from mmdet.structures import SampleList
from mmdet.utils import OptConfigType, InstanceList, OptInstanceList
from mmdet.models.utils import (multi_apply, unpack_gt_instances,
                                filter_scores_and_topk)
from mmyolo.registry import MODELS
from mmyolo.models.dense_heads import YOLOv8HeadModule, YOLOv8Head
from mmyolo.models.utils import gt_instances_preprocess
from mmcv.cnn.bricks import build_norm_layer

from ..layers.lsk import ResidualLSKBlock
from ..layers.mona import Mona2d


def _dense_contrastive_similarity(x: Tensor, w: Tensor,
                                  use_einsum: bool) -> Tensor:
    """Compute dense image-text similarity from normalized inputs."""
    if use_einsum:
        return torch.einsum('bchw,bkc->bkhw', x, w)

    batch, channel, height, width = x.shape
    num_classes = w.shape[1]
    x = x.permute(0, 2, 3, 1).reshape(batch, -1, channel)
    x = torch.matmul(x, w.permute(0, 2, 1))
    return x.reshape(batch, height, width, num_classes).permute(0, 3, 1, 2)


def _normalize_text_features(
        w: Tensor, text_norm: Optional[nn.Module] = None) -> Tensor:
    """Normalize each text embedding with L2 or an optional BatchNorm1d."""
    if text_norm is None:
        return F.normalize(w, dim=-1, p=2)

    if w.dim() != 3:
        raise ValueError('Text features must have shape [B, K, D].')
    batch_size, num_classes, embed_dims = w.shape
    if embed_dims != text_norm.num_features:
        raise ValueError(
            'Text feature dimension must match text BatchNorm channels.')
    w = text_norm(w.reshape(batch_size * num_classes, embed_dims))
    return w.reshape(batch_size, num_classes, embed_dims)


def _prepare_route_map(route: Tensor, x: Tensor, detach: bool) -> Tensor:
    """Align a class-agnostic spatial route with a visual feature map."""
    if route is None or route.dim() != 4:
        raise ValueError('CSPR route maps must have shape [B, 1, H, W].')
    if route.shape[1] != 1:
        route = route.mean(dim=1, keepdim=True)
    if route.shape[0] == 1 and x.shape[0] > 1:
        route = route.expand(x.shape[0], -1, -1, -1)
    if route.shape[0] != x.shape[0]:
        raise ValueError('CSPR route and visual batch sizes do not match.')
    if route.shape[-2:] != x.shape[-2:]:
        route = F.interpolate(
            route.float(),
            size=x.shape[-2:],
            mode='bilinear',
            align_corners=False)
    if detach:
        route = route.detach()
    return route.to(device=x.device, dtype=x.dtype).clamp(0.0, 1.0)


def _routed_contrastive_logits(
        x: Tensor,
        base_w: Tensor,
        prompt_w: Tensor,
        gate: Tensor,
        ranking: Tensor,
        logit_scale: Tensor,
        bias: Tensor,
        use_einsum: bool,
        mode: str,
        topk: int,
        detach_gate: bool,
        text_norm: Optional[nn.Module] = None) -> Tensor:
    """Route prompted text densely in training and sparsely in inference."""
    if base_w.shape != prompt_w.shape:
        raise ValueError('CSPR base and prompted text shapes must match.')
    if base_w.shape[0] != x.shape[0]:
        raise ValueError('CSPR text and visual batch sizes must match.')
    if text_norm is None:
        base_w = _normalize_text_features(base_w)
        prompt_w = _normalize_text_features(prompt_w)
    else:
        # Normalize both banks in one BN call so they use identical batch
        # statistics and update the running statistics only once.
        num_classes = base_w.shape[1]
        text = _normalize_text_features(
            torch.cat((base_w, prompt_w), dim=1), text_norm)
        base_w, prompt_w = text.split(num_classes, dim=1)

    gate = _prepare_route_map(gate, x, detach_gate)
    scale = logit_scale.exp()
    base_logits = (
        _dense_contrastive_similarity(x, base_w, use_einsum) * scale + bias)

    if mode == 'dense':
        prompt_logits = (
            _dense_contrastive_similarity(x, prompt_w, use_einsum) * scale +
            bias)
        return base_logits + gate * (prompt_logits - base_logits)
    if mode != 'sparse_topk':
        raise ValueError(f'Unsupported CSPR mode: {mode}.')

    ranking = _prepare_route_map(ranking, x, detach=True)
    batch_size, channels, height, width = x.shape
    num_positions = height * width
    selected_count = min(max(int(topk), 0), num_positions)
    if selected_count == 0:
        return base_logits

    indices = ranking.flatten(1).topk(selected_count, dim=1).indices
    flat_visual = x.flatten(2).transpose(1, 2)
    visual_indices = indices.unsqueeze(-1).expand(-1, -1, channels)
    selected_visual = torch.gather(flat_visual, 1, visual_indices)
    selected_prompt_logits = (
        torch.einsum('bqd,bkd->bkq', selected_visual, prompt_w) * scale + bias)

    num_classes = base_logits.shape[1]
    score_indices = indices.unsqueeze(1).expand(-1, num_classes, -1)
    flat_base = base_logits.flatten(2)
    selected_base_logits = torch.gather(flat_base, 2, score_indices)
    selected_gate = torch.gather(gate.flatten(2), 2, indices.unsqueeze(1))
    selected_logits = (
        selected_base_logits +
        selected_gate * (selected_prompt_logits - selected_base_logits))

    routed_logits = flat_base.clone()
    routed_logits.scatter_(2, score_indices, selected_logits)
    return routed_logits.reshape(batch_size, num_classes, height, width)


@MODELS.register_module()
class ContrastiveHead(BaseModule):
    """Contrastive Head for YOLO-World
    compute the region-text scores according to the
    similarity between image and text features
    Args:
        embed_dims (int): embed dim of text and image features
    """
    def __init__(self,
                 embed_dims: int,
                 init_cfg: OptConfigType = None,
                 use_einsum: bool = True) -> None:

        super().__init__(init_cfg=init_cfg)

        self.bias = nn.Parameter(torch.zeros([]))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.use_einsum = use_einsum

    def forward(self, x: Tensor, w: Tensor) -> Tensor:
        """Forward function of contrastive learning."""
        x = F.normalize(x, dim=1, p=2)
        w = F.normalize(w, dim=-1, p=2)
        x = _dense_contrastive_similarity(x, w, self.use_einsum)
        x = x * self.logit_scale.exp() + self.bias
        return x

    def forward_routed(self, x: Tensor, base_w: Tensor, prompt_w: Tensor,
                       gate: Tensor, ranking: Tensor, mode: str, topk: int,
                       detach_gate: bool) -> Tensor:
        """Compute CSPR logits while normalizing visual features once."""
        x = F.normalize(x, dim=1, p=2)
        return _routed_contrastive_logits(
            x, base_w, prompt_w, gate, ranking, self.logit_scale, self.bias,
            self.use_einsum, mode, topk, detach_gate)


@MODELS.register_module()
class BNContrastiveHead(BaseModule):
    """ Batch Norm Contrastive Head for YOLO-World
    using batch norm instead of l2-normalization
    Args:
        embed_dims (int): embed dim of text and image features
        norm_cfg (dict): normalization params
    """
    def __init__(self,
                 embed_dims: int,
                 norm_cfg: ConfigDict,
                 init_cfg: OptConfigType = None,
                 use_einsum: bool = True,
                 text_norm_cfg: Optional[ConfigDict] = None) -> None:

        super().__init__(init_cfg=init_cfg)
        self.norm = build_norm_layer(norm_cfg, embed_dims)[1]
        self.text_norm = None
        if text_norm_cfg is not None:
            self.text_norm = build_norm_layer(text_norm_cfg, embed_dims)[1]
            if not isinstance(self.text_norm, nn.BatchNorm1d):
                raise TypeError('text_norm_cfg must build BatchNorm1d.')
        self.bias = nn.Parameter(torch.zeros([]))
        # use -1.0 is more stable
        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))
        self.use_einsum = use_einsum

    def forward(self, x: Tensor, w: Tensor) -> Tensor:
        """Forward function of contrastive learning."""
        x = self.norm(x)
        w = _normalize_text_features(w, self.text_norm)
        x = _dense_contrastive_similarity(x, w, self.use_einsum)
        x = x * self.logit_scale.exp() + self.bias
        return x

    def forward_routed(self, x: Tensor, base_w: Tensor, prompt_w: Tensor,
                       gate: Tensor, ranking: Tensor, mode: str, topk: int,
                       detach_gate: bool) -> Tensor:
        """Compute CSPR logits while applying batch normalization once."""
        x = self.norm(x)
        return _routed_contrastive_logits(
            x, base_w, prompt_w, gate, ranking, self.logit_scale, self.bias,
            self.use_einsum, mode, topk, detach_gate, self.text_norm)


@MODELS.register_module()
class RepBNContrastiveHead(BaseModule):
    """ Batch Norm Contrastive Head for YOLO-World
    using batch norm instead of l2-normalization
    Args:
        embed_dims (int): embed dim of text and image features
        norm_cfg (dict): normalization params
    """
    def __init__(self,
                 embed_dims: int,
                 num_guide_embeds: int,
                 norm_cfg: ConfigDict,
                 init_cfg: OptConfigType = None) -> None:

        super().__init__(init_cfg=init_cfg)
        self.norm = build_norm_layer(norm_cfg, embed_dims)[1]
        self.conv = nn.Conv2d(embed_dims, num_guide_embeds, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        """Forward function of contrastive learning."""
        x = self.norm(x)
        x = self.conv(x)
        return x


@MODELS.register_module()
class YOLOWorldHeadModule(YOLOv8HeadModule):
    """Head Module for YOLO-World

    Args:
        embed_dims (int): embed dim for text feautures and image features
        use_bn_head (bool): use batch normalization head
        lsk_cfg (dict, optional): Small LSK residuals on selected classification
            inputs. None preserves the original head and checkpoint keys.
        mona_cfg (dict, optional): Mona on selected classification inputs.
            None or enabled=False preserves the original checkpoint keys.
    """
    def __init__(self,
                 *args,
                 embed_dims: int,
                 use_bn_head: bool = False,
                 use_einsum: bool = True,
                 freeze_all: bool = False,
                 text_norm_cfg: Optional[dict] = None,
                 spatial_prompt_routing_cfg: Optional[dict] = None,
                 lsk_cfg: Optional[dict] = None,
                 mona_cfg: Optional[dict] = None,
                 **kwargs) -> None:
        self.embed_dims = embed_dims
        self.use_bn_head = use_bn_head
        self.use_einsum = use_einsum
        self.freeze_all = freeze_all
        self.text_norm_cfg = text_norm_cfg
        self.lsk_cfg = copy.deepcopy(lsk_cfg)
        self.mona_cfg = copy.deepcopy(mona_cfg)
        if self.text_norm_cfg is not None and not self.use_bn_head:
            raise ValueError('text_norm_cfg requires use_bn_head=True.')

        routing_cfg = (
            {} if spatial_prompt_routing_cfg is None else
            dict(spatial_prompt_routing_cfg))
        self.spatial_prompt_routing_enabled = bool(
            routing_cfg.pop('enabled', False))
        self.spatial_prompt_train_mode = routing_cfg.pop(
            'train_mode', 'dense')
        self.spatial_prompt_test_mode = routing_cfg.pop(
            'test_mode', 'sparse_topk')
        self.spatial_prompt_topk = tuple(
            int(value) for value in routing_cfg.pop(
                'topk_per_level', (512, 256, 128)))
        self.spatial_prompt_gate_source = routing_cfg.pop(
            'gate_source', 'foreground_prior')
        self.spatial_prompt_ranking_source = routing_cfg.pop(
            'ranking_source', 'reliability')
        self.spatial_prompt_detach_gate = bool(
            routing_cfg.pop('detach_gate', True))
        if routing_cfg:
            raise ValueError(
                'Unknown spatial_prompt_routing_cfg fields: '
                f'{tuple(routing_cfg)}')
        if self.spatial_prompt_train_mode not in ('dense', 'sparse_topk'):
            raise ValueError('CSPR train_mode must be dense or sparse_topk.')
        if self.spatial_prompt_test_mode not in ('dense', 'sparse_topk'):
            raise ValueError('CSPR test_mode must be dense or sparse_topk.')
        if self.spatial_prompt_gate_source != 'foreground_prior':
            raise ValueError(
                'CSPR currently requires gate_source=foreground_prior.')
        if self.spatial_prompt_ranking_source != 'reliability':
            raise ValueError(
                'CSPR currently requires ranking_source=reliability.')
        if any(value < 0 for value in self.spatial_prompt_topk):
            raise ValueError('CSPR topk_per_level values must be non-negative.')
        super().__init__(*args, **kwargs)
        if (self.spatial_prompt_routing_enabled and
                len(self.spatial_prompt_topk) != self.num_levels):
            raise ValueError(
                'CSPR topk_per_level must match the number of feature levels.')

    def init_weights(self, prior_prob=0.01):
        """Initialize the weight and bias of PPYOLOE head."""
        super().init_weights()
        for cls_pred, cls_contrast, stride in zip(self.cls_preds,
                                                  self.cls_contrasts,
                                                  self.featmap_strides):
            cls_pred[-1].bias.data[:] = 0.0  # reset bias
            if hasattr(cls_contrast, 'bias'):
                nn.init.constant_(
                    cls_contrast.bias.data,
                    math.log(5 / self.num_classes / (640 / stride)**2))

    def _init_layers(self) -> None:
        """initialize conv layers in YOLOv8 head."""
        # Init decouple head
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.cls_contrasts = nn.ModuleList()

        reg_out_channels = max(
            (16, self.in_channels[0] // 4, self.reg_max * 4))
        cls_out_channels = max(self.in_channels[0], self.num_classes)

        for i in range(self.num_levels):#（8 16 32）
            self.reg_preds.append(
                nn.Sequential(
                    ConvModule(in_channels=self.in_channels[i],
                               out_channels=reg_out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               norm_cfg=self.norm_cfg,
                               act_cfg=self.act_cfg),
                    ConvModule(in_channels=reg_out_channels,
                               out_channels=reg_out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               norm_cfg=self.norm_cfg,
                               act_cfg=self.act_cfg),
                    nn.Conv2d(in_channels=reg_out_channels,
                              out_channels=4 * self.reg_max,
                              kernel_size=1)))
            self.cls_preds.append(
                nn.Sequential(
                    ConvModule(in_channels=self.in_channels[i],
                               out_channels=cls_out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               norm_cfg=self.norm_cfg,
                               act_cfg=self.act_cfg),
                    ConvModule(in_channels=cls_out_channels,
                               out_channels=cls_out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               norm_cfg=self.norm_cfg,
                               act_cfg=self.act_cfg),
                    nn.Conv2d(in_channels=cls_out_channels,
                              out_channels=self.embed_dims,
                              kernel_size=1)))
            if self.use_bn_head:
                self.cls_contrasts.append(
                    BNContrastiveHead(self.embed_dims,
                                      self.norm_cfg,
                                      use_einsum=self.use_einsum,
                                      text_norm_cfg=self.text_norm_cfg))
            else:
                self.cls_contrasts.append(
                    ContrastiveHead(self.embed_dims,
                                    use_einsum=self.use_einsum))

        self.cls_lsk = None
        if self.lsk_cfg is not None:
            lsk_cfg = dict(self.lsk_cfg)
            levels = tuple(lsk_cfg.pop('active_levels', (2,)))
            init_scale = lsk_cfg.pop('init_scale', 1e-2)
            if lsk_cfg:
                raise ValueError(f'Unknown lsk_cfg fields: {tuple(lsk_cfg)}')
            if (not levels or len(set(levels)) != len(levels)
                    or any(not isinstance(i, int) or isinstance(i, bool)
                           or i < 0 or i >= self.num_levels for i in levels)):
                raise ValueError('LSK active_levels must be unique valid feature indices.')
            self.cls_lsk = nn.ModuleList([
                ResidualLSKBlock(channels, init_scale=init_scale)
                if level in levels else nn.Identity()
                for level, channels in enumerate(self.in_channels)
            ])

        self.cls_mona = None
        if self.mona_cfg is not None:
            mona_cfg = dict(self.mona_cfg)
            enabled = mona_cfg.pop('enabled', True)
            levels = tuple(mona_cfg.pop('active_levels', (0,)))
            if enabled:
                if self.cls_lsk is not None:
                    raise ValueError('Enable either Mona or LSK on classification inputs.')
                if (not levels or len(set(levels)) != len(levels)
                        or any(not isinstance(i, int) or isinstance(i, bool)
                               or i < 0 or i >= self.num_levels for i in levels)):
                    raise ValueError('Mona active_levels must be unique valid feature indices.')
                self.cls_mona = nn.ModuleList([
                    Mona2d(channels, **mona_cfg)
                    if level in levels else nn.Identity()
                    for level, channels in enumerate(self.in_channels)
                ])

        proj = torch.arange(self.reg_max, dtype=torch.float)
        self.register_buffer('proj', proj, persistent=False)

        if self.freeze_all:
            self._freeze_all()

    def _freeze_all(self):
        """Freeze the model."""
        for m in self.modules():
            if isinstance(m, _BatchNorm):
                m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_all:
            self._freeze_all()

    def forward(self,
                img_feats: Tuple[Tensor],
                txt_feats: Tensor,
                txt_masks: Tensor,
                base_txt_feats: Optional[Tensor] = None,
                foreground_maps: Optional[Sequence[Tensor]] = None,
                reliability_maps: Optional[Sequence[Tensor]] = None
                ) -> Tuple[List]:
        """Forward features from the upstream network."""
        assert len(img_feats) == self.num_levels
        cls_adapters = self.cls_mona if self.cls_mona is not None else self.cls_lsk
        if self.spatial_prompt_routing_enabled:
            if (base_txt_feats is None or foreground_maps is None or
                    reliability_maps is None):
                raise RuntimeError(
                    'CSPR requires base text, foreground maps, and '
                    'reliability maps.')
            if (len(foreground_maps) != self.num_levels or
                    len(reliability_maps) != self.num_levels):
                raise ValueError(
                    'CSPR requires one foreground and reliability map per '
                    'feature level.')
            mode = (
                self.spatial_prompt_train_mode
                if self.training else self.spatial_prompt_test_mode)
            outputs = [
                self.forward_single(
                    img_feat,
                    txt_feats,
                    txt_masks,
                    cls_pred,
                    reg_pred,
                    cls_contrast,
                    base_txt_feat=base_txt_feats,
                    foreground_map=foreground_map,
                    reliability_map=reliability_map,
                    routing_mode=mode,
                    routing_topk=self.spatial_prompt_topk[level_idx],
                    cls_adapter=(cls_adapters[level_idx]
                                 if cls_adapters is not None else None))
                for level_idx, (
                    img_feat, cls_pred, reg_pred, cls_contrast,
                    foreground_map, reliability_map) in enumerate(
                        zip(img_feats, self.cls_preds, self.reg_preds,
                            self.cls_contrasts, foreground_maps,
                            reliability_maps))
            ]
            return tuple([
                output[output_idx] for output in outputs
            ] for output_idx in range(len(outputs[0])))

        if cls_adapters is not None:
            outputs = [
                self.forward_single(
                    img_feat, txt_feats, txt_masks, cls_pred, reg_pred,
                    cls_contrast, cls_adapter=cls_adapter)
                for img_feat, cls_pred, reg_pred, cls_contrast, cls_adapter in zip(
                    img_feats, self.cls_preds, self.reg_preds,
                    self.cls_contrasts, cls_adapters)
            ]
            return tuple(map(list, zip(*outputs)))

        txt_feats = [txt_feats for _ in range(self.num_levels)]
        txt_masks = [txt_masks for _ in range(self.num_levels)]
        return multi_apply(self.forward_single, img_feats, txt_feats,
                           txt_masks, self.cls_preds, self.reg_preds,
                           self.cls_contrasts)

    def forward_single(self, img_feat: Tensor, txt_feat: Tensor,
                       txt_masks: Tensor, cls_pred: nn.ModuleList,
                       reg_pred: nn.ModuleList,
                       cls_contrast: nn.ModuleList,
                       base_txt_feat: Optional[Tensor] = None,
                       foreground_map: Optional[Tensor] = None,
                       reliability_map: Optional[Tensor] = None,
                       routing_mode: str = 'dense',
                       routing_topk: int = 0,
                       cls_adapter: Optional[nn.Module] = None) -> Tuple:
        """Forward feature of a single scale level."""
        b, _, h, w = img_feat.shape
        cls_feat = img_feat if cls_adapter is None else cls_adapter(img_feat)
        cls_embed = cls_pred(cls_feat)
        if base_txt_feat is None:
            cls_logit = cls_contrast(cls_embed, txt_feat)
        else:
            if not hasattr(cls_contrast, 'forward_routed'):
                raise TypeError(
                    'The configured contrastive head does not support CSPR.')
            cls_logit = cls_contrast.forward_routed(
                cls_embed,
                base_txt_feat,
                txt_feat,
                foreground_map,
                reliability_map,
                mode=routing_mode,
                topk=routing_topk,
                detach_gate=self.spatial_prompt_detach_gate)

        if txt_masks is not None:

            # # 临时增加这一行，强制将 mask 设为 1
            # txt_masks = torch.ones_like(txt_masks)
            txt_masks = txt_masks.view(b, -1, 1, 1).expand(-1, -1, h, w)
            # Preserve the original FP32 mask while avoiding FP16 overflow
            # under AMP. Both values have zero sigmoid probability.
            mask_fill_value = max(-10e6, torch.finfo(cls_logit.dtype).min)
            if self.training:
                cls_logit = cls_logit * txt_masks
                cls_logit[txt_masks == 0] = mask_fill_value
            else:
                cls_logit[txt_masks == 0] = mask_fill_value

        bbox_dist_preds = reg_pred(img_feat)
        if self.reg_max > 1:
            bbox_dist_preds = bbox_dist_preds.reshape(
                [-1, 4, self.reg_max, h * w]).permute(0, 3, 1, 2)

            # TODO: The get_flops script cannot handle the situation of
            #  matmul, and needs to be fixed later
            # bbox_preds = bbox_dist_preds.softmax(3).matmul(self.proj)
            bbox_preds = bbox_dist_preds.softmax(3).matmul(
                self.proj.view([-1, 1])).squeeze(-1)
            bbox_preds = bbox_preds.transpose(1, 2).reshape(b, -1, h, w)
        else:
            bbox_preds = bbox_dist_preds
        if self.training:
            return cls_logit, bbox_preds, bbox_dist_preds
        return cls_logit, bbox_preds


@MODELS.register_module()
class RepYOLOWorldHeadModule(YOLOWorldHeadModule):
    def __init__(self,
                 *args,
                 embed_dims: int,
                 num_guide: int,
                 freeze_all: bool = False,
                 **kwargs) -> None:
        super().__init__(*args,
                         embed_dims=embed_dims,
                         use_bn_head=True,
                         use_einsum=False,
                         freeze_all=freeze_all,
                         **kwargs)

        # using rep head
        cls_contrasts = []
        for _ in range(self.num_levels):
            cls_contrasts.append(
                RepBNContrastiveHead(embed_dims=embed_dims,
                                     num_guide_embeds=num_guide,
                                     norm_cfg=self.norm_cfg))
        self.cls_contrasts = nn.ModuleList(cls_contrasts)

    def forward_single(self, img_feat: Tensor, cls_pred: nn.ModuleList,
                       reg_pred: nn.ModuleList,
                       cls_contrast: nn.ModuleList,
                       cls_adapter: Optional[nn.Module] = None) -> Tuple:
        """Forward features from the upstream network."""
        b, _, h, w = img_feat.shape
        cls_feat = img_feat if cls_adapter is None else cls_adapter(img_feat)
        cls_embed = cls_pred(cls_feat)
        cls_logit = cls_contrast(cls_embed)
        bbox_dist_preds = reg_pred(img_feat)
        if self.reg_max > 1:
            bbox_dist_preds = bbox_dist_preds.reshape(
                [-1, 4, self.reg_max, h * w]).permute(0, 3, 1, 2)

            # TODO: The get_flops script cannot handle the situation of
            #  matmul, and needs to be fixed later
            # bbox_preds = bbox_dist_preds.softmax(3).matmul(self.proj)
            bbox_preds = bbox_dist_preds.softmax(3).matmul(
                self.proj.view([-1, 1])).squeeze(-1)
            bbox_preds = bbox_preds.transpose(1, 2).reshape(b, -1, h, w)
        else:
            bbox_preds = bbox_dist_preds
        if self.training:
            return cls_logit, bbox_preds, bbox_dist_preds
        else:
            return cls_logit, bbox_preds

    def forward(self, img_feats: Tuple[Tensor]) -> Tuple[List]:
        assert len(img_feats) == self.num_levels
        cls_adapters = self.cls_mona if self.cls_mona is not None else self.cls_lsk
        if cls_adapters is not None:
            return multi_apply(self.forward_single, img_feats, self.cls_preds,
                               self.reg_preds, self.cls_contrasts, cls_adapters)
        return multi_apply(self.forward_single, img_feats, self.cls_preds,
                           self.reg_preds, self.cls_contrasts)


@MODELS.register_module()
class YOLOWorldHead(YOLOv8Head):
    """YOLO-World Head
    """
    def __init__(self,
                 world_size=-1,
                 proposal_cfg: Optional[dict] = None,
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.world_size = world_size
        self.proposal_cfg = None
        if proposal_cfg is not None:
            self.proposal_cfg = dict(
                pre_nms_topk=3000, nms_iou_threshold=0.7, max_per_img=300)
            self.proposal_cfg.update(dict(proposal_cfg))

    """YOLO World v8 head."""

    def classification_position_weights(self, assigned_scores, foreground, img_metas):
        """Extension point; the original head retains its exact supervision."""
        return None

    def loss(self, img_feats: Tuple[Tensor], txt_feats: Tensor,
             txt_masks: Tensor,
             batch_data_samples: Union[list, dict],
             spatial_prompt_context: Optional[dict] = None) -> dict:
        """Run the head and calculate detection losses."""
        outs = self(
            img_feats,
            txt_feats,
            txt_masks,
            spatial_prompt_context=spatial_prompt_context)
        cls_scores, bbox_preds, bbox_dist_preds = outs
        return self.loss_by_feat(
            cls_scores,
            bbox_preds,
            bbox_dist_preds,
            txt_masks,
            batch_data_samples['bboxes_labels'],
            batch_data_samples['img_metas'])

    def loss_and_predict(
        self,
        img_feats: Tuple[Tensor],
        txt_feats: Tensor,
        txt_masks: Tensor,
        batch_data_samples: SampleList,
        proposal_cfg: Optional[ConfigDict] = None,
        spatial_prompt_context: Optional[dict] = None
    ) -> Tuple[dict, InstanceList]:
        """Perform forward propagation of the head, then calculate loss and
        predictions from the features and data samples.
        """
        outputs = unpack_gt_instances(batch_data_samples)
        (batch_gt_instances, batch_gt_instances_ignore,
         batch_img_metas) = outputs

        outs = self(
            img_feats,
            txt_feats,
            txt_masks,
            spatial_prompt_context=spatial_prompt_context)
        cls_scores, bbox_preds, bbox_dist_preds = outs

        losses = self.loss_by_feat(
            cls_scores,
            bbox_preds,
            bbox_dist_preds,
            txt_masks,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore)

        predictions = self.predict_by_feat(
            cls_scores,
            bbox_preds,
            batch_img_metas=batch_img_metas,
            cfg=proposal_cfg)
        return losses, predictions

    def forward(self,
                img_feats: Tuple[Tensor],
                txt_feats: Tensor,
                txt_masks: Tensor,
                spatial_prompt_context: Optional[dict] = None) -> Tuple[List]:
        """Forward features from the upstream network."""
        context = (
            {} if spatial_prompt_context is None else
            dict(spatial_prompt_context))
        allowed_fields = {
            'base_txt_feats', 'foreground_maps', 'reliability_maps'
        }
        unknown_fields = set(context) - allowed_fields
        if unknown_fields:
            raise ValueError(
                f'Unknown CSPR context fields: {tuple(unknown_fields)}')
        return self.head_module(img_feats, txt_feats, txt_masks, **context)

    def predict(self,
                img_feats: Tuple[Tensor],
                txt_feats: Tensor,
                txt_masks: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = False,
                spatial_prompt_context: Optional[dict] = None) -> InstanceList:
        """Perform forward propagation of the detection head and predict
        detection results on the features of the upstream network.
        执行检测头的前向传播，并在上游网络特征上预测检测结果。
        """
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        outs = self(
            img_feats,
            txt_feats,
            txt_masks,
            spatial_prompt_context=spatial_prompt_context)
        cls_scores, bbox_preds = outs
        predictions = self.predict_by_feat(
            cls_scores,
            bbox_preds,
            batch_img_metas=batch_img_metas,
            rescale=rescale)
        return predictions

    def aug_test(self,
                 aug_batch_feats,
                 aug_batch_img_metas,
                 rescale=False,
                 with_ori_nms=False,
                 **kwargs):
        """Test function with test time augmentation."""
        raise NotImplementedError('aug_test is not implemented yet.')

    def loss_by_feat(
            self,
            cls_scores: Sequence[Tensor],
            bbox_preds: Sequence[Tensor],
            bbox_dist_preds: Sequence[Tensor],
            batch_text_masks: Tensor,
            batch_gt_instances: Sequence[InstanceData],
            batch_img_metas: Sequence[dict],
            batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        """Calculate the loss based on the features extracted by the detection
        head.

        Args:
            cls_scores (Sequence[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_priors * num_classes.
            bbox_preds (Sequence[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_priors * 4.
            bbox_dist_preds (Sequence[Tensor]): Box distribution logits for
                each scale level with shape (bs, reg_max + 1, H*W, 4).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
        Returns:
            dict[str, Tensor]: A dictionary of losses.
        """
        num_imgs = len(batch_img_metas)

        current_featmap_sizes = [
            cls_score.shape[2:] for cls_score in cls_scores
        ]
        # If the shape does not equal, generate new one
        if current_featmap_sizes != self.featmap_sizes_train:
            self.featmap_sizes_train = current_featmap_sizes

            mlvl_priors_with_stride = self.prior_generator.grid_priors(
                self.featmap_sizes_train,
                dtype=cls_scores[0].dtype,
                device=cls_scores[0].device,
                with_stride=True)

            self.num_level_priors = [len(n) for n in mlvl_priors_with_stride]
            self.flatten_priors_train = torch.cat(mlvl_priors_with_stride,
                                                  dim=0)
            self.stride_tensor = self.flatten_priors_train[..., [2]]

        # gt info
        gt_info = gt_instances_preprocess(batch_gt_instances, num_imgs)
        gt_labels = gt_info[:, :, :1]
        gt_bboxes = gt_info[:, :, 1:]  # xyxy
        pad_bbox_flag = (gt_bboxes.sum(-1, keepdim=True) > 0).float()

        # pred info
        flatten_cls_preds = [
            cls_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                 self.num_classes)
            for cls_pred in cls_scores
        ]
        flatten_pred_bboxes = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]
        # (bs, n, 4 * reg_max)
        flatten_pred_dists = [
            bbox_pred_org.reshape(num_imgs, -1, self.head_module.reg_max * 4)
            for bbox_pred_org in bbox_dist_preds
        ]

        flatten_dist_preds = torch.cat(flatten_pred_dists, dim=1)
        flatten_cls_preds = torch.cat(flatten_cls_preds, dim=1)
        flatten_pred_bboxes = torch.cat(flatten_pred_bboxes, dim=1)
        flatten_pred_bboxes = self.bbox_coder.decode(
            self.flatten_priors_train[..., :2], flatten_pred_bboxes,
            self.stride_tensor[..., 0])

        assigned_result = self.assigner(
            (flatten_pred_bboxes.detach()).type(gt_bboxes.dtype),
            flatten_cls_preds.detach().sigmoid(), self.flatten_priors_train,
            gt_labels, gt_bboxes, pad_bbox_flag)

        assigned_bboxes = assigned_result['assigned_bboxes']
        assigned_scores = assigned_result['assigned_scores']
        fg_mask_pre_prior = assigned_result['fg_mask_pre_prior']

        assigned_scores_sum = assigned_scores.sum().clamp(min=1)

        partial_weight = self.classification_position_weights(
            assigned_scores, fg_mask_pre_prior, batch_img_metas)
        if batch_text_masks is not None:
            cls_weight = batch_text_masks.view(num_imgs, 1, -1).expand(
                -1, flatten_cls_preds.shape[1], -1).to(flatten_cls_preds)

            loss_cls = self.loss_cls(flatten_cls_preds, assigned_scores)
            if partial_weight is not None:
                loss_cls = loss_cls * partial_weight
            _loss_cls = (loss_cls * cls_weight).sum(dim=-1)
            loss_cls = _loss_cls.sum()
        else:
            loss_cls = self.loss_cls(flatten_cls_preds, assigned_scores)
            if partial_weight is not None:
                loss_cls = loss_cls * partial_weight
            loss_cls = loss_cls.sum()
        loss_cls /= assigned_scores_sum

        # rescale bbox
        assigned_bboxes_scaled = assigned_bboxes / self.stride_tensor
        flatten_pred_bboxes_scaled = flatten_pred_bboxes / self.stride_tensor

        # select positive samples mask
        num_pos = fg_mask_pre_prior.sum()
        if num_pos > 0:
            # when num_pos > 0, assigned_scores_sum will >0, so the loss_bbox
            # will not report an error
            # iou loss
            prior_bbox_mask = fg_mask_pre_prior.unsqueeze(-1).repeat([1, 1, 4])
            pred_bboxes_pos = torch.masked_select(
                flatten_pred_bboxes_scaled,
                prior_bbox_mask).reshape([-1, 4])
            assigned_bboxes_pos = torch.masked_select(
                assigned_bboxes_scaled,
                prior_bbox_mask).reshape([-1, 4])
            bbox_weight = torch.masked_select(assigned_scores.sum(-1),
                                              fg_mask_pre_prior).unsqueeze(-1)
            loss_bbox = self.loss_bbox(
                pred_bboxes_pos, assigned_bboxes_pos,
                weight=bbox_weight) / assigned_scores_sum

            # dfl loss
            pred_dist_pos = flatten_dist_preds[fg_mask_pre_prior]
            assigned_ltrb = self.bbox_coder.encode(
                self.flatten_priors_train[..., :2] / self.stride_tensor,
                assigned_bboxes_scaled,
                max_dis=self.head_module.reg_max - 1,
                eps=0.01)
            assigned_ltrb_pos = torch.masked_select(
                assigned_ltrb, prior_bbox_mask).reshape([-1, 4])
            loss_dfl = self.loss_dfl(pred_dist_pos.reshape(
                -1, self.head_module.reg_max),
                                     assigned_ltrb_pos.reshape(-1),
                                     weight=bbox_weight.expand(-1,
                                                               4).reshape(-1),
                                     avg_factor=assigned_scores_sum)
        else:
            loss_bbox = flatten_pred_bboxes_scaled.sum() * 0
            loss_dfl = flatten_pred_bboxes_scaled.sum() * 0
        if self.world_size == -1:
            _, world_size = get_dist_info()
        else:
            world_size = self.world_size
            world_size = self.world_size

        return dict(loss_cls=loss_cls * num_imgs * world_size,
                    loss_bbox=loss_bbox * num_imgs * world_size,
                    loss_dfl=loss_dfl * num_imgs * world_size)

    def predict_proposals_by_feat(
            self,
            bbox_preds: Sequence[Tensor],
            prior_maps: Optional[Sequence[Tensor]],
            batch_img_metas: Sequence[dict],
            rescale: bool = True) -> List[InstanceData]:
        """Decode class-agnostic proposals scored only by visual priors."""
        num_imgs = len(batch_img_metas)
        if not prior_maps or len(prior_maps) != len(bbox_preds):
            return [
                InstanceData(
                    bboxes=bbox_preds[0].new_zeros((0, 4)),
                    scores=bbox_preds[0].new_zeros((0, )),
                    labels=bbox_preds[0].new_zeros((0, ), dtype=torch.long))
                for _ in range(num_imgs)
            ]

        featmap_sizes = [bbox_pred.shape[2:] for bbox_pred in bbox_preds]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=bbox_preds[0].dtype,
            device=bbox_preds[0].device)
        flatten_priors = torch.cat(mlvl_priors)
        strides = torch.cat([
            flatten_priors.new_full((size.numel() * self.num_base_priors, ),
                                    stride)
            for size, stride in zip(featmap_sizes, self.featmap_strides)
        ])
        flat_bbox_preds = torch.cat([
            pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for pred in bbox_preds
        ], dim=1)
        decoded = self.bbox_coder.decode(
            flatten_priors[None], flat_bbox_preds, strides)
        flat_scores = []
        for prior, size in zip(prior_maps, featmap_sizes):
            if prior.shape[-2:] != size:
                prior = F.interpolate(
                    prior, size=size, mode='bilinear', align_corners=False)
            flat_scores.append(prior.reshape(num_imgs, -1))
        flat_scores = torch.cat(flat_scores, dim=1).clamp(0.0, 1.0)

        results = []
        pre_topk = int(self.proposal_cfg['pre_nms_topk'])
        max_per_img = int(self.proposal_cfg['max_per_img'])
        nms_thr = float(self.proposal_cfg['nms_iou_threshold'])
        for boxes, scores, img_meta in zip(decoded, flat_scores,
                                           batch_img_metas):
            topk = min(pre_topk, scores.numel())
            scores, indices = scores.topk(topk)
            boxes = boxes[indices]
            _, keep = nms(boxes, scores, nms_thr)
            keep = keep[:max_per_img]
            boxes, scores = boxes[keep], scores[keep]

            if rescale and boxes.numel() > 0:
                pad_param = img_meta.get('pad_param', None)
                if pad_param is not None:
                    boxes -= boxes.new_tensor([
                        pad_param[2], pad_param[0], pad_param[2], pad_param[0]
                    ])
                boxes /= boxes.new_tensor(
                    img_meta['scale_factor']).repeat((1, 2))
            ori_h, ori_w = img_meta['ori_shape'][:2]
            boxes[:, 0::2].clamp_(0, ori_w)
            boxes[:, 1::2].clamp_(0, ori_h)
            results.append(
                InstanceData(
                    bboxes=boxes,
                    scores=scores,
                    labels=scores.new_zeros(scores.shape, dtype=torch.long)))
        return results

    def predict_by_feat(self,  # 将头部输出特征转换为检测结果
                        cls_scores: List[Tensor],  # 分类分支特征列表
                        bbox_preds: List[Tensor],  # 回归分支特征列表
                        objectnesses: Optional[List[Tensor]] = None,  # 目标存在性分支特征
                        batch_img_metas: Optional[List[dict]] = None,  # 批次图像元信息
                        cfg: Optional[ConfigDict] = None,  # 推理配置
                        rescale: bool = True,  # 是否映射回原图尺度
                        with_nms: bool = True) -> List[InstanceData]:  # 是否执行NMS

        """Transform a batch of output features extracted by the head into bbox results."""  # 函数文档说明



        # #检测是否是由于文本特征过弱导致的分类分数过低问题
        # for cls_logit in cls_scores:
        #     # 在此处添加打印检查
        #     if cls_logit.max().item() < 0.001:  # 假设 0.001 为你的 score_thr
        #         print(f"DEBUG: Max abstract cls_logit {cls_logit.max().item()} is lower than threshold.")
        #


        assert len(cls_scores) == len(bbox_preds)  # 分类与回归预测数量应一致
        if objectnesses is None:  # 判断是否包含objectness分支
            with_objectnesses = False  # 标记缺少objectness
        else:
            with_objectnesses = True  # 标记存在objectness
            assert len(cls_scores) == len(objectnesses)  # 三分支层数需一致

        cfg = self.test_cfg if cfg is None else cfg  # 若未传入cfg则使用默认测试配置
        cfg = copy.deepcopy(cfg)  # 拷贝配置以免修改原配置

        multi_label = cfg.multi_label  # 读取multi_label参数
        multi_label &= self.num_classes > 1  # 在单类情况下禁用multi_label
        cfg.multi_label = multi_label  # 回写multi_label结果

        # Optional Base-only inference gate. IDs index the current text/class
        # columns, not the annotation category IDs or the Base training order.
        base_score_thr = cfg.get('base_score_thr', None)
        base_class_ids = cfg.get('base_class_ids', None)
        if base_score_thr is not None:
            if (isinstance(base_score_thr, bool)
                    or not isinstance(base_score_thr, (int, float))
                    or not math.isfinite(base_score_thr)
                    or not 0 <= base_score_thr <= 1):
                raise ValueError('base_score_thr must be a finite score in [0, 1].')
            if (not isinstance(base_class_ids, (list, tuple))
                    or not base_class_ids
                    or any(isinstance(index, bool) or not isinstance(index, int)
                           or not 0 <= index < self.num_classes
                           for index in base_class_ids)
                    or len(set(base_class_ids)) != len(base_class_ids)):
                raise ValueError(
                    'base_class_ids must contain unique valid class-column indices.')

        num_imgs = len(batch_img_metas)  # 批次图像数量
        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]  # 各层特征图尺寸

        if featmap_sizes != self.featmap_sizes:  # 尺寸变化时需重新生成先验
            self.mlvl_priors = self.prior_generator.grid_priors(  # 动态生成网格先验
                featmap_sizes,
                dtype=cls_scores[0].dtype,
                device=cls_scores[0].device)
            self.featmap_sizes = featmap_sizes  # 更新缓存的特征图尺寸
        flatten_priors = torch.cat(self.mlvl_priors)  # 拼接各尺度先验

        mlvl_strides = [  # 为每层生成步长张量
            flatten_priors.new_full(
                (featmap_size.numel() * self.num_base_priors, ), stride)  # 使用对应步长填充
            for featmap_size, stride in zip(featmap_sizes, self.featmap_strides)  # 遍历每层尺寸与步长
        ]
        flatten_stride = torch.cat(mlvl_strides)  # 展平特征步长

        flatten_cls_scores = [  # 保存展平后的分类特征
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                  self.num_classes)  # 调整维度并拉平
            for cls_score in cls_scores  # 遍历各尺度分类特征
        ]
        flatten_bbox_preds = [  # 保存展平后的回归特征
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)  # 调整维度并拉平
            for bbox_pred in bbox_preds  # 遍历各尺度回归特征
        ]

        flatten_cls_scores = torch.cat(flatten_cls_scores, dim=1).sigmoid()  # 拼接并映射分类分数到0-1
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)  # 拼接所有回归预测
        flatten_decoded_bboxes = self.bbox_coder.decode(  # 将归一化回归量解码为实际框
            flatten_priors[None], flatten_bbox_preds, flatten_stride)

        if with_objectnesses:  # 若存在objectness分支则需同样展平
            flatten_objectness = [
                objectness.permute(0, 2, 3, 1).reshape(num_imgs, -1)  # 展平objectness特征
                for objectness in objectnesses  # 遍历各尺度objectness
            ]
            flatten_objectness = torch.cat(flatten_objectness, dim=1).sigmoid()  # 拼接并归一化objectness得分
        else:
            flatten_objectness = [None for _ in range(num_imgs)]  # 不存在objectness时填充None

        results_list = []  # 初始化结果列表
        for (bboxes, scores, objectness,
             img_meta) in zip(flatten_decoded_bboxes, flatten_cls_scores,
                              flatten_objectness, batch_img_metas):  # 遍历每张图像的预测
            ori_shape = img_meta['ori_shape']  # 原图尺寸
            scale_factor = img_meta['scale_factor']  # 尺度缩放因子
            if 'pad_param' in img_meta:  # 判断是否存在padding参数
                pad_param = img_meta['pad_param']  # 读取padding信息
            else:
                pad_param = None  # 无padding信息时置空

            score_thr = cfg.get('score_thr', -1)  # 读取分数阈值
            if (objectness is not None and score_thr > 0 and not cfg.get(
                    'yolox_style', False)):
                conf_inds = objectness > score_thr  # objectness筛选掩码
                bboxes = bboxes[conf_inds, :]  # 过滤框
                scores = scores[conf_inds, :]  # 过滤分类分数
                objectness = objectness[conf_inds]  # 过滤objectness

            if objectness is not None:
                scores *= objectness[:, None]  # 分类分数乘以objectness

            if scores.shape[0] == 0:  # 若没有候选框
                empty_results = InstanceData()  # 创建空结果
                empty_results.bboxes = bboxes  # 占位框
                empty_results.scores = scores[:, 0]  # 占位分数
                empty_results.labels = scores[:, 0].int()  # 占位标签
                results_list.append(empty_results)  # 保存空结果
                continue  # 处理下一张图像

            if base_score_thr is not None:
                # Reject individual Base entries before global top-k/NMS;
                # keep all eligible labels, original scores and box coordinates.
                # DIOR 2.17 passes no objectness factor, so scores are sigmoid
                # class scores here. If supplied, factors were applied above.
                base_scores = scores[:, base_class_ids]
                scores[:, base_class_ids] = base_scores.masked_fill(
                    base_scores < base_score_thr, float('-inf'))

            nms_pre = cfg.get('nms_pre', 100000)  # NMS前保留数量
            if cfg.multi_label is False:  # 单标签模式
                scores, labels = scores.max(1, keepdim=True)  # 取最大类分数及标签
                scores, _, keep_idxs, results = filter_scores_and_topk(  # 依据阈值筛选候选
                    scores,
                    score_thr,
                    nms_pre,
                    results=dict(labels=labels[:, 0]))
                labels = results['labels']  # 提取筛选后的标签
            else:
                scores, labels, keep_idxs, _ = filter_scores_and_topk(  # 多标签模式直接筛选
                    scores, score_thr, nms_pre)

            results = InstanceData(scores=scores,  # 根据筛选结果构建实例数据
                                   labels=labels,
                                   bboxes=bboxes[keep_idxs])

            if rescale:  # 需要映射回原图时处理
                if pad_param is not None:  # 先去除padding偏移
                    results.bboxes -= results.bboxes.new_tensor([
                        pad_param[2], pad_param[0], pad_param[2], pad_param[0]
                    ])
                results.bboxes /= results.bboxes.new_tensor(  # 按缩放因子还原
                    scale_factor).repeat((1, 2))

            if cfg.get('yolox_style', False):  # yolox模式不做max_per_img裁剪
                cfg.max_per_img = len(results)  # 使用全部候选

            results = self._bbox_post_process(results=results,  # 执行后处理（含NMS与裁剪）
                                              cfg=cfg,
                                              rescale=False,
                                              with_nms=with_nms,
                                              img_meta=img_meta)
            results.bboxes[:, 0::2].clamp_(0, ori_shape[1])  # 约束x坐标到图像宽度
            results.bboxes[:, 1::2].clamp_(0, ori_shape[0])  # 约束y坐标到图像高度

            results_list.append(results)  # 保存当前图像结果
        return results_list  # 返回全部图像的检测结果

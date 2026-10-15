"""Block-wise Mona for frozen adaptation or joint YOLOv8 training.

This is a CNN adaptation of Mona, rather than a reproduction of the paper's
two adapters per Swin block. Original state-dict paths are retained so a
trained detector can be loaded without reinitializing its existing layers.
"""

from copy import deepcopy

import torch
from mmcv.cnn import ConvModule
from mmyolo.models.backbones import YOLOv8CSPDarknet
from mmyolo.models.layers import CSPLayerWithTwoConv, SPPFBottleneck
from mmyolo.models.utils import make_divisible, make_round
from mmyolo.registry import MODELS

from .mm_backbone import MultiModalYOLOBackbone
from ..layers.mona import Mona2d


class MonaCSPLayerWithTwoConv(CSPLayerWithTwoConv):
    """Adapt each internal bottleneck before it feeds the next block/concat."""

    def __init__(self, *args, mona_cfg=None, **kwargs):
        super().__init__(*args, **kwargs)
        cfg = dict(mona_cfg or {})
        enabled = cfg.pop('enabled', True)
        self.mona_adapters = torch.nn.ModuleList([
            Mona2d(self.mid_channels, **cfg) for _ in self.blocks
        ]) if enabled else None

    def forward(self, x):
        if self.mona_adapters is None:
            return super().forward(x)
        features = list(self.main_conv(x).split(
            (self.mid_channels, self.mid_channels), dim=1))
        for block, adapter in zip(self.blocks, self.mona_adapters):
            features.append(adapter(block(features[-1])))
        return self.final_conv(torch.cat(features, dim=1))


@MODELS.register_module()
class MonaYOLOv8CSPDarknet(YOLOv8CSPDarknet):
    """Train adapters with either a frozen or fully trainable image backbone.

    The default retains the earlier frozen-backbone behavior. Joint training
    uses ``freeze_original=False, frozen_stages=-1, norm_eval=False``.
    No enclosing no_grad/detach is used: earlier adapters need gradients
    propagated through subsequent frozen convolutions.
    """

    def __init__(self, mona_cfg=None, frozen_stages=4, norm_eval=True,
                 freeze_original=True, **kwargs):
        self.freeze_original = bool(freeze_original)
        required_stages = 4 if self.freeze_original else -1
        if frozen_stages != required_stages:
            raise ValueError('MonaYOLOv8CSPDarknet requires frozen_stages='
                             f'{required_stages} with '
                             f'freeze_original={self.freeze_original}.')
        if self.freeze_original and not norm_eval:
            raise ValueError('A frozen Mona backbone requires norm_eval=True '
                             'to preserve its trained BatchNorm statistics.')
        if kwargs.get('arch', 'P5') != 'P5':
            raise ValueError('This Mona adaptation currently supports P5.')
        self.mona_cfg = dict(bottleneck_dim=64, dropout=0.1, zero_init=True)
        self.mona_cfg.update(dict(mona_cfg or {}))
        self.mona_enabled = bool(self.mona_cfg.pop('enabled', True))
        # Upstream YOLOv8 mutates its last-stage class setting in __init__.
        # Keep this new backbone from changing other YOLOv8 instances.
        self.arch_settings = deepcopy(type(self).arch_settings)
        super().__init__(frozen_stages=frozen_stages, norm_eval=norm_eval,
                         **kwargs)
        self._freeze_stages()

    def build_stage_layer(self, stage_idx, setting):
        # Preserve the upstream stage layout and all original parameter names.
        in_channels, out_channels, num_blocks, add_identity, use_spp = setting
        in_channels = make_divisible(in_channels, self.widen_factor)
        out_channels = make_divisible(out_channels, self.widen_factor)
        num_blocks = make_round(num_blocks, self.deepen_factor)
        stage = [
            ConvModule(in_channels, out_channels, 3, stride=2, padding=1,
                       norm_cfg=self.norm_cfg, act_cfg=self.act_cfg),
            MonaCSPLayerWithTwoConv(
                out_channels, out_channels, num_blocks=num_blocks,
                add_identity=add_identity, norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                mona_cfg=dict(enabled=self.mona_enabled, **self.mona_cfg)),
        ]
        if use_spp:
            stage.append(SPPFBottleneck(
                out_channels, out_channels, kernel_sizes=5,
                norm_cfg=self.norm_cfg, act_cfg=self.act_cfg))
        return stage

    def _freeze_stages(self):
        if not self.freeze_original:
            # Respect EMA's requires_grad_(False), rather than re-enabling it.
            return super()._freeze_stages()
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for layer_name in self.layers:
            getattr(self, layer_name).eval()
        for module in self.modules():
            if isinstance(module, Mona2d):
                module.requires_grad_(True)
                module.train(self.training)


@MODELS.register_module()
class MonaMultiModalYOLOBackbone(MultiModalYOLOBackbone):
    """Avoid the ordinary wrapper accidentally freezing its new adapters.

    The default preserves the earlier frozen text/Prompt Bank behavior.
    Joint training uses frozen_stages=-1 and freeze_text_model=False; the
    text encoder's own frozen_modules setting keeps CLIP frozen while its
    added Prompt Bank trains.
    """

    def __init__(self, image_model, text_model, frozen_stages=4,
                 freeze_text_model=True, **kwargs):
        if frozen_stages not in (-1, 4):
            raise ValueError('MonaMultiModalYOLOBackbone supports '
                             'frozen_stages=-1 (joint) or 4 (frozen).')
        self.freeze_text_model = bool(freeze_text_model)
        super().__init__(image_model=image_model, text_model=text_model,
                         frozen_stages=frozen_stages, **kwargs)

    def _freeze_stages(self):
        if not isinstance(self.image_model, MonaYOLOv8CSPDarknet):
            raise TypeError('Use MonaYOLOv8CSPDarknet with the Mona wrapper.')
        if self.frozen_stages != self.image_model.frozen_stages:
            raise ValueError('Mona wrapper and image backbone must use the '
                             'same frozen_stages setting.')
        self.image_model._freeze_stages()
        if self.freeze_text_model and self.text_model is not None:
            self.text_model.requires_grad_(False)
            self.text_model.eval()

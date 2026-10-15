"""V1: relative C2fAttention evidence for dual-stream WSR, without Regional."""

import torch
import torch.nn.functional as F
from mmyolo.models.utils import make_divisible
from mmyolo.registry import MODELS

from ..layers.yolo_bricks import MaxSigmoidAttnBlock, WeakSupportRetriever
from .dual_stream_pafpn import DualStreamResidual, PriorFreeWeakSupportRetriever
from .yolo_world_pafpn import YOLOWorldPAFPN, YOLOWorldWSRPAFPN


@MODELS.register_module()
class SemanticDualStreamYOLOWorldWSRPAFPN(YOLOWorldWSRPAFPN):
    """Use raw A3 for routing and normalized A3/A4/A5 for WSR support.

    These detached maps are relative semantic evidence, not calibrated
    foreground probabilities. No visual objectness head or mask loss is used.
    """

    def __init__(self, *args, dual_stream_cfg=None, semantic_reliability_cfg=None,
                 **kwargs):
        if kwargs.get('fgbg_cfg') is not None:
            raise ValueError('Semantic V1 requires fgbg_cfg=None.')
        if kwargs.get('text_residual_cfg') is not None:
            raise ValueError('Semantic V1 does not use a text residual.')
        super().__init__(*args, **kwargs)
        if (not isinstance(self.wsr, WeakSupportRetriever)
                or isinstance(self.wsr, PriorFreeWeakSupportRetriever)):
            raise ValueError('Semantic V1 requires reliability-guided WSR.')
        if len(self.out_channels) != 3:
            raise ValueError('Semantic V1 expects exactly P3, P4 and P5.')
        cfg = dict(temperature=1.0, std_floor=1e-3, fusion_weights=(0.5, 0.25))
        supplied = semantic_reliability_cfg or {}
        if set(supplied) - set(cfg):
            raise ValueError('Unknown semantic reliability options.')
        cfg.update(supplied)
        self.temperature = float(cfg['temperature'])
        self.std_floor = float(cfg['std_floor'])
        self.fusion_weights = tuple(cfg['fusion_weights'])
        if self.temperature <= 0 or self.std_floor <= 0:
            raise ValueError('Temperature and std_floor must be positive.')
        if (len(self.fusion_weights) != 2
                or any(not 0 <= w <= 1 for w in self.fusion_weights)):
            raise ValueError('Two fusion weights in [0, 1] are required.')
        for attention in self._route_blocks():
            if not isinstance(attention, MaxSigmoidAttnBlock):
                raise TypeError('Semantic V1 requires MaxSigmoidAttnBlock.')
            attention.capture_route = True
        dual_cfg = dict(learnable_scale=True)
        dual_cfg.update(dual_stream_cfg or {})
        if not dual_cfg['learnable_scale']:
            raise ValueError('Semantic V1 uses a zero-init learnable scale.')
        self.dual_stream = DualStreamResidual(
            semantic_channels=make_divisible(self.out_channels[0], self.widen_factor),
            visual_channels=make_divisible(self.in_channels[0], self.widen_factor),
            **dual_cfg)
        self.collect_diagnostics = False
        self.last_diagnostics = {}
        self.last_routes = None
        self.last_semantic_maps = None
        self.last_reliability = None

    def _route_blocks(self):
        # Return references, without registering the attention modules twice.
        return (self.top_down_layers[-1].attn_block,
                self.bottom_up_layers[0].attn_block,
                self.bottom_up_layers[1].attn_block)

    def normalize_semantic(self, route):
        route = route.detach().float()
        mean = route.mean(dim=(-2, -1), keepdim=True)
        std = route.std(dim=(-2, -1), unbiased=False, keepdim=True)
        return ((route - mean) / std.clamp_min(self.std_floor)
                / self.temperature).sigmoid()

    def fuse_semantic(self, maps):
        s3, s4, s5 = (m.detach().float() for m in maps)
        s4, s5 = (F.interpolate(m, size=s3.shape[-2:], mode='bilinear',
                                align_corners=False) for m in (s4, s5))
        alpha, beta = self.fusion_weights
        return 1 - (1 - s3) * (1 - alpha * s4) * (1 - beta * s5)

    def forward(self, img_feats, txt_feats=None, txt_masks=None):
        self._prompt_bank_context = None
        self.last_routes = self.last_semantic_maps = self.last_reliability = None
        self.last_diagnostics = {}
        blocks = self._route_blocks()
        for attention in blocks:
            attention.last_route = None
            attention.route_text_mask = txt_masks
        try:
            # The parent WSR forward skips retrieval when Regional is absent.
            # Run the original PAFPN once, then explicitly apply V1 and WSR.
            results = YOLOWorldPAFPN.forward(
                self, img_feats, txt_feats, txt_masks=txt_masks)
            routes = tuple(attention.last_route for attention in blocks)
        finally:
            for attention in blocks:
                attention.route_text_mask = None
                attention.last_route = None
        if any(route is None for route in routes):
            raise RuntimeError('C2fAttention did not export all three route maps.')
        self.last_routes = tuple(route.detach() for route in routes)
        self.last_semantic_maps = tuple(self.normalize_semantic(a) for a in routes)
        reliability = self.fuse_semantic(self.last_semantic_maps)
        self.last_reliability = reliability
        self.dual_stream.collect_diagnostics = self.collect_diagnostics
        p3 = self.dual_stream(results[0], img_feats[0], routes[0])
        enhanced = self.wsr(p3, results[1], results[2], reliability)
        if self.collect_diagnostics:
            with torch.no_grad():
                self.last_diagnostics.update(self.dual_stream.last_diagnostics)
                self.last_diagnostics['wsr_residual_ratio'] = (
                    (enhanced.float() - p3.float()).square().mean().sqrt()
                    / p3.float().square().mean().sqrt().clamp_min(1e-6))
                for name, value in [*zip(('A3', 'A4', 'A5'), routes),
                                    ('R', reliability)]:
                    flat = value.detach().float().flatten()
                    self.last_diagnostics[name + '_mean'] = flat.mean()
                    self.last_diagnostics[name + '_std'] = flat.std(unbiased=False)
                    for label, quantile in zip(
                            ('p10', 'p50', 'p90'),
                            torch.quantile(flat, flat.new_tensor([0.1, 0.5, 0.9]))):
                        self.last_diagnostics[name + '_' + label] = quantile
        results = (enhanced, *results[1:])
        # GuardPrompt still consumes post-WSR features with no visual gradient.
        # Its reliability switches remain disabled in the 2.13 config.
        self._prompt_bank_context = (
            tuple(feat.detach() for feat in results), self.last_semantic_maps)
        return results

    def get_proposal_prior_maps(self):
        """Return relative semantic evidence without fabricating Regional masks."""
        return self.last_semantic_maps

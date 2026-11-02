"""OVC/WSR neck with optional final text-to-image residual attention."""

import copy
import torch
from mmyolo.models.utils import make_divisible
from mmyolo.registry import MODELS

from .semantic_dual_stream_pafpn import SemanticDualStreamYOLOWorldWSRPAFPN
from .yolo_world_pafpn import YOLOWorldPAFPN
from ..layers.ovc_layers import (DualSourceWeakSupportRetriever, OVCResidual,
                                 VisualObjectAdapter)


class _DisabledDualStream(torch.nn.Module):
    """Pass P3 through while retaining the interface used by OVC training."""

    def __init__(self):
        super().__init__()
        # The OVC optimizer reads out_proj.parameters().
        self.out_proj = torch.nn.Identity()
        self.scale_mode = 'disabled'
        self.collect_diagnostics = False
        self.last_diagnostics = {}

    def forward(self, semantic, *args):
        return semantic


@MODELS.register_module()
class OVCYOLOWorldWSRPAFPN(SemanticDualStreamYOLOWorldWSRPAFPN):
    def __init__(self, *args, ovc_enabled=True, adapter_enabled=True,
                 objectness_reweight=True, context_floor=0.25,
                 dual_stream_cfg=None, dual_stream_enabled=True, wsr_enabled=True,
                 text_cross_attn_cfg=None, dsg_cfg=None, **kwargs):
        if not ovc_enabled:
            legacy_wsr = dict(kwargs.get('wsr_cfg') or {})
            if legacy_wsr.get('type') == 'DualSourceWeakSupportRetriever':
                legacy_wsr.update(type='WeakSupportRetriever', reliability_log_weight=0.5)
                legacy_wsr.pop('num_visual_tokens', None)
                kwargs['wsr_cfg'] = legacy_wsr
        # The parent constructs the legacy residual before OVC replaces it.
        legacy_dual_cfg = {k: v for k, v in (dual_stream_cfg or {}).items()
                           if k not in ('scale_mode', 'fixed_scale', 'scale_warmup_iters')}
        super().__init__(*args, dual_stream_cfg=legacy_dual_cfg, **kwargs)
        self.wsr_enabled = bool(wsr_enabled)
        if not self.wsr_enabled:
            # Keep the inherited architecture intact while removing WSR and
            # all of its parameters from this ablation model.
            self.wsr = None
        self.ovc_enabled = ovc_enabled
        self.adapter_enabled = adapter_enabled and ovc_enabled
        self.dual_stream_enabled = bool(dual_stream_enabled)
        if (self.ovc_enabled and self.wsr_enabled
                and not isinstance(self.wsr, DualSourceWeakSupportRetriever)):
            raise ValueError('OVC requires DualSourceWeakSupportRetriever.')
        if (self.ovc_enabled and self.wsr_enabled
                and not self.adapter_enabled and self.wsr.num_visual_tokens):
            raise ValueError('Visual support slots require an Adapter.')
        # Also construct then remove the whole ablated branch to preserve the
        # initialization RNG stream of the remaining modules under the same seed.
        if self.adapter_enabled or (self.ovc_enabled and not self.dual_stream_enabled):
            cfg = dict(hidden_channels=64, norm_groups=8, residual_scale=0.1,
                       learnable_scale=True)
            cfg.update(dual_stream_cfg or {})
            c3 = make_divisible(self.in_channels[0], self.widen_factor)
            p3 = make_divisible(self.out_channels[0], self.widen_factor)
            self.visual_adapter = VisualObjectAdapter(c3, cfg['hidden_channels'],
                                                     cfg['norm_groups'])
            self.dual_stream = OVCResidual(
                semantic_channels=p3, visual_channels=c3,
                objectness_reweight=objectness_reweight,
                context_floor=context_floor, **cfg)
            if not self.adapter_enabled:
                del self.visual_adapter
        if not self.dual_stream_enabled:
            # Construct the original branch first to preserve the seed's
            # initialization order for all other modules, then remove it.
            self.dual_stream = _DisabledDualStream()
        self.main_visual = self.main_objectness_logits = None
        self.last_visual_scores = None
        self.last_support_features = None
        self.dsg = None
        if dsg_cfg is not None:
            cfg = copy.deepcopy(dsg_cfg)
            cfg.setdefault('text_channels', self.guide_channels)
            self.dsg = MODELS.build(cfg)
            if self.dsg.text_channels != self.guide_channels:
                raise ValueError('DSG output must match the neck guide channels.')
        self.text_cross_attn = None
        if text_cross_attn_cfg is not None:
            cfg = copy.deepcopy(text_cross_attn_cfg)
            cfg.setdefault('in_channels', [
                make_divisible(c, self.widen_factor) for c in self.out_channels])
            cfg.setdefault('text_channels', self.guide_channels)
            self.text_cross_attn = MODELS.build(cfg)

    def clear_auxiliary_cache(self):
        self.main_visual = self.main_objectness_logits = None

    def forward(self, img_feats, txt_feats=None, txt_masks=None):
        self.clear_auxiliary_cache()
        self.last_visual_scores = None
        self.last_support_features = None
        guide_feats = txt_feats
        if self.dsg is not None:
            self.dsg.collect_diagnostics = self.collect_diagnostics
            guide_feats = self.dsg(txt_feats, txt_masks)
        if not self.ovc_enabled:
            results = super().forward(img_feats, guide_feats, txt_masks)
            if self.dsg is not None and self.collect_diagnostics:
                self.last_diagnostics.update(self.dsg.last_diagnostics)
            return self._apply_text_cross_attention(results, txt_feats, txt_masks)
        self._prompt_bank_context = None
        self.last_diagnostics = {}
        if self.dsg is not None and self.collect_diagnostics:
            self.last_diagnostics.update(self.dsg.last_diagnostics)
        self.last_routes = self.last_semantic_maps = self.last_reliability = None
        blocks = self._route_blocks()
        for block in blocks:
            block.last_route = None
            block.route_text_mask = txt_masks
        try:
            results = YOLOWorldPAFPN.forward(self, img_feats, guide_feats, txt_masks)
            routes = tuple(block.last_route for block in blocks)
        finally:
            for block in blocks:
                block.last_route = block.route_text_mask = None
        if any(route is None for route in routes):
            raise RuntimeError('Missing C2fAttention route map.')
        self.last_routes = tuple(route.detach() for route in routes)
        self.last_semantic_maps = tuple(self.normalize_semantic(a) for a in routes)
        semantic = self.last_semantic_maps[0]
        self.last_reliability = semantic
        self.dual_stream.collect_diagnostics = self.collect_diagnostics
        if self.adapter_enabled:
            visual, logits = self.visual_adapter(img_feats[0])
            self.main_visual, self.main_objectness_logits = visual, logits
            objectness = logits.detach().float().sigmoid()
            self.last_visual_scores = objectness * (1 - semantic)
            p3 = self.dual_stream(results[0], visual, semantic, objectness)
        else:
            p3 = self.dual_stream(results[0], img_feats[0], semantic)
        self.last_support_features = p3.detach()
        if self.wsr_enabled:
            enhanced = self.wsr(p3, results[1], results[2], semantic,
                                self.last_visual_scores)
        else:
            enhanced = p3
        if self.collect_diagnostics:
            with torch.no_grad():
                self.last_diagnostics.update(self.dual_stream.last_diagnostics)
                if self.wsr_enabled:
                    self.last_diagnostics['wsr_residual_ratio'] = (
                        (enhanced.float() - p3.float()).square().mean().sqrt()
                        / p3.float().square().mean().sqrt().clamp_min(1e-6))
                for name, value in [('S3', semantic),
                                    ('O', self.main_objectness_logits.detach().float().sigmoid()
                                     if self.adapter_enabled else None)]:
                    if value is not None:
                        self.last_diagnostics[name + '_mean'] = value.mean()
                        self.last_diagnostics[name + '_std'] = value.std(unbiased=False)
        results = (enhanced, *results[1:])
        self._prompt_bank_context = (
            tuple(feat.detach() for feat in results), self.last_semantic_maps)
        return self._apply_text_cross_attention(results, txt_feats, txt_masks)

    def _apply_text_cross_attention(self, results, txt_feats, txt_masks):
        # Cache PromptBank inputs before this step. Only the main detection
        # features receive the new residual; OVC/WSR routing stays upstream.
        if self.text_cross_attn is not None:
            self.text_cross_attn.collect_diagnostics = self.collect_diagnostics
            results = self.text_cross_attn(results, txt_feats, txt_masks=txt_masks)
            if self.collect_diagnostics:
                self.last_diagnostics.update(self.text_cross_attn.last_diagnostics)
        return results

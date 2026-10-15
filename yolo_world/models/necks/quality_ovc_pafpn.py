"""Localization quality with replacement or O-preserving additive OVC."""

import torch
import torch.nn.functional as F
from mmyolo.models.utils import make_divisible
from mmyolo.registry import MODELS

from .ovc_pafpn import OVCYOLOWorldWSRPAFPN
from .yolo_world_pafpn import YOLOWorldPAFPN
from ..layers.localization_quality import LocalizationVisualAdapter


@MODELS.register_module()
class QualityOVCYOLOWorldWSRPAFPN(OVCYOLOWorldWSRPAFPN):
    def __init__(self, *args, quality_head_cfg=None, quality_wsr_enabled=True,
                 quality_wsr_start_iters=1000, quality_wsr_warmup_iters=1000,
                 objectness_reweight=False, retain_objectness=False, **kwargs):
        if objectness_reweight and not retain_objectness:
            raise ValueError('DIOR 3.3 replaces O; objectness_reweight must be False.')
        super().__init__(*args, objectness_reweight=objectness_reweight, **kwargs)
        self.retain_objectness = bool(retain_objectness)
        if not self.adapter_enabled:
            raise ValueError('Localization supervision requires the visual Adapter.')
        if min(quality_wsr_start_iters, quality_wsr_warmup_iters) < 0:
            raise ValueError('Quality support schedules must be non-negative.')
        if not self.retain_objectness:
            self.visual_adapter = LocalizationVisualAdapter(self.visual_adapter.adapter)
        cfg = dict(type='ClassAgnosticLocalizationHead', hidden_channels=64,
                   norm_groups=8, reg_max=16, featmap_strides=(8, 16, 32))
        cfg.update(quality_head_cfg or {})
        cfg.update(in_channels=[make_divisible(c, self.widen_factor) for c in self.out_channels],
                   visual_channels=self.visual_adapter.adapter[0].out_channels)
        self.localization_head = MODELS.build(cfg)
        self.quality_wsr_enabled = bool(quality_wsr_enabled)
        self.quality_wsr_start_iters = int(quality_wsr_start_iters)
        self.quality_wsr_warmup_iters = int(quality_wsr_warmup_iters)
        self.register_buffer('quality_iteration', torch.zeros((), dtype=torch.long))
        self.clear_auxiliary_cache()

    def clear_auxiliary_cache(self):
        super().clear_auxiliary_cache()
        self.coarse_boxes = self.coarse_distributions = self.coarse_quality_logits = None
        self.quality_features = None

    def quality_support_fraction(self):
        if not self.quality_wsr_enabled:
            return 0.0
        if not self.training:
            return 1.0
        elapsed = max(0, int(self.quality_iteration) + 1 - self.quality_wsr_start_iters)
        if self.quality_wsr_warmup_iters:
            return min(elapsed / self.quality_wsr_warmup_iters, 1.0)
        return float(elapsed > 0)

    @torch.no_grad()
    def support_prior(self, p3, quality, objectness_prior=None):
        fraction = self.quality_support_fraction()
        if objectness_prior is not None and fraction == 0:
            # Exact original O ranking during warmup or the quality ablation.
            return objectness_prior
        quality = [F.interpolate(q.detach().float().sigmoid(), p3.shape[-2:],
                                 mode='bilinear', align_corners=False) for q in quality]
        # This max-fused map ranks support positions; it is NOT a box's IoU.
        quality = torch.stack(quality).amax(0)
        if objectness_prior is not None:
            # Add quality to O*(1-S3); never replace the original O predictor,
            # its dual-stream gate, or any WSR support source/budget.
            return objectness_prior * ((1 - fraction) + fraction * quality)
        # Fixed support budget even when quality is disabled or warming up.
        # Only the former visual slots change source; all P3 queries survive.
        kernel = self.wsr.detail_kernel if self.wsr_enabled else 5
        detail = p3.detach().float()
        detail = (detail - F.avg_pool2d(detail, kernel, stride=1, padding=kernel // 2)).square().mean(1, keepdim=True).sqrt()
        low, high = detail.amin((-2, -1), keepdim=True), detail.amax((-2, -1), keepdim=True)
        detail = (detail - low) / (high - low).clamp_min(1e-6)
        return (1 - fraction) * detail + fraction * quality

    def forward(self, img_feats, txt_feats=None, txt_masks=None):
        self.clear_auxiliary_cache()
        self.last_visual_scores = None
        self.last_support_features = None
        self._prompt_bank_context = None
        self.last_diagnostics = {}
        self.last_routes = self.last_semantic_maps = self.last_reliability = None
        blocks = self._route_blocks()
        for block in blocks:
            block.last_route = None
            block.route_text_mask = txt_masks
        try:
            results = YOLOWorldPAFPN.forward(self, img_feats, txt_feats, txt_masks)
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
        visual, logits = self.visual_adapter(img_feats[0])
        self.main_visual, self.main_objectness_logits = visual, logits
        objectness = logits.detach().float().sigmoid() if logits is not None else None
        objectness_prior = objectness * (1 - semantic) if objectness is not None else None
        p3 = self.dual_stream(results[0], visual, semantic, objectness)
        pre = (p3, *results[1:])
        coarse, raw, quality = self.localization_head(pre, visual)
        self.coarse_boxes, self.coarse_distributions, self.coarse_quality_logits = coarse, raw, quality
        self.last_visual_scores = self.support_prior(p3, quality, objectness_prior)
        self.last_support_features = p3.detach()
        enhanced = (self.wsr(p3, results[1], results[2], semantic, self.last_visual_scores)
                    if self.wsr_enabled else p3)
        results = (enhanced, *results[1:])
        self.quality_features = tuple(f.detach() for f in results)
        self._prompt_bank_context = (self.quality_features, self.last_semantic_maps)
        if self.collect_diagnostics:
            with torch.no_grad():
                self.last_diagnostics.update(self.dual_stream.last_diagnostics)
                self.last_diagnostics.update(
                    quality_support_fraction=p3.new_tensor(self.quality_support_fraction()),
                    q0_mean=torch.cat([q.detach().float().sigmoid().flatten() for q in quality]).mean(),
                    S3_mean=semantic.mean(), S3_std=semantic.std(unbiased=False))
                if objectness is not None:
                    self.last_diagnostics.update(
                        O_mean=objectness.mean(), O_std=objectness.std(unbiased=False))
                if self.wsr_enabled:
                    self.last_diagnostics['wsr_residual_ratio'] = (
                        (enhanced.float() - p3.float()).square().mean().sqrt()
                        / p3.float().square().mean().sqrt().clamp_min(1e-6))
        return results

    def quality_for_final_boxes(self, boxes):
        if self.main_visual is None or self.quality_features is None:
            raise RuntimeError('Run the quality neck before evaluating final boxes.')
        return self.localization_head.quality_for_boxes(self.quality_features, self.main_visual, boxes)

    def get_proposal_prior_maps(self):
        if self.coarse_quality_logits is None:
            return None
        return tuple(q.detach().float().sigmoid() for q in self.coarse_quality_logits)

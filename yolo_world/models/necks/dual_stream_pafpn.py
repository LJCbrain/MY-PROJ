"""P3 semantic/visual soft routing before same-image support retrieval."""

import torch
from torch import nn
import torch.nn.functional as F

from mmyolo.models.utils import make_divisible
from mmyolo.registry import MODELS

from ..layers.yolo_bricks import MaxSigmoidAttnBlock, WeakSupportRetriever
from .yolo_world_pafpn import YOLOWorldPAFPN, YOLOWorldWSRPAFPN


class DualStreamResidual(nn.Module):
    """Fuse text-aware P3 and pre-text backbone C3 using a detached route."""

    def __init__(self, semantic_channels, visual_channels, hidden_channels=64,
                 norm_groups=8, residual_scale=0.1, learnable_scale=False):
        super().__init__()
        if norm_groups <= 0 or hidden_channels <= 0 or hidden_channels % norm_groups:
            raise ValueError('hidden_channels must be divisible by norm_groups.')
        if residual_scale <= 0:
            raise ValueError('residual_scale must be positive for zero-init fusion.')
        self.residual_scale = float(residual_scale)
        self.learnable_scale = bool(learnable_scale)
        self.collect_diagnostics = False
        self.last_diagnostics = {}
        if self.learnable_scale:
            self.residual_alpha = nn.Parameter(torch.zeros(()))

        def projection(channels):
            return nn.Sequential(
                nn.Conv2d(channels, hidden_channels, 1, bias=False),
                nn.GroupNorm(norm_groups, hidden_channels),
                nn.SiLU())

        def transform():
            return nn.Sequential(
                nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1,
                          groups=hidden_channels, bias=False),
                nn.SiLU(),
                nn.Conv2d(hidden_channels, hidden_channels, 1))

        self.semantic_proj = projection(semantic_channels)
        self.visual_proj = projection(visual_channels)
        self.semantic_transform = transform()
        self.visual_transform = transform()
        self.out_proj = nn.Conv2d(hidden_channels * 2, semantic_channels, 1)
        if self.learnable_scale:
            # Only the residual coefficient starts at zero. A nonzero output
            # projection lets that coefficient receive a gradient immediately.
            nn.init.kaiming_normal_(self.out_proj.weight, nonlinearity='linear')
        else:
            # Preserve the original 2.12 initialization and checkpoint layout.
            nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, semantic, visual, route):
        size = semantic.shape[-2:]
        visual = self.visual_proj(visual)
        if visual.shape[-2:] != size:
            visual = F.interpolate(visual, size=size, mode='bilinear',
                                   align_corners=False)
        route = route.detach().to(device=semantic.device, dtype=semantic.dtype)
        if route.shape[-2:] != size:
            route = F.interpolate(route, size=size, mode='bilinear',
                                  align_corners=False)
        if route.shape[:2] != (semantic.shape[0], 1):
            raise ValueError('Dual-stream route must have shape B x 1 x H x W.')
        primary = self.semantic_transform(self.semantic_proj(semantic) * route)
        complement = self.visual_transform(visual * (1 - route))
        update = self.out_proj(torch.cat((primary, complement), dim=1))
        scale = (self.residual_scale * self.residual_alpha.tanh()
                 if self.learnable_scale else self.residual_scale)
        residual = scale * update
        self.last_diagnostics = {}
        if self.collect_diagnostics:
            with torch.no_grad():
                self.last_diagnostics = dict(
                    dual_lambda=torch.as_tensor(scale).detach().clone(),
                    dual_residual_ratio=(residual.float().square().mean().sqrt()
                                         / semantic.detach().float().square()
                                         .mean().sqrt().clamp_min(1e-6)))
        return semantic + residual


@MODELS.register_module()
class PriorFreeWeakSupportRetriever(WeakSupportRetriever):
    """Use detail peaks and a coverage grid, without a foreground prior.

    Keep the inherited Q/K/V, learned gate and residual parameter names so WSR
    checkpoints remain loadable. Uniform scores only adapt the old forward
    API; they neither rank supports nor bias attention.
    """

    def __init__(self, *args, num_reliable_tokens=0, num_detail_tokens=192,
                 reliability_log_weight=0.0, **kwargs):
        if num_reliable_tokens != 0 or reliability_log_weight != 0:
            raise ValueError('Prior-free WSR cannot use reliability tokens/weights.')
        super().__init__(
            *args, num_reliable_tokens=0, num_detail_tokens=num_detail_tokens,
            reliability_log_weight=0.0, **kwargs)
        if self.num_support_tokens < 2:
            raise ValueError('WSR needs at least two support slots for entropy.')

    def _select_support_indices(self, p3, reliability=None):
        # Selection is discrete; avoid building an unused autograd graph.
        with torch.no_grad():
            b, _, h, w = p3.shape
            n = h * w
            target = self.num_support_tokens
            detail = p3.detach().float()
            detail = detail - F.avg_pool2d(
                detail, self.detail_kernel, stride=1,
                padding=self.detail_kernel // 2)
            scores = detail.square().mean(dim=1, keepdim=True).sqrt()
            peaks = scores
            if self.reliability_nms_kernel > 1:
                pooled = F.max_pool2d(
                    scores, self.reliability_nms_kernel, stride=1,
                    padding=self.reliability_nms_kernel // 2)
                peaks = scores.masked_fill(scores < pooled, -torch.inf)
            values, indices = peaks.flatten(1).topk(
                min(self.num_detail_tokens, n), dim=1)
            # Fewer true peaks than the budget: fill later using detail rank,
            # rather than accepting arbitrary indices with -inf peak scores.
            indices = indices.masked_fill(~torch.isfinite(values), n)
            grid = self._grid_indices(h, w, p3.device).expand(b, -1)
            fallback = scores.flatten(1).topk(min(target, n), dim=1).indices
            candidates = torch.cat((indices, grid, fallback), dim=1)
            count = candidates.shape[1]
            order = torch.arange(count, device=p3.device)
            same = candidates.unsqueeze(2) == candidates.unsqueeze(1)
            first = torch.where(same, order.view(1, 1, count), count).min(2).values
            keep = (first == order[None]) & (candidates < n)
            ranks = torch.where(keep, order[None], count)
            unique_count = min(target, n)
            positions = ranks.topk(unique_count, dim=1, largest=False).indices
            selected = candidates.gather(1, positions)
            # Tiny feature maps cannot supply 256 distinct positions.
            repeat = torch.arange(target, device=p3.device) % unique_count
            return selected[:, repeat]

    def forward(self, p3, p4, p5, reliability=None):
        if reliability is not None:
            raise ValueError('Prior-free WSR does not accept a reliability map.')
        neutral = p3.new_ones((p3.shape[0], 1, *p3.shape[-2:]))
        return super().forward(p3, p4, p5, neutral)


@MODELS.register_module()
class DualStreamYOLOWorldWSRPAFPN(YOLOWorldWSRPAFPN):
    """E2 dual stream between Regional feature processing and WSR.

    The original pre-fusion reliability guides support selection and attention;
    it never masks the dual-stream output. Explicit prior-free configurations
    remain supported for the earlier ablation.
    """

    def __init__(self, *args, dual_stream_cfg=None, **kwargs):
        if kwargs.get('text_residual_cfg') is not None:
            raise ValueError('This neck does not use a prior-based text residual.')
        super().__init__(*args, **kwargs)
        if not isinstance(self.wsr, WeakSupportRetriever):
            raise ValueError('This neck requires a WeakSupportRetriever.')
        prior_free = isinstance(self.wsr, PriorFreeWeakSupportRetriever)
        if prior_free != (self.fgbg_gate is None):
            raise ValueError(
                'Use Regional with WeakSupportRetriever, or explicitly disable '
                'the prior with PriorFreeWeakSupportRetriever.')
        attention = self.top_down_layers[-1].attn_block
        if not isinstance(attention, MaxSigmoidAttnBlock):
            raise TypeError('Dual-stream routing requires MaxSigmoidAttnBlock.')
        attention.capture_route = True
        self.dual_stream = DualStreamResidual(
            semantic_channels=make_divisible(self.out_channels[0], self.widen_factor),
            visual_channels=make_divisible(self.in_channels[0], self.widen_factor),
            **(dual_stream_cfg or {}))
        self.last_route = None

    def forward(self, img_feats, txt_feats=None, txt_masks=None):
        self._prompt_bank_context = None
        self.last_route = None
        attention = self.top_down_layers[-1].attn_block
        attention.last_route = None
        attention.route_text_mask = txt_masks
        try:
            # Run exactly one original PAFPN and its optional Regional gate.
            # Insert dual fusion before WSR, preserving the pre-fusion prior.
            results = YOLOWorldPAFPN.forward(
                self, img_feats, txt_feats, txt_masks=txt_masks)
            route = attention.last_route
        finally:
            attention.route_text_mask = None
            attention.last_route = None
        if route is None:
            raise RuntimeError('P3 attention did not export a routing map.')
        self.last_route = route.detach()
        reliability_maps = (
            self._fgbg_parts.get('reliability_maps') if self._fgbg_parts else None)
        if self.fgbg_gate is not None and not reliability_maps:
            raise RuntimeError('Regional did not return WSR reliability maps.')
        p3 = self.dual_stream(results[0], img_feats[0], route)
        if reliability_maps:
            reliability = self._fgbg_parts.get(
                'fused_reliability', reliability_maps[0])
            p3 = self.wsr(p3, results[1], results[2], reliability.detach())
        else:
            p3 = self.wsr(p3, results[1], results[2])
        results = (p3, *results[1:])
        # Preserve GuardPrompt's post-WSR context and gradient isolation.
        self._prompt_bank_context = (
            tuple(feat.detach() for feat in results),
            tuple(prior.detach() for prior in reliability_maps)
            if reliability_maps else None)
        return results

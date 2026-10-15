"""Single-model visual representation and dual-source retrieval for DIOR 2.14."""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from mmyolo.registry import MODELS

from .yolo_bricks import WeakSupportRetriever
from ..necks.dual_stream_pafpn import DualStreamResidual


def _stable_descending_indices(scores):
    """Keep equal-score pixel order on PyTorch versions without stable argsort."""
    try:
        return torch.sort(scores, dim=-1, descending=True, stable=True).indices
    except TypeError as error:
        if 'stable' not in str(error):
            raise
        # Older builds may also lack stable sort. Preserve ties exactly instead
        # of perturbing scores or silently switching to an unstable ordering.
        order = np.argsort(-scores.detach().cpu().numpy(), axis=-1, kind='mergesort')
        return torch.as_tensor(order.copy(), device=scores.device, dtype=torch.long)


class VisualObjectAdapter(nn.Module):
    def __init__(self, in_channels, hidden_channels=64, norm_groups=8):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
            nn.GroupNorm(norm_groups, hidden_channels), nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1,
                      groups=hidden_channels, bias=False),
            nn.SiLU(), nn.Conv2d(hidden_channels, hidden_channels, 1))
        self.objectness = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1,
                      groups=hidden_channels, bias=False),
            nn.SiLU(), nn.Conv2d(hidden_channels, 1, 1))

    def forward(self, c3):
        visual = self.adapter(c3.detach())
        return visual, self.objectness(visual)


class OVCResidual(DualStreamResidual):
    """Use already projected visual features; detection never trains Adapter."""
    def __init__(self, *args, objectness_reweight=True, context_floor=0.25,
                 scale_mode='learnable', fixed_scale=0.02, scale_warmup_iters=1000, **kwargs):
        super().__init__(*args, **kwargs)
        if scale_mode not in ('learnable', 'fixed'):
            raise ValueError('scale_mode must be learnable or fixed.')
        self.scale_mode = scale_mode
        if scale_mode == 'fixed':
            if fixed_scale <= 0 or scale_warmup_iters < 0:
                raise ValueError('Invalid fixed scale or warmup.')
            if hasattr(self, 'residual_alpha'):
                del self.residual_alpha
            self.fixed_scale = float(fixed_scale)
            self.scale_warmup_iters = int(scale_warmup_iters)
            self.register_buffer('scale_iteration', torch.zeros((), dtype=torch.long))
            nn.init.kaiming_normal_(self.out_proj.weight, nonlinearity='linear')
            nn.init.zeros_(self.out_proj.bias)
        # The Adapter owns the visual projection, not the detection branch.
        del self.visual_proj
        if not 0 <= context_floor <= 1:
            raise ValueError('context_floor must be in [0, 1].')
        self.objectness_reweight = objectness_reweight
        self.context_floor = context_floor

    def forward(self, semantic, visual, route, objectness):
        route = route.detach().to(semantic)
        visual = visual.detach().to(semantic)
        if visual.shape[-2:] != semantic.shape[-2:]:
            visual = F.interpolate(visual, semantic.shape[-2:], mode='bilinear',
                                   align_corners=False)
        primary = self.semantic_transform(self.semantic_proj(semantic) * route)
        complement = self.visual_transform(visual * (1 - route))
        if self.objectness_reweight:
            gate = self.context_floor + (1 - self.context_floor) * objectness.detach()
            complement = complement * gate.to(complement)
        update = self.out_proj(torch.cat((primary, complement), dim=1))
        if self.scale_mode == 'fixed':
            fraction = (min((int(self.scale_iteration) + 1) / self.scale_warmup_iters, 1.)
                        if self.training and self.scale_warmup_iters else 1.)
            scale = semantic.new_tensor(self.fixed_scale * fraction)
        else:
            scale = self.residual_scale * self.residual_alpha.tanh()
        residual = scale * update
        self.last_diagnostics = {}
        if self.collect_diagnostics:
            with torch.no_grad():
                self.last_diagnostics = dict(
                    dual_lambda=scale.detach().clone(),
                    dual_update_rms=update.detach().float().square().mean().sqrt(),
                    dual_residual_ratio=(residual.float().square().mean().sqrt()
                                         / semantic.detach().float().square()
                                         .mean().sqrt().clamp_min(1e-6)))
        return semantic + residual


@MODELS.register_module()
class DualSourceWeakSupportRetriever(WeakSupportRetriever):
    """Reserve visual slots, without feeding semantic scores into retrieval.

    Source ids: 0 semantic, 1 visual, 2 detail, 3 grid, 4 final fallback.
    Q/K/V and checkpoint names are inherited from the existing retriever.
    """
    def __init__(self, *args, num_visual_tokens=32,
                 reliability_log_weight=0.0, **kwargs):
        super().__init__(*args, reliability_log_weight=reliability_log_weight,
                         **kwargs)
        if reliability_log_weight != 0:
            raise ValueError('Dual-source retrieval uses ranking, not log priors.')
        if not 0 <= num_visual_tokens <= self.num_reliable_tokens:
            raise ValueError('Visual slots must fit the reliable support budget.')
        if self.num_support_tokens < 2:
            raise ValueError('At least two support slots are required.')
        self.num_visual_tokens = num_visual_tokens
        self._visual_scores = None
        self.last_support_indices = None
        self.last_support_sources = None

    def forward(self, p3, p4, p5, reliability, visual_scores=None):
        if self.num_visual_tokens and visual_scores is None:
            raise ValueError('Visual support slots require visual scores.')
        self._visual_scores = visual_scores
        self.last_support_indices = self.last_support_sources = None
        try:
            return super().forward(p3, p4, p5, reliability)
        finally:
            self._visual_scores = None

    @torch.no_grad()
    def _select_support_indices(self, p3, reliability):
        b, _, h, w = p3.shape
        n = h * w
        target = min(self.num_support_tokens, n)
        detail = p3.detach().float()
        detail = detail - F.avg_pool2d(detail, self.detail_kernel, stride=1,
                                      padding=self.detail_kernel // 2)
        detail = detail.square().mean(1, keepdim=True).sqrt()
        visual = (self._visual_scores.detach().float()
                  if self._visual_scores is not None else torch.zeros_like(reliability))
        maps = [reliability.detach().float(), visual, detail]
        peaks = []
        for scores in maps:
            if self.reliability_nms_kernel > 1:
                maximum = F.max_pool2d(scores, self.reliability_nms_kernel,
                                      stride=1, padding=self.reliability_nms_kernel // 2)
                scores = scores.masked_fill(scores < maximum, -torch.inf)
            peaks.append(scores.flatten(1))
        budgets = [self.num_reliable_tokens - self.num_visual_tokens,
                   self.num_visual_tokens, self.num_detail_tokens]
        grid = self._grid_indices(h, w, p3.device).unique(sorted=False)
        all_indices, all_sources = [], []
        for batch_idx in range(b):
            used = torch.zeros(n, dtype=torch.bool, device=p3.device)
            selected, sources = [], []

            def take(order, quota, source):
                order = order[~used[order]][:max(quota, 0)]
                used[order] = True
                selected.append(order)
                sources.append(torch.full_like(order, source))
                return order.numel()

            for source, (scores, peak, quota) in enumerate(zip(maps, peaks, budgets)):
                peak_row = peak[batch_idx]
                order = _stable_descending_indices(peak_row)
                order = order[torch.isfinite(peak_row[order])]
                count = take(order, quota, source)
                dense = _stable_descending_indices(scores[batch_idx].flatten())
                take(dense, quota - count, source)
            take(grid, self.num_grid_tokens, 3)
            count = sum(x.numel() for x in selected)
            take(_stable_descending_indices(reliability[batch_idx].flatten()),
                 target - count, 4)
            indices = torch.cat(selected)[:target]
            source_ids = torch.cat(sources)[:target]
            repeat = torch.arange(self.num_support_tokens, device=p3.device) % target
            all_indices.append(indices[repeat])
            all_sources.append(source_ids[repeat])
        self.last_support_indices = torch.stack(all_indices)
        self.last_support_sources = torch.stack(all_sources)
        return self.last_support_indices

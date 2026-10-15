"""Weak box supervision and same-instance consistency, with no teacher."""

from contextlib import contextmanager
import math

import torch
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
from torchvision.ops import roi_align
from torchvision.transforms import functional as TVF


@contextmanager
def auxiliary_backbone_context(module):
    """Preserve batch-stat normalization, but do not update running buffers."""
    changed = [(m, m.track_running_stats) for m in module.modules()
               if isinstance(m, _BatchNorm) and m.training]
    try:
        for m, _ in changed:
            m.track_running_stats = False
        with torch.no_grad():
            yield
    finally:
        for m, old in changed:
            m.track_running_stats = old


def augment_view(inputs, preprocessor, amount=0.2, hue=0.02, grayscale_prob=0.1):
    """Undo actual input normalization; torchvision adjustments expect RGB 0..1."""
    if not (0 <= amount <= 1 and 0 <= hue <= 0.5 and 0 <= grayscale_prob <= 1):
        raise ValueError('Invalid photometric augmentation range.')
    normalized = getattr(preprocessor, '_enable_normalize', False)
    pixels = inputs.detach().float()
    if normalized:
        pixels = pixels * preprocessor.std + preprocessor.mean
    # YOLOW loads OpenCV BGR unless the preprocessor converts channels.
    rgb_input = getattr(preprocessor, '_channel_conversion', False)
    if not rgb_input:
        pixels = pixels[:, [2, 1, 0]]
    pixels = (pixels / 255.0).clamp(0, 1)
    views = []
    for rgb in pixels:
        factors = torch.empty(4).uniform_(-1, 1).tolist()
        rgb = TVF.adjust_brightness(rgb, 1 + amount * factors[0])
        rgb = TVF.adjust_contrast(rgb, 1 + amount * factors[1])
        rgb = TVF.adjust_saturation(rgb, 1 + amount * factors[2])
        rgb = TVF.adjust_hue(rgb, hue * factors[3])
        if torch.rand(()) < grayscale_prob:
            rgb = TVF.rgb_to_grayscale(rgb, num_output_channels=3)
        views.append(rgb)
    output = torch.stack(views).clamp(0, 1) * 255.0
    if not rgb_input:
        output = output[:, [2, 1, 0]]
    if normalized:
        output = (output - preprocessor.mean) / preprocessor.std
    return output.to(inputs)


def collect_rois(samples, input_shape, batch_size, device):
    """Return clipped Nx5 [image, x1,y1,x2,y2] in current input coordinates."""
    if isinstance(samples, dict):
        packed = samples.get('bboxes_labels')
        rois = (packed[:, [0, 2, 3, 4, 5]].detach().to(device=device, dtype=torch.float32)
                if packed is not None else torch.empty(0, 5, device=device))
    else:
        rows = []
        for i, sample in enumerate(samples):
            boxes = sample.gt_instances.bboxes
            boxes = boxes.tensor if hasattr(boxes, 'tensor') else boxes
            boxes = boxes.detach().to(device=device, dtype=torch.float32)
            rows.append(torch.cat((boxes.new_full((len(boxes), 1), i), boxes), 1))
        rois = torch.cat(rows) if rows else torch.empty(0, 5, device=device)
    rois = rois.clone()
    height, width = input_shape
    rois[:, [1, 3]] = rois[:, [1, 3]].clamp(0, width)
    rois[:, [2, 4]] = rois[:, [2, 4]].clamp(0, height)
    valid = (torch.isfinite(rois).all(1) & (rois[:, 0] >= 0)
             & (rois[:, 0] < batch_size) & (rois[:, 0] == rois[:, 0].floor())
             & (rois[:, 3] > rois[:, 1]) & (rois[:, 4] > rois[:, 2]))
    return rois[valid]


def scale_rois(rois, input_shape, feature_shape):
    result = rois.clone()
    result[:, [1, 3]] *= feature_shape[1] / input_shape[1]
    result[:, [2, 4]] *= feature_shape[0] / input_shape[0]
    return result


@torch.no_grad()
def box_trimaps(rois, batch_size, shape, input_shape, core_ratio=0.8,
                ignore_ratio=1.1):
    """Fractional core-cell coverage, inverse-instance-area positive weighting.

    This is weak box supervision, not an instance segmentation target.
    Any core overlap makes a positive cell; ignore covers the expanded box.
    """
    height, width = shape
    positive = torch.zeros(batch_size, 1, height, width, device=rois.device)
    ignored = torch.zeros_like(positive, dtype=torch.bool)
    scaled = scale_rois(rois, input_shape, shape)
    for image_idx, x1, y1, x2, y2 in scaled.cpu().tolist():
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        bw, bh = x2 - x1, y2 - y1
        for ratio, is_core in ((ignore_ratio, False), (core_ratio, True)):
            left, right = max(0, cx - bw * ratio / 2), min(width, cx + bw * ratio / 2)
            top, bottom = max(0, cy - bh * ratio / 2), min(height, cy + bh * ratio / 2)
            l, r = max(0, math.floor(left)), min(width, math.ceil(right))
            t, b = max(0, math.floor(top)), min(height, math.ceil(bottom))
            if r <= l or b <= t:
                continue
            if not is_core:
                ignored[int(image_idx), 0, t:b, l:r] = True
            else:
                xs = torch.arange(l, r, device=rois.device, dtype=torch.float32)
                ys = torch.arange(t, b, device=rois.device, dtype=torch.float32)
                dx = (torch.minimum(xs + 1, xs.new_tensor(right))
                      - torch.maximum(xs, xs.new_tensor(left))).clamp_min(0)
                dy = (torch.minimum(ys + 1, ys.new_tensor(bottom))
                      - torch.maximum(ys, ys.new_tensor(top))).clamp_min(0)
                coverage = dy[:, None] * dx[None, :]
                positive[int(image_idx), 0, t:b, l:r] += (
                    coverage / max((right - left) * (bottom - top), 1e-12))
    target = (positive > 0).float()
    negative = ~ignored & ~(positive > 0)
    return target, positive, negative


def balanced_objectness_loss(logits, target, positive_weight, negative):
    loss = F.binary_cross_entropy_with_logits(logits.float(), target.float(), reduction='none')
    pos_sum = positive_weight.sum()
    neg_sum = negative.sum()
    pos_loss = (loss * positive_weight).sum() / pos_sum.clamp_min(1e-12)
    neg_loss = (loss * negative).sum() / neg_sum.clamp_min(1)
    active = (pos_sum > 0).to(loss) + (neg_sum > 0).to(loss)
    return (pos_loss + neg_loss) / active.clamp_min(1)


def region_consistency(visual_a, visual_b, rois, input_shape, max_per_image=32,
                       normalized_regularization=False, collect_spectrum=False):
    """Symmetric losses: both views update the SAME Adapter parameters."""
    rois = scale_rois(rois, input_shape, visual_a.shape[-2:])
    valid = (rois[:, 3] - rois[:, 1] >= 2) & (rois[:, 4] - rois[:, 2] >= 2)
    rois = rois[valid]
    rows = []
    for i in range(visual_a.shape[0]):
        group = rois[rois[:, 0] == i]
        if len(group) > max_per_image:
            group = group[torch.randperm(len(group), device=rois.device)[:max_per_image]]
        rows.append(group)
    rois = torch.cat(rows)
    zero = (visual_a.float().sum() + visual_b.float().sum()) * 0
    if not len(rois):
        stats = dict(roi_count=zero.detach(), roi_std=zero.detach(),
                     cross_instance_cosine=zero.detach())
        if normalized_regularization:
            stats.update(covariance_loss=zero, normalized_roi_std=zero.detach(),
                         roi_norm=zero.detach(), variance_active=zero.detach(),
                         covariance_active=zero.detach(), roi_effective_rank=zero.detach())
        return zero, zero, stats
    embeddings = [roi_align(v.float(), rois, (2, 2), spatial_scale=1.0,
                            sampling_ratio=2, aligned=True).flatten(1)
                  for v in (visual_a, visual_b)]
    normed = [F.normalize(z, dim=1, eps=1e-6) for z in embeddings]
    consistency = (1 - (normed[0] * normed[1]).sum(1)).mean()
    variance = zero
    std = zero.detach()
    cross = zero.detach()
    if len(rois) >= 2:
        stds = [(z.var(0, unbiased=False) + 1e-4).sqrt() for z in embeddings]
        variance = sum((0.1 - s).clamp_min(0).mean() for s in stds) / 2
        std = sum(s.detach().mean() for s in stds) / 2
        with torch.no_grad():
            sim = normed[0] @ normed[0].T
            cross = (sim.sum() - sim.diag().sum()) / (len(rois) * (len(rois) - 1))
    if normalized_regularization:
        normalized = [z * math.sqrt(z.shape[1]) for z in normed]
        normalized_std = zero.detach()
        covariance = zero
        if len(rois) >= 2:
            stds = [(z.var(0, unbiased=False) + 1e-4).sqrt() for z in normalized]
            variance = sum((.5 - s).clamp_min(0).mean() for s in stds) / 2
            normalized_std = sum(s.detach().mean() for s in stds) / 2
        if len(rois) >= 8:
            terms = []
            for z in normalized:
                centered = z - z.mean(0, keepdim=True)
                cov = centered.T @ centered / (len(rois) - 1)
                off_diagonal = cov - torch.diag_embed(cov.diag())
                terms.append(off_diagonal.square().sum() / z.shape[1])
            covariance = sum(terms) / 2
        stats = dict(
            covariance_loss=covariance, roi_count=zero.detach().new_tensor(len(rois)),
            roi_std=std, normalized_roi_std=normalized_std, cross_instance_cosine=cross,
            roi_norm=sum(z.detach().norm(dim=1).mean() for z in embeddings) / 2,
            variance_active=zero.detach().new_tensor(int(len(rois) >= 2)),
            covariance_active=zero.detach().new_tensor(int(len(rois) >= 8)))
        if collect_spectrum:
            with torch.no_grad():
                z = normalized[0].detach()
                z = z - z.mean(0, keepdim=True)
                singular = torch.linalg.svdvals(z)
                prob = singular / singular.sum().clamp_min(1e-8)
                rank = (-(prob * prob.clamp_min(1e-8).log()).sum()).exp()
                stats['roi_effective_rank'] = torch.where(singular.sum() > 1e-8, rank, rank * 0)
        return consistency, variance, stats
    return consistency, variance, dict(
        roi_count=zero.detach().new_tensor(len(rois)), roi_std=std,
        cross_instance_cosine=cross)

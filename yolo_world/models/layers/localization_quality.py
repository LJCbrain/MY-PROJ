"""Class-free point regression and box-conditioned IoU quality for DIOR 3.3.

No class logits or class labels are used by the auxiliary matcher. Quality is
the IoU of the *decoded box*, not a foreground label or a centerness target.
"""

import math

import torch
from torch import nn
import torch.nn.functional as F
from mmyolo.registry import MODELS


class LocalizationVisualAdapter(nn.Module):
    """Retain the 2.17 Adapter weights/keys, without its binary O predictor."""

    def __init__(self, adapter):
        super().__init__()
        self.adapter = adapter

    def forward(self, c3):
        return self.adapter(c3.detach()), None


def flatten_maps(maps, channels):
    return torch.cat([m.permute(0, 2, 3, 1).reshape(m.shape[0], -1, channels)
                      for m in maps], dim=1)


def point_geometry(maps, strides):
    """YOLO's offset=0.5 reference points, in current input pixels."""
    points, steps, levels = [], [], []
    for level, (pred, stride) in enumerate(zip(maps, strides)):
        h, w = pred.shape[-2:]
        ys = (torch.arange(h, device=pred.device, dtype=torch.float32) + .5) * stride
        xs = (torch.arange(w, device=pred.device, dtype=torch.float32) + .5) * stride
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        points.append(torch.stack((xx, yy), -1).reshape(-1, 2))
        steps.append(xx.new_full((h * w,), stride))
        levels.append(torch.full((h * w,), level, device=pred.device, dtype=torch.long))
    return torch.cat(points), torch.cat(steps), torch.cat(levels)


def decode_distances(distances, points, strides):
    pixels = distances.float() * strides[None, :, None]
    return torch.cat((points[None] - pixels[..., :2],
                      points[None] + pixels[..., 2:]), dim=-1)


def pairwise_iou(gt, boxes):
    left = torch.maximum(gt[:, None, :2], boxes[None, :, :2])
    right = torch.minimum(gt[:, None, 2:], boxes[None, :, 2:])
    inter = (right - left).clamp_min(0).prod(-1)
    area_gt = (gt[:, 2:] - gt[:, :2]).clamp_min(0).prod(-1)
    area_box = (boxes[:, 2:] - boxes[:, :2]).clamp_min(0).prod(-1)
    return inter / (area_gt[:, None] + area_box[None] - inter).clamp_min(1e-7)


@torch.no_grad()
def geometric_box_targets(boxes, points, strides, levels, gt, reg_max,
                          topk=10, center_radius=2.5, target_scale=4.0,
                          gt_chunk_size=64):
    """Assign representable, central points on a suitable scale by box IoU.

    Each GT first chooses the available level nearest sqrt(area)/(4*stride).
    Its top-k points are ranked by decoded-box IoU (center distance breaks
    near ties). Conflicts choose the best geometric score. All-box max IoU
    targets are computed independently for quality, including badly localized
    and zero-overlap boxes. GT chunks bound memory on dense DIOR images.
    """
    n = len(points)
    assigned = torch.full((n,), -1, device=points.device, dtype=torch.long)
    best_score = boxes.new_full((n,), -torch.inf)
    quality = boxes.new_zeros(n)
    inside_any = torch.zeros(n, device=points.device, dtype=torch.bool)
    level_strides = torch.stack([strides[levels == i][0]
                                 for i in range(int(levels.max()) + 1)])
    for start in range(0, len(gt), gt_chunk_size):
        group = gt[start:start + gt_chunk_size]
        iou = pairwise_iou(group, boxes)
        quality = torch.maximum(quality, iou.max(0).values)
        left = points[None] - group[:, None, :2]
        right = group[:, None, 2:] - points[None]
        distance = torch.cat((left, right), -1)
        inside = distance.amin(-1) > 0
        inside_any |= inside.any(0)
        representable = distance.amax(-1) / strides[None] < reg_max - 1 - .01
        centers = (group[:, :2] + group[:, 2:]) / 2
        central_distance = (points[None] - centers[:, None]).abs() / strides[None, :, None]
        eligible = inside & representable
        central = eligible & (central_distance.amax(-1) <= center_radius)
        eligible = torch.where(central.any(1, keepdim=True), central, eligible)
        available = torch.stack([eligible[:, levels == i].any(1)
                                  for i in range(len(level_strides))], 1)
        size = (group[:, 2:] - group[:, :2]).prod(-1).sqrt().clamp_min(1e-6)
        scale_cost = (size[:, None] / (target_scale * level_strides[None])).log().abs()
        preferred = scale_cost.masked_fill(~available, torch.inf).argmin(1)
        eligible &= levels[None] == preferred[:, None]
        score = iou + .001 / (1 + central_distance.square().sum(-1))
        score = score.masked_fill(~eligible, -torch.inf)
        chosen = score.topk(min(topk, n), dim=1).indices
        keep = torch.zeros_like(eligible).scatter_(1, chosen, True) & eligible
        chunk_score, index = score.masked_fill(~keep, -torch.inf).max(0)
        replace = chunk_score > best_score
        assigned[replace] = index[replace] + start
        best_score = torch.maximum(best_score, chunk_score)
    return assigned, quality.clamp(0, 1), inside_any


def soft_iou_quality_loss(logits, target, valid):
    """Balance overlapping/zero-overlap regions using continuous IoU targets.

    On partial images, unmatched regions are excluded by the caller. This is
    not an additional Base/background binary classification objective.
    """
    loss = F.binary_cross_entropy_with_logits(logits.float(), target.float(), reduction='none')
    overlap = valid & (target > 0)
    zero_overlap = valid & ~overlap
    groups = []
    for mask in (overlap, zero_overlap):
        if mask.any():
            groups.append(loss[mask].mean())
    return torch.stack(groups).mean() if groups else logits.float().sum() * 0


@MODELS.register_module()
class ClassAgnosticLocalizationHead(nn.Module):
    """Auxiliary coarse DFL boxes and a shared pre/post-WSR IoU evaluator.

    Fused neck features are detached; live visual Adapter features remain in
    this loss path. Box geometry is detached before quality evaluation so the
    quality loss cannot move boxes to manipulate its own target.
    """

    def __init__(self, in_channels, visual_channels=64, hidden_channels=64,
                 norm_groups=8, reg_max=16, featmap_strides=(8, 16, 32),
                 match_topk=10, center_radius=2.5, target_scale=4.0):
        super().__init__()
        if (not in_channels or len(in_channels) != len(featmap_strides) or reg_max < 2
                or norm_groups < 1 or hidden_channels < 1
                or hidden_channels % norm_groups or match_topk < 1
                or center_radius <= 0 or target_scale <= 0
                or any(s <= 0 for s in featmap_strides)):
            raise ValueError('Invalid localization head geometry or channels.')
        self.reg_max = reg_max
        self.featmap_strides = tuple(featmap_strides)
        self.match_topk = match_topk
        self.center_radius = center_radius
        self.target_scale = target_scale
        self.stems = nn.ModuleList([nn.Sequential(
            nn.Conv2d(c + visual_channels, hidden_channels, 1, bias=False),
            nn.GroupNorm(norm_groups, hidden_channels), nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1,
                      groups=hidden_channels, bias=False), nn.SiLU()) for c in in_channels])
        self.regressors = nn.ModuleList([nn.Conv2d(hidden_channels, 4 * reg_max, 1)
                                        for _ in in_channels])
        self.quality_evaluators = nn.ModuleList([nn.Sequential(
            nn.Conv2d(hidden_channels + 4, hidden_channels, 1, bias=False),
            nn.GroupNorm(norm_groups, hidden_channels), nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1,
                      groups=hidden_channels, bias=False), nn.SiLU(),
            nn.Conv2d(hidden_channels, 1, 1)) for _ in in_channels])
        self.register_buffer('projection', torch.arange(reg_max, dtype=torch.float32))
        self.bbox_loss = MODELS.build(dict(type='IoULoss', iou_mode='ciou',
                                          bbox_format='xyxy', reduction='mean',
                                          loss_weight=1.0, return_iou=False))

    def encode(self, features, visual):
        encoded = []
        for stem, feature in zip(self.stems, features):
            v = F.interpolate(visual, feature.shape[-2:], mode='bilinear', align_corners=False)
            encoded.append(stem(torch.cat((feature.detach(), v.to(feature)), dim=1)))
        return encoded

    def evaluate_quality(self, encoded, distances):
        if len(encoded) != len(distances):
            raise ValueError('Quality features and box levels must match.')
        logits = []
        for evaluator, feature, distance in zip(self.quality_evaluators, encoded, distances):
            if feature.shape[-2:] != distance.shape[-2:]:
                raise ValueError('Quality features and boxes must share reference points.')
            geometry = distance.detach().float().clamp(0, 64).log1p() / math.log(self.reg_max)
            logits.append(evaluator(torch.cat((feature, geometry.to(feature)), 1)))
        return tuple(logits)

    def forward(self, features, visual):
        encoded = self.encode(features, visual)
        raw = tuple(layer(f) for layer, f in zip(self.regressors, encoded))
        # Softmax/expectation stay FP32 under AMP for large pixel coordinates.
        distances = tuple((r.float().reshape(r.shape[0], 4, self.reg_max, *r.shape[-2:])
                           .softmax(dim=2) * self.projection[None, None, :, None, None])
                          .sum(dim=2) for r in raw)
        return distances, raw, self.evaluate_quality(encoded, distances)

    def quality_for_boxes(self, features, visual, distances):
        return self.evaluate_quality(self.encode(features, visual), distances)

    def loss(self, coarse, raw, q0, refined, q1, rois, partial, collect_diagnostics=False):
        """Refined YOLO boxes are targets only; the main TAL remains unchanged."""
        with torch.autocast(device_type=coarse[0].device.type, enabled=False):
            points, strides, levels = point_geometry(coarse, self.featmap_strides)
            d0, d1 = flatten_maps(coarse, 4), flatten_maps(refined, 4).detach()
            boxes0 = decode_distances(d0, points, strides)
            boxes1 = decode_distances(d1, points, strides)
            logits0 = flatten_maps(q0, 1).squeeze(-1).float()
            logits1 = flatten_maps(q1, 1).squeeze(-1).float()
            raw_flat = flatten_maps(raw, 4 * self.reg_max).float()
            zero = raw_flat.sum() * 0
            bbox_losses, dfl_losses = [], []
            targets0, targets1, valid_rows = [], [], []
            matched = covered = total_gt = 0
            for image in range(len(d0)):
                gt = rois[rois[:, 0] == image, 1:]
                assignment, target0, inside = geometric_box_targets(
                    boxes0[image].detach(), points, strides, levels, gt, self.reg_max,
                    self.match_topk, self.center_radius, self.target_scale)
                # The final box has its own IoU target, never the coarse-box IoU.
                target1 = boxes1[image].new_zeros(len(points))
                for start in range(0, len(gt), 64):
                    target1 = torch.maximum(target1, pairwise_iou(
                        gt[start:start + 64], boxes1[image]).max(0).values)
                positive = assignment >= 0
                count = int(positive.sum())
                matched += count
                total_gt += len(gt)
                if count:
                    indices = assignment[positive]
                    covered += len(indices.unique())
                    gt_pos = gt[indices]
                    bbox_losses.append(self.bbox_loss(boxes0[image, positive], gt_pos))
                    distances = torch.cat((points[positive] - gt_pos[:, :2],
                                           gt_pos[:, 2:] - points[positive]), -1)
                    targets = (distances / strides[positive, None]).clamp(0, self.reg_max - 1 - .01)
                    left = targets.long()
                    right_weight = targets - left
                    reg = raw_flat[image, positive].reshape(-1, self.reg_max)
                    dfl_losses.append((F.cross_entropy(reg, left.flatten(), reduction='none')
                                       * (1 - right_weight.flatten())
                                       + F.cross_entropy(reg, (left + 1).flatten(), reduction='none')
                                       * right_weight.flatten()).mean())
                valid = (inside | (target0 > 0) | (target1 > 0)) if partial[image] else torch.ones_like(inside)
                targets0.append(target0)
                targets1.append(target1)
                valid_rows.append(valid)
            t0, t1, valid = torch.stack(targets0), torch.stack(targets1), torch.stack(valid_rows)
            losses = dict(
                bbox=torch.stack(bbox_losses).mean() if bbox_losses else zero,
                dfl=torch.stack(dfl_losses).mean() if dfl_losses else zero,
                q0=soft_iou_quality_loss(logits0, t0, valid),
                q1=soft_iou_quality_loss(logits1, t1, valid))
            stats = {}
            if collect_diagnostics:
                stats.update(quality_matched_points=zero.detach().new_tensor(matched),
                             quality_gt_coverage=zero.detach().new_tensor(covered / max(total_gt, 1)),
                             quality_supervised_points=valid.sum().detach())
                for name, logits, target in (('q0', logits0, t0), ('q1', logits1, t1)):
                    mask = valid & (target > 0)
                    predicted, truth = logits.detach().sigmoid()[mask], target[mask]
                    if predicted.numel():
                        pc, tc = predicted - predicted.mean(), truth - truth.mean()
                        corr = (pc * tc).sum() / (pc.square().sum() * tc.square().sum()).sqrt().clamp_min(1e-7)
                        stats[name + '_iou_mae'] = (predicted - truth).abs().mean()
                        stats[name + '_iou_corr'] = corr
                        stats[name + '_target_iou'] = truth.mean()
                    else:
                        for suffix in ('_iou_mae', '_iou_corr', '_target_iou'):
                            stats[name + suffix] = zero.detach()
            return losses, stats

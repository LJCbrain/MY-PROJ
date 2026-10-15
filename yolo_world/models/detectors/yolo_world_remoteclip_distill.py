"""DIOR v3.0: supervise the existing post-BN classification features by GT.

    Same augmented image -> GT crop -> frozen teacher projected embedding
    Existing detector   -> assigned positive post-BN vectors -> cosine loss

No extra inference branch, assignment pass, BN pass, or pseudo Novel labels.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn
from mmyolo.models.task_modules.assigners import BatchTaskAlignedAssigner
from mmyolo.registry import MODELS, TASK_UTILS

from .yolo_world_ovc import YOLOWorldOVCPartialDetector
from ..layers.remoteclip_teacher import RemoteCLIPTeacherCache

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


@TASK_UTILS.register_module()
class DistillBatchTaskAlignedAssigner(BatchTaskAlignedAssigner):
    """Expose the exact GT indices already chosen by the original assigner."""
    def get_targets(self, gt_labels, gt_bboxes, assigned_gt_idxs,
                    fg_mask_pre_prior, batch_size, num_gt):
        self._distill_indices = assigned_gt_idxs.detach()
        return super().get_targets(gt_labels, gt_bboxes, assigned_gt_idxs,
                                   fg_mask_pre_prior, batch_size, num_gt)

    @torch.no_grad()
    def forward(self, pred_bboxes, pred_scores, priors, gt_labels,
                gt_bboxes, pad_bbox_flag):
        self._distill_indices = None
        try:
            result = super().forward(pred_bboxes, pred_scores, priors,
                                     gt_labels, gt_bboxes, pad_bbox_flag)
            indices = self._distill_indices
            if indices is None:
                indices = torch.full(pred_scores.shape[:2], -1,
                                     device=pred_scores.device, dtype=torch.long)
            result['distill_gt_indices'] = indices
            result['distill_gt_bboxes'] = gt_bboxes.detach()
            return result
        finally:
            self._distill_indices = None


def teacher_crop(rgb, box, context=1.0, min_size=2.0, image_size=224):
    """Crop a box from the SAME augmented RGB [0,1] image used by the student."""
    height, width = rgb.shape[-2:]
    x1, y1, x2, y2 = [float(v) for v in box]
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
        return None
    # Check visible GT extent, not the expanded context extent.
    if min(width, x2) - max(0, x1) < min_size or min(height, y2) - max(0, y1) < min_size:
        return None
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half_w, half_h = (x2 - x1) * context / 2, (y2 - y1) * context / 2
    left, right = max(0, math.floor(cx - half_w)), min(width, math.ceil(cx + half_w))
    top, bottom = max(0, math.floor(cy - half_h)), min(height, math.ceil(cy + half_h))
    crop = rgb[:, top:bottom, left:right]
    h, w = crop.shape[-2:]
    mean = rgb.new_tensor(CLIP_MEAN).view(3, 1, 1)
    std = rgb.new_tensor(CLIP_STD).view(3, 1, 1)
    # Subtract mean before padding: zero padding then represents mean RGB.
    side = max(h, w)
    pad_x, pad_y = (side - w) // 2, (side - h) // 2
    square = F.pad(crop - mean, (pad_x, side - w - pad_x,
                                pad_y, side - h - pad_y)) + mean
    resized = F.interpolate(square[None], (image_size, image_size),
                            mode='bicubic', align_corners=False, antialias=True)[0]
    return (resized.clamp(0, 1) - mean) / std


def gather_positive_vectors(features, batch_index, flat_indices):
    """Match the detection head's P3/P4/P5, row-major flattening without a
    second dense concatenation of all 768-channel feature maps.
    """
    outputs, offset = [], 0
    for feature in features:
        height, width = feature.shape[-2:]
        count = height * width
        local = flat_indices[(flat_indices >= offset) &
                             (flat_indices < offset + count)] - offset
        if local.numel():
            outputs.append(feature[batch_index, :, local // width, local % width].t())
        offset += count
    if sum(len(part) for part in outputs) != len(flat_indices):
        raise ValueError('Positive-prior indices do not match classification maps.')
    return torch.cat(outputs, dim=0)


def equal_gt_cosine_loss(student_groups, teacher_vectors):
    """Every selected GT has equal weight, regardless of its positive count."""
    if len(student_groups) != len(teacher_vectors) or not student_groups:
        raise ValueError('Teacher/student GT groups must be nonempty and aligned.')
    losses = []
    for student, teacher in zip(student_groups, teacher_vectors):
        if student.shape[-1] != teacher.shape[-1]:
            raise ValueError('Student and RemoteCLIP projection dimensions differ.')
        similarity = (F.normalize(student.float(), dim=-1) *
                      F.normalize(teacher.detach().float(), dim=-1)).sum(-1)
        losses.append((1 - similarity).mean())
    return torch.stack(losses).mean()


@MODELS.register_module()
class YOLOWorldRemoteCLIPDistillDetector(YOLOWorldOVCPartialDetector):
    def __init__(self, *args, remoteclip_distill_cfg, **kwargs):
        super().__init__(*args, **kwargs)
        defaults = dict(checkpoint='pretrained_models/RemoteCLIP-ViT-L-14.pt',
                        loss_weight=0.1, warmup_iters=2000,
                        max_regions_per_image=8, teacher_batch_size=8,
                        context=1.0, min_box_size=2.0, precision='fp16', seed=300)
        unknown = set(remoteclip_distill_cfg) - set(defaults)
        if unknown:
            raise ValueError(f'Unknown RemoteCLIP distillation options: {unknown}')
        defaults.update(dict(remoteclip_distill_cfg))
        self.remoteclip_distill_cfg = defaults
        cfg = defaults
        if (not math.isfinite(cfg['loss_weight']) or cfg['loss_weight'] < 0 or
                cfg['warmup_iters'] < 0 or cfg['max_regions_per_image'] < 1 or
                cfg['teacher_batch_size'] < 1 or not math.isfinite(cfg['context']) or
                cfg['context'] < 1 or not math.isfinite(cfg['min_box_size']) or
                cfg['min_box_size'] <= 0 or cfg['precision'] not in ('fp16', 'fp32')):
            raise ValueError('Invalid RemoteCLIP distillation settings.')
        if (self.bbox_head.train_cfg is not None and
                not isinstance(self.bbox_head.assigner, DistillBatchTaskAlignedAssigner)):
            raise TypeError('Use DistillBatchTaskAlignedAssigner for exact GT correspondence.')
        for contrast in self.bbox_head.head_module.cls_contrasts:
            if not isinstance(getattr(contrast, 'norm', None), nn.modules.batchnorm._BatchNorm):
                raise TypeError('RemoteCLIP distillation requires the existing BN contrastive head.')
        self._remoteclip_teacher = RemoteCLIPTeacherCache(cfg['checkpoint'], cfg['precision'])
        self.register_buffer('_remoteclip_distill_step', torch.zeros((), dtype=torch.long))

    def _rgb_images(self, batch_inputs):
        # YOLOW preprocessing already converts BGR to RGB. Undo its affine
        # normalization, retaining mosaic/affine/flip/color augmentation.
        images = batch_inputs.detach().float()
        preprocessor = self.data_preprocessor
        if hasattr(preprocessor, 'mean') and hasattr(preprocessor, 'std'):
            images = images * preprocessor.std.float() + preprocessor.mean.float()
        return (images / 255.0).clamp(0, 1)

    def _distill_loss(self, batch_inputs, features, assignment):
        cfg = self.remoteclip_distill_cfg
        zero = features[0].float().sum() * 0
        rgb = self._rgb_images(batch_inputs)
        groups, crops = [], []
        foreground = assignment['fg_mask_pre_prior'].bool()
        indices = assignment['distill_gt_indices']
        boxes = assignment['distill_gt_bboxes']
        expected_priors = sum(f.shape[-2] * f.shape[-1] for f in features)
        if expected_priors != indices.shape[1]:
            raise ValueError('Assigner and classification feature sizes differ.')
        # Local RNG does not perturb detector augmentations/dropout; the buffer
        # persists in checkpoints so resumed sampling/warmup continues correctly.
        generator = torch.Generator().manual_seed(
            cfg['seed'] + int(self._remoteclip_distill_step))
        skipped = 0
        for batch_index in range(len(batch_inputs)):
            selected = torch.unique(indices[batch_index, foreground[batch_index]])
            order = torch.randperm(len(selected), generator=generator).tolist()
            accepted = 0
            for position in order:
                gt_index = int(selected[position])
                crop = teacher_crop(rgb[batch_index], boxes[batch_index, gt_index],
                                    cfg['context'], cfg['min_box_size'])
                if crop is None:
                    skipped += 1
                    continue
                positives = torch.where(foreground[batch_index] &
                                        (indices[batch_index] == gt_index))[0]
                groups.append(gather_positive_vectors(features, batch_index, positives))
                crops.append(crop)
                accepted += 1
                if accepted >= cfg['max_regions_per_image']:
                    break
        warmup = (min((int(self._remoteclip_distill_step) + 1) /
                      cfg['warmup_iters'], 1.0) if cfg['warmup_iters'] else 1.0)
        weight = cfg['loss_weight'] * warmup
        raw = zero
        if crops:
            teacher = self._remoteclip_teacher.get(batch_inputs.device)
            dtype = next(teacher.parameters()).dtype
            outputs = []
            with torch.no_grad():
                for start in range(0, len(crops), cfg['teacher_batch_size']):
                    pixels = torch.stack(crops[start:start + cfg['teacher_batch_size']])
                    vectors = teacher(pixel_values=pixels.to(dtype=dtype)).image_embeds.float()
                    if not torch.isfinite(vectors).all():
                        raise FloatingPointError('Nonfinite RemoteCLIP teacher vectors.')
                    outputs.append(vectors)
            raw = equal_gt_cosine_loss(groups, torch.cat(outputs))
        return dict(
            loss_remoteclip_distill=raw * weight,
            remoteclip_distill_raw=raw.detach(),
            remoteclip_distill_cosine=(1 - raw.detach()) if crops else zero.detach(),
            remoteclip_distill_regions=zero.detach().new_tensor(len(crops)),
            remoteclip_distill_positives=zero.detach().new_tensor(sum(len(g) for g in groups)),
            remoteclip_distill_skipped=zero.detach().new_tensor(skipped),
            remoteclip_distill_weight=zero.detach().new_tensor(weight))

    def loss(self, batch_inputs, batch_data_samples):
        if self.remoteclip_distill_cfg['loss_weight'] == 0:
            return super().loss(batch_inputs, batch_data_samples)
        captures, assignment, handles = {}, {}, []
        try:
            # Capture the actual BN outputs with gradients, without calling BN
            # again or changing any tensor that produces detection logits.
            for index, contrast in enumerate(self.bbox_head.head_module.cls_contrasts):
                handles.append(contrast.norm.register_forward_hook(
                    lambda module, args, output, i=index: captures.__setitem__(i, output)))
            handles.append(self.bbox_head.assigner.register_forward_hook(
                lambda module, args, output: assignment.update(output)))
            losses = super().loss(batch_inputs, batch_data_samples)
            features = [captures[i] for i in range(len(self.bbox_head.head_module.cls_contrasts))]
            # Cosine/reduction/crop normalization must stay FP32 under AMP.
            with torch.autocast(device_type=batch_inputs.device.type, enabled=False):
                losses.update(self._distill_loss(batch_inputs, features, assignment))
            self._remoteclip_distill_step.add_(1)
            return losses
        finally:
            for handle in handles:
                handle.remove()
            captures.clear()
            assignment.clear()

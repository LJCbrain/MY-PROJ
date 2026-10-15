import math

import torch
from mmyolo.registry import MODELS
from .yolo_world_head import YOLOWorldHead


@MODELS.register_module()
class PartialLabelYOLOWorldHead(YOLOWorldHead):
    def __init__(self, *args, cls_foreground_weight=1.0,
                 cls_background_weight=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.cls_foreground_weight = float(cls_foreground_weight)
        self.cls_background_weight = float(cls_background_weight)
        for weight in (self.cls_foreground_weight, self.cls_background_weight):
            if not math.isfinite(weight) or weight < 0:
                raise ValueError('Classification position weights must be finite and non-negative.')

    def classification_position_weights(self, assigned_scores, foreground, img_metas):
        partial = torch.tensor([meta.get('partial_labels', False) for meta in img_metas],
                               device=assigned_scores.device, dtype=torch.bool)
        active = ~partial[:, None] | foreground.bool()
        self.last_partial_diagnostics = dict(
            partial_fraction=partial.float().mean().detach(),
            cls_positive_positions=foreground.sum().detach(),
            cls_negative_positions=(active & ~foreground.bool()).sum().detach())
        weights = torch.where(
            foreground.bool(), self.cls_foreground_weight,
            self.cls_background_weight).to(assigned_scores)
        return (active.to(assigned_scores) * weights).unsqueeze(-1)

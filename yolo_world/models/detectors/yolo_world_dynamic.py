# Copyright (c) Tencent Inc. All rights reserved.
from typing import Union

import torch
import torch.nn.functional as F
from mmdet.structures import SampleList
from torch import Tensor

from mmyolo.registry import MODELS

from .yolo_world import YOLOWorldDetector


@MODELS.register_module()
class YOLOWorldDynamicFusionDetector(YOLOWorldDetector):
    """YOLO-World detector with dynamic-fusion router regularization."""

    def __init__(self,
                 *args,
                 loss_router_balance_weight: float = 0.0,
                 **kwargs) -> None:
        self.loss_router_balance_weight = float(loss_router_balance_weight)
        super().__init__(*args, **kwargs)

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        losses = super().loss(batch_inputs, batch_data_samples)
        losses.update(self._loss_dynamic_fusion())
        losses.update(self._scale_semantic_diagnostics())
        return losses

    def _loss_dynamic_fusion(self) -> dict:
        if self.loss_router_balance_weight <= 0 or not self.with_neck:
            return {}
        parts = getattr(self.neck, '_dynamic_fusion_parts', None)
        if not parts:
            return {}
        router_weights = parts.get('router_weights', None)
        if not router_weights:
            return {}

        mean_weights = torch.stack([
            weights.mean(dim=(0, 2, 3)) for weights in router_weights
        ]).mean(dim=0)
        target = mean_weights.new_full(mean_weights.shape,
                                       1.0 / mean_weights.numel())
        return dict(
            loss_router_balance=F.mse_loss(mean_weights, target) *
            self.loss_router_balance_weight)

    def _scale_semantic_diagnostics(self) -> dict:
        """Expose detached scale diagnostics without affecting optimization."""
        diagnostics = {}
        level_names = ('p3', 'p4', 'p5')

        head_module = getattr(self.bbox_head, 'head_module', None)
        prototype_parts = getattr(head_module, '_prototype_parts', None)
        if prototype_parts:
            for level, parts in enumerate(prototype_parts):
                if not parts or level >= len(level_names):
                    continue
                name = level_names[level]
                diagnostics[f'semantic_alpha_{name}'] = parts['alpha']
                diagnostics[f'semantic_topk_similarity_{name}'] = (
                    parts['topk_similarity'])
                diagnostics[f'semantic_residual_norm_{name}'] = (
                    parts['residual_norm'])

        context_parts = getattr(self.neck, '_scale_context_parts', None)
        if context_parts:
            dilations = context_parts.get('level_dilations', ())
            kernel_weights = context_parts.get('kernel_weights', ())
            residual_norms = context_parts.get('residual_norms', ())
            for level, name in enumerate(level_names):
                if level < len(residual_norms):
                    diagnostics[f'context_residual_norm_{name}'] = (
                        residual_norms[level].mean())
                if level >= len(kernel_weights):
                    continue
                weights = kernel_weights[level]
                if weights is None:
                    continue
                for index, dilation in enumerate(dilations[level]):
                    diagnostics[f'context_kernel_{name}_d{dilation}'] = (
                        weights[:, index].mean())
        return diagnostics


@MODELS.register_module()
class YOLOWorldScaleDecoupledDetector(YOLOWorldDynamicFusionDetector):
    """Detector entry point for scale-decoupled prototype fusion."""

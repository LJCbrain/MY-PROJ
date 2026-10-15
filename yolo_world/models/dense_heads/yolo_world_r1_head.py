"""Dense box-alignment refinement head for YOLO-World.

This module is intentionally kept separate from ``yolo_world_head.py`` so the
existing YOLO-World heads and every configuration that uses them remain
unchanged.  R1 is a class-agnostic second box regression inside the dense head:

    image feature -> coarse DFL box -> box-guided feature sampling
                  -> distance residual -> final box

The original TAL assignment, classification loss, coarse IoU loss, and DFL
loss are retained.  A separate IoU loss supervises the refined final boxes.
No quality score, text feature, foreground prior, or proposal Top-K is used by
R1.
"""

from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.config import ConfigDict
from mmengine.dist import get_dist_info
from mmengine.model import BaseModule
from mmengine.structures import InstanceData
from mmdet.structures import SampleList
from mmdet.utils import InstanceList, OptInstanceList
from mmdet.models.utils import unpack_gt_instances
from mmyolo.models.utils import gt_instances_preprocess
from mmyolo.registry import MODELS
from torch import Tensor

from .yolo_world_head import YOLOWorldHead, YOLOWorldHeadModule


class DenseBoxAlignmentRefiner(BaseModule):
    """Refine dense l/t/r/b distances using features sampled from a coarse box.

    The sampler uses either five points (box center and four edge midpoints) or
    nine points (a 3x3 grid spanning the predicted box).  It runs at every
    location of every feature level; it does not select Top-K proposals.
    """

    def __init__(
            self,
            channels: int,
            reg_max: int,
            num_sample_points: int = 5,
            min_distance: float = 0.01,
            detach_coarse_boxes: bool = True,
            norm_cfg: ConfigDict = dict(
                type='BN', momentum=0.03, eps=0.001),
            act_cfg: ConfigDict = dict(type='SiLU', inplace=True),
            init_cfg=None) -> None:
        super().__init__(init_cfg=init_cfg)
        if num_sample_points not in (5, 9):
            raise ValueError('R1 num_sample_points must be 5 or 9.')
        if min_distance <= 0:
            raise ValueError('R1 min_distance must be positive.')

        self.reg_max = int(reg_max)
        self.num_sample_points = int(num_sample_points)
        self.min_distance = float(min_distance)
        self.detach_coarse_boxes = bool(detach_coarse_boxes)

        # Zero logits start with uniform weighting over the sampled positions.
        self.sample_weight = nn.Conv2d(
            channels, self.num_sample_points, kernel_size=1)
        self.fuse = nn.Sequential(
            ConvModule(
                2 * channels + 4,
                channels,
                kernel_size=3,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg),
            ConvModule(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg),
        )
        self.residual_pred = nn.Conv2d(channels, 4, kernel_size=1)
        self.reset_identity()

    def reset_identity(self) -> None:
        """Make a newly initialized R1 branch exactly reproduce coarse boxes."""
        nn.init.constant_(self.sample_weight.weight, 0.0)
        nn.init.constant_(self.sample_weight.bias, 0.0)
        nn.init.constant_(self.residual_pred.weight, 0.0)
        nn.init.constant_(self.residual_pred.bias, 0.0)

    def _box_sample_offsets(
            self, distances: Tensor) -> List[Tuple[Tensor, Tensor]]:
        """Return sample offsets in feature-cell units relative to each prior."""
        left, top, right, bottom = distances.unbind(dim=1)
        center_x = 0.5 * (right - left)
        center_y = 0.5 * (bottom - top)

        if self.num_sample_points == 5:
            return [
                (center_x, center_y),
                (-left, center_y),
                (right, center_y),
                (center_x, -top),
                (center_x, bottom),
            ]

        xs = (-left, center_x, right)
        ys = (-top, center_y, bottom)
        return [(x_offset, y_offset) for y_offset in ys for x_offset in xs]

    @staticmethod
    def _make_grid(x: Tensor, y: Tensor, height: int,
                   width: int) -> Tensor:
        """Convert feature coordinates to an align_corners=False sample grid."""
        x_norm = (2.0 * (x + 0.5) / max(width, 1)) - 1.0
        y_norm = (2.0 * (y + 0.5) / max(height, 1)) - 1.0
        return torch.stack((x_norm, y_norm), dim=-1)

    def _sample_box_features(self, feature: Tensor,
                             distances: Tensor) -> Tensor:
        batch_size, _, height, width = feature.shape
        sample_distances = (
            distances.detach() if self.detach_coarse_boxes else distances)
        sample_distances = sample_distances.clamp_min(self.min_distance)

        dtype = feature.dtype
        device = feature.device
        base_y, base_x = torch.meshgrid(
            torch.arange(height, device=device, dtype=dtype),
            torch.arange(width, device=device, dtype=dtype),
            indexing='ij')
        base_x = base_x.unsqueeze(0).expand(batch_size, -1, -1)
        base_y = base_y.unsqueeze(0).expand(batch_size, -1, -1)

        weights = self.sample_weight(feature).softmax(dim=1)
        aligned = torch.zeros_like(feature)
        for point_idx, (x_offset, y_offset) in enumerate(
                self._box_sample_offsets(sample_distances)):
            grid = self._make_grid(
                base_x + x_offset, base_y + y_offset, height, width)
            sampled = F.grid_sample(
                feature,
                grid,
                mode='bilinear',
                padding_mode='border',
                align_corners=False)
            aligned = aligned + weights[:, point_idx:point_idx + 1] * sampled
        return aligned

    def forward(self, feature: Tensor, coarse_distances: Tensor,
                residual_scale: float) -> Tensor:
        aligned = self._sample_box_features(feature, coarse_distances)
        geometry_scale = float(max(self.reg_max - 1, 1))
        geometry = (coarse_distances.detach() /
                    geometry_scale).clamp(min=0.0, max=2.0)
        residual_feature = self.fuse(
            torch.cat((feature, aligned, geometry), dim=1))
        residual = torch.tanh(self.residual_pred(residual_feature))
        # The baseline DFL/IoU path already supervises ``coarse_distances``.
        # Detaching the residual base prevents the additional refine loss from
        # merely increasing the old regression-loss weight.  Gradients still
        # reach the shared regression feature through ``residual_feature``.
        residual_base = (
            coarse_distances.detach()
            if self.training and self.detach_coarse_boxes
            else coarse_distances)
        refined = residual_base + float(residual_scale) * residual
        return refined.clamp_min(self.min_distance)


@MODELS.register_module()
class YOLOWorldR1HeadModule(YOLOWorldHeadModule):
    """Independent YOLO-World head module with dense R1 box refinement."""

    def __init__(self,
                 *args,
                 refine_cfg: Optional[dict] = None,
                 **kwargs) -> None:
        cfg = {} if refine_cfg is None else dict(refine_cfg)
        self.refine_max_residual = float(cfg.pop('max_residual', 2.0))
        self.refine_warmup_iters = int(cfg.pop('warmup_iters', 1000))
        self.refine_num_sample_points = int(
            cfg.pop('num_sample_points', 5))
        self.refine_min_distance = float(cfg.pop('min_distance', 0.01))
        self.refine_detach_coarse_boxes = bool(
            cfg.pop('detach_coarse_boxes', True))
        if cfg:
            raise ValueError(
                f'Unknown R1 refine_cfg fields: {tuple(cfg.keys())}')
        if self.refine_max_residual < 0:
            raise ValueError('R1 max_residual must be non-negative.')
        if self.refine_warmup_iters < 0:
            raise ValueError('R1 warmup_iters must be non-negative.')

        super().__init__(*args, **kwargs)

        self.refine_blocks = nn.ModuleList([
            DenseBoxAlignmentRefiner(
                channels=reg_pred[-1].in_channels,
                reg_max=self.reg_max,
                num_sample_points=self.refine_num_sample_points,
                min_distance=self.refine_min_distance,
                detach_coarse_boxes=self.refine_detach_coarse_boxes,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
            for reg_pred in self.reg_preds
        ])
        self.register_buffer(
            '_refine_step', torch.zeros((), dtype=torch.long),
            persistent=True)
        if self.freeze_all:
            self._freeze_all()

    def init_weights(self, prior_prob=0.01):
        super().init_weights(prior_prob=prior_prob)
        for block in self.refine_blocks:
            block.reset_identity()

    def _residual_scale(self) -> float:
        if not self.training or self.refine_warmup_iters <= 0:
            return self.refine_max_residual
        self._refine_step.add_(1)
        progress = min(
            float(self._refine_step.item()) /
            float(self.refine_warmup_iters), 1.0)
        return self.refine_max_residual * progress

    def forward(self,
                img_feats: Tuple[Tensor],
                txt_feats: Tensor,
                txt_masks: Tensor,
                base_txt_feats: Optional[Tensor] = None,
                foreground_maps: Optional[Sequence[Tensor]] = None,
                reliability_maps: Optional[Sequence[Tensor]] = None
                ) -> Tuple[List]:
        """Run dense classification, coarse regression, and R1 at all levels."""
        assert len(img_feats) == self.num_levels
        residual_scale = self._residual_scale()

        if self.spatial_prompt_routing_enabled:
            if (base_txt_feats is None or foreground_maps is None or
                    reliability_maps is None):
                raise RuntimeError(
                    'CSPR requires base text, foreground maps, and '
                    'reliability maps.')
            if (len(foreground_maps) != self.num_levels or
                    len(reliability_maps) != self.num_levels):
                raise ValueError(
                    'CSPR requires one foreground and reliability map per '
                    'feature level.')
            mode = (
                self.spatial_prompt_train_mode
                if self.training else self.spatial_prompt_test_mode)
            outputs = [
                self._forward_single_r1(
                    img_feat,
                    txt_feats,
                    txt_masks,
                    cls_pred,
                    reg_pred,
                    cls_contrast,
                    refine_block,
                    residual_scale,
                    base_txt_feat=base_txt_feats,
                    foreground_map=foreground_map,
                    reliability_map=reliability_map,
                    routing_mode=mode,
                    routing_topk=self.spatial_prompt_topk[level_idx])
                for level_idx, (
                    img_feat, cls_pred, reg_pred, cls_contrast, refine_block,
                    foreground_map, reliability_map) in enumerate(
                        zip(img_feats, self.cls_preds, self.reg_preds,
                            self.cls_contrasts, self.refine_blocks,
                            foreground_maps, reliability_maps))
            ]
        else:
            outputs = [
                self._forward_single_r1(
                    img_feat,
                    txt_feats,
                    txt_masks,
                    cls_pred,
                    reg_pred,
                    cls_contrast,
                    refine_block,
                    residual_scale)
                for (img_feat, cls_pred, reg_pred, cls_contrast,
                     refine_block) in zip(
                         img_feats, self.cls_preds, self.reg_preds,
                         self.cls_contrasts, self.refine_blocks)
            ]

        return tuple([
            output[output_idx] for output in outputs
        ] for output_idx in range(len(outputs[0])))

    def _forward_single_r1(
            self,
            img_feat: Tensor,
            txt_feat: Tensor,
            txt_masks: Tensor,
            cls_pred: nn.Module,
            reg_pred: nn.Module,
            cls_contrast: nn.Module,
            refine_block: DenseBoxAlignmentRefiner,
            residual_scale: float,
            base_txt_feat: Optional[Tensor] = None,
            foreground_map: Optional[Tensor] = None,
            reliability_map: Optional[Tensor] = None,
            routing_mode: str = 'dense',
            routing_topk: int = 0) -> Tuple[Tensor, ...]:
        batch_size, _, height, width = img_feat.shape

        cls_embed = cls_pred(img_feat)
        if base_txt_feat is None:
            cls_logit = cls_contrast(cls_embed, txt_feat)
        else:
            if not hasattr(cls_contrast, 'forward_routed'):
                raise TypeError(
                    'The configured contrastive head does not support CSPR.')
            cls_logit = cls_contrast.forward_routed(
                cls_embed,
                base_txt_feat,
                txt_feat,
                foreground_map,
                reliability_map,
                mode=routing_mode,
                topk=routing_topk,
                detach_gate=self.spatial_prompt_detach_gate)

        if txt_masks is not None:
            expanded_masks = txt_masks.view(
                batch_size, -1, 1, 1).expand(-1, -1, height, width)
            cls_logit = cls_logit * expanded_masks
            cls_logit[expanded_masks == 0] = -10e6

        # Preserve the exact two-convolution coarse regression tower and expose
        # its hidden feature only to the new R1 branch.
        reg_feature = reg_pred[0](img_feat)
        reg_feature = reg_pred[1](reg_feature)
        bbox_dist_preds = reg_pred[2](reg_feature)

        if self.reg_max > 1:
            coarse_bbox_preds = bbox_dist_preds.reshape(
                [-1, 4, self.reg_max, height * width]).permute(0, 3, 1, 2)
            coarse_bbox_preds = coarse_bbox_preds.softmax(3).matmul(
                self.proj.view([-1, 1])).squeeze(-1)
            coarse_bbox_preds = coarse_bbox_preds.transpose(1, 2).reshape(
                batch_size, 4, height, width)
        else:
            coarse_bbox_preds = bbox_dist_preds

        refined_bbox_preds = refine_block(
            reg_feature, coarse_bbox_preds, residual_scale)
        if self.training:
            return (cls_logit, refined_bbox_preds, bbox_dist_preds,
                    coarse_bbox_preds)
        return cls_logit, refined_bbox_preds


@MODELS.register_module()
class YOLOWorldR1Head(YOLOWorldHead):
    """Independent YOLO-World bbox head supervising coarse and refined boxes."""

    def __init__(self,
                 *args,
                 refine_loss_weight: float = 0.5,
                 **kwargs) -> None:
        self.refine_loss_weight = float(refine_loss_weight)
        if self.refine_loss_weight < 0:
            raise ValueError('refine_loss_weight must be non-negative.')
        super().__init__(*args, **kwargs)

    def loss(self,
             img_feats: Tuple[Tensor],
             txt_feats: Tensor,
             txt_masks: Tensor,
             batch_data_samples: Union[list, dict],
             spatial_prompt_context: Optional[dict] = None) -> dict:
        outs = self(
            img_feats,
            txt_feats,
            txt_masks,
            spatial_prompt_context=spatial_prompt_context)
        (cls_scores, refined_bbox_preds, bbox_dist_preds,
         coarse_bbox_preds) = outs
        return self.loss_by_feat(
            cls_scores,
            refined_bbox_preds,
            bbox_dist_preds,
            coarse_bbox_preds,
            txt_masks,
            batch_data_samples['bboxes_labels'],
            batch_data_samples['img_metas'])

    def loss_and_predict(
            self,
            img_feats: Tuple[Tensor],
            txt_feats: Tensor,
            txt_masks: Tensor,
            batch_data_samples: SampleList,
            proposal_cfg: Optional[ConfigDict] = None,
            spatial_prompt_context: Optional[dict] = None
    ) -> Tuple[dict, InstanceList]:
        (batch_gt_instances, batch_gt_instances_ignore,
         batch_img_metas) = unpack_gt_instances(batch_data_samples)
        outs = self(
            img_feats,
            txt_feats,
            txt_masks,
            spatial_prompt_context=spatial_prompt_context)
        (cls_scores, refined_bbox_preds, bbox_dist_preds,
         coarse_bbox_preds) = outs
        losses = self.loss_by_feat(
            cls_scores,
            refined_bbox_preds,
            bbox_dist_preds,
            coarse_bbox_preds,
            txt_masks,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore)
        predictions = self.predict_by_feat(
            cls_scores,
            refined_bbox_preds,
            batch_img_metas=batch_img_metas,
            cfg=proposal_cfg)
        return losses, predictions

    def loss_by_feat(
            self,
            cls_scores: Sequence[Tensor],
            refined_bbox_preds: Sequence[Tensor],
            bbox_dist_preds: Sequence[Tensor],
            coarse_bbox_preds: Sequence[Tensor],
            batch_text_masks: Tensor,
            batch_gt_instances: Sequence[InstanceData],
            batch_img_metas: Sequence[dict],
            batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        """Keep baseline TAL/losses on coarse boxes and add final-box IoU loss."""
        losses = super().loss_by_feat(
            cls_scores,
            coarse_bbox_preds,
            bbox_dist_preds,
            batch_text_masks,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore)
        if self.refine_loss_weight <= 0:
            return losses
        losses['loss_bbox_refine'] = self._loss_bbox_refine(
            cls_scores,
            coarse_bbox_preds,
            refined_bbox_preds,
            batch_gt_instances,
            batch_img_metas)
        return losses

    def _loss_bbox_refine(
            self,
            cls_scores: Sequence[Tensor],
            coarse_bbox_preds: Sequence[Tensor],
            refined_bbox_preds: Sequence[Tensor],
            batch_gt_instances: Sequence[InstanceData],
            batch_img_metas: Sequence[dict]) -> Tensor:
        """Supervise R1 with assignments produced by the unchanged coarse TAL."""
        num_imgs = len(batch_img_metas)
        gt_info = gt_instances_preprocess(batch_gt_instances, num_imgs)
        gt_labels = gt_info[:, :, :1]
        gt_bboxes = gt_info[:, :, 1:]
        pad_bbox_flag = (gt_bboxes.sum(-1, keepdim=True) > 0).float()

        flatten_cls_preds = torch.cat([
            cls_pred.permute(0, 2, 3, 1).reshape(
                num_imgs, -1, self.num_classes)
            for cls_pred in cls_scores
        ], dim=1)
        flatten_coarse = torch.cat([
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in coarse_bbox_preds
        ], dim=1)
        flatten_refined = torch.cat([
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in refined_bbox_preds
        ], dim=1)

        decoded_coarse = self.bbox_coder.decode(
            self.flatten_priors_train[..., :2],
            flatten_coarse,
            self.stride_tensor[..., 0])
        decoded_refined = self.bbox_coder.decode(
            self.flatten_priors_train[..., :2],
            flatten_refined,
            self.stride_tensor[..., 0])

        assigned_result = self.assigner(
            decoded_coarse.detach().type(gt_bboxes.dtype),
            flatten_cls_preds.detach().sigmoid(),
            self.flatten_priors_train,
            gt_labels,
            gt_bboxes,
            pad_bbox_flag)
        assigned_bboxes = assigned_result['assigned_bboxes']
        assigned_scores = assigned_result['assigned_scores']
        positive_mask = assigned_result['fg_mask_pre_prior']
        assigned_scores_sum = assigned_scores.sum().clamp(min=1)

        if positive_mask.sum() > 0:
            box_mask = positive_mask.unsqueeze(-1).expand(-1, -1, 4)
            refined_feature_boxes = (
                decoded_refined / self.stride_tensor).masked_select(
                    box_mask).reshape(-1, 4)
            assigned_feature_boxes = (
                assigned_bboxes / self.stride_tensor).masked_select(
                    box_mask).reshape(-1, 4)
            bbox_weight = assigned_scores.sum(-1).masked_select(
                positive_mask).unsqueeze(-1)
            loss_refine = self.loss_bbox(
                refined_feature_boxes,
                assigned_feature_boxes,
                weight=bbox_weight) / assigned_scores_sum
        else:
            loss_refine = decoded_refined.sum() * 0.0

        if self.world_size == -1:
            _, world_size = get_dist_info()
        else:
            world_size = self.world_size
        return (loss_refine * num_imgs * world_size *
                self.refine_loss_weight)

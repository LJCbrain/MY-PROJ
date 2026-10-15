"""YOLO-World with independent visual proposals and no-RoI semantic reuse."""

import torch
import torch.nn.functional as F
from mmdet.structures.bbox import bbox_overlaps
from mmengine.structures import InstanceData
from mmyolo.registry import MODELS

from ..dense_heads.cavp_head import map_cavp_boxes, stable_score_order
from .yolo_world import YOLOWorldDetector


def read_cavp_logits(cls_scores, proposals, image_index, strides,
                     text_mask=None, mode='source_center', source_weight=0.5):
    """Sample logits in input coordinates, with half-stride grid centers."""
    num_classes = cls_scores[0].shape[1]
    logits = cls_scores[0].new_empty(
        (len(proposals), num_classes), dtype=torch.float32)
    for level, (level_scores, stride) in enumerate(zip(cls_scores, strides)):
        indices = (proposals.level_ids == level).nonzero(
            as_tuple=False).flatten()
        if not indices.numel():
            continue
        score_map = level_scores[image_index].float()
        source = score_map[:, proposals.grid_y[indices],
                           proposals.grid_x[indices]].transpose(0, 1)
        if mode == 'source':
            logits[indices] = source
            continue
        height, width = score_map.shape[-2:]
        boxes = proposals.bboxes[indices]
        centers = (boxes[:, :2] + boxes[:, 2:]) * 0.5
        # Feature index u = x / stride - 0.5; align_corners=False maps it
        # to normalized x = 2 * (u + 0.5) / width - 1.
        grid = centers / centers.new_tensor([stride * width, stride * height])
        grid = (2 * grid - 1).reshape(1, -1, 1, 2)
        center = F.grid_sample(
            score_map.unsqueeze(0), grid, mode='bilinear',
            padding_mode='border', align_corners=False)[0, :, :, 0].T
        logits[indices] = source_weight * source + (1 - source_weight) * center
    if text_mask is not None:
        logits = logits.masked_fill(~text_mask.to(
            device=logits.device, dtype=torch.bool).reshape(1, -1), -torch.inf)
    return logits


def fuse_cavp_detections(main, rescue, duplicate_iou=0.7, max_per_img=300):
    """Prefer main boxes for same-class duplicates, then enforce one budget."""
    before = len(rescue)
    if len(main) and len(rescue):
        overlaps = bbox_overlaps(rescue.bboxes.float(), main.bboxes.float())
        same_class = rescue.labels[:, None] == main.labels[None, :]
        duplicate = ((overlaps > duplicate_iou) & same_class).any(dim=1)
        rescue = rescue[~duplicate]
    boxes = torch.cat((main.bboxes.float(), rescue.bboxes.float()))
    scores = torch.cat((main.scores.float(), rescue.scores.float()))
    labels = torch.cat((main.labels, rescue.labels))
    # Main items come first, so stable ties also favor the main path.
    keep = stable_score_order(scores)[:max_per_img]
    stats = dict(main_count=len(main), rescue_count=before,
                 rescue_duplicates=before - len(rescue),
                 displaced_main=len(main) - int((keep < len(main)).sum()),
                 retained_rescue=int((keep >= len(main)).sum()))
    return InstanceData(bboxes=boxes[keep], scores=scores[keep],
                        labels=labels[keep]), stats


@MODELS.register_module()
class YOLOWorldCAVPDetector(YOLOWorldDetector):
    """Keep the semantic detector intact; train a detached visual branch."""

    def __init__(self, *args, cavp_neck, cavp_head, cavp_cfg=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.cavp_neck = MODELS.build(cavp_neck)
        self.cavp_head = MODELS.build(cavp_head)
        self.cavp_cfg = dict(
            enabled=True, fusion_enabled=True, loss_weight=1.0,
            pre_nms_topk=3000, proposal_nms_iou=0.7, proposal_max_per_img=300,
            semantic_readout='source_center', source_weight=0.5,
            semantic_score_thr=0.001, semantic_margin=0.0,
            score_temperature=1.0, score_bias=0.0,
            duplicate_iou=0.7, max_per_img=300)
        if cavp_cfg:
            unknown = set(cavp_cfg) - set(self.cavp_cfg)
            if unknown:
                raise ValueError(f'Unknown CAVP settings: {sorted(unknown)}')
            self.cavp_cfg.update(cavp_cfg)
        cfg = self.cavp_cfg
        if cfg['semantic_readout'] not in ('source', 'source_center'):
            raise ValueError('CAVP readout must be source or source_center.')
        if not 0 <= cfg['source_weight'] <= 1 or cfg['score_temperature'] <= 0:
            raise ValueError('Invalid CAVP source weight or temperature.')
        if cfg['loss_weight'] <= 0:
            raise ValueError('CAVP loss_weight must be positive.')
        for key in ('pre_nms_topk', 'proposal_max_per_img', 'max_per_img'):
            if cfg[key] <= 0:
                raise ValueError(f'{key} must be positive.')
        if tuple(self.cavp_head.featmap_strides) != tuple(
                self.bbox_head.featmap_strides):
            raise ValueError('CAVP and semantic grids must use matching strides.')
        if not cfg['enabled']:
            self.cavp_neck.requires_grad_(False)
            self.cavp_head.requires_grad_(False)

    def _extract_dual_features(self, batch_inputs, batch_data_samples):
        main, text, masks, raw = super().extract_feat(
            batch_inputs, batch_data_samples, return_backbone_feats=True)
        visual = self.cavp_neck(raw)
        if any(a.shape[-2:] != b.shape[-2:] for a, b in zip(main, visual)):
            raise ValueError('CAVP and semantic feature map sizes do not match.')
        return main, text, masks, visual

    def loss(self, batch_inputs, batch_data_samples):
        if not self.cavp_cfg['enabled']:
            return super().loss(batch_inputs, batch_data_samples)
        self.bbox_head.num_classes = self.num_train_classes
        main, text, masks, visual = self._extract_dual_features(
            batch_inputs, batch_data_samples)
        losses = self.bbox_head.loss(main, text, masks, batch_data_samples)
        losses.update(self._loss_fgbg(batch_inputs, batch_data_samples))
        losses.update(self._loss_text_prompt_bank())
        losses.update({key: value * self.cavp_cfg['loss_weight']
                       for key, value in self.cavp_head.loss(
                           visual, batch_data_samples).items()})
        return losses

    def _forward(self, batch_inputs, batch_data_samples=None):
        if not self.cavp_cfg['enabled']:
            return dict(main=super()._forward(batch_inputs, batch_data_samples),
                        cavp=None)
        main, text, masks, visual = self._extract_dual_features(
            batch_inputs, batch_data_samples)
        return dict(main=self.bbox_head(main, text, masks),
                    cavp=self.cavp_head(visual))

    def _label_proposals(self, cls_scores, proposals, index, text_masks,
                         meta, rescale):
        cfg = self.cavp_cfg
        mask = None if text_masks is None else (
            text_masks[index] if text_masks.ndim > 1 else text_masks)
        logits = read_cavp_logits(
            cls_scores, proposals, index, self.cavp_head.featmap_strides,
            text_mask=mask, mode=cfg['semantic_readout'],
            source_weight=cfg['source_weight'])
        probabilities = (logits / cfg['score_temperature'] +
                         cfg['score_bias']).sigmoid()
        scores, labels = probabilities.max(dim=1)
        valid = torch.isfinite(scores) & (scores > cfg['semantic_score_thr'])
        if cfg['semantic_margin'] > 0:
            top = probabilities.topk(min(2, probabilities.shape[1]), dim=1).values
            second = top[:, 1] if top.shape[1] == 2 else torch.zeros_like(scores)
            valid &= (scores - second) >= cfg['semantic_margin']
        boxes = map_cavp_boxes(proposals.bboxes, meta, rescale=rescale)
        valid &= ((boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1]))
        return InstanceData(bboxes=boxes[valid], scores=scores[valid],
                            labels=labels[valid])

    def predict(self, batch_inputs, batch_data_samples, rescale=True):
        if not self.cavp_cfg['enabled']:
            return super().predict(batch_inputs, batch_data_samples, rescale)
        main, text, masks, visual = self._extract_dual_features(
            batch_inputs, batch_data_samples)
        self.bbox_head.num_classes = text.shape[-2]
        # Compute the semantic head once; retain its unfiltered dense logits.
        cls_scores, bbox_preds = self.bbox_head(main, text, masks)
        metas = [sample.metainfo for sample in batch_data_samples]
        main_results = self.bbox_head.predict_by_feat(
            cls_scores, bbox_preds, batch_img_metas=metas, rescale=rescale)
        object_logits, cavp_boxes = self.cavp_head(visual)
        cfg = self.cavp_cfg
        proposals = self.cavp_head.predict_proposals(
            object_logits, cavp_boxes, metas,
            pre_nms_topk=cfg['pre_nms_topk'],
            nms_iou_threshold=cfg['proposal_nms_iou'],
            max_per_img=cfg['proposal_max_per_img'])
        final_results = []
        for index, (sample, original, proposal, meta) in enumerate(zip(
                batch_data_samples, main_results, proposals, metas)):
            rescue = self._label_proposals(
                cls_scores, proposal, index, masks, meta, rescale)
            if cfg['fusion_enabled']:
                final, stats = fuse_cavp_detections(
                    original, rescue, cfg['duplicate_iou'], cfg['max_per_img'])
            else:
                final = original
                stats = dict(main_count=len(original), rescue_count=len(rescue),
                             rescue_duplicates=0, displaced_main=0,
                             retained_rescue=0)
            mapped = proposal.clone()
            mapped.bboxes = map_cavp_boxes(proposal.bboxes, meta, rescale)
            sample.proposal_instances = mapped
            sample.main_pred_instances = original
            sample.rescue_pred_instances = rescue
            sample.cavp_stats = stats
            final_results.append(final)
        self._check_pred_label_range(final_results, self.bbox_head.num_classes)
        return self.add_pred_to_datasample(batch_data_samples, final_results)

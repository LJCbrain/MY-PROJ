"""Independent one-class assignment, DFL regression and visual proposals."""

import copy

import numpy as np
import torch
from mmcv.ops import nms
from mmengine.structures import InstanceData
from mmyolo.models.dense_heads import YOLOv8Head
from mmyolo.registry import MODELS


def stable_score_order(scores):
    """Descending 1-D score indices, including on older PyTorch releases.

    Some releases support stable torch.sort but not Tensor.argsort. For
    releases without either stable API, use an exact CPU mergesort instead
    of perturbing scores, which could reorder close but unequal values.
    """
    try:
        return torch.sort(scores, descending=True, stable=True)[1]
    except TypeError:
        values = scores.detach().cpu().to(torch.float64).numpy()
        order = np.argsort(-values, kind='mergesort')
        return torch.as_tensor(order, dtype=torch.long, device=scores.device)


def map_cavp_boxes(boxes, meta, rescale=True):
    """Map input-space boxes without modifying the stored proposals."""
    boxes = boxes.clone()
    if rescale:
        pad = meta.get('pad_param', (0, 0, 0, 0))
        boxes -= boxes.new_tensor([pad[2], pad[0], pad[2], pad[0]])
        scale = boxes.new_tensor(meta['scale_factor']).flatten()
        if scale.numel() == 2:
            scale = scale.repeat(2)
        boxes /= scale
        height, width = meta['ori_shape'][:2]
    else:
        height, width = meta['img_shape'][:2]
    boxes[:, 0::2].clamp_(0, width)
    boxes[:, 1::2].clamp_(0, height)
    return boxes


@MODELS.register_module()
class CAVPHead(YOLOv8Head):
    """A YOLOv8 one-class head whose only class denotes objectness."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.num_classes != 1:
            raise ValueError('CAVPHead requires num_classes=1.')

    @staticmethod
    def object_targets(batch_data_samples):
        if isinstance(batch_data_samples, dict):
            samples = dict(batch_data_samples)
            targets = batch_data_samples['bboxes_labels'].clone()
            targets[:, 1] = 0
            samples['bboxes_labels'] = targets
        else:
            samples = copy.deepcopy(batch_data_samples)
            for sample in samples:
                sample.gt_instances.labels.zero_()
        return samples

    def loss(self, x, batch_data_samples):
        losses = super().loss(x, self.object_targets(batch_data_samples))
        return {
            'loss_cavp_obj': losses['loss_cls'],
            'loss_cavp_bbox': losses['loss_bbox'],
            'loss_cavp_dfl': losses['loss_dfl'],
        }

    def predict_proposals(self, object_logits, bbox_preds, batch_img_metas,
                          pre_nms_topk=3000, nms_iou_threshold=0.7,
                          max_per_img=300, scores_are_logits=True):
        """Return input-space proposals and their original dense indices.

        Evaluation adapters may supply existing probability maps with
        ``scores_are_logits=False`` to share the exact CAVP postprocessing.
        The normal CAVP path always uses logits.
        """
        sizes = [item.shape[-2:] for item in object_logits]
        counts = [h * w for h, w in sizes]
        device = object_logits[0].device
        priors = torch.cat(self.prior_generator.grid_priors(
            sizes, dtype=torch.float32, device=device))
        strides = torch.cat([
            priors.new_full((count,), stride)
            for count, stride in zip(counts, self.featmap_strides)])
        batch_size = object_logits[0].shape[0]
        scores = torch.cat([
            item.reshape(batch_size, -1) for item in object_logits
        ], dim=1).float()
        if scores_are_logits:
            scores = scores.sigmoid()
        distances = torch.cat([
            item.permute(0, 2, 3, 1).reshape(batch_size, -1, 4)
            for item in bbox_preds
        ], dim=1).float()
        boxes = self.bbox_coder.decode(priors[None], distances, strides)
        ends = torch.tensor(counts, device=device).cumsum(0)
        starts = torch.cat((ends.new_zeros(1), ends[:-1]))
        widths = torch.tensor([w for _, w in sizes], device=device)
        results = []
        for image_boxes, image_scores, meta in zip(
                boxes, scores, batch_img_metas):
            finite = torch.isfinite(image_boxes).all(-1)
            image_boxes = map_cavp_boxes(image_boxes, meta, rescale=False)
            valid = (finite & torch.isfinite(image_scores) &
                     (image_boxes[:, 2] > image_boxes[:, 0]) &
                     (image_boxes[:, 3] > image_boxes[:, 1]))
            indices = valid.nonzero(as_tuple=False).flatten()
            order = stable_score_order(image_scores[indices])
            indices = indices[order[:pre_nms_topk]]
            if indices.numel():
                _, keep = nms(image_boxes[indices], image_scores[indices],
                              nms_iou_threshold)
                indices = indices[keep[:max_per_img]]
            levels = torch.bucketize(indices, ends[:-1], right=True)
            local = indices - starts[levels]
            results.append(InstanceData(
                bboxes=image_boxes[indices], scores=image_scores[indices],
                labels=torch.zeros_like(indices), flat_indices=indices,
                level_ids=levels, grid_x=local % widths[levels],
                grid_y=local // widths[levels]))
        return results

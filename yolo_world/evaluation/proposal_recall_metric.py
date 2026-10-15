"""Class-agnostic proposal recall for DIOR Scheme A."""

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
from mmengine.evaluator import BaseMetric
from mmyolo.registry import METRICS


@METRICS.register_module()
class PriorProposalRecallMetric(BaseMetric):
    """Recall of prior-scored decoded proposals at fixed IoU thresholds."""

    default_prefix = 'proposal'

    def __init__(self,
                 ann_file: str,
                 base_classes: Sequence[str],
                 novel_classes: Sequence[str],
                 iou_thresholds: Sequence[float] = (0.3, 0.5),
                 proposal_num: int = 300,
                 proposal_nums: Optional[Sequence[int]] = None,
                 report_classwise: bool = False,
                 collect_device: str = 'cpu',
                 prefix: str = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.ann_file = ann_file
        self.base_classes = tuple(base_classes)
        self.novel_classes = tuple(novel_classes)
        self._base_class_set = set(self.base_classes)
        self._novel_class_set = set(self.novel_classes)
        self.iou_thresholds = tuple(float(v) for v in iou_thresholds)
        if proposal_nums is None:
            proposal_nums = (proposal_num, )
        self.proposal_nums = tuple(
            sorted({int(value) for value in proposal_nums}))
        if not self.proposal_nums or self.proposal_nums[0] <= 0:
            raise ValueError('proposal_nums must contain positive integers.')
        self.proposal_num = self.proposal_nums[-1]
        self.report_classwise = bool(report_classwise)
        self._ground_truth = self._load_ground_truth(ann_file)

    def _load_ground_truth(self, ann_file: str) -> dict:
        payload = json.loads(Path(ann_file).read_text(encoding='utf-8'))
        names = {cat['id']: cat['name'] for cat in payload['categories']}
        ground_truth = defaultdict(list)
        for ann in payload['annotations']:
            x, y, width, height = ann['bbox']
            name = names[ann['category_id']]
            group = ('base' if name in self._base_class_set else
                     'novel' if name in self._novel_class_set else None)
            if group is None:
                continue
            ground_truth[int(ann['image_id'])].append(
                ([x, y, x + width, y + height], group, name))
        return ground_truth

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        del data_batch
        for sample in data_samples:
            if isinstance(sample, dict):
                image_id = int(
                    sample['img_id'] if 'img_id' in sample else
                    sample['metainfo']['img_id'])
                proposals = sample.get('proposal_instances', None)
            else:
                image_id = int(sample.metainfo['img_id'])
                proposals = getattr(sample, 'proposal_instances', None)
            if proposals is None:
                boxes = np.zeros((0, 4), dtype=np.float32)
            else:
                proposal_boxes = (
                    proposals['bboxes'] if isinstance(proposals, dict) else
                    proposals.bboxes)
                boxes = proposal_boxes[:self.proposal_num].detach().cpu().numpy()
            self.results.append(dict(image_id=image_id, boxes=boxes))

    @staticmethod
    def _iou(gt_box: np.ndarray, proposals: np.ndarray) -> np.ndarray:
        if proposals.size == 0:
            return np.zeros((0, ), dtype=np.float32)
        left_top = np.maximum(gt_box[None, :2], proposals[:, :2])
        right_bottom = np.minimum(gt_box[None, 2:], proposals[:, 2:])
        wh = np.clip(right_bottom - left_top, 0, None)
        intersection = wh[:, 0] * wh[:, 1]
        gt_area = np.prod(np.clip(gt_box[2:] - gt_box[:2], 0, None))
        proposal_area = np.prod(
            np.clip(proposals[:, 2:] - proposals[:, :2], 0, None), axis=1)
        return intersection / np.clip(
            gt_area + proposal_area - intersection, 1e-6, None)

    def compute_metrics(self, results: list) -> Dict[str, float]:
        proposals_by_image = {
            int(result['image_id']): np.asarray(
                result['boxes'], dtype=np.float32)
            for result in results
        }
        totals = {group: 0 for group in ('all', 'base', 'novel')}
        class_totals = {name: 0 for name in self.novel_classes}
        recalled = {
            (proposal_num, threshold): {
                group: 0
                for group in ('all', 'base', 'novel')
            }
            for proposal_num in self.proposal_nums
            for threshold in self.iou_thresholds
        }
        class_recalled = {
            (proposal_num, threshold): {
                name: 0
                for name in self.novel_classes
            }
            for proposal_num in self.proposal_nums
            for threshold in self.iou_thresholds
        }
        for image_id, instances in self._ground_truth.items():
            proposals = proposals_by_image.get(
                image_id, np.zeros((0, 4), dtype=np.float32))
            for box, group, class_name in instances:
                totals['all'] += 1
                totals[group] += 1
                if group == 'novel':
                    class_totals[class_name] += 1
                overlaps = self._iou(np.asarray(box, dtype=np.float32), proposals)
                for proposal_num in self.proposal_nums:
                    topk_overlaps = overlaps[:proposal_num]
                    best_iou = (
                        float(topk_overlaps.max())
                        if topk_overlaps.size else 0.0)
                    for threshold in self.iou_thresholds:
                        if best_iou < threshold:
                            continue
                        key = (proposal_num, threshold)
                        recalled[key]['all'] += 1
                        recalled[key][group] += 1
                        if group == 'novel':
                            class_recalled[key][class_name] += 1

        metrics = {}
        for proposal_num in self.proposal_nums:
            for threshold in self.iou_thresholds:
                key = (proposal_num, threshold)
                iou_name = int(round(threshold * 100))
                for group in ('all', 'base', 'novel'):
                    denominator = max(totals[group], 1)
                    metrics[
                        f'recall@{proposal_num}_iou{iou_name}_{group}'] = (
                            recalled[key][group] / denominator)
                if self.report_classwise:
                    for class_name in self.novel_classes:
                        safe_name = re.sub(
                            r'[^0-9A-Za-z]+', '_', class_name).strip('_')
                        denominator = max(class_totals[class_name], 1)
                        metrics[
                            f'recall@{proposal_num}_iou{iou_name}_novel_'
                            f'{safe_name}'] = (
                                class_recalled[key][class_name] / denominator)
        return metrics

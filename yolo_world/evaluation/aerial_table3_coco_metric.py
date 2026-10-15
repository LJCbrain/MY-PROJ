"""Metrics required by the upper block of RT-OVAD Table III."""

from __future__ import annotations

import os.path as osp
import tempfile
from typing import Dict, Sequence

import numpy as np
from mmdet.datasets.api_wrappers import COCOeval
from mmengine.fileio import load
from mmyolo.registry import METRICS

from .dior_coco_metric import DIORCocoMetric


@METRICS.register_module()
class AerialTableIIICocoMetric(DIORCocoMetric):
    """COCO AP50/Recall@0.5 metrics for Table III GZSD and ZSD.

    ``DIORCocoMetric`` already reports AP50 for the configured base and
    novel category groups.  This extension adds the corresponding recall
    values at IoU=0.5 and their harmonic mean.  Table III follows the
    DescReg protocol and reports RE@100, so ``recall_max_dets`` defaults to
    100.  The detector may still retain more predictions; COCOeval evaluates
    only its highest-scoring 100 detections per image for this recall metric.
    """

    def __init__(self,
                 *args,
                 recall_max_dets: int = 100,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if recall_max_dets <= 0:
            raise ValueError('recall_max_dets must be positive.')
        self.recall_max_dets = int(recall_max_dets)

    def compute_metrics(self, results: list) -> Dict[str, float]:
        eval_results = super().compute_metrics(results)
        metrics = getattr(self, 'metrics', getattr(self, 'metric', []))
        if isinstance(metrics, str):
            metrics = [metrics]
        if self.format_only or 'bbox' not in metrics:
            return eval_results

        eval_results.update(self._compute_bbox_recall50_group_metrics(results))
        return eval_results

    def _compute_bbox_recall50_group_metrics(
            self, results: list) -> Dict[str, float]:
        tmp_dir = None
        if self.outfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            outfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            outfile_prefix = f'{self.outfile_prefix}.table3_recall'

        try:
            pred_results = self._extract_predictions(results)
            result_files = self.results2json(pred_results, outfile_prefix)
            bbox_result_file = result_files.get('bbox')
            if bbox_result_file is None:
                return self._empty_recall_group_results()

            predictions = load(bbox_result_file)
            if not predictions:
                return self._empty_recall_group_results()

            coco_api = self._coco_api
            cat_ids = self._get_cat_ids(coco_api)
            img_ids = self._get_img_ids(coco_api)
            coco_dt = coco_api.loadRes(predictions)
            coco_eval = COCOeval(coco_api, coco_dt, 'bbox')
            coco_eval.params.catIds = cat_ids
            coco_eval.params.imgIds = img_ids
            coco_eval.params.iouThrs = np.array([0.5])
            coco_eval.params.maxDets = [self.recall_max_dets]
            coco_eval.evaluate()
            coco_eval.accumulate()

            class_recalls = self._per_class_recall50(coco_api, cat_ids,
                                                      coco_eval)
            base_recall = self._mean_ap(class_recalls, self.base_classes)
            novel_recall = self._mean_ap(class_recalls, self.novel_classes)
            hm = self._harmonic_mean(base_recall, novel_recall)
            return {
                'bbox_recall_50_base': float(f'{base_recall:.3f}'),
                'bbox_recall_50_novel': float(f'{novel_recall:.3f}'),
                'bbox_recall_50_HM': float(f'{hm:.3f}'),
            }
        finally:
            if tmp_dir is not None:
                tmp_dir.cleanup()

    @staticmethod
    def _per_class_recall50(coco_api, cat_ids: Sequence[int],
                            coco_eval: COCOeval) -> Dict[str, float]:
        recalls = coco_eval.eval['recall']
        if hasattr(coco_api, 'loadCats'):
            categories = coco_api.loadCats(cat_ids)
        elif hasattr(coco_api, 'load_cats'):
            categories = coco_api.load_cats(cat_ids)
        else:
            categories = [coco_api.cats[cat_id] for cat_id in cat_ids]

        class_recalls = {}
        for index, category in enumerate(categories):
            recall = float(recalls[0, index, 0, -1])
            class_recalls[category['name']] = max(0.0, recall)
        return class_recalls

    @staticmethod
    def _empty_recall_group_results() -> Dict[str, float]:
        return {
            'bbox_recall_50_base': 0.0,
            'bbox_recall_50_novel': 0.0,
            'bbox_recall_50_HM': 0.0,
        }

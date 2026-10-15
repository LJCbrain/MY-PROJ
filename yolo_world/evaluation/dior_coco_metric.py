import os.path as osp
import tempfile
from typing import Dict, List, Sequence

import numpy as np
from mmdet.datasets.api_wrappers import COCOeval
from mmdet.evaluation.metrics import CocoMetric
from mmengine.fileio import load
from mmyolo.registry import METRICS


@METRICS.register_module()
class DIORCocoMetric(CocoMetric):
    """COCO metric with DIOR base/novel AP50 and harmonic mean."""

    def __init__(self,
                 base_classes: Sequence[str],
                 novel_classes: Sequence[str],
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.base_classes = tuple(base_classes)
        self.novel_classes = tuple(novel_classes)

    def compute_metrics(self, results: list) -> Dict[str, float]:
        eval_results = super().compute_metrics(results)
        metrics = getattr(self, 'metrics', getattr(self, 'metric', []))
        if isinstance(metrics, str):
            metrics = [metrics]
        if self.format_only or 'bbox' not in metrics:
            return eval_results

        group_results = self._compute_bbox_ap50_group_metrics(results)
        eval_results.update(group_results)
        return eval_results

    def _compute_bbox_ap50_group_metrics(self, results: list) -> Dict[str,
                                                                       float]:
        tmp_dir = None
        if self.outfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            outfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            outfile_prefix = f'{self.outfile_prefix}.dior_groups'

        try:
            pred_results = self._extract_predictions(results)
            result_files = self.results2json(pred_results, outfile_prefix)
            bbox_result_file = result_files.get('bbox', None)
            if bbox_result_file is None:
                return {}

            predictions = load(bbox_result_file)
            if len(predictions) == 0:
                return self._empty_group_results()

            coco_api = self._coco_api
            cat_ids = self._get_cat_ids(coco_api)
            img_ids = self._get_img_ids(coco_api)
            coco_dt = coco_api.loadRes(predictions)
            coco_eval = COCOeval(coco_api, coco_dt, 'bbox')
            coco_eval.params.catIds = cat_ids
            coco_eval.params.imgIds = img_ids
            coco_eval.params.maxDets = list(self.proposal_nums)
            if self.iou_thrs is not None:
                coco_eval.params.iouThrs = self.iou_thrs
            coco_eval.evaluate()
            coco_eval.accumulate()

            class_ap50s = self._per_class_ap50(coco_api, cat_ids, coco_eval)
            base_ap50 = self._mean_ap(class_ap50s, self.base_classes)
            novel_ap50 = self._mean_ap(class_ap50s, self.novel_classes)
            hm = self._harmonic_mean(base_ap50, novel_ap50)
            return {
                'bbox_mAP_50_base': float(f'{base_ap50:.3f}'),
                'bbox_mAP_50_novel': float(f'{novel_ap50:.3f}'),
                'bbox_mAP_50_HM': float(f'{hm:.3f}'),
            }
        finally:
            if tmp_dir is not None:
                tmp_dir.cleanup()

    def _get_cat_ids(self, coco_api) -> List[int]:
        if getattr(self, 'cat_ids', None) is not None:
            return self.cat_ids
        classes = self.dataset_meta['classes']
        if hasattr(coco_api, 'get_cat_ids'):
            return coco_api.get_cat_ids(cat_names=classes)
        return coco_api.getCatIds(catNms=list(classes))

    def _get_img_ids(self, coco_api) -> List[int]:
        if getattr(self, 'img_ids', None) is not None:
            return self.img_ids
        if hasattr(coco_api, 'get_img_ids'):
            return coco_api.get_img_ids()
        return coco_api.getImgIds()

    @staticmethod
    def _extract_predictions(results: list) -> list:
        if not results:
            return results
        first = results[0]
        if isinstance(first, tuple):
            return [pred for _, pred in results]
        return results

    def _per_class_ap50(self, coco_api, cat_ids: Sequence[int],
                        coco_eval: COCOeval) -> Dict[str, float]:
        precisions = coco_eval.eval['precision']
        iou_thresholds = coco_eval.params.iouThrs
        ap50_index = int(np.argmin(np.abs(iou_thresholds - 0.5)))
        class_ap50s = {}
        if hasattr(coco_api, 'loadCats'):
            cats = coco_api.loadCats(cat_ids)
        elif hasattr(coco_api, 'load_cats'):
            cats = coco_api.load_cats(cat_ids)
        else:
            cats = [coco_api.cats[cat_id] for cat_id in cat_ids]
        for idx, cat in enumerate(cats):
            precision = precisions[ap50_index, :, idx, 0, -1]
            precision = precision[precision > -1]
            class_ap50s[cat['name']] = float(np.mean(precision)
                                             ) if precision.size else 0.0
        return class_ap50s

    @staticmethod
    def _mean_ap(class_aps: Dict[str, float], classes: Sequence[str]) -> float:
        aps = [class_aps[name] for name in classes if name in class_aps]
        return float(np.mean(aps)) if aps else 0.0

    @staticmethod
    def _harmonic_mean(base_score: float, novel_score: float) -> float:
        if base_score <= 0.0 or novel_score <= 0.0:
            return 0.0
        return 2.0 * base_score * novel_score / (base_score + novel_score)

    @staticmethod
    def _empty_group_results() -> Dict[str, float]:
        return {
            'bbox_mAP_50_base': 0.0,
            'bbox_mAP_50_novel': 0.0,
            'bbox_mAP_50_HM': 0.0,
        }

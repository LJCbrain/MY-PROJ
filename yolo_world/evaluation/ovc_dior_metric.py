"""Full-precision group AP50 and per-Novel AP50 for the 2.14 experiments."""
from mmyolo.registry import METRICS

from .dior_coco_metric import DIORCocoMetric


@METRICS.register_module()
class OVCDIORCocoMetric(DIORCocoMetric):
    def __init__(self, *args, report_all_classes=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.report_all_classes = report_all_classes

    def _per_class_ap50(self, coco_api, cat_ids, coco_eval):
        values = super()._per_class_ap50(coco_api, cat_ids, coco_eval)
        self._ovc_class_ap50 = values
        return values

    def _compute_bbox_ap50_group_metrics(self, results):
        self._ovc_class_ap50 = {}
        values = super()._compute_bbox_ap50_group_metrics(results)
        if not values:
            return values
        classes = self._ovc_class_ap50
        base = self._mean_ap(classes, self.base_classes)
        novel = self._mean_ap(classes, self.novel_classes)
        values.update(bbox_mAP_50_base=base, bbox_mAP_50_novel=novel,
                      bbox_mAP_50_HM=self._harmonic_mean(base, novel))
        values.update({'bbox_AP50_novel/' + name: classes.get(name, 0.)
                       for name in self.novel_classes})
        if self.report_all_classes:
            values.update({'bbox_AP50/' + name: value
                           for name, value in classes.items()})
        return values

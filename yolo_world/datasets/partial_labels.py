"""Explicit incomplete-annotation metadata for DIOR 2.15 only."""
import torch
from mmyolo.datasets import YOLOv5CocoDataset
from mmyolo.registry import DATASETS
from mmengine.dataset import COLLATE_FUNCTIONS

from .utils import yolow_collate


@DATASETS.register_module()
class PartialLabelYOLOv5CocoDataset(YOLOv5CocoDataset):
    def parse_data_info(self, raw_data_info):
        data = super().parse_data_info(raw_data_info)
        data['partial_labels'] = bool(raw_data_info['raw_img_info'].get('partial_labels', False))
        return data


@COLLATE_FUNCTIONS.register_module()
def yolow_partial_collate(data_batch, use_ms_training=False):
    result = yolow_collate(data_batch, use_ms_training=use_ms_training)
    result['data_samples']['partial_labels'] = torch.tensor(
        [sample['data_samples'].metainfo.get('partial_labels', False) for sample in data_batch],
        dtype=torch.bool)
    return result

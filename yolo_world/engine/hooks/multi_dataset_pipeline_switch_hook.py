"""Pipeline switching for a ``ConcatDataset`` of multimodal datasets."""

from __future__ import annotations

from copy import deepcopy

from mmengine.dataset import Compose
from mmengine.hooks import Hook
from mmyolo.registry import HOOKS


@HOOKS.register_module()
class MultiDatasetPipelineSwitchHook(Hook):
    """Switch every child pipeline of a concatenated training dataset.

    The stock ``PipelineSwitchHook`` only updates a dataset that directly
    exposes ``pipeline``.  DIOR+DOTA training uses ``ConcatDataset`` whose
    two ``MultiModalDataset`` children own the actual pipelines.
    """

    def __init__(self, switch_epoch: int, switch_pipeline: list) -> None:
        self.switch_epoch = int(switch_epoch)
        self.switch_pipeline = switch_pipeline
        self._switched = False

    def before_train_epoch(self, runner) -> None:
        if self._switched or runner.epoch != self.switch_epoch:
            return

        dataset = runner.train_dataloader.dataset
        children = getattr(dataset, 'datasets', None)
        if children is None:
            children = [dataset]

        switched = 0
        for child in children:
            if not hasattr(child, 'pipeline'):
                continue
            child.pipeline = Compose(deepcopy(self.switch_pipeline))
            switched += 1
        if switched != len(children):
            raise TypeError(
                'Every child of the DIOR+DOTA training dataset must expose '
                'a pipeline for MultiDatasetPipelineSwitchHook.')

        self._switched = True
        runner.logger.info(
            'Switched %d DIOR/DOTA child dataset pipelines at epoch %d.',
            switched, runner.epoch)

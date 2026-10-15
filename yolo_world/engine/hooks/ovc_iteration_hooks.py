"""Iteration equivalents of the original 1082-step DIOR epoch schedule."""
from copy import deepcopy
from types import SimpleNamespace

from mmengine.dataset import Compose
from mmengine.hooks import Hook
from mmyolo.engine.hooks import YOLOv5ParamSchedulerHook
from mmyolo.registry import HOOKS


@HOOKS.register_module()
class OVCIterationParamSchedulerHook(YOLOv5ParamSchedulerHook):
    def __init__(self, virtual_epoch_iters=1082, **kwargs):
        super().__init__(**kwargs)
        self.virtual_epoch_iters = virtual_epoch_iters

    def before_train_iter(self, runner, batch_idx, data_batch=None):
        size = self.virtual_epoch_iters
        virtual_epoch = runner.iter // size
        warmup = max(round(self.warmup_epochs * size), self.warmup_mim_iter)
        if runner.iter <= warmup:
            proxy = SimpleNamespace(iter=runner.iter, epoch=virtual_epoch,
                                    train_dataloader=range(size), optim_wrapper=runner.optim_wrapper)
            super().before_train_iter(proxy, batch_idx, data_batch)
        else:
            # The original hook updates at epoch END, using that epoch's index.
            factor = self.scheduler_fn(max(warmup // size, virtual_epoch - 1))
            for i, group in enumerate(runner.optim_wrapper.optimizer.param_groups):
                group['lr'] = self._base_lr[i] * factor
                if 'momentum' in group:
                    group['momentum'] = self._base_momentum[i]

    def after_train_epoch(self, runner):
        pass


@HOOKS.register_module()
class OVCIterationPipelineSwitchHook(Hook):
    priority = 48

    def __init__(self, switch_iter, switch_pipeline):
        self.switch_iter = int(switch_iter)
        self.switch_pipeline = switch_pipeline
        self._switched = False

    def _switch(self, runner):
        if self._switched:
            return
        loader = runner.train_dataloader
        loader.dataset.pipeline = Compose(deepcopy(self.switch_pipeline))
        # Drop prefetched Mosaic batches and rebuild worker copies of the dataset.
        wrapper = runner.train_loop.dataloader_iterator
        iterator = wrapper._iterator
        if hasattr(iterator, '_shutdown_workers'):
            iterator._shutdown_workers()
        if hasattr(loader, '_iterator'):
            loader._iterator = None
        wrapper._iterator = iter(loader)
        self._switched = True
        runner.logger.info('OVC: clean-image pipeline enabled at completed step %d.', self.switch_iter)

    def before_train(self, runner):
        if runner.iter >= self.switch_iter:
            self._switch(runner)

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        if runner.iter + 1 >= self.switch_iter:
            self._switch(runner)

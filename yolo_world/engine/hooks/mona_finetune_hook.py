"""Preserve trained state when starting DIOR 3.6 from a 2.17 checkpoint."""

from collections import Counter
import re

from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmyolo.registry import HOOKS
from torch.nn.modules.batchnorm import _BatchNorm


@HOOKS.register_module()
class MonaFineTuneHook(Hook):
    # Validate the checkpoint before EMAHook loads its copy (priority 49).
    priority = 48
    _counter_names = ('_routing_step', '_semantic_anchor_step')
    _new_prefix = 'bbox_head.head_module.cls_mona.'
    _run_name = 'DIOR 3.6'

    def __init__(self, source_iter=25968, freeze_bn=True):
        self.source_iter = int(source_iter)
        if self.source_iter < 0:
            raise ValueError('source_iter must be non-negative.')
        self.freeze_bn = bool(freeze_bn)
        self._loaded = False
        self._saved_counters = {}

    @staticmethod
    def _model(runner):
        model = runner.model
        return model.module if is_model_wrapper(model) else model

    @staticmethod
    def _has_mona(model):
        return model.bbox_head.head_module.cls_mona is not None

    def after_load_checkpoint(self, runner, checkpoint):
        model = self._model(runner)
        state = checkpoint.get('state_dict', checkpoint)
        loaded = {re.sub(r'^module\.', '', key): value
                  for key, value in state.items()}
        expected = model.state_dict()
        meta = checkpoint.get('meta', {})
        if getattr(runner, '_resume', False) and 'mona_finetune' not in meta:
            raise RuntimeError(
                'Starting plan A or B from 2.17 requires resume=False. '
                f'Resume is reserved for a checkpoint from the same {self._run_name} run.')
        missing = [key for key in expected if key not in loaded
                   and not key.startswith(self._new_prefix)]
        unexpected = [key for key in loaded if key not in expected]
        mismatched = [key for key in expected if key in loaded
                      and expected[key].shape != loaded[key].shape]
        if missing or unexpected or mismatched:
            raise RuntimeError(
                f'{self._run_name} requires a compatible trained checkpoint; only new '
                'Mona parameters may be absent. '
                f'Missing={missing[:12]}, unexpected={unexpected[:12]}, '
                f'shape_mismatch={mismatched[:12]}.')
        # Resuming 3.6 must restore the adapter, rather than silently restart it.
        if getattr(runner, '_resume', False):
            missing_mona = [key for key in expected if key not in loaded]
            if missing_mona:
                raise RuntimeError(
                    f'Use load_from with resume=False to start {self._run_name} from 2.17. '
                    'Resume requires the same model and optimizer groups.')

        previous = meta.get('mona_finetune', {})
        self.source_iter = int(previous.get(
            'source_iter', meta.get('iter', self.source_iter)))
        self._saved_counters = previous.get('model_counters', {})
        self.source_checkpoint = previous.get(
            'source_checkpoint', runner.cfg.get('load_from'))
        self._loaded = True
        runner.logger.info(
            self._run_name + ': checkpoint compatible; %d new Mona tensors, '
            'source completed iteration=%d.',
            sum(key not in loaded for key in expected), self.source_iter)

    def _restore_counters(self, model):
        for name, buffer in model.named_buffers():
            if name.rsplit('.', 1)[-1] in self._counter_names:
                buffer.fill_(int(self._saved_counters.get(name, self.source_iter)))

    def _freeze_bn(self, model):
        if self.freeze_bn:
            for module in model.modules():
                if isinstance(module, _BatchNorm):
                    # Affine parameters keep their configured learning rates.
                    module.eval()

    def before_train(self, runner):
        if not self._loaded:
            raise RuntimeError(self._run_name + ' must load a trained checkpoint before training.')
        model = self._model(runner)
        self._restore_counters(model)
        self._freeze_bn(model)
        for hook in runner.hooks:
            if hasattr(hook, 'ema_model'):
                self._restore_counters(hook.ema_model.module)
        variant = 'B (Mona)' if self._has_mona(model) else 'A (no Mona)'
        groups = Counter()
        for group in runner.optim_wrapper.optimizer.param_groups:
            count = sum(p.numel() for p in group['params'] if p.requires_grad)
            if count:
                groups[group.get('initial_lr', group['lr'])] += count
        runner.logger.info(
            self._run_name + ': plan %s; trainable parameter counts by peak LR=%s; '
            'freeze BN running statistics=%s. Nonpersistent prompt counters '
            'restored in the detector and EMA.', variant, dict(groups), self.freeze_bn)

    def before_train_iter(self, runner, batch_idx, data_batch=None):
        # IterBasedTrainLoop calls model.train() again before EVERY iteration.
        self._freeze_bn(self._model(runner))

    def before_save_checkpoint(self, runner, checkpoint):
        model = self._model(runner)
        counters = {name: int(buffer.item())
                    for name, buffer in model.named_buffers()
                    if name.rsplit('.', 1)[-1] in self._counter_names}
        checkpoint.setdefault('meta', {})['mona_finetune'] = dict(
            source_iter=self.source_iter,
            source_checkpoint=self.source_checkpoint,
            model_counters=counters,
            variant='B' if self._has_mona(model) else 'A')

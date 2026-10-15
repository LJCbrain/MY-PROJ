"""Validate original YOLO-World loading and the first-stage Mona mode."""

from collections import Counter
import re

from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmyolo.registry import HOOKS


@HOOKS.register_module()
class MonaJointTrainingHook(Hook):
    priority = 48  # Inspect state before EMAHook handles its copy at 49.
    _meta_key = 'mona_joint_training'
    _counter_names = ('_routing_step', '_semantic_anchor_step')
    _mona_prefixes = tuple(
        f'backbone.image_model.stage{i}.1.mona_adapters.'
        for i in range(1, 5))
    _new_module_prefixes = (
        'neck.visual_adapter.', 'neck.dual_stream.', 'neck.wsr.',
        'backbone.text_model.prompt_bank.')
    _new_buffers = ('_ovc_iteration', '_ovc_variance_batches', '_ovc_covariance_batches')

    def __init__(self):
        self._loaded = False
        self._saved_counters = {}
        self._fresh = True

    @staticmethod
    def _model(runner):
        model = runner.model
        return model.module if is_model_wrapper(model) else model

    def after_load_checkpoint(self, runner, checkpoint):
        model = self._model(runner)
        state = checkpoint.get('state_dict', checkpoint)
        loaded = {re.sub(r'^module\.', '', k): v for k, v in state.items()}
        expected = model.state_dict()
        previous = checkpoint.get('meta', {}).get(self._meta_key)
        self._fresh = previous is None
        if getattr(runner, '_resume', False) and self._fresh:
            raise RuntimeError('Joint Mona training must start from original '
                               'YOLO-World pretraining with resume=False. '
                               'Resume requires a checkpoint from the same '
                               'joint-training experiment.')
        if self._fresh and any(
                key.startswith(self._new_module_prefixes) or
                key.startswith(self._mona_prefixes) for key in loaded):
            raise RuntimeError('Use original YOLO-World pretraining for the '
                               'first joint-training run. This checkpoint '
                               'already contains trained DIOR custom modules '
                               'or Mona; use its matching experiment config.')
        missing = [key for key in expected if key not in loaded]
        permitted = self._mona_prefixes + self._new_module_prefixes
        invalid_missing = [key for key in missing
                           if not (self._fresh and (
                               key.startswith(permitted) or key in self._new_buffers))]
        unexpected = [key for key in loaded if key not in expected]
        mismatched = [key for key in expected if key in loaded
                      and expected[key].shape != loaded[key].shape]
        if invalid_missing or unexpected or mismatched:
            raise RuntimeError('Joint Mona checkpoint is incompatible; all '
                               'original image/neck/head/CLIP weights must '
                               'load. Only the declared new modules may be '
                               'initialized in a fresh run. '
                               f'Missing={invalid_missing[:12]}, '
                               f'unexpected={unexpected[:12]}, '
                               f'shape_mismatch={mismatched[:12]}.')
        self._saved_counters = (previous or {}).get('model_counters', {})
        self.source_checkpoint = (previous or {}).get(
            'source_checkpoint', runner.cfg.get('load_from'))
        self._loaded = True
        runner.logger.info(
            'DIOR 3.7 joint: checkpoint compatible; %d new Mona tensors and '
            '%d other new tensors initialized; fresh first-stage run=%s. '
            'Pretraining iteration metadata is not used as DIOR iteration.',
            sum(key.startswith(self._mona_prefixes) for key in missing),
            sum(not key.startswith(self._mona_prefixes) for key in missing),
            self._fresh)

    def _restore_counters(self, model):
        for name, buffer in model.named_buffers():
            if name.rsplit('.', 1)[-1] in self._counter_names:
                if not self._fresh and name not in self._saved_counters:
                    raise RuntimeError('Joint checkpoint lacks prompt counter ' + name)
                buffer.fill_(int(self._saved_counters.get(name, 0)))

    def before_train(self, runner):
        from ...models.layers.mona import Mona2d

        if not self._loaded:
            raise RuntimeError('Joint Mona training requires pretrained '
                               'YOLO-World weights before training.')
        model = self._model(runner)
        image = model.backbone.image_model
        frozen = [name for name, p in image.named_parameters() if not p.requires_grad]
        if image.freeze_original or image.frozen_stages != -1 or frozen:
            raise RuntimeError('Joint mode must train the original image '
                               f'backbone and Mona together. Frozen={frozen[:12]}')
        text = model.backbone.text_model
        if any(p.requires_grad for p in text.model.parameters()):
            raise RuntimeError('CLIP text parameters must stay frozen in joint mode.')
        if not text.training or not text.prompt_bank.training or any(
                not p.requires_grad for p in text.prompt_bank.parameters()):
            raise RuntimeError('The new Prompt Bank must be trainable in joint mode.')
        if self._fresh:
            for name in self._new_buffers:
                buffer = getattr(model, name, None)
                if buffer is not None:
                    buffer.zero_()
            for module in image.modules():
                if isinstance(module, Mona2d) and (
                        module.project2.weight.detach().count_nonzero() or
                        module.project2.bias.detach().count_nonzero()):
                    raise RuntimeError('Fresh joint Mona must start with '
                                       'a zero output projection.')
        self._restore_counters(model)
        for hook in runner.hooks:
            if hasattr(hook, 'ema_model'):
                self._restore_counters(hook.ema_model.module)
        groups = Counter()
        for group in runner.optim_wrapper.optimizer.param_groups:
            count = sum(p.numel() for p in group['params'] if p.requires_grad)
            if count:
                groups[group.get('initial_lr', group['lr'])] += count
        runner.logger.info(
            'DIOR 3.7 joint: image backbone and Mona trainable; '
            'CLIP frozen; Prompt Bank trainable; trainable parameter counts '
            'by peak LR=%s. Prompt counters start at zero for fresh training; '
            'BatchNorm follows the configured training mode.', dict(groups))

    def before_save_checkpoint(self, runner, checkpoint):
        model = self._model(runner)
        checkpoint.setdefault('meta', {})[self._meta_key] = dict(
            source_checkpoint=self.source_checkpoint,
            model_counters={name: int(buffer.item()) for name, buffer in model.named_buffers()
                            if name.rsplit('.', 1)[-1] in self._counter_names})

"""Keep auxiliary gradient clipping from rescaling the main detector."""

import math
import torch
from mmengine.optim import AmpOptimWrapper, OptimWrapper
from mmyolo.registry import OPTIM_WRAPPERS, OPTIM_WRAPPER_CONSTRUCTORS

from .yolow_v5_optim_constructor import YOLOWv5OptimizerConstructor


@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class OVCOptimizerConstructor(YOLOWv5OptimizerConstructor):
    def __call__(self, model):
        wrapper = super().__call__(model)
        if hasattr(model, 'module'):
            model = model.module
        adapter = getattr(model.neck, 'visual_adapter', None)
        auxiliary = {id(p) for p in adapter.parameters()} if adapter is not None else set()
        for group in wrapper.optimizer.param_groups:
            flags = {id(p) in auxiliary for p in group['params']}
            if len(flags) > 1:
                raise ValueError('OVC needs paramwise groups to isolate Adapter clipping.')
            group['ovc_auxiliary'] = True in flags
        if hasattr(model.neck, 'dual_stream'):
            wrapper._ovc_fusion_params = tuple(model.neck.dual_stream.out_proj.parameters())
        return wrapper


@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class OVCFineTuneOptimizerConstructor(OVCOptimizerConstructor):
    """Apply LR multipliers without losing the configured norm/bias decay.

    The original constructor lets any custom LR key bypass norm/bias rules.
    This variant keeps explicit custom decay overrides, and applies the usual
    norm/bias defaults when a custom key only changes the learning rate.
    Existing experiments continue to use OVCOptimizerConstructor unchanged.
    """

    def __call__(self, model):
        wrapper = super().__call__(model)
        if hasattr(model, 'module'):
            model = model.module
        groups = {id(p): group for group in wrapper.optimizer.param_groups
                  for p in group['params']}
        custom = self.paramwise_cfg.get('custom_keys', {})
        keys = sorted(sorted(custom), key=len, reverse=True)
        weight_decay = wrapper.optimizer.defaults.get('weight_decay', 0.0)
        norm_types = (torch.nn.modules.batchnorm._BatchNorm,
                      torch.nn.modules.instancenorm._InstanceNorm,
                      torch.nn.GroupNorm, torch.nn.LayerNorm)
        for prefix, module in model.named_modules():
            for name, param in module.named_parameters(recurse=False):
                if not param.requires_grad:
                    continue
                full_name = f'{prefix}.{name}' if prefix else name
                matched = next((custom[key] for key in keys if key in full_name), {})
                if 'decay_mult' in matched:
                    continue
                if isinstance(module, norm_types):
                    multiplier = self.paramwise_cfg.get('norm_decay_mult')
                elif name == 'bias':
                    multiplier = self.paramwise_cfg.get('bias_decay_mult')
                else:
                    multiplier = None
                if multiplier is not None:
                    groups[id(param)]['weight_decay'] = weight_decay * multiplier
        return wrapper


class _GroupedClipping:
    def _clip_grad(self):
        for auxiliary, name in ((False, 'main'), (True, 'aux')):
            params = [p for group in self.optimizer.param_groups
                      if bool(group.get('ovc_auxiliary', False)) == auxiliary
                      for p in group['params'] if p.requires_grad and p.grad is not None]
            if params:
                norm = self.clip_func(params, **self.clip_grad_kwargs)
                if norm is not None:
                    self.message_hub.update_scalar('train/ovc_' + name + '_grad_norm',
                                                   float(norm))


@OPTIM_WRAPPERS.register_module()
class OVCGroupedOptimWrapper(_GroupedClipping, OptimWrapper):
    """FP32 counterpart, useful for CPU validation of the same gradient paths."""


@OPTIM_WRAPPERS.register_module()
class OVCGroupedAmpOptimWrapper(_GroupedClipping, AmpOptimWrapper):

    def step(self, **kwargs):
        scale = self.loss_scaler.get_scale()
        super().step(**kwargs)
        skipped = int(self.loss_scaler.get_scale() < scale)
        self.message_hub.update_scalar('train/ovc_amp_skipped', skipped)


class _OVCTracking:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ovc_monitor = dict(amp_skipped_total=0, nonfinite_gradient_steps=0)

    def _clip_grad(self):
        super()._clip_grad()
        finite = True
        for name in ('main', 'aux'):
            key = 'train/ovc_' + name + '_grad_norm'
            if key in self.message_hub.log_scalars:
                finite &= math.isfinite(float(self.message_hub.get_scalar(key).current()))
        self.ovc_monitor['nonfinite_gradient_steps'] += int(not finite)
        grads = [p.grad.detach().float().norm() for p in getattr(self, '_ovc_fusion_params', ())
                 if p.grad is not None]
        if grads:
            self.message_hub.update_scalar('train/ovc_fusion_grad_norm_postclip',
                                            float(torch.stack(grads).norm()))

    def step(self, **kwargs):
        old_scale = self.loss_scaler.get_scale() if hasattr(self, 'loss_scaler') else None
        super().step(**kwargs)
        if old_scale is not None:
            scale = self.loss_scaler.get_scale()
            self.ovc_monitor['amp_skipped_total'] += int(scale < old_scale)
            self.message_hub.update_scalar('train/ovc_amp_scale', scale)
        for k, v in self.ovc_monitor.items():
            self.message_hub.update_scalar('train/ovc_' + k, v)

    def state_dict(self):
        state = super().state_dict()
        state['ovc_monitor'] = self.ovc_monitor.copy()
        return state

    def load_state_dict(self, state_dict):
        state = state_dict.copy()
        self.ovc_monitor.update(state.pop('ovc_monitor', {}))
        super().load_state_dict(state)


@OPTIM_WRAPPERS.register_module()
class OVCTrackedAmpOptimWrapper(_OVCTracking, OVCGroupedAmpOptimWrapper):
    pass


@OPTIM_WRAPPERS.register_module()
class OVCTrackedOptimWrapper(_OVCTracking, OVCGroupedOptimWrapper):
    pass

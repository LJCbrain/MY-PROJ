"""Isolate localization + Adapter clipping from the original detector."""

from mmyolo.registry import OPTIM_WRAPPER_CONSTRUCTORS

from .ovc_optim_wrapper import OVCOptimizerConstructor


@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class QualityOVCOptimizerConstructor(OVCOptimizerConstructor):
    def __call__(self, model):
        wrapper = super().__call__(model)
        while hasattr(model, 'module'):
            model = model.module
        auxiliary = {id(p) for module in (model.neck.visual_adapter, model.neck.localization_head)
                     for p in module.parameters()}
        for group in wrapper.optimizer.param_groups:
            flags = {id(p) in auxiliary for p in group['params']}
            if len(flags) > 1:
                raise ValueError('Quality auxiliary parameters require separate clipping groups.')
            group['ovc_auxiliary'] = True in flags
        return wrapper

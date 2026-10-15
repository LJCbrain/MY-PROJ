"""Log V1 routing and residual strength only at the requested interval."""

from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmyolo.registry import HOOKS


@HOOKS.register_module()
class SemanticWSRDiagnosticsHook(Hook):
    priority = 'NORMAL'

    def __init__(self, interval=50):
        if interval <= 0:
            raise ValueError('Diagnostic interval must be positive.')
        self.interval = interval

    @staticmethod
    def _neck(runner):
        model = runner.model
        while is_model_wrapper(model):
            model = model.module
        return model.neck

    def before_train_iter(self, runner, batch_idx, data_batch=None):
        neck = self._neck(runner)
        # LoggerHook uses the within-epoch index, which resets each epoch.
        neck.collect_diagnostics = self.every_n_inner_iters(batch_idx, self.interval)

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        neck = self._neck(runner)
        if neck.collect_diagnostics:
            for name, value in neck.last_diagnostics.items():
                runner.message_hub.update_scalar('train/v1_' + name, value.item())
        neck.collect_diagnostics = False
        neck.dual_stream.collect_diagnostics = False

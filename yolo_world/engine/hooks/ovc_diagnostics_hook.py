"""Log OVC features without changing training or reading an EMA model."""

from mmyolo.registry import HOOKS
from .semantic_wsr_diagnostics_hook import SemanticWSRDiagnosticsHook


@HOOKS.register_module()
class OVCDiagnosticsHook(SemanticWSRDiagnosticsHook):
    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        model = runner.model
        while hasattr(model, 'module'):
            model = model.module
        neck = model.neck
        if neck.collect_diagnostics:
            values = {**neck.last_diagnostics,
                      **getattr(model, 'last_aux_diagnostics', {})}
            for name, value in values.items():
                runner.message_hub.update_scalar('train/ovc_' + name, value.item())
        neck.collect_diagnostics = False
        neck.dual_stream.collect_diagnostics = False

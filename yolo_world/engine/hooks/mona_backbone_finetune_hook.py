"""Validate best-weight loading for the block-wise backbone Mona variant."""

from mmyolo.registry import HOOKS

from .mona_finetune_hook import MonaFineTuneHook


@HOOKS.register_module()
class MonaBackboneFineTuneHook(MonaFineTuneHook):
    _run_name = 'DIOR 3.7'
    # Only the new adapter keys can be missing from the original 2.17 best.
    _new_prefix = tuple(
        f'backbone.image_model.stage{i}.1.mona_adapters.'
        for i in range(1, 5))

    @staticmethod
    def _has_mona(model):
        return model.backbone.image_model.mona_enabled

    def before_train(self, runner):
        super().before_train(runner)
        model = self._model(runner)
        image = model.backbone.image_model
        unexpected = [name for name, param in image.named_parameters()
                      if param.requires_grad and '.mona_adapters.' not in name]
        if unexpected:
            raise RuntimeError('Original image backbone parameters must stay '
                               f'frozen: {unexpected[:12]}')
        trainable = sum(p.numel() for p in image.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in image.parameters() if not p.requires_grad)
        text = model.backbone.text_model
        text_trainable = sum(p.numel() for p in text.parameters()
                             if p.requires_grad) if text is not None else 0
        runner.logger.info(
            'DIOR 3.7: image backbone original frozen parameters=%d; '
            'trainable Mona parameters=%d; text/Prompt Bank trainable '
            'parameters=%d.', frozen, trainable, text_trainable)

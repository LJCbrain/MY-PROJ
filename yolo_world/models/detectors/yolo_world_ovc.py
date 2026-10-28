"""Single detector, shared two-view Adapter, no teacher or semantic holdout."""

import time

import torch
from mmyolo.registry import MODELS

from .yolo_world import YOLOWorldDetector
from ..layers.ovc_training import (augment_view, auxiliary_backbone_context,
                                   balanced_objectness_loss, box_trimaps,
                                   collect_rois, region_consistency)


@MODELS.register_module()
class YOLOWorldOVCDetector(YOLOWorldDetector):
    def __init__(self, *args, ovc_training_cfg=None, **kwargs):
        super().__init__(*args, **kwargs)
        defaults = dict(consistency_enabled=True, objectness_weight=0.1,
                        consistency_weight=0.05, variance_weight=0.01,
                        warmup_iters=1000, max_rois_per_image=32,
                        color_amount=0.2, hue=0.02, grayscale_prob=0.1,
                        normalized_regularization=False, covariance_weight=0.001)
        supplied = ovc_training_cfg or {}
        if set(supplied) - set(defaults):
            raise ValueError('Unknown OVC training settings.')
        defaults.update(supplied)
        self.ovc_training_cfg = defaults
        if any(defaults[k] < 0 for k in ('objectness_weight', 'consistency_weight',
                                         'variance_weight', 'covariance_weight', 'warmup_iters')):
            raise ValueError('Loss weights and warmup must be non-negative.')
        if defaults['max_rois_per_image'] <= 0:
            raise ValueError('At least one ROI per image is required.')
        self.register_buffer('_ovc_iteration', torch.zeros((), dtype=torch.long))
        if defaults['normalized_regularization']:
            self.register_buffer('_ovc_variance_batches', torch.zeros((), dtype=torch.long))
            self.register_buffer('_ovc_covariance_batches', torch.zeros((), dtype=torch.long))
        self.last_aux_diagnostics = {}

    def auxiliary_visual_forward(self, inputs):
        """One extra image-only pass. Never call neck/text/head or touch caches."""
        with auxiliary_backbone_context(self.backbone.image_model):
            c3 = self.backbone.forward_image(inputs)[0]
        return self.neck.visual_adapter(c3)

    def loss(self, batch_inputs, batch_data_samples):
        self.last_aux_diagnostics = {}
        if not getattr(self.neck, 'adapter_enabled', False):
            return super().loss(batch_inputs, batch_data_samples)
        try:
            residual = self.neck.dual_stream
            if getattr(residual, 'scale_mode', 'learnable') == 'fixed':
                residual.scale_iteration.copy_(self._ovc_iteration)
            losses = super().loss(batch_inputs, batch_data_samples)
            va, oa = self.neck.main_visual, self.neck.main_objectness_logits
            cfg = self.ovc_training_cfg
            augmented = augment_view(batch_inputs, self.data_preprocessor,
                                     cfg['color_amount'], cfg['hue'],
                                     cfg['grayscale_prob'])
            vb, ob = self.auxiliary_visual_forward(augmented)
            rois = collect_rois(batch_data_samples, batch_inputs.shape[-2:],
                                len(batch_inputs), batch_inputs.device)
            targets = box_trimaps(rois, len(batch_inputs), va.shape[-2:],
                                  batch_inputs.shape[-2:])
            partial = batch_data_samples.get('partial_labels') if isinstance(batch_data_samples, dict) else None
            if partial is not None:
                partial = partial.to(device=va.device, dtype=torch.bool)
                targets = (targets[0], targets[1], targets[2] & ~partial[:, None, None, None])
            # BCE and ROI losses must be FP32 even inside AmpOptimWrapper.
            with torch.autocast(device_type=batch_inputs.device.type, enabled=False):
                obj = (balanced_objectness_loss(oa, *targets)
                       + balanced_objectness_loss(ob, *targets)) / 2
                vc = var = covariance = obj * 0
                stats = {}
                if cfg['consistency_enabled']:
                    vc, var, stats = region_consistency(
                        va, vb, rois, batch_inputs.shape[-2:], cfg['max_rois_per_image'],
                        normalized_regularization=cfg.get('normalized_regularization', False),
                        collect_spectrum=self.neck.collect_diagnostics)
                    covariance = stats.pop('covariance_loss', covariance)
            warmup = (min((int(self._ovc_iteration) + 1) / cfg['warmup_iters'], 1.0)
                      if cfg['warmup_iters'] else 1.0)
            losses['loss_ovc_objectness'] = obj * cfg['objectness_weight'] * warmup
            if cfg['consistency_enabled']:
                losses['loss_ovc_consistency'] = vc * cfg['consistency_weight'] * warmup
                losses['loss_ovc_variance'] = var * cfg['variance_weight'] * warmup
                if cfg.get('normalized_regularization', False):
                    losses['loss_ovc_covariance'] = covariance * cfg['covariance_weight'] * warmup
            with torch.no_grad():
                if hasattr(self, '_ovc_variance_batches'):
                    self._ovc_variance_batches.add_(stats.get('variance_active', obj.new_zeros(())).long())
                    self._ovc_covariance_batches.add_(stats.get('covariance_active', obj.new_zeros(())).long())
                    stats.update(variance_batches=self._ovc_variance_batches.detach().clone(),
                                 covariance_batches=self._ovc_covariance_batches.detach().clone())
                self.last_aux_diagnostics = {
                    **stats, **getattr(self.bbox_head, 'last_partial_diagnostics', {}),
                    'objectness_raw': obj.detach(), 'covariance_raw': covariance.detach(),
                    'consistency_raw': vc.detach(), 'variance_raw': var.detach(),
                    'positive_cells': targets[0].sum(),
                    'negative_cells': targets[2].sum(),
                    'warmup': obj.detach().new_tensor(warmup)}
            self._ovc_iteration.add_(1)
            if getattr(residual, 'scale_mode', 'learnable') == 'fixed':
                residual.scale_iteration.copy_(self._ovc_iteration)
            return losses
        finally:
            # Do not retain a completed training graph on the neck/EMA copy.
            self.neck.clear_auxiliary_cache()


@MODELS.register_module()
class YOLOWorldOVCPartialDetector(YOLOWorldOVCDetector):
    """Attach O-ranked, class-agnostic proposals without a second image pass."""
    def __init__(self, *args, clip_roi_cfg=None, **kwargs):
        super().__init__(*args, **kwargs)
        from ..layers.clip_roi import clip_roi_config
        self.clip_roi_cfg = clip_roi_config(clip_roi_cfg)
        self._clip_roi_runtime = None
        self.last_clip_roi_timings = []
        self.last_clip_roi_debug = None

    def _apply(self, fn, *args, **kwargs):
        # Runtime-only CLIP weights/caches must not stay on an obsolete device.
        self._clip_roi_runtime = None
        return super()._apply(fn, *args, **kwargs)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        self._clip_roi_runtime = None
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def predict(self, batch_inputs, batch_data_samples, rescale=True):
        enabled = self.clip_roi_cfg['enabled']
        if enabled:
            if not rescale:
                raise ValueError('CLIP RoI requires rescale=True (original-image boxes).')
            from ..layers.clip_roi import FrozenCLIPRegionScorer, sync
            sync(batch_inputs.device)
            start = time.perf_counter()
        capture = {}
        def capture_dense(module, args, output):
            capture['boxes'] = output[1]
            if enabled:
                capture['classes'] = output[0]
        handle = self.bbox_head.register_forward_hook(
            capture_dense)
        try:
            results = super().predict(batch_inputs, batch_data_samples, rescale=rescale)
        finally:
            handle.remove()
        objectness = self.neck.main_objectness_logits.detach().float().sigmoid()
        boxes = capture['boxes']
        proposals = self.bbox_head.predict_proposals_by_feat(
            boxes, [objectness] * len(boxes),
            [sample.metainfo for sample in results], rescale=rescale)
        for sample, proposal in zip(results, proposals):
            sample.proposal_instances = proposal
        if enabled:
            sync(batch_inputs.device)
            detector_s = time.perf_counter() - start
            if self._clip_roi_runtime is None:
                self._clip_roi_runtime = FrozenCLIPRegionScorer(self.clip_roi_cfg)
            runtime = self._clip_roi_runtime
            self.last_clip_roi_timings = []
            self.last_clip_roi_debug = [] if self.clip_roi_cfg['debug'] else None
            for index, sample in enumerate(results):
                prediction, timing = runtime.predict(
                    self.bbox_head,
                    [x[index:index+1] for x in capture['classes']],
                    [x[index:index+1] for x in boxes],
                    objectness[index:index+1], sample.metainfo, batch_inputs.device)
                sample.pred_instances = prediction
                self.last_clip_roi_timings.append(timing)
                if self.last_clip_roi_debug is not None:
                    self.last_clip_roi_debug.append(runtime.last_debug)
            sync(batch_inputs.device)
            predict_s = time.perf_counter() - start
            for sample, timing in zip(results, self.last_clip_roi_timings):
                timing.update(detector_batch_s=detector_s, predict_batch_s=predict_s)
                # DumpResults preserves these measurements; batch times include
                # all images and exclude the runner's data loading/preprocessor.
                sample.set_metainfo(dict(clip_roi_timing=timing))
        return results

"""Original YOLO head + quality supervision, optionally retaining all OVC."""

import math

import torch
from mmyolo.registry import MODELS

from .yolo_world import YOLOWorldDetector
from .yolo_world_ovc import YOLOWorldOVCDetector
from ..layers.ovc_training import (augment_view, balanced_objectness_loss,
                                   box_trimaps, collect_rois, region_consistency)


@MODELS.register_module()
class YOLOWorldQualityOVCDetector(YOLOWorldOVCDetector):
    def __init__(self, *args, quality_training_cfg=None, quality_score_enabled=True,
                 quality_score_power=1.0, ovc_training_cfg=None, **kwargs):
        ovc_cfg = dict(ovc_training_cfg or {})
        super().__init__(*args, ovc_training_cfg=ovc_cfg, **kwargs)
        if not self.neck.retain_objectness:
            if ovc_cfg.get('objectness_weight', 0) != 0:
                raise ValueError('DIOR 3.3 has no binary O loss; set objectness_weight=0.')
            self.ovc_training_cfg['objectness_weight'] = 0.0
        defaults = dict(bbox_weight=1.0, dfl_weight=.25,
                        q0_weight=.1, q1_weight=.1, warmup_iters=1000)
        if set(quality_training_cfg or {}) - set(defaults):
            raise ValueError('Unknown localization quality training settings.')
        defaults.update(quality_training_cfg or {})
        if any(not math.isfinite(v) or v < 0 for v in defaults.values()):
            raise ValueError('Quality loss weights and warmup must be finite and non-negative.')
        if not math.isfinite(quality_score_power) or quality_score_power <= 0:
            raise ValueError('quality_score_power must be positive and finite.')
        if (tuple(self.neck.localization_head.featmap_strides)
                != tuple(self.bbox_head.featmap_strides)
                or self.bbox_head.prior_generator.offset != .5):
            raise ValueError('Auxiliary and main head reference points must match.')
        self.quality_training_cfg = defaults
        self.quality_score_enabled = bool(quality_score_enabled)
        self.quality_score_power = float(quality_score_power)

    def _sync_quality_iteration(self):
        self.neck.quality_iteration.copy_(self._ovc_iteration)
        residual = self.neck.dual_stream
        if getattr(residual, 'scale_mode', 'learnable') == 'fixed':
            residual.scale_iteration.copy_(self._ovc_iteration)

    @staticmethod
    def _partial_flags(samples, batch_size, device):
        if isinstance(samples, dict):
            supplied = samples.get('partial_labels')
            if supplied is not None:
                return torch.as_tensor(supplied, device=device, dtype=torch.bool)
            metas = samples.get('img_metas', [{} for _ in range(batch_size)])
        else:
            metas = [sample.metainfo for sample in samples]
        return torch.tensor([m.get('partial_labels', False) for m in metas],
                            device=device, dtype=torch.bool)

    @staticmethod
    def _warmup(iteration, duration):
        return min((int(iteration) + 1) / duration, 1.) if duration else 1.

    def loss(self, batch_inputs, batch_data_samples):
        self.last_aux_diagnostics = {}
        self._sync_quality_iteration()
        capture = {}
        # Preserve every original head/text loss and the main TAL assignment.
        handle = self.bbox_head.register_forward_hook(
            lambda module, args, output: capture.update(boxes=tuple(b.detach() for b in output[1])))
        try:
            losses = YOLOWorldDetector.loss(self, batch_inputs, batch_data_samples)
        except BaseException:
            self.neck.clear_auxiliary_cache()
            raise
        finally:
            handle.remove()
        try:
            refined = capture['boxes']
            q1 = self.neck.quality_for_final_boxes(refined)
            rois = collect_rois(batch_data_samples, batch_inputs.shape[-2:],
                                len(batch_inputs), batch_inputs.device)
            partial = self._partial_flags(batch_data_samples, len(batch_inputs), batch_inputs.device)
            aux, quality_stats = self.neck.localization_head.loss(
                self.neck.coarse_boxes, self.neck.coarse_distributions,
                self.neck.coarse_quality_logits, refined, q1, rois, partial,
                collect_diagnostics=self.neck.collect_diagnostics)
            cfg = self.quality_training_cfg
            ramp = self._warmup(self._ovc_iteration, cfg['warmup_iters'])
            for name in ('bbox', 'dfl', 'q0', 'q1'):
                losses['loss_localization_' + name] = aux[name] * cfg[name + '_weight'] * ramp

            # Retain the 2.17 two-view losses. Additive mode also retains the
            # same O predictor, trimaps, weights, and one auxiliary image pass.
            ovc = self.ovc_training_cfg
            va = self.neck.main_visual
            zero = va.float().sum() * 0
            obj = vc = var = covariance = zero
            stats = {}
            objectness_stats = {}
            if self.neck.retain_objectness or ovc['consistency_enabled']:
                augmented = augment_view(batch_inputs, self.data_preprocessor,
                                         ovc['color_amount'], ovc['hue'], ovc['grayscale_prob'])
                vb, ob = self.auxiliary_visual_forward(augmented)
                with torch.autocast(device_type=batch_inputs.device.type, enabled=False):
                    if self.neck.retain_objectness:
                        targets = box_trimaps(rois, len(batch_inputs), va.shape[-2:],
                                              batch_inputs.shape[-2:])
                        targets = (targets[0], targets[1],
                                   targets[2] & ~partial[:, None, None, None])
                        obj = (balanced_objectness_loss(self.neck.main_objectness_logits, *targets)
                               + balanced_objectness_loss(ob, *targets)) / 2
                        objectness_stats = dict(
                            objectness_raw=obj.detach(), positive_cells=targets[0].sum(),
                            negative_cells=targets[2].sum())
                    if ovc['consistency_enabled']:
                        vc, var, stats = region_consistency(
                            va, vb, rois, batch_inputs.shape[-2:], ovc['max_rois_per_image'],
                            normalized_regularization=ovc['normalized_regularization'],
                            collect_spectrum=self.neck.collect_diagnostics)
                        covariance = stats.pop('covariance_loss', zero)
            ovc_ramp = self._warmup(self._ovc_iteration, ovc['warmup_iters'])
            if self.neck.retain_objectness:
                losses['loss_ovc_objectness'] = obj * ovc['objectness_weight'] * ovc_ramp
            if ovc['consistency_enabled']:
                losses['loss_ovc_consistency'] = vc * ovc['consistency_weight'] * ovc_ramp
                losses['loss_ovc_variance'] = var * ovc['variance_weight'] * ovc_ramp
                if ovc['normalized_regularization']:
                    losses['loss_ovc_covariance'] = covariance * ovc['covariance_weight'] * ovc_ramp
            if hasattr(self, '_ovc_variance_batches'):
                self._ovc_variance_batches.add_(stats.get('variance_active', zero.detach()).long())
                self._ovc_covariance_batches.add_(stats.get('covariance_active', zero.detach()).long())
                stats.update(variance_batches=self._ovc_variance_batches.detach().clone(),
                             covariance_batches=self._ovc_covariance_batches.detach().clone())
            self.last_aux_diagnostics = {
                **stats, **quality_stats, **objectness_stats,
                **getattr(self.bbox_head, 'last_partial_diagnostics', {}),
                'consistency_raw': vc.detach(), 'variance_raw': var.detach(),
                'covariance_raw': covariance.detach(),
                'quality_warmup': zero.detach().new_tensor(ramp),
                'warmup': zero.detach().new_tensor(ovc_ramp),
                **{'localization_' + name + '_raw': value.detach() for name, value in aux.items()}}
            self._ovc_iteration.add_(1)
            self._sync_quality_iteration()
            return losses
        finally:
            self.neck.clear_auxiliary_cache()

    def predict(self, batch_inputs, batch_data_samples, rescale=True):
        try:
            features, text, mask = self.extract_feat(batch_inputs, batch_data_samples)
            classes = text.shape[1] if text.dim() == 3 else text.shape[0]
            self.bbox_head.num_classes = classes
            scores, boxes = self.bbox_head(features, text, mask)
            quality_logits = None
            if self.quality_score_enabled:
                quality_logits = self.neck.quality_for_final_boxes(boxes)
                if self.quality_score_power != 1:
                    quality_logits = tuple(torch.logit(
                        q.float().sigmoid().pow(self.quality_score_power).clamp(1e-7, 1 - 1e-7))
                        for q in quality_logits)
            metas = [sample.metainfo for sample in batch_data_samples]
            # The original head sigmoids this optional scalar factor and
            # multiplies p*q BEFORE score_thr, top-k, NMS and max_per_img.
            predictions = self.bbox_head.predict_by_feat(
                scores, boxes, objectnesses=quality_logits,
                batch_img_metas=metas, rescale=rescale)
            self._check_pred_label_range(predictions, classes)
            proposals = self.bbox_head.predict_proposals_by_feat(
                self.neck.coarse_boxes, self.neck.get_proposal_prior_maps(),
                metas, rescale=rescale)
            if self.neck.retain_objectness:
                # Preserve 2.17's proposal metric: actual YOLO boxes ranked
                # by O. Keep auxiliary B0/q0 proposals separate for diagnosis.
                objectness = self.neck.main_objectness_logits.detach().float().sigmoid()
                original_proposals = self.bbox_head.predict_proposals_by_feat(
                    boxes, [objectness] * len(boxes), metas, rescale=rescale)
            results = self.add_pred_to_datasample(batch_data_samples, predictions)
            for index, (sample, proposal) in enumerate(zip(results, proposals)):
                if self.neck.retain_objectness:
                    sample.proposal_instances = original_proposals[index]
                    sample.quality_proposal_instances = proposal
                else:
                    sample.proposal_instances = proposal
            return results
        finally:
            self.neck.clear_auxiliary_cache()

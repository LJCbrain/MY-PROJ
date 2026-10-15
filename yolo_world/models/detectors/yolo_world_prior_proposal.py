"""Class-agnostic proposal attachment for DIOR validation diagnostics."""

from torch import Tensor
from mmdet.structures import SampleList
from mmyolo.registry import MODELS

from .yolo_world import YOLOWorldDetector


@MODELS.register_module()
class YOLOWorldPriorProposalDetector(YOLOWorldDetector):
    """YOLO-World detector that exposes visual-prior proposals at evaluation.

    The detector loss is inherited unchanged.  The proposals are only attached
    to validation/test samples so ``PriorProposalRecallMetric`` can measure
    whether the class-agnostic visual prior covers base and novel objects.
    """

    def _get_prior_maps(self):
        if not self.with_neck or not hasattr(
                self.neck, 'get_proposal_prior_maps'):
            return None
        return self.neck.get_proposal_prior_maps()

    def _get_spatial_prompt_context(self):
        """Collect the current base text and detached spatial routing maps."""
        enabled = bool(
            getattr(
                getattr(self.bbox_head, 'head_module', None),
                'spatial_prompt_routing_enabled',
                False))
        if not enabled:
            return None
        if not hasattr(self.backbone, 'get_last_base_text_features'):
            raise RuntimeError(
                'CSPR requires access to the latest unprompted text features.')
        base_txt_feats = self.backbone.get_last_base_text_features()
        if base_txt_feats is None:
            raise RuntimeError(
                'CSPR base text is unavailable. The prompted text forward '
                'must run before the detection head.')
        if not self.with_neck or not hasattr(
                self.neck, 'get_spatial_prompt_routing_context'):
            raise RuntimeError(
                'CSPR requires a neck that exposes spatial routing maps.')
        context = self.neck.get_spatial_prompt_routing_context()
        context['base_txt_feats'] = base_txt_feats.detach()
        return context

    def loss(self, batch_inputs: Tensor, batch_data_samples: SampleList):
        """Calculate detection, foreground, and prompt losses with CSPR."""
        self.bbox_head.num_classes = self.num_train_classes
        img_feats, txt_feats, txt_masks = self.extract_feat(
            batch_inputs, batch_data_samples)
        losses = self.bbox_head.loss(
            img_feats,
            txt_feats,
            txt_masks,
            batch_data_samples,
            spatial_prompt_context=self._get_spatial_prompt_context())
        losses.update(self._loss_fgbg(batch_inputs, batch_data_samples))
        losses.update(self._loss_text_prompt_bank())
        return losses

    def _forward(self, batch_inputs: Tensor,
                 batch_data_samples: SampleList = None):
        """Run a tensor-mode forward with the current CSPR context."""
        img_feats, txt_feats, txt_masks = self.extract_feat(
            batch_inputs, batch_data_samples)
        return self.bbox_head.forward(
            img_feats,
            txt_feats,
            txt_masks,
            spatial_prompt_context=self._get_spatial_prompt_context())

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        img_feats, txt_feats, txt_masks = self.extract_feat(
            batch_inputs, batch_data_samples)
        current_num_classes = (
            txt_feats.shape[1] if txt_feats.dim() == 3 else txt_feats.shape[0])
        self.bbox_head.num_classes = current_num_classes
        outs = self.bbox_head(
            img_feats,
            txt_feats,
            txt_masks,
            spatial_prompt_context=self._get_spatial_prompt_context())
        cls_scores, bbox_preds = outs
        batch_img_metas = [sample.metainfo for sample in batch_data_samples]
        results_list = self.bbox_head.predict_by_feat(
            cls_scores,
            bbox_preds,
            batch_img_metas=batch_img_metas,
            rescale=rescale)
        self._check_pred_label_range(results_list, current_num_classes)
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)

        proposal_cfg = getattr(self.bbox_head, 'proposal_cfg', None)
        if proposal_cfg:
            proposal_list = self.bbox_head.predict_proposals_by_feat(
                bbox_preds, self._get_prior_maps(), batch_img_metas,
                rescale=rescale)
            for data_sample, proposals in zip(batch_data_samples,
                                              proposal_list):
                data_sample.proposal_instances = proposals
        return batch_data_samples

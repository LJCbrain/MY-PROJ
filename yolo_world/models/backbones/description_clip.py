"""Opt-in DIORv3.2 text backbone; the DIORv2.17 implementation is untouched."""

import math
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from mmyolo.registry import MODELS

from .mm_backbone import (HuggingCLIPLanguageBackbone,
                          text_topology_kl_loss)
from ..layers.description_regularization import (
    description_triplet_loss, load_description_matrix,
    read_description_manifest, similarity_from_embeddings)


@MODELS.register_module()
class DescriptionRegularizedCLIPLanguageBackbone(HuggingCLIPLanguageBackbone):
    """Use fixed aerial descriptions to rank trainable CLIP class prototypes.

    Inference never reads descriptions/matrices or evaluates the extra loss.
    The optional full semantic mapping is shared by classification and relation
    learning, and stays active at inference. Raw CLIP neck guidance is untouched.
    Disabling BOTH mapping and description loss gives the original 2.17 path.
    Existing CLIP/prompt names remain compatible with baseline checkpoints.
    """

    def __init__(self, *args, description_regularization_cfg=None,
                 semantic_adapter_cfg=None, **kwargs):
        super().__init__(*args, **kwargs)
        adapter_cfg = dict(semantic_adapter_cfg or {})
        enabled = adapter_cfg.pop('enabled', False)
        bias = adapter_cfg.pop('bias', True)
        if adapter_cfg:
            raise ValueError('Unknown semantic adapter settings: '
                             + ', '.join(adapter_cfg))
        if not isinstance(enabled, bool) or not isinstance(bias, bool):
            raise ValueError('Semantic adapter enabled/bias must be boolean.')
        if enabled and not self.has_prompt_bank:
            raise ValueError('Semantic adapter requires the prompted detection path.')
        self.semantic_adapter = None
        if enabled:
            dimension = self.model.text_projection.out_features
            # A complete learned map, NOT x + A(x), and NOT identity initialized.
            # Keep it outside self.model so frozen_modules=['all'] freezes CLIP
            # without freezing this new trainable classifier projection.
            self.semantic_adapter = nn.Linear(dimension, dimension, bias=bias)
            nn.init.xavier_uniform_(self.semantic_adapter.weight)
            if self.semantic_adapter.bias is not None:
                nn.init.zeros_(self.semantic_adapter.bias)
        cfg = dict(description_regularization_cfg or {})
        self.description_path = cfg.pop('description_path', None)
        self.description_matrix_path = cfg.pop('matrix_path', None)
        self.description_weight = float(cfg.pop('loss_weight', 0.0))
        self.description_temperature = float(cfg.pop('temperature', 0.03))
        self.description_topk = cfg.pop('topk', 3)
        self.description_margin_scale = float(cfg.pop('margin_scale', 1.0))
        self.description_warmup_iters = cfg.pop('warmup_iters', 1000)
        if cfg:
            raise ValueError('Unknown description regularization settings: '
                             + ', '.join(cfg))
        if not math.isfinite(self.description_weight) or self.description_weight < 0:
            raise ValueError('Description loss_weight must be finite and non-negative.')
        if (not math.isfinite(self.description_temperature) or
                self.description_temperature <= 0):
            raise ValueError('Description temperature must be finite and positive.')
        if not isinstance(self.description_topk, int) or self.description_topk <= 0:
            raise ValueError('Description topk must be a positive integer.')
        if (not math.isfinite(self.description_margin_scale) or
                self.description_margin_scale < 0):
            raise ValueError('Description margin_scale must be finite and non-negative.')
        if (not isinstance(self.description_warmup_iters, int) or
                self.description_warmup_iters < 0):
            raise ValueError('Description warmup_iters must be a non-negative integer.')
        if self.description_weight > 0:
            if not self.has_prompt_bank or len(self.semantic_anchor_texts) < 3:
                raise ValueError('Description regularization needs a prompt bank '
                                 'and at least three semantic anchor texts.')
            if self.semantic_anchor_sample_size != 0:
                raise ValueError('DIOR 3.2 requires all-class anchors: sample_size=0.')
            if not self.description_path and not self.description_matrix_path:
                raise ValueError('Provide description_path or matrix_path.')
        # Lazily initialized CPU teacher, deliberately outside EMA/state_dict.
        # Only the integer warmup counter is saved and synchronized by the
        # baseline IntegerSafeExpMomentumEMA hook.
        self._description_similarity = None
        self.register_buffer('_description_step', torch.zeros((), dtype=torch.long))

    def _map_class_prototypes(self, prototypes):
        if self.semantic_adapter is None:
            return prototypes
        return F.normalize(self.semantic_adapter(prototypes), dim=-1)

    def forward(self, text, image_features=None, reliability_maps=None,
                base_text_features=None, positive_class_mask=None):
        prototypes, text_mask = super().forward(
            text, image_features=image_features,
            reliability_maps=reliability_maps,
            base_text_features=base_text_features,
            positive_class_mask=positive_class_mask)
        if image_features is not None:
            # Only the classifier path is projected. Calls without image
            # features supply frozen CLIP guidance/routing, not class scores.
            prototypes = self._map_class_prototypes(prototypes)
        return prototypes, text_mask

    def _get_description_similarity(self):
        if self._description_similarity is not None:
            return self._description_similarity
        source_sha256 = None
        descriptions = None
        if self.description_path:
            _, descriptions, source_sha256 = read_description_manifest(
                self.description_path, self.semantic_anchor_texts)
        if self.description_matrix_path:
            if not Path(self.description_matrix_path).is_file():
                raise FileNotFoundError(self.description_matrix_path)
            matrix = load_description_matrix(
                self.description_matrix_path, self.semantic_anchor_texts,
                source_sha256)
        else:
            inputs = self.tokenizer(text=descriptions, return_tensors='pt',
                                    padding=True, truncation=False)
            if inputs['input_ids'].shape[1] > self.model.config.max_position_embeddings:
                raise ValueError('An aerial description exceeds CLIP context length; '
                                 'shorten it rather than silently truncating it.')
            inputs = inputs.to(device=self.model.device)
            was_training = self.model.training
            self.model.eval()
            try:
                with torch.no_grad(), torch.autocast(
                        device_type=self.model.device.type, enabled=False):
                    embeddings = self.model(**inputs).text_embeds
                matrix = similarity_from_embeddings(embeddings)
            finally:
                self.model.train(was_training)
        self._description_similarity = matrix
        return matrix

    def _add_semantic_anchor_loss(self, image_features, reliability_maps):
        if self.description_weight <= 0:
            # Exact original path, including its RNG and warmup behaviour.
            return super()._add_semantic_anchor_loss(image_features,
                                                     reliability_maps)
        if not self.training:
            return
        similarity = self._get_description_similarity()
        original = self._get_semantic_anchor_embeddings()
        batch_size = image_features[0].shape[0]
        count = len(self.semantic_anchor_texts)
        class_embeddings = original.to(image_features[0].device).unsqueeze(0)
        class_embeddings = class_embeddings.expand(batch_size, -1, -1)
        mask = torch.ones(batch_size, count, device=class_embeddings.device,
                          dtype=torch.bool)
        tokens, _ = self.prompt_bank(
            image_features, reliability_maps, class_embeddings=class_embeddings,
            text_mask=mask, positive_class_mask=None, compute_losses=False,
            routing_grad=False, update_stats=False)
        inputs = self.tokenizer(text=list(self.semantic_anchor_texts) * batch_size,
                                return_tensors='pt', padding=True)
        inputs = inputs.to(device=self.model.device)
        prompted = self._forward_prompted_text(inputs, tokens, count)
        prompted = F.normalize(prompted, dim=-1).reshape(batch_size, count, -1)
        original = original.to(device=prompted.device, dtype=prompted.dtype)
        if self.semantic_anchor_weight > 0:
            anchor = 1.0 - (prompted * original.unsqueeze(0)).sum(dim=-1)
            self._prompt_bank_losses['loss_guard_prompt_anchor'] = (
                anchor.mean() * self._semantic_anchor_scale())
        if self.text_topology_weight > 0:
            topology = text_topology_kl_loss(
                prompted, original, valid_mask=mask,
                temperature=self.text_topology_temperature)
            self._prompt_bank_losses['loss_text_topology'] = (
                topology * self.text_topology_weight)
        # Anchor the prompted CLIP output above, but teach the actual classifier
        # prototypes below. Do not anchor the new projection back to raw CLIP.
        student = self._map_class_prototypes(prompted)
        loss = description_triplet_loss(
            student, similarity, temperature=self.description_temperature,
            topk=self.description_topk, margin_scale=self.description_margin_scale)
        warmup = (min((int(self._description_step) + 1) /
                      self.description_warmup_iters, 1.0)
                  if self.description_warmup_iters else 1.0)
        self._prompt_bank_losses['loss_text_description'] = (
            loss * self.description_weight * warmup)
        self._description_step.add_(1)

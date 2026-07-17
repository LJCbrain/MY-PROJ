# Copyright (c) Tencent Inc. All rights reserved.
import itertools
import os
from typing import Dict, List, Optional, Sequence, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.batchnorm import _BatchNorm
from mmengine.model import BaseModule
from mmyolo.registry import MODELS
from mmdet.utils import OptMultiConfig, ConfigType
from transformers import (AutoTokenizer, AutoModel, CLIPTextConfig)
from transformers import CLIPTextModelWithProjection as CLIPTP


class MultiScalePromptBank(nn.Module):
    """Shared image-conditioned prompt bank selected across feature scales."""

    def __init__(self,
                 image_channels: Sequence[int],
                 text_hidden_size: int,
                 num_prompts: int = 10,
                 prompt_length: int = 12,
                 num_scales: int = 3,
                 key_dim: Optional[int] = None,
                 feat_indices: Optional[Sequence[int]] = None,
                 use_reliability: bool = True,
                 loss_match_weight: float = 0.7,
                 loss_orthogonal_weight: float = 0.3,
                 init_std: float = 0.02) -> None:
        super().__init__()
        self.num_prompts = num_prompts
        self.prompt_length = prompt_length
        self.num_scales = num_scales
        self.key_dim = key_dim or text_hidden_size
        self.use_reliability = bool(use_reliability)
        self.loss_match_weight = float(loss_match_weight)
        self.loss_orthogonal_weight = float(loss_orthogonal_weight)

        if feat_indices is None:
            feat_indices = tuple(range(num_scales))
        self.feat_indices = tuple(feat_indices)
        if len(self.feat_indices) != num_scales:
            raise ValueError('feat_indices must match num_scales.')
        if len(image_channels) != num_scales:
            raise ValueError('image_channels must match num_scales.')

        query_channels = [ch * 2 if self.use_reliability else ch
                          for ch in image_channels]
        self.image_projs = nn.ModuleList(
            [nn.Linear(ch, self.key_dim) for ch in query_channels])
        self.keys = nn.Parameter(torch.empty(num_prompts, self.key_dim))
        self.prompts = nn.Parameter(
            torch.empty(num_prompts, prompt_length, text_hidden_size))
        self.reset_parameters(init_std)

    def reset_parameters(self, init_std: float) -> None:
        nn.init.normal_(self.keys, std=init_std)
        nn.init.normal_(self.prompts, std=init_std)
        for proj in self.image_projs:
            nn.init.xavier_uniform_(proj.weight)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)

    def forward(
            self,
            img_feats: Sequence[Tensor],
            reliability_maps: Optional[Sequence[Tensor]] = None
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        selected_prompts = []
        match_losses = []
        selected_indices = []
        selected_scores = []

        for scale_idx, feat_idx in enumerate(self.feat_indices):
            feat = img_feats[feat_idx]
            reliability = None
            if reliability_maps is not None:
                reliability = reliability_maps[feat_idx]
            pooled = self._pool_feature(feat, reliability)

            query = F.normalize(self.image_projs[scale_idx](pooled), dim=-1)
            keys = F.normalize(self.keys, dim=-1)
            similarity = query @ keys.t()
            indices = similarity.argmax(dim=-1)
            prompts = self.prompts[indices]

            selected_prompts.append(prompts)
            selected_indices.append(indices.detach())
            selected_scores.append(
                similarity.gather(1, indices[:, None]).squeeze(1).detach())
            match_losses.append(
                1.0 - similarity.gather(1, indices[:, None]).squeeze(1))

        prompt_tokens = torch.cat(selected_prompts, dim=1)
        losses = {}
        if self.loss_match_weight > 0:
            losses['loss_mspb_match'] = (
                torch.stack(match_losses, dim=0).sum(dim=0).mean() *
                self.loss_match_weight)
        if self.loss_orthogonal_weight > 0:
            losses['loss_mspb_orthogonal'] = (
                self._orthogonal_loss() * self.loss_orthogonal_weight)

        self.last_selected_indices = torch.stack(selected_indices, dim=1)
        self.last_selected_scores = torch.stack(selected_scores, dim=1)
        return prompt_tokens, losses

    def _pool_feature(self, feat: Tensor,
                      reliability: Optional[Tensor]) -> Tensor:
        if feat.dim() == 3:
            pooled = feat.mean(dim=1)
            return torch.cat([pooled, pooled], dim=-1) \
                if self.use_reliability else pooled
        if feat.dim() != 4:
            raise ValueError('image features must be 3D or 4D tensors.')

        global_pooled = feat.mean(dim=(2, 3))
        if not self.use_reliability:
            return global_pooled
        if reliability is None:
            return torch.cat([global_pooled, global_pooled], dim=-1)
        if reliability.dim() != 4:
            raise ValueError('reliability maps must be 4D tensors.')
        if reliability.shape[-2:] != feat.shape[-2:]:
            reliability = F.interpolate(
                reliability,
                size=feat.shape[-2:],
                mode='bilinear',
                align_corners=False)
        reliability = reliability.to(dtype=feat.dtype).clamp_min(0)
        reliable_pooled = (feat * reliability).sum(dim=(2, 3))
        reliable_pooled = reliable_pooled / reliability.sum(
            dim=(2, 3)).clamp_min(torch.finfo(feat.dtype).eps)
        return torch.cat([global_pooled, reliable_pooled], dim=-1)

    def _orthogonal_loss(self) -> Tensor:
        prompts = F.normalize(self.prompts.flatten(1), dim=-1)
        similarity = prompts @ prompts.t()
        off_diagonal = ~torch.eye(
            self.num_prompts, device=similarity.device, dtype=torch.bool)
        return similarity.masked_select(off_diagonal).abs().sum() / (
            self.num_prompts * (self.num_prompts - 1))


@MODELS.register_module()
class HuggingVisionBackbone(BaseModule):  #用 Transformer 做图像骨干
    def __init__(self,
                 model_name: str,
                 out_indices: Sequence[int] = (0, 1, 2, 3),
                 norm_eval: bool = True,
                 frozen_modules: Sequence[str] = (),
                 init_cfg: OptMultiConfig = None) -> None:

        super().__init__(init_cfg=init_cfg)

        self.norm_eval = norm_eval
        self.frozen_modules = frozen_modules
        self.model = AutoModel.from_pretrained(model_name)

        self._freeze_modules()

    def forward(self, image: Tensor) -> Tuple[Tensor]:
        encoded_dict = self.image_model(pixel_values=image,
                                        output_hidden_states=True)
        hidden_states = encoded_dict.hidden_states
        img_feats = encoded_dict.get('reshaped_hidden_states', hidden_states)
        img_feats = [img_feats[i] for i in self.image_out_indices]
        return tuple(img_feats)

    def _freeze_modules(self):
        for name, module in self.model.named_modules():
            for frozen_name in self.frozen_modules:
                if name.startswith(frozen_name):
                    module.eval()
                    for param in module.parameters():
                        param.requires_grad = False
                    break

    def train(self, mode=True):
        super().train(mode)
        self._freeze_modules()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()


@MODELS.register_module()
class HuggingCLIPLanguageBackbone(BaseModule):
    def __init__(self,
                 model_name: str,
                 frozen_modules: Sequence[str] = (),
                 dropout: float = 0.0,
                 add_mask: bool = False,
                 training_use_cache: bool = False,
                 prompt_bank_cfg: Optional[ConfigType] = None,
                 init_cfg: OptMultiConfig = None) -> None:

        super().__init__(init_cfg=init_cfg)

        self.frozen_modules = frozen_modules
        self.training_use_cache = training_use_cache
        self.add_mask = add_mask
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        clip_config = CLIPTextConfig.from_pretrained(model_name,
                                                     attention_dropout=dropout)
        self.model = CLIPTP.from_pretrained(model_name, config=clip_config)
        self.prompt_bank = None
        self._prompt_bank_losses = {}
        self.semantic_anchor_texts = ()
        self.semantic_anchor_embed_path = None
        self.semantic_anchor_weight = 0.0
        self.semantic_anchor_warmup_iters = 0
        # This cache is intentionally not a buffer: it is initialized lazily
        # after EMA constructs its shadow model and must not be EMA-swapped.
        self._semantic_anchor_embeddings = None
        self.register_buffer(
            '_semantic_anchor_step', torch.zeros((), dtype=torch.long),
            persistent=False)
        if prompt_bank_cfg is not None:
            prompt_bank_cfg = dict(prompt_bank_cfg)
            enabled = prompt_bank_cfg.pop('enabled', True)
            if enabled:
                self.semantic_anchor_texts = tuple(
                    prompt_bank_cfg.pop('semantic_anchor_texts', ()))
                self.semantic_anchor_embed_path = prompt_bank_cfg.pop(
                    'semantic_anchor_embed_path', None)
                self.semantic_anchor_weight = float(prompt_bank_cfg.pop(
                    'semantic_anchor_weight', 0.0))
                self.semantic_anchor_warmup_iters = int(
                    prompt_bank_cfg.pop('semantic_anchor_warmup_iters', 0))
                prompt_bank_cfg.setdefault('text_hidden_size',
                                           clip_config.hidden_size)
                self.prompt_bank = MultiScalePromptBank(**prompt_bank_cfg)
        self._freeze_modules()

    def forward_tokenizer(self, texts):
        if not hasattr(self, 'text'):
            text = list(itertools.chain(*texts))#列表扁平化
            text = self.tokenizer(text=text, return_tensors='pt', padding=True)
            self.text = text.to(device=self.model.device)
        return self.text

    @property
    def has_prompt_bank(self) -> bool:
        return self.prompt_bank is not None

    def forward(self,
                text: List[List[str]],
                image_features: Optional[Sequence[Tensor]] = None,
                reliability_maps: Optional[Sequence[Tensor]] = None) -> Tensor:
        num_per_batch = [len(t) for t in text]
        assert max(num_per_batch) == min(num_per_batch), (
            'number of sequences not equal in batch')
        text = list(itertools.chain(*text))
        if self.add_mask:
            text_mask = torch.tensor([x != self.pad_value for x in text],
                                     requires_grad=False).to(self.model.device)
        text = self.tokenizer(text=text, return_tensors='pt', padding=True)
        text = text.to(device=self.model.device)

        self._prompt_bank_losses = {}
        if self.has_prompt_bank and image_features is not None:
            prompt_tokens, self._prompt_bank_losses = self.prompt_bank(
                image_features, reliability_maps)
            txt_feats = self._forward_prompted_text(
                text, prompt_tokens, num_per_batch[0])
            self._add_semantic_anchor_loss(prompt_tokens)
        elif len(self.frozen_modules) > 0:
            with torch.no_grad():
                txt_outputs = self.model(**text)
                txt_feats = txt_outputs.text_embeds
        else:
            txt_outputs = self.model(**text)
            txt_feats = txt_outputs.text_embeds

        txt_feats = txt_feats / txt_feats.norm(p=2, dim=-1, keepdim=True)
        txt_feats = txt_feats.reshape(-1, num_per_batch[0],
                                      txt_feats.shape[-1])
        if self.add_mask:
            text_mask = text_mask.reshape(-1, num_per_batch[0]).to(txt_feats)
        else:
            text_mask = None
        return txt_feats, text_mask

    def _forward_prompted_text(self, text_inputs, prompt_tokens,
                               num_classes: int) -> Tensor:
        input_ids = text_inputs['input_ids']
        attention_mask = text_inputs.get('attention_mask', None)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        token_embedding = self.model.text_model.embeddings.token_embedding
        token_embeds = token_embedding(input_ids)
        prompt_tokens = prompt_tokens.to(
            device=token_embeds.device, dtype=token_embeds.dtype)

        batch_texts = input_ids.shape[0]
        prompt_len = prompt_tokens.shape[1]
        max_positions = getattr(self.model.config, 'max_position_embeddings',
                                input_ids.shape[1] + prompt_len)
        valid_lens = attention_mask.sum(dim=-1).long()
        max_valid_len = int(valid_lens.max().item())
        max_seq_len = min(max_positions, prompt_len + max_valid_len)
        available_class_len = max_seq_len - prompt_len - 1
        if available_class_len <= 0:
            raise ValueError('prompt_length is too large for CLIP max length.')

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        pad_ids = input_ids.new_full((batch_texts, max_seq_len), pad_id)
        combined_embeds = token_embedding(pad_ids)
        prompted_attention = attention_mask.new_zeros(
            (batch_texts, max_seq_len))
        eos_positions = input_ids.new_zeros(batch_texts)
        eos_id = self.tokenizer.eos_token_id

        for row in range(batch_texts):
            image_idx = row // num_classes
            valid_len = int(valid_lens[row].item())
            class_ids = input_ids[row, 1:valid_len]
            if class_ids.numel() > available_class_len:
                if available_class_len == 1:
                    class_ids = input_ids.new_tensor([eos_id])
                else:
                    class_ids = torch.cat([
                        class_ids[:available_class_len - 1],
                        input_ids.new_tensor([eos_id])
                    ])
            class_embeds = token_embedding(class_ids)
            seq_embeds = torch.cat([
                token_embeds[row, :1],
                prompt_tokens[image_idx],
                class_embeds,
            ],
                                   dim=0)
            seq_len = seq_embeds.shape[0]
            combined_embeds[row, :seq_len] = seq_embeds
            prompted_attention[row, :seq_len] = 1
            eos_positions[row] = seq_len - 1

        return self._encode_prompted_embeddings(
            combined_embeds, prompted_attention, eos_positions)

    def _add_semantic_anchor_loss(self, prompt_tokens: Tensor) -> None:
        if (not self.training or self.semantic_anchor_weight <= 0 or
                not self.semantic_anchor_texts):
            return
        anchor_count = len(self.semantic_anchor_texts)
        anchor_texts = list(self.semantic_anchor_texts) * prompt_tokens.shape[0]
        anchor_inputs = self.tokenizer(
            text=anchor_texts, return_tensors='pt', padding=True)
        anchor_inputs = anchor_inputs.to(device=self.model.device)
        prompted_embeddings = self._forward_prompted_text(
            anchor_inputs, prompt_tokens, anchor_count)
        prompted_embeddings = F.normalize(prompted_embeddings, dim=-1)
        prompted_embeddings = prompted_embeddings.reshape(
            prompt_tokens.shape[0], anchor_count, -1)
        original_embeddings = self._get_semantic_anchor_embeddings()
        original_embeddings = original_embeddings.to(
            device=prompted_embeddings.device,
            dtype=prompted_embeddings.dtype)
        anchor_loss = 1.0 - (prompted_embeddings *
                             original_embeddings.unsqueeze(0)).sum(dim=-1)
        self._prompt_bank_losses['loss_mspb_anchor'] = (
            anchor_loss.mean() * self._semantic_anchor_scale())

    def _get_semantic_anchor_embeddings(self) -> Tensor:
        if self._semantic_anchor_embeddings is not None:
            return self._semantic_anchor_embeddings

        embeddings = None
        if (self.semantic_anchor_embed_path is not None and
                os.path.isfile(self.semantic_anchor_embed_path)):
            payload = torch.load(
                self.semantic_anchor_embed_path, map_location='cpu')
            if isinstance(payload, dict):
                saved_texts = tuple(payload.get('texts', ()))
                if saved_texts and saved_texts != self.semantic_anchor_texts:
                    raise ValueError(
                        'semantic anchor texts do not match the offline table.')
                embeddings = payload.get('embeddings', None)
            else:
                embeddings = payload

        if embeddings is None:
            anchor_inputs = self.tokenizer(
                text=list(self.semantic_anchor_texts),
                return_tensors='pt', padding=True)
            anchor_inputs = anchor_inputs.to(device=self.model.device)
            with torch.no_grad():
                embeddings = self.model(**anchor_inputs).text_embeds

        if not torch.is_tensor(embeddings):
            raise TypeError('semantic anchor table must contain a tensor.')
        if embeddings.shape[0] != len(self.semantic_anchor_texts):
            raise ValueError('semantic anchor table has an unexpected class count.')
        self._semantic_anchor_embeddings = F.normalize(
            embeddings.detach().float(), dim=-1)
        return self._semantic_anchor_embeddings

    def _semantic_anchor_scale(self) -> float:
        if self.semantic_anchor_warmup_iters <= 0:
            return self.semantic_anchor_weight
        self._semantic_anchor_step.add_(1)
        warmup_scale = min(
            float(self._semantic_anchor_step.item()) /
            float(self.semantic_anchor_warmup_iters), 1.0)
        return self.semantic_anchor_weight * warmup_scale

    def _encode_prompted_embeddings(self, inputs_embeds: Tensor,
                                    attention_mask: Tensor,
                                    eos_positions: Tensor) -> Tensor:
        text_model = self.model.text_model
        position_ids = torch.arange(
            inputs_embeds.shape[1], device=inputs_embeds.device)
        position_ids = position_ids.unsqueeze(0).expand(
            inputs_embeds.shape[0], -1)
        position_embeds = text_model.embeddings.position_embedding(
            position_ids)
        hidden_states = inputs_embeds + position_embeds

        causal_attention_mask = self._build_causal_attention_mask(
            hidden_states.shape[0], hidden_states.shape[1],
            hidden_states.dtype, hidden_states.device)
        expanded_attention_mask = self._expand_attention_mask(
            attention_mask, hidden_states.dtype, hidden_states.shape[1])

        try:
            encoder_outputs = text_model.encoder(
                inputs_embeds=hidden_states,
                attention_mask=expanded_attention_mask,
                causal_attention_mask=causal_attention_mask,
                return_dict=True)
        except TypeError:
            encoder_outputs = text_model.encoder(
                inputs_embeds=hidden_states,
                attention_mask=expanded_attention_mask,
                return_dict=True)

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = text_model.final_layer_norm(last_hidden_state)
        pooled_output = last_hidden_state[
            torch.arange(last_hidden_state.shape[0],
                         device=last_hidden_state.device), eos_positions]
        return self.model.text_projection(pooled_output)

    @staticmethod
    def _build_causal_attention_mask(batch_size: int, seq_len: int,
                                     dtype: torch.dtype,
                                     device: torch.device) -> Tensor:
        mask = torch.full((seq_len, seq_len),
                          torch.finfo(dtype).min,
                          device=device,
                          dtype=dtype)
        mask = torch.triu(mask, diagonal=1)
        return mask[None, None, :, :].expand(batch_size, 1, seq_len, seq_len)

    @staticmethod
    def _expand_attention_mask(mask: Tensor, dtype: torch.dtype,
                               target_len: int) -> Tensor:
        expanded_mask = mask[:, None, None, :].to(dtype=dtype)
        expanded_mask = expanded_mask.expand(-1, 1, target_len, -1)
        return (1.0 - expanded_mask) * torch.finfo(dtype).min

    def get_prompt_bank_losses(self) -> Dict[str, Tensor]:
        return self._prompt_bank_losses

    def _freeze_modules(self):

        if len(self.frozen_modules) == 0:
            # not freeze
            return
        if self.frozen_modules[0] == "all":
            self.model.eval()
            for _, module in self.model.named_modules():
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False
            return
        for name, module in self.model.named_modules():
            for frozen_name in self.frozen_modules:
                if name.startswith(frozen_name):
                    module.eval()
                    for param in module.parameters():
                        param.requires_grad = False
                    break

    def train(self, mode=True):
        super().train(mode)
        self._freeze_modules()


@MODELS.register_module()
class PseudoLanguageBackbone(BaseModule):  #伪语言骨干：离线词汇嵌入
    """Pseudo Language Backbone
    Args:
        text_embed_path (str): path to the text embedding file
    """
    def __init__(self,
                 text_embed_path: str = "",
                 test_embed_path: str = None,
                 init_cfg: OptMultiConfig = None):
        super().__init__(init_cfg)
        # {text:embed}
        self.text_embed = torch.load(text_embed_path, map_location='cpu')
        if test_embed_path is None:
            self.test_embed = self.text_embed
        else:
            self.test_embed = torch.load(test_embed_path)
        self.register_buffer("buff", torch.zeros([
            1,
        ]))

    def forward_cache(self, text: List[List[str]]) -> Tensor:
        if not hasattr(self, "cache"):
            self.cache = self.forward_text(text)
        return self.cache

    def forward(self, text: List[List[str]]) -> Tensor:
        if self.training:
            return self.forward_text(text)
        else:
            return self.forward_cache(text)

    def forward_text(self, text: List[List[str]]) -> Tensor:
        num_per_batch = [len(t) for t in text]
        assert max(num_per_batch) == min(num_per_batch), (
            'number of sequences not equal in batch')
        text = list(itertools.chain(*text))
        if self.training:
            text_embed_dict = self.text_embed
        else:
            text_embed_dict = self.test_embed
        text_embeds = torch.stack(
            [text_embed_dict[x.split("/")[0]] for x in text])
        # requires no grad and force to float
        text_embeds = text_embeds.to(
            self.buff.device).requires_grad_(False).float()
        text_embeds = text_embeds.reshape(-1, num_per_batch[0],
                                          text_embeds.shape[-1])
        return text_embeds


@MODELS.register_module()
class MultiModalYOLOBackbone(BaseModule):
    def __init__(self,
                 image_model: ConfigType,
                 text_model: ConfigType,
                 frozen_stages: int = -1,
                 with_text_model: bool = True,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg)
        self.with_text_model = with_text_model
        self.image_model = MODELS.build(image_model)
        if self.with_text_model:
            self.text_model = MODELS.build(text_model)
        else:
            self.text_model = None
        self.frozen_stages = frozen_stages
        self._freeze_stages()

    def _freeze_stages(self):
        """Freeze the parameters of the specified stage so that they are no
        longer updated."""
        if self.frozen_stages >= 0:
            for i in range(self.frozen_stages + 1):
                m = getattr(self.image_model, self.image_model.layers[i])
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def train(self, mode: bool = True):
        """Convert the model into training mode while keep normalization layer
        frozen."""
        super().train(mode)
        self._freeze_stages()

    def forward(self, image: Tensor,
                text: List[List[str]]) -> Tuple[Tuple[Tensor], Tensor]:
        img_feats = self.image_model(image)
        if text is not None and self.with_text_model:
            if getattr(self.text_model, 'has_prompt_bank', False):
                txt_feats = self.text_model(text, image_features=img_feats)
            else:
                txt_feats = self.text_model(text)
            return img_feats, txt_feats
        else:
            return img_feats, None

    def forward_text(self, text: List[List[str]]) -> Tensor:
        assert self.with_text_model, "forward_text() requires a text model"
        txt_feats = self.text_model(text)
        return txt_feats

    def forward_prompted_text(
            self, text: List[List[str]], image_features: Sequence[Tensor],
            reliability_maps: Optional[Sequence[Tensor]] = None) -> Tensor:
        assert self.with_text_model, (
            'forward_prompted_text() requires a text model')
        if not self.has_text_prompt_bank:
            raise RuntimeError('text prompt bank is not enabled.')
        return self.text_model(
            text,
            image_features=image_features,
            reliability_maps=reliability_maps)

    def forward_image(self, image: Tensor) -> Tuple[Tensor]:
        return self.image_model(image)

    @property
    def has_text_prompt_bank(self) -> bool:
        return (self.with_text_model and self.text_model is not None and
                getattr(self.text_model, 'has_prompt_bank', False))

    def get_text_model_losses(self) -> Dict[str, Tensor]:
        if self.has_text_prompt_bank:
            return self.text_model.get_prompt_bank_losses()
        return {}

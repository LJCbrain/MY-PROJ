"""Text-to-image residual attention for the final multi-scale neck features."""

import math
from typing import Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from mmyolo.registry import MODELS


@MODELS.register_module()
class MultiScaleTextCrossAttention(nn.Module):
    """Read shared text K/V with a separate visual Q at each feature level.

    Only the attention branch is normalized. The original visual feature is
    the identity path, and text is neither modified nor detached here. The
    caller supplies the frozen base text embeddings, not prompted text.
    """

    def __init__(self,
                 in_channels: Sequence[int],
                 text_channels: int = 768,
                 embed_channels: int = 256,
                 num_heads: int = 8,
                 active_levels: Optional[Sequence[int]] = None,
                 init_scale: float = 1e-3,
                 attention_dropout: float = 0.0):
        super().__init__()
        self.in_channels = tuple(in_channels)
        if not self.in_channels or min(self.in_channels) <= 0:
            raise ValueError('in_channels must contain positive channel counts.')
        if min(text_channels, embed_channels, num_heads) <= 0:
            raise ValueError('Text, embedding and head dimensions must be positive.')
        if embed_channels % num_heads:
            raise ValueError('embed_channels must be divisible by num_heads.')
        levels = (tuple(range(len(self.in_channels))) if active_levels is None
                  else tuple(active_levels))
        if (not levels or len(set(levels)) != len(levels)
                or any(not isinstance(i, int) or i < 0
                       or i >= len(self.in_channels) for i in levels)):
            raise ValueError('active_levels must be unique valid feature indices.')
        if not math.isfinite(init_scale):
            raise ValueError('init_scale must be finite.')
        if not 0 <= attention_dropout < 1:
            raise ValueError('attention_dropout must be in [0, 1).')

        self.text_channels = text_channels
        self.embed_channels = embed_channels
        self.num_heads = num_heads
        self.head_channels = embed_channels // num_heads
        self.active_levels = tuple(sorted(levels))
        self.attention_dropout = attention_dropout
        self.text_norm = nn.LayerNorm(text_channels)
        self.key_proj = nn.Linear(text_channels, embed_channels)
        self.value_proj = nn.Linear(text_channels, embed_channels)
        # Inactive levels have no unused trainable parameters under DDP.
        self.image_norms = nn.ModuleDict({
            str(i): nn.LayerNorm(self.in_channels[i]) for i in self.active_levels})
        self.query_projs = nn.ModuleDict({
            str(i): nn.Linear(self.in_channels[i], embed_channels)
            for i in self.active_levels})
        self.output_projs = nn.ModuleDict({
            str(i): nn.Linear(embed_channels, self.in_channels[i])
            for i in self.active_levels})
        self.residual_scales = nn.ParameterDict({
            str(i): nn.Parameter(torch.full((self.in_channels[i],), init_scale))
            for i in self.active_levels})
        self.collect_diagnostics = False
        self.last_diagnostics = {}

    def forward(self, img_feats: Sequence[Tensor], txt_feats: Optional[Tensor],
                txt_masks: Optional[Tensor] = None) -> Tuple[Tensor, ...]:
        self.last_diagnostics = {}
        if len(img_feats) != len(self.in_channels):
            raise ValueError('Image feature count must match in_channels.')
        batch_size = img_feats[0].shape[0]
        for feat, channels in zip(img_feats, self.in_channels):
            if (feat.ndim != 4 or feat.shape[0] != batch_size
                    or feat.shape[1] != channels):
                raise ValueError('Image features must have the configured B x C x H x W shape.')
        if txt_feats is None:
            return tuple(img_feats)
        if (txt_feats.ndim != 3 or txt_feats.shape[0] != batch_size
                or txt_feats.shape[-1] != self.text_channels):
            raise ValueError('Text features must have shape B x N x text_channels.')
        if txt_masks is None:
            valid = torch.ones(txt_feats.shape[:2], dtype=torch.bool,
                               device=txt_feats.device)
        else:
            if txt_masks.shape != txt_feats.shape[:2]:
                raise ValueError('Text mask must have shape B x N.')
            valid = txt_masks.to(device=txt_feats.device, dtype=torch.bool)

        # A dummy masked token keeps empty-vocabulary training finite and keeps
        # the parameter graph connected. Its final visual update is zero.
        if txt_feats.shape[1] == 0:
            txt_feats = txt_feats.new_zeros((batch_size, 1, self.text_channels))
            valid = valid.new_zeros((batch_size, 1))
        has_text = valid.any(dim=-1)
        text = self.text_norm(txt_feats.masked_fill(~valid.unsqueeze(-1), 0))
        keys = self.key_proj(text).reshape(
            batch_size, -1, self.num_heads, self.head_channels).transpose(1, 2)
        values = self.value_proj(text).reshape(
            batch_size, -1, self.num_heads, self.head_channels).transpose(1, 2)
        safe_valid = valid.clone()
        safe_valid[:, 0] |= ~has_text
        outputs = list(img_feats)
        for level in self.active_levels:
            name = str(level)
            feat = img_feats[level]
            tokens = feat.flatten(2).transpose(1, 2)
            queries = self.query_projs[name](self.image_norms[name](tokens))
            queries = queries.reshape(
                batch_size, -1, self.num_heads, self.head_channels).transpose(1, 2)
            # Keep the dot products and normalization finite under AMP. The
            # projections still use the caller's autocast policy.
            with torch.autocast(device_type=feat.device.type, enabled=False):
                scores = torch.matmul(
                    queries.float() * (self.head_channels ** -0.5),
                    keys.float().transpose(-2, -1))
                scores = scores.masked_fill(
                    ~safe_valid[:, None, None, :], float('-inf'))
                weights = F.softmax(scores, dim=-1)
                weights = F.dropout(weights, p=self.attention_dropout,
                                    training=self.training)
                context = torch.matmul(weights, values.float())
            context = context.transpose(1, 2).reshape(
                batch_size, -1, self.embed_channels)
            delta = self.output_projs[name](context.to(queries.dtype))
            # Apply after the output projection so its bias cannot alter an
            # image with no valid text. No spatial foreground gate is used.
            delta = delta.masked_fill(~has_text[:, None, None], 0)
            delta = delta * self.residual_scales[name].to(delta.dtype)
            update = delta.transpose(1, 2).reshape_as(feat).to(feat.dtype)
            outputs[level] = feat + update
            if self.collect_diagnostics:
                with torch.no_grad():
                    prefix = f'text_ca_p{level + 3}'
                    self.last_diagnostics[prefix + '_residual_ratio'] = (
                        update.float().square().mean().sqrt()
                        / feat.float().square().mean().sqrt().clamp_min(1e-6))
                    self.last_diagnostics[prefix + '_scale_abs_mean'] = (
                        self.residual_scales[name].detach().abs().mean())
        return tuple(outputs)

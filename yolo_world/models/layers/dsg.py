"""Residual propagation over frozen class-text graphs, with no extra loss."""

import math
from typing import Optional, Sequence

import torch
from torch import nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmyolo.registry import MODELS


@MODELS.register_module()
class DynamicSemanticGraph(BaseModule):
    def __init__(self, text_channels: int, hidden_channels: int = 256,
                 similarity_threshold: float = 0.5,
                 similarity_quantile: Optional[float] = None,
                 max_residual_scale: float = 0.1, init_scale: float = 0.0,
                 novel_indices: Sequence[int] = (), init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        if text_channels <= 0 or hidden_channels <= 0:
            raise ValueError('DSG channel dimensions must be positive.')
        if not 0 <= similarity_threshold <= 1:
            raise ValueError('DSG similarity threshold must be in [0, 1].')
        if similarity_quantile is not None and not 0 <= similarity_quantile <= 1:
            raise ValueError('DSG similarity quantile must be in [0, 1].')
        if (not math.isfinite(max_residual_scale) or
                not 0 < max_residual_scale < 1 or
                not math.isfinite(init_scale) or
                abs(init_scale) >= max_residual_scale):
            raise ValueError('DSG init_scale must lie inside its scale bound.')
        self.text_channels = text_channels
        self.similarity_threshold = float(similarity_threshold)
        self.similarity_quantile = similarity_quantile
        self.max_residual_scale = float(max_residual_scale)
        self.novel_indices = tuple(novel_indices)
        if any(not isinstance(index, int) or index < 0
               for index in self.novel_indices):
            raise ValueError('DSG novel indices must be nonnegative integers.')
        self.fc1 = nn.Linear(text_channels, hidden_channels, bias=False)
        self.fc2 = nn.Linear(hidden_channels, text_channels, bias=False)
        self.gamma = nn.Parameter(torch.tensor(
            math.atanh(init_scale / max_residual_scale)))
        self.collect_diagnostics = False
        self.last_diagnostics = {}
        self._graph_settings = None
        # These are derived caches, not model buffers. Even nonpersistent
        # buffers participate in EMAHook's positional buffer swaps.
        for name in ('_graph_text', '_graph_mask', '_graph_adjacency',
                     '_graph_threshold'):
            setattr(self, name, None)

    def clear_graph_cache(self):
        self._graph_text = self._graph_mask = self._graph_adjacency = None
        self._graph_threshold = self._graph_settings = None
        self.last_diagnostics = {}

    def _apply(self, fn, *args, **kwargs):
        # Plain tensor attributes do not follow Module.to()/cuda()/half().
        # Rebuild the cache on demand in the new device/dtype instead.
        self.clear_graph_cache()
        return super()._apply(fn, *args, **kwargs)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Both PyTorch and MMEngine recurse through this long-standing API.
        # Clearing derived buffers here also supports older PyTorch versions.
        self.clear_graph_cache()
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys,
            unexpected_keys, error_msgs)

    @property
    def residual_scale(self):
        return self.max_residual_scale * self.gamma.tanh()

    def _adjacency(self, text, valid):
        cached = self._graph_text
        settings = (self.similarity_threshold, self.similarity_quantile)
        if (cached is not None and cached.shape == text.shape and
                cached.device == text.device and cached.dtype == text.dtype and
                self._graph_settings == settings and
                torch.equal(cached, text) and
                torch.equal(self._graph_mask, valid)):
            return self._graph_adjacency
        similarity = text @ text.transpose(-1, -2)
        count = text.shape[1]
        diagonal = torch.eye(count, device=text.device, dtype=torch.bool)[None]
        pairs = valid[:, :, None] & valid[:, None, :]
        eligible = pairs & ~diagonal
        threshold = similarity.new_full((text.shape[0], 1, 1),
                                         self.similarity_threshold)
        if self.similarity_quantile is not None:
            # CLIP similarities need not share BERT's scale. Calibrate only
            # on frozen, valid text pairs, separately for each vocabulary.
            for index in range(text.shape[0]):
                values = similarity[index][eligible[index]]
                if values.numel():
                    threshold[index, 0, 0] = torch.maximum(
                        threshold[index, 0, 0],
                        torch.quantile(values, self.similarity_quantile))
        edges = eligible & (similarity >= threshold)
        adjacency = torch.where(edges, similarity, torch.zeros_like(similarity))
        adjacency = adjacency + torch.diag_embed(valid.to(text.dtype))
        degree = adjacency.sum(-1)
        inv_sqrt = degree.clamp_min(1e-6).rsqrt() * valid
        adjacency = adjacency * inv_sqrt[:, :, None] * inv_sqrt[:, None, :]
        self._graph_text = text.detach().clone()
        self._graph_mask = valid.detach().clone()
        self._graph_adjacency = adjacency.detach()
        self._graph_threshold = threshold.detach()
        self._graph_settings = settings
        return self._graph_adjacency

    def forward(self, text, txt_masks=None):
        if text is None or text.ndim != 3 or text.shape[-1] != self.text_channels:
            raise ValueError('DSG requires text with shape B x N x text_channels.')
        if not text.shape[0] or not text.shape[1]:
            raise ValueError('DSG requires a nonempty vocabulary and batch.')
        batch, count = text.shape[:2]
        if self.novel_indices and max(self.novel_indices) >= count:
            raise ValueError('DSG novel indices exceed the graph vocabulary.')
        if txt_masks is not None and txt_masks.shape != (batch, count):
            raise ValueError('DSG text mask must have shape B x N.')
        # The detector expands one fixed vocabulary with stride zero. Compute
        # its GCN once and broadcast the result, preserving detector gradients.
        shared = (batch == 1 or
                  (text.stride(0) == 0 and
                   (txt_masks is None or txt_masks.stride(0) == 0)))
        source = text[:1] if shared else text
        valid = (torch.ones(source.shape[:2], device=text.device, dtype=torch.bool)
                 if txt_masks is None else
                 (txt_masks[:1] if shared else txt_masks).to(
                     device=text.device, dtype=torch.bool))
        with torch.no_grad(), torch.autocast(device_type=text.device.type,
                                             enabled=False):
            clean = torch.where(valid[..., None], source.detach().float(),
                                torch.zeros_like(source, dtype=torch.float32))
            if not torch.isfinite(clean).all():
                raise ValueError('DSG valid text embeddings must be finite.')
            base = F.normalize(clean, dim=-1, eps=1e-6)
            adjacency = self._adjacency(base, valid)
        work_dtype = self.fc1.weight.dtype
        hidden = F.relu(self.fc1(adjacency.to(work_dtype) @ base.to(work_dtype)))
        delta = self.fc2(adjacency.to(hidden.dtype) @ hidden)
        with torch.autocast(device_type=text.device.type, enabled=False):
            delta = F.normalize(delta.float(), dim=-1, eps=1e-6)
            scale = self.max_residual_scale * self.gamma.float().tanh()
            update = scale * delta
            enhanced = F.normalize(base + update, dim=-1, eps=1e-6)
            enhanced = torch.where(valid[..., None], enhanced,
                                   torch.zeros_like(enhanced))
        self.last_diagnostics = {}
        if self.collect_diagnostics:
            with torch.no_grad():
                eye = torch.eye(count, device=text.device, dtype=torch.bool)[None]
                degrees = ((adjacency > 0) & ~eye).sum(-1).float()
                denominator = valid.sum().clamp_min(1)
                self.last_diagnostics = dict(
                    dsg_scale=scale.detach(),
                    dsg_similarity_threshold=self._graph_threshold.mean(),
                    dsg_residual_ratio=(update.square().sum() /
                                        base.square().sum().clamp_min(1e-6)).sqrt(),
                    dsg_neighbor_degree_mean=(degrees * valid).sum() / denominator,
                    dsg_isolated_fraction=((degrees == 0) & valid).sum() / denominator,
                    dsg_text_cosine=((base * enhanced).sum(-1) * valid).sum()
                    / denominator)
                if self.novel_indices:
                    novel = degrees[:, self.novel_indices]
                    self.last_diagnostics['dsg_novel_degree_mean'] = novel.mean()
                    self.last_diagnostics['dsg_novel_isolated_fraction'] = (
                        novel == 0).float().mean()
                    for index in self.novel_indices:
                        self.last_diagnostics[f'dsg_novel_{index}_degree'] = (
                            degrees[:, index].mean())
        enhanced = enhanced.to(text.dtype)
        return enhanced.expand(batch, -1, -1) if shared else enhanced

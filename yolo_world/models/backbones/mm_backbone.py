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


class SemanticBoundedSphericalFusion(nn.Module):
    """Keep prompted text inside a geodesic trust region around CLIP text."""

    def __init__(self,
                 max_angle_deg: float = 15.0,
                 eps: float = 1e-6,
                 small_angle_eps: float = 1e-4,
                 compute_fp32: bool = True,
                 collect_stats: bool = True) -> None:
        super().__init__()
        if not 0.0 <= max_angle_deg <= 180.0:
            raise ValueError('max_angle_deg must be in [0, 180].')
        self.max_angle_deg = float(max_angle_deg)
        self.eps = float(eps)
        self.small_angle_eps = float(small_angle_eps)
        self.compute_fp32 = bool(compute_fp32)
        self.collect_stats = bool(collect_stats)
        self.last_stats: Dict[str, Tensor] = {}

    def forward(self,
                original: Tensor,
                prompted: Tensor,
                text_mask: Optional[Tensor] = None) -> Tensor:
        if original.shape != prompted.shape:
            raise ValueError('original and prompted text must share a shape.')
        if original.dim() != 3:
            raise ValueError('text features must have shape [B, N, D].')

        output_dtype = prompted.dtype
        work_dtype = torch.float32 if self.compute_fp32 else prompted.dtype
        original_unit = F.normalize(
            original.detach().to(work_dtype), dim=-1, eps=self.eps)
        prompted_unit = F.normalize(
            prompted.to(work_dtype), dim=-1, eps=self.eps)
        dot = (original_unit * prompted_unit).sum(dim=-1).clamp(-1.0, 1.0)
        safe_dot = dot.clamp(-1.0 + self.eps, 1.0 - self.eps)
        theta = torch.acos(safe_dot)
        near_same = dot >= torch.cos(
            dot.new_tensor(self.small_angle_eps))
        theta = torch.where(near_same, torch.zeros_like(theta), theta)

        max_angle = theta.new_tensor(self.max_angle_deg * torch.pi / 180.0)
        if self.max_angle_deg >= 180.0:
            fused = prompted_unit
            alpha = torch.ones_like(theta)
            clipped = torch.zeros_like(theta, dtype=torch.bool)
        elif self.max_angle_deg <= 0.0:
            fused = original_unit + prompted_unit * 0.0
            alpha = torch.zeros_like(theta)
            clipped = theta > self.small_angle_eps
        else:
            target_theta = theta.clamp_max(max_angle)
            alpha = torch.where(
                theta > self.small_angle_eps,
                target_theta / theta.clamp_min(self.eps),
                torch.ones_like(theta))
            clipped = theta > max_angle

            tangent = prompted_unit - dot.unsqueeze(-1) * original_unit
            tangent_norm = tangent.norm(dim=-1, keepdim=True)
            tangent_unit = tangent / tangent_norm.clamp_min(self.eps)
            projected = (
                torch.cos(target_theta).unsqueeze(-1) * original_unit +
                torch.sin(target_theta).unsqueeze(-1) * tangent_unit)
            projected = torch.where(
                (tangent_norm > self.eps).expand_as(projected),
                projected,
                original_unit)
            fused = torch.where(
                clipped.unsqueeze(-1), projected, prompted_unit)
            fused = F.normalize(fused, dim=-1, eps=self.eps)

        if text_mask is not None:
            valid_mask = text_mask.to(
                device=fused.device, dtype=torch.bool)
            if valid_mask.shape != fused.shape[:2]:
                raise ValueError('text_mask must have shape [B, N].')
            fused = torch.where(
                valid_mask.unsqueeze(-1), fused, original_unit)
        else:
            valid_mask = torch.ones_like(theta, dtype=torch.bool)

        if self.collect_stats:
            self._collect_stats(theta, alpha, clipped, valid_mask)
        return fused.to(output_dtype)

    def _collect_stats(self, theta: Tensor, alpha: Tensor, clipped: Tensor,
                       valid_mask: Tensor) -> None:
        valid_theta = theta.detach().masked_select(valid_mask)
        valid_alpha = alpha.detach().masked_select(valid_mask)
        valid_clipped = clipped.detach().masked_select(valid_mask)
        if valid_theta.numel() == 0:
            zero = theta.detach().new_zeros(())
            self.last_stats = {
                'semantic_angle_mean_deg': zero,
                'semantic_angle_p95_deg': zero,
                'semantic_angle_max_deg': zero,
                'semantic_clip_rate': zero,
                'semantic_alpha_mean': zero,
            }
            return

        radians_to_degrees = 180.0 / torch.pi
        angle_deg = valid_theta.float() * radians_to_degrees
        self.last_stats = {
            'semantic_angle_mean_deg': angle_deg.mean(),
            'semantic_angle_p95_deg': torch.quantile(angle_deg, 0.95),
            'semantic_angle_max_deg': angle_deg.max(),
            'semantic_clip_rate': valid_clipped.float().mean(),
            'semantic_alpha_mean': valid_alpha.float().mean(),
        }


class SemanticallyGuardedClassConditionedPrompting(nn.Module):
    """Semantically Guarded Class-Conditioned Prompting router."""

    def __init__(self,
                 image_channels: Sequence[int],
                 text_hidden_size: int,
                 num_prompts: int = 10,
                 prompt_length: int = 12,
                 num_scales: int = 3,
                 key_dim: Optional[int] = None,
                 feat_indices: Optional[Sequence[int]] = None,
                 use_reliability: bool = True,
                 use_reliability_prior: bool = True,
                 use_reliable_global_pool: Optional[bool] = None,
                 reliability_prior_weight: float = 1.0,
                 reliability_prior_warmup_iters: int = 0,
                 reliability_log_min: float = -4.0,
                 class_semantic_residual_weight: float = 0.0,
                 loss_match_weight: float = 0.7,
                 loss_match_start_iter: int = 0,
                 loss_match_warmup_iters: int = 0,
                 loss_orthogonal_weight: float = 0.3,
                 loss_balance_weight: float = 0.05,
                 loss_balance_warmup_weight: Optional[float] = None,
                 loss_balance_warmup_iters: int = 0,
                 per_scale_balance: bool = False,
                 balance_use_all_text: bool = False,
                 routing_temperature: float = 0.1,
                 balance_temperature: Optional[float] = None,
                 routing_entropy_floor: float = 0.75,
                 spatial_temperature: float = 0.1,
                 class_chunk_size: int = 64,
                 orthogonal_key_init: bool = False,
                 key_orthogonalization: str = 'none',
                 normalize_match_by_num_scales: bool = False,
                 collect_key_stats: bool = False,
                 eps: float = 1e-6,
                 init_std: float = 0.02) -> None:
        super().__init__()
        if num_prompts <= 0:
            raise ValueError('num_prompts must be positive.')
        if routing_temperature <= 0 or spatial_temperature <= 0:
            raise ValueError('routing temperatures must be positive.')
        if balance_temperature is not None and balance_temperature <= 0:
            raise ValueError('balance_temperature must be positive.')
        if reliability_prior_weight < 0:
            raise ValueError('reliability_prior_weight must be non-negative.')
        if class_semantic_residual_weight < 0:
            raise ValueError(
                'class_semantic_residual_weight must be non-negative.')
        if key_orthogonalization not in ('none', 'qr'):
            raise ValueError(
                'key_orthogonalization must be either none or qr.')
        self.num_prompts = int(num_prompts)
        self.prompt_length = int(prompt_length)
        self.num_scales = int(num_scales)
        self.key_dim = key_dim or text_hidden_size
        self.use_reliability = bool(use_reliability)
        self.use_reliability_prior = bool(use_reliability_prior)
        if use_reliable_global_pool is None:
            use_reliable_global_pool = self.use_reliability
        self.use_reliable_global_pool = bool(use_reliable_global_pool)
        self.reliability_prior_weight = float(reliability_prior_weight)
        self.reliability_prior_warmup_iters = max(
            int(reliability_prior_warmup_iters), 0)
        self.reliability_log_min = float(reliability_log_min)
        self.class_semantic_residual_weight = float(
            class_semantic_residual_weight)
        self.loss_match_weight = float(loss_match_weight)
        self.loss_match_start_iter = max(int(loss_match_start_iter), 0)
        self.loss_match_warmup_iters = max(
            int(loss_match_warmup_iters), 0)
        self.loss_orthogonal_weight = float(loss_orthogonal_weight)
        self.loss_balance_weight = float(loss_balance_weight)
        if loss_balance_warmup_weight is None:
            loss_balance_warmup_weight = self.loss_balance_weight
        self.loss_balance_warmup_weight = float(
            loss_balance_warmup_weight)
        self.loss_balance_warmup_iters = max(
            int(loss_balance_warmup_iters), 0)
        self.per_scale_balance = bool(per_scale_balance)
        self.balance_use_all_text = bool(balance_use_all_text)
        self.routing_temperature = float(routing_temperature)
        self.balance_temperature = float(
            balance_temperature
            if balance_temperature is not None else routing_temperature)
        self.routing_entropy_floor = float(routing_entropy_floor)
        self.spatial_temperature = float(spatial_temperature)
        self.class_chunk_size = max(int(class_chunk_size), 1)
        self.orthogonal_key_init = bool(orthogonal_key_init)
        self.key_orthogonalization = key_orthogonalization
        self.normalize_match_by_num_scales = bool(
            normalize_match_by_num_scales)
        self.collect_key_stats = bool(collect_key_stats)
        self.eps = float(eps)
        if (self.class_semantic_residual_weight > 0 and
                self.key_dim != text_hidden_size):
            raise ValueError(
                'class semantic residual requires key_dim to equal '
                'text_hidden_size.')

        if feat_indices is None:
            feat_indices = tuple(range(num_scales))
        self.feat_indices = tuple(feat_indices)
        if len(self.feat_indices) != num_scales:
            raise ValueError('feat_indices must match num_scales.')
        if len(image_channels) != num_scales:
            raise ValueError('image_channels must match num_scales.')

        self.class_score_projs = nn.ModuleList([
            nn.Conv2d(ch, self.key_dim, kernel_size=1, bias=False)
            for ch in image_channels
        ])
        query_channels = [
            ch * 2 if self.use_reliable_global_pool else ch
            for ch in image_channels
        ]
        self.image_projs = nn.ModuleList(
            [nn.Linear(ch, self.key_dim) for ch in query_channels])
        self.keys = nn.Parameter(
            torch.empty(self.num_prompts, self.key_dim))
        self.prompts = nn.Parameter(
            torch.empty(self.num_prompts, self.prompt_length,
                        text_hidden_size))
        self.last_selected_indices: Optional[Tensor] = None
        self.last_selected_scores: Optional[Tensor] = None
        self.last_routing_usage: Optional[Tensor] = None
        self.last_routing_entropy: Optional[Tensor] = None
        self.last_max_key_share: Optional[Tensor] = None
        self.last_active_keys: Optional[Tensor] = None
        self.register_buffer(
            '_routing_step', torch.zeros((), dtype=torch.long),
            persistent=False)
        self.reset_parameters(init_std)

    def reset_parameters(self, init_std: float) -> None:
        if self.orthogonal_key_init:
            nn.init.orthogonal_(self.keys)
            with torch.no_grad():
                self.keys.mul_(init_std * self.key_dim**0.5)
        else:
            nn.init.normal_(self.keys, std=init_std)
        nn.init.normal_(self.prompts, std=init_std)
        for proj in self.class_score_projs:
            nn.init.xavier_uniform_(proj.weight)
        for proj in self.image_projs:
            nn.init.xavier_uniform_(proj.weight)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)

    def _normalized_routing_keys(self) -> Tensor:
        keys = self.keys.float()
        if self.key_orthogonalization == 'qr':
            with torch.cuda.amp.autocast(enabled=False):
                basis, _ = torch.linalg.qr(keys.t(), mode='reduced')
                return basis.t()
        return F.normalize(keys, dim=-1, eps=self.eps)

    def forward(
            self,
            img_feats: Sequence[Tensor],
            reliability_maps: Optional[Sequence[Tensor]],
            class_embeddings: Tensor,
            text_mask: Optional[Tensor] = None,
            positive_class_mask: Optional[Tensor] = None,
            compute_losses: bool = True,
            routing_grad: bool = True,
            update_stats: bool = True
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        if class_embeddings.dim() != 3:
            raise ValueError(
                'class_embeddings must have shape [B, N, D].')
        batch_size, num_classes = class_embeddings.shape[:2]
        if text_mask is None:
            text_mask = torch.ones(
                batch_size,
                num_classes,
                device=class_embeddings.device,
                dtype=torch.bool)
        else:
            text_mask = text_mask.to(
                device=class_embeddings.device, dtype=torch.bool)
        if text_mask.shape != (batch_size, num_classes):
            raise ValueError('text_mask must have shape [B, N].')

        match_mask = text_mask
        if positive_class_mask is not None:
            positive_class_mask = positive_class_mask.to(
                device=class_embeddings.device, dtype=torch.bool)
            if positive_class_mask.shape != match_mask.shape:
                raise ValueError(
                    'positive_class_mask must have shape [B, N].')
            match_mask = match_mask & positive_class_mask
        balance_mask = text_mask if self.balance_use_all_text else match_mask

        selected_prompts = []
        match_losses = []
        assignments = []
        hard_assignments = []
        balance_probabilities = []
        selected_indices = []
        selected_scores = []
        selected_margins = []
        normalized_keys = self._normalized_routing_keys()

        for scale_idx, feat_idx in enumerate(self.feat_indices):
            feat = img_feats[feat_idx]
            reliability = None
            if reliability_maps is not None:
                reliability = reliability_maps[feat_idx]
            pooled = self._class_conditioned_pool(
                feat, reliability, class_embeddings, text_mask, scale_idx)
            visual_query = F.normalize(
                self.image_projs[scale_idx](pooled),
                dim=-1,
                eps=self.eps)
            if self.class_semantic_residual_weight > 0:
                class_query = F.normalize(
                    class_embeddings.detach().to(visual_query),
                    dim=-1,
                    eps=self.eps)
                query = F.normalize(
                    visual_query +
                    self.class_semantic_residual_weight * class_query,
                    dim=-1,
                    eps=self.eps)
            else:
                query = visual_query
            similarity = torch.matmul(
                query,
                normalized_keys.to(
                    device=query.device, dtype=query.dtype).t())
            indices = similarity.argmax(dim=-1)
            hard_assignment = F.one_hot(
                indices, num_classes=self.num_prompts).to(similarity)
            balance_probability = F.softmax(
                similarity / self.balance_temperature, dim=-1)

            if self.training and routing_grad:
                soft_assignment = F.softmax(
                    similarity / self.routing_temperature, dim=-1)
                assignment = (
                    hard_assignment - soft_assignment.detach() +
                    soft_assignment)
            else:
                assignment = hard_assignment

            prompts = torch.einsum(
                'bnk,kmd->bnmd', assignment, self.prompts)
            prompts = prompts * text_mask[:, :, None, None].to(prompts)
            selected_similarity = similarity.gather(
                -1, indices.unsqueeze(-1)).squeeze(-1)

            selected_prompts.append(prompts)
            assignments.append(assignment)
            hard_assignments.append(hard_assignment)
            balance_probabilities.append(balance_probability)
            selected_indices.append(indices.detach())
            selected_scores.append(selected_similarity.detach())
            if self.num_prompts > 1:
                top_two = similarity.topk(2, dim=-1).values
                selected_margins.append(
                    (top_two[..., 0] - top_two[..., 1]).detach())
            else:
                selected_margins.append(
                    torch.ones_like(selected_similarity).detach())
            match_losses.append(1.0 - selected_similarity)

        prompt_tokens = torch.cat(selected_prompts, dim=2)
        losses: Dict[str, Tensor] = {}
        if compute_losses:
            losses.update(self._routing_losses(
                match_losses,
                assignments,
                hard_assignments,
                balance_probabilities,
                match_mask,
                balance_mask))
            if self.loss_orthogonal_weight > 0:
                losses['loss_guard_prompt_orthogonal'] = (
                    self._orthogonal_loss() *
                    self.loss_orthogonal_weight)

        if update_stats:
            self.last_selected_indices = torch.stack(
                selected_indices, dim=1)
            self.last_selected_scores = torch.stack(
                selected_scores, dim=1)
            losses.update(self._update_routing_stats(
                selected_indices,
                balance_mask,
                selected_margins=selected_margins,
                normalized_keys=normalized_keys))
        if self.training and compute_losses:
            self._routing_step.add_(1)
        return prompt_tokens, losses

    def _class_conditioned_pool(
            self, feat: Tensor, reliability: Optional[Tensor],
            class_embeddings: Tensor, text_mask: Tensor,
            scale_idx: int) -> Tensor:
        if feat.dim() != 4:
            raise ValueError('image features must be 4D tensors.')
        if feat.shape[0] != class_embeddings.shape[0]:
            raise ValueError('image and text batch sizes do not match.')

        feat_float = feat.float()
        visual = self.class_score_projs[scale_idx](feat).float()
        visual = F.normalize(visual, dim=1, eps=self.eps)
        classes = F.normalize(
            class_embeddings.detach().float(), dim=-1, eps=self.eps)
        reliability_prior = self._prepare_reliability(
            reliability, feat.shape[-2:], feat.device)
        if self.use_reliability_prior:
            spatial_prior = torch.log(
                reliability_prior.clamp_min(self.eps))
            spatial_prior = spatial_prior.clamp(
                min=self.reliability_log_min, max=0.0)
            spatial_prior = (
                spatial_prior * self._reliability_prior_scale())
        else:
            spatial_prior = torch.zeros_like(reliability_prior)
        flat_feat = feat_float.flatten(2)

        pooled_chunks = []
        for start in range(0, classes.shape[1], self.class_chunk_size):
            end = min(start + self.class_chunk_size, classes.shape[1])
            class_chunk = classes[:, start:end]
            logits = torch.einsum(
                'bdhw,bnd->bnhw', visual, class_chunk)
            logits = (
                logits / self.spatial_temperature +
                spatial_prior)
            weights = F.softmax(logits.flatten(2), dim=-1)
            pooled = torch.einsum(
                'bnq,bcq->bnc', weights, flat_feat)
            pooled_chunks.append(pooled)
        class_pooled = torch.cat(pooled_chunks, dim=1)
        class_pooled = class_pooled * text_mask.unsqueeze(-1).to(
            class_pooled)

        if not self.use_reliable_global_pool:
            return class_pooled.to(feat.dtype)
        reliable_pooled = self._reliable_global_pool(
            feat_float, reliability_prior)
        reliable_pooled = reliable_pooled[:, None].expand(
            -1, class_embeddings.shape[1], -1)
        pooled = torch.cat([class_pooled, reliable_pooled], dim=-1)
        return pooled.to(feat.dtype)

    def _reliability_prior_scale(self) -> float:
        if (not self.training or
                self.reliability_prior_warmup_iters <= 0):
            return self.reliability_prior_weight
        progress = min(
            float(self._routing_step.item()) /
            float(self.reliability_prior_warmup_iters),
            1.0)
        return self.reliability_prior_weight * progress

    def _prepare_reliability(self, reliability: Optional[Tensor],
                             feat_shape: Tuple[int, int],
                             device: torch.device) -> Tensor:
        if reliability is None:
            return torch.ones(
                1, 1, *feat_shape, device=device, dtype=torch.float32)
        if reliability.dim() != 4:
            raise ValueError('reliability maps must be 4D tensors.')
        if reliability.shape[1] != 1:
            reliability = reliability.mean(dim=1, keepdim=True)
        if reliability.shape[-2:] != feat_shape:
            reliability = F.interpolate(
                reliability.float(),
                size=feat_shape,
                mode='bilinear',
                align_corners=False)
        reliability = reliability.float().clamp_min(0)
        has_support = reliability.sum(dim=(2, 3), keepdim=True) > self.eps
        return torch.where(
            has_support, reliability, torch.ones_like(reliability))

    def _reliable_global_pool(self, feat: Tensor,
                              reliability: Tensor) -> Tensor:
        if reliability.shape[0] == 1 and feat.shape[0] != 1:
            reliability = reliability.expand(feat.shape[0], -1, -1, -1)
        numerator = (feat * reliability).sum(dim=(2, 3))
        denominator = reliability.sum(dim=(2, 3)).clamp_min(self.eps)
        return numerator / denominator

    def _routing_losses(
            self,
            match_losses: Sequence[Tensor],
            assignments: Sequence[Tensor],
            hard_assignments: Sequence[Tensor],
            balance_probabilities: Sequence[Tensor],
            match_mask: Tensor,
            balance_mask: Tensor) -> Dict[str, Tensor]:
        losses: Dict[str, Tensor] = {}
        valid_classes = match_mask.to(match_losses[0])
        valid_count = valid_classes.sum()
        zero = sum(item.sum() for item in match_losses) * 0.0

        if self.loss_match_weight > 0:
            if bool((valid_count > 0).item()):
                match = torch.stack(match_losses, dim=1)
                denominator = valid_count
                if self.normalize_match_by_num_scales:
                    denominator = (
                        denominator *
                        match.new_tensor(float(len(match_losses))))
                match = ((match * valid_classes[:, None]).sum() /
                          denominator.clamp_min(1.0))
            else:
                match = zero
            losses['loss_guard_prompt_match'] = (
                match * self._match_loss_weight())

        if self.loss_balance_weight > 0:
            balance_classes = balance_mask.to(match_losses[0])
            balance_count = balance_classes.sum()
            if (bool((balance_count > 0).item()) and
                    self.num_prompts > 1):
                if self.per_scale_balance:
                    balance = self._per_scale_switch_balance(
                        hard_assignments,
                        balance_probabilities,
                        balance_classes,
                        balance_count)
                else:
                    balance = self._joint_entropy_balance(
                        assignments, balance_classes, balance_count)
            else:
                balance = zero
            losses['loss_guard_prompt_balance'] = (
                balance * self._balance_loss_weight())
        return losses

    def _joint_entropy_balance(
            self,
            assignments: Sequence[Tensor],
            valid_classes: Tensor,
            valid_count: Tensor) -> Tensor:
        stacked = torch.stack(assignments, dim=1)
        mask = valid_classes[:, None, :, None]
        assignment_count = valid_count * len(assignments)
        usage = (stacked * mask).sum(dim=(0, 1, 2))
        usage = usage / assignment_count.clamp_min(1.0)
        usage = (usage + self.eps) / (
            usage.sum() + self.num_prompts * self.eps)
        entropy = -(usage * usage.log()).sum()
        entropy = entropy / torch.log(
            entropy.new_tensor(float(self.num_prompts)))
        max_active = min(
            self.num_prompts,
            max(int(assignment_count.detach().item()), 1))
        max_entropy = (
            0.0 if max_active <= 1 else
            float(torch.log(torch.tensor(float(max_active))) /
                  torch.log(torch.tensor(float(self.num_prompts)))))
        effective_floor = min(self.routing_entropy_floor, max_entropy)
        return F.relu(
            entropy.new_tensor(effective_floor) - entropy).square()

    def _per_scale_switch_balance(
            self,
            hard_assignments: Sequence[Tensor],
            balance_probabilities: Sequence[Tensor],
            valid_classes: Tensor,
            valid_count: Tensor) -> Tensor:
        mask = valid_classes.unsqueeze(-1)
        scale_losses = []
        for hard_assignment, probability in zip(
                hard_assignments, balance_probabilities):
            hard_usage = (hard_assignment * mask).sum(dim=(0, 1))
            hard_usage = hard_usage / valid_count.clamp_min(1.0)
            soft_usage = (probability * mask).sum(dim=(0, 1))
            soft_usage = soft_usage / valid_count.clamp_min(1.0)
            scale_losses.append(
                self.num_prompts *
                (hard_usage.detach() * soft_usage).sum())
        return torch.stack(scale_losses).mean()

    def _match_loss_weight(self) -> float:
        step = int(self._routing_step.item())
        if step < self.loss_match_start_iter:
            return 0.0
        if self.loss_match_warmup_iters <= 0:
            return self.loss_match_weight
        progress = min(
            float(step - self.loss_match_start_iter) /
            float(self.loss_match_warmup_iters),
            1.0)
        return self.loss_match_weight * progress

    def _balance_loss_weight(self) -> float:
        if self.loss_balance_warmup_iters <= 0:
            return self.loss_balance_weight
        progress = min(
            float(self._routing_step.item()) /
            float(self.loss_balance_warmup_iters),
            1.0)
        return (
            self.loss_balance_warmup_weight +
            (self.loss_balance_weight -
             self.loss_balance_warmup_weight) * progress)

    def _update_routing_stats(
            self,
            selected_indices: Sequence[Tensor],
            route_mask: Tensor,
            selected_margins: Optional[Sequence[Tensor]] = None,
            normalized_keys: Optional[Tensor] = None) -> Dict[str, Tensor]:
        usages = []
        entropies = []
        max_shares = []
        active_counts = []
        metrics: Dict[str, Tensor] = {}
        for scale_idx, indices in enumerate(selected_indices):
            selected = indices.masked_select(route_mask)
            if selected.numel() == 0:
                usage = self.keys.detach().new_zeros(self.num_prompts)
                entropy = usage.new_zeros(())
                max_share = usage.new_zeros(())
                active_keys = usage.new_zeros(())
            else:
                counts = torch.bincount(
                    selected, minlength=self.num_prompts).float()
                usage = counts / counts.sum().clamp_min(1.0)
                nonzero = usage > 0
                entropy = -(usage[nonzero] * usage[nonzero].log()).sum()
                if self.num_prompts > 1:
                    entropy = entropy / torch.log(
                        entropy.new_tensor(float(self.num_prompts)))
                max_share = usage.max()
                active_keys = nonzero.sum().float()

            scale_name = f'p{scale_idx + 3}'
            usages.append(usage)
            entropies.append(entropy)
            max_shares.append(max_share)
            active_counts.append(active_keys)
            metrics.update({
                f'guard_prompt_routing_entropy_{scale_name}':
                    entropy.detach(),
                f'guard_prompt_max_key_share_{scale_name}':
                    max_share.detach(),
                f'guard_prompt_active_keys_{scale_name}':
                    active_keys.detach(),
            })

        usage_tensor = torch.stack(usages)
        entropy_tensor = torch.stack(entropies)
        max_share_tensor = torch.stack(max_shares)
        active_tensor = torch.stack(active_counts)
        self.last_routing_usage = usage_tensor.detach()
        self.last_routing_entropy = entropy_tensor.detach()
        self.last_max_key_share = max_share_tensor.detach()
        self.last_active_keys = active_tensor.detach()

        # Preserve the old aggregate names, but make them expose the worst
        # scale so a collapsed route can no longer be hidden by other scales.
        metrics.update({
            'guard_prompt_routing_entropy':
                entropy_tensor.min().detach(),
            'guard_prompt_max_key_share':
                max_share_tensor.max().detach(),
            'guard_prompt_active_keys':
                active_tensor.min().detach(),
        })
        if self.collect_key_stats and normalized_keys is not None:
            metrics.update(self._key_geometry_metrics(normalized_keys))
            if selected_margins is not None:
                stacked_margins = torch.stack(selected_margins, dim=1)
                margin_mask = route_mask[:, None].expand_as(stacked_margins)
                if bool(margin_mask.any().item()):
                    margin_mean = stacked_margins.masked_select(
                        margin_mask).float().mean()
                else:
                    margin_mean = stacked_margins.sum() * 0.0
                metrics['key_top1_top2_margin'] = margin_mean.detach()
        return metrics

    def _key_geometry_metrics(
            self, normalized_keys: Tensor) -> Dict[str, Tensor]:
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
            keys = normalized_keys.detach().to(dtype=torch.float32)
            gram = torch.mm(keys, keys.t())
            if self.num_prompts <= 1:
                mean_abs = gram.new_zeros(())
                max_abs = gram.new_zeros(())
            else:
                off_diagonal = ~torch.eye(
                    self.num_prompts, device=gram.device, dtype=torch.bool)
                similarities = gram.masked_select(off_diagonal).abs()
                mean_abs = similarities.mean()
                max_abs = similarities.max()

            eigenvalues = torch.linalg.eigvalsh(gram).clamp_min(0.0)
            probabilities = eigenvalues / eigenvalues.sum().clamp_min(
                self.eps)
            entropy = -(
                probabilities *
                probabilities.clamp_min(self.eps).log()).sum()
            effective_rank = entropy.exp()
            prompt_norm = self.prompts.detach().to(
                dtype=torch.float32).norm(dim=-1).mean()

        return {
            'key_cos_mean_abs': mean_abs,
            'key_cos_max_abs': max_abs,
            'key_effective_rank': effective_rank,
            'prompt_token_norm_mean': prompt_norm,
        }

    def _orthogonal_loss(self) -> Tensor:
        if self.num_prompts <= 1:
            return self.prompts.sum() * 0.0
        prompts = F.normalize(
            self.prompts.flatten(1), dim=-1, eps=self.eps)
        similarity = prompts @ prompts.t()
        off_diagonal = ~torch.eye(
            self.num_prompts, device=similarity.device, dtype=torch.bool)
        return similarity.masked_select(off_diagonal).abs().sum() / (
            self.num_prompts * (self.num_prompts - 1))


# Preserve the original public name and state-dict paths for compatibility.
MultiScalePromptBank = SemanticallyGuardedClassConditionedPrompting


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
                 pad_value: str = '',
                 training_use_cache: bool = False,
                 prompt_bank_cfg: Optional[ConfigType] = None,
                 semantic_fusion_cfg: Optional[ConfigType] = None,
                 init_cfg: OptMultiConfig = None) -> None:

        super().__init__(init_cfg=init_cfg)

        self.frozen_modules = frozen_modules
        self.training_use_cache = training_use_cache
        self.add_mask = add_mask
        self.pad_value = pad_value
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        clip_config = CLIPTextConfig.from_pretrained(model_name,
                                                     attention_dropout=dropout)
        self.model = CLIPTP.from_pretrained(model_name, config=clip_config)
        self.prompt_bank = None
        self.semantic_fusion = None
        self._prompt_bank_losses = {}
        self._last_base_text_features = None
        self.semantic_anchor_texts = ()
        self.semantic_anchor_embed_path = None
        self.semantic_anchor_weight = 0.0
        self.semantic_anchor_warmup_iters = 0
        self.semantic_anchor_sample_size = 0
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
                self.semantic_anchor_sample_size = int(
                    prompt_bank_cfg.pop('semantic_anchor_sample_size', 0))
                prompt_bank_cfg.setdefault('text_hidden_size',
                                           clip_config.hidden_size)
                self.prompt_bank = MultiScalePromptBank(**prompt_bank_cfg)
        if semantic_fusion_cfg is not None:
            semantic_fusion_cfg = dict(semantic_fusion_cfg)
            enabled = semantic_fusion_cfg.pop('enabled', True)
            if enabled:
                self.semantic_fusion = SemanticBoundedSphericalFusion(
                    **semantic_fusion_cfg)
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
                reliability_maps: Optional[Sequence[Tensor]] = None,
                base_text_features: Optional[Tensor] = None,
                positive_class_mask: Optional[Tensor] = None) -> Tensor:
        num_per_batch = [len(t) for t in text]
        assert max(num_per_batch) == min(num_per_batch), (
            'number of sequences not equal in batch')
        batch_size = len(text)
        num_classes = num_per_batch[0]
        flat_texts = list(itertools.chain(*text))
        valid_text_mask = torch.tensor(
            [x != self.pad_value for x in flat_texts],
            device=self.model.device,
            dtype=torch.bool).reshape(batch_size, num_classes)
        text_inputs = self.tokenizer(
            text=flat_texts, return_tensors='pt', padding=True)
        text_inputs = text_inputs.to(device=self.model.device)

        self._prompt_bank_losses = {}
        self._last_base_text_features = None
        if self.has_prompt_bank and image_features is not None:
            if base_text_features is None:
                with torch.no_grad():
                    base_outputs = self.model(**text_inputs)
                    base_text_features = base_outputs.text_embeds
                base_text_features = F.normalize(
                    base_text_features, dim=-1).reshape(
                        batch_size, num_classes, -1)
            else:
                if base_text_features.shape[:2] != (
                        batch_size, num_classes):
                    raise ValueError(
                        'base_text_features must have shape [B, N, D].')
                base_text_features = F.normalize(
                    base_text_features.detach(), dim=-1)
            self._last_base_text_features = base_text_features.detach()
            prompt_tokens, self._prompt_bank_losses = self.prompt_bank(
                image_features,
                reliability_maps,
                class_embeddings=base_text_features,
                text_mask=valid_text_mask,
                positive_class_mask=positive_class_mask)
            txt_feats = self._forward_prompted_text(
                text_inputs, prompt_tokens, num_classes)
            txt_feats = F.normalize(txt_feats, dim=-1).reshape(
                batch_size, num_classes, -1)
            self._add_semantic_anchor_loss(
                image_features, reliability_maps)
            if self.semantic_fusion is not None:
                txt_feats = self.semantic_fusion(
                    base_text_features, txt_feats, valid_text_mask)
                self._prompt_bank_losses.update({
                    f'guard_prompt_{name}': value
                    for name, value in self.semantic_fusion.last_stats.items()
                })
        elif len(self.frozen_modules) > 0:
            with torch.no_grad():
                txt_outputs = self.model(**text_inputs)
                txt_feats = txt_outputs.text_embeds
        else:
            txt_outputs = self.model(**text_inputs)
            txt_feats = txt_outputs.text_embeds

        if txt_feats.dim() == 2:
            txt_feats = F.normalize(txt_feats, dim=-1).reshape(
                batch_size, num_classes, -1)
        else:
            txt_feats = F.normalize(txt_feats, dim=-1)
        if self.add_mask:
            text_mask = valid_text_mask.to(txt_feats)
        else:
            text_mask = None
        return txt_feats, text_mask

    def get_last_base_text_features(self) -> Optional[Tensor]:
        """Return the unprompted text used by the latest prompted forward."""
        return self._last_base_text_features

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
        if prompt_tokens.dim() not in (3, 4):
            raise ValueError(
                'prompt_tokens must have shape [B, P, D] or [B, N, P, D].')
        prompt_len = prompt_tokens.shape[-2]
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
            class_idx = row % num_classes
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
                (prompt_tokens[image_idx, class_idx]
                 if prompt_tokens.dim() == 4 else
                 prompt_tokens[image_idx]),
                class_embeds,
            ],
                                   dim=0)
            seq_len = seq_embeds.shape[0]
            combined_embeds[row, :seq_len] = seq_embeds
            prompted_attention[row, :seq_len] = 1
            eos_positions[row] = seq_len - 1

        return self._encode_prompted_embeddings(
            combined_embeds, prompted_attention, eos_positions)

    def _add_semantic_anchor_loss(
            self, image_features: Sequence[Tensor],
            reliability_maps: Optional[Sequence[Tensor]]) -> None:
        if (not self.training or self.semantic_anchor_weight <= 0 or
                not self.semantic_anchor_texts):
            return
        original_embeddings = self._get_semantic_anchor_embeddings()
        anchor_indices = torch.arange(len(self.semantic_anchor_texts))
        if (0 < self.semantic_anchor_sample_size <
                len(self.semantic_anchor_texts)):
            anchor_indices = torch.randperm(
                len(self.semantic_anchor_texts))[
                    :self.semantic_anchor_sample_size]
        anchor_texts_once = [
            self.semantic_anchor_texts[int(index)]
            for index in anchor_indices
        ]
        original_embeddings = original_embeddings[anchor_indices]
        anchor_count = len(anchor_texts_once)
        batch_size = image_features[0].shape[0]
        class_embeddings = original_embeddings.to(
            device=image_features[0].device).unsqueeze(0).expand(
                batch_size, -1, -1)
        anchor_mask = torch.ones(
            batch_size,
            anchor_count,
            device=class_embeddings.device,
            dtype=torch.bool)
        prompt_tokens, _ = self.prompt_bank(
            image_features,
            reliability_maps,
            class_embeddings=class_embeddings,
            text_mask=anchor_mask,
            positive_class_mask=None,
            compute_losses=False,
            routing_grad=False,
            update_stats=False)

        anchor_texts = anchor_texts_once * batch_size
        anchor_inputs = self.tokenizer(
            text=anchor_texts, return_tensors='pt', padding=True)
        anchor_inputs = anchor_inputs.to(device=self.model.device)
        prompted_embeddings = self._forward_prompted_text(
            anchor_inputs, prompt_tokens, anchor_count)
        prompted_embeddings = F.normalize(prompted_embeddings, dim=-1)
        prompted_embeddings = prompted_embeddings.reshape(
            batch_size, anchor_count, -1)
        original_embeddings = original_embeddings.to(
            device=prompted_embeddings.device,
            dtype=prompted_embeddings.dtype)
        anchor_loss = 1.0 - (prompted_embeddings *
                             original_embeddings.unsqueeze(0)).sum(dim=-1)
        self._prompt_bank_losses['loss_guard_prompt_anchor'] = (
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
            reliability_maps: Optional[Sequence[Tensor]] = None,
            base_text_features: Optional[Tensor] = None,
            positive_class_mask: Optional[Tensor] = None) -> Tensor:
        assert self.with_text_model, (
            'forward_prompted_text() requires a text model')
        if not self.has_text_prompt_bank:
            raise RuntimeError('text prompt bank is not enabled.')
        return self.text_model(
            text,
            image_features=image_features,
            reliability_maps=reliability_maps,
            base_text_features=base_text_features,
            positive_class_mask=positive_class_mask)

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

    def get_last_base_text_features(self) -> Optional[Tensor]:
        """Return the current unprompted text features for CSPR."""
        if not self.has_text_prompt_bank:
            return None
        return self.text_model.get_last_base_text_features()


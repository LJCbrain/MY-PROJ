"""Frozen RemoteCLIP ViT-L/14 image backbone for DIORv3.1.

Only the image feature extractor is replaced.  Intermediate spatial tokens are
resized to the stride-8/16/32 interfaces consumed by the unchanged DIORv2.17
neck.  The frozen ViT is kept outside ``state_dict`` and the optimizer so EMA
and training checkpoints do not duplicate its 1.7 GB local checkpoint.
"""

import math
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmyolo.registry import MODELS

from ..layers.remoteclip_teacher import load_remoteclip_visual


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
_SHARED_VISUAL_MODELS = {}


def _clear_shared_visual_models():
    """Test helper; training keeps one frozen model per process/device/dtype."""
    _SHARED_VISUAL_MODELS.clear()


class _SharedVisualCache:
    """Share one immutable tower between the training model and EMA copy."""

    def __init__(self, checkpoint, precision):
        self.checkpoint = str(checkpoint)
        self.precision = precision

    def __deepcopy__(self, memo):
        result = type(self)(self.checkpoint, self.precision)
        memo[id(self)] = result
        return result

    def get(self, device):
        dtype = (torch.float16
                 if self.precision == 'fp16' and device.type == 'cuda'
                 else torch.float32)
        key = (str(Path(self.checkpoint).resolve()), str(device), str(dtype))
        if key not in _SHARED_VISUAL_MODELS:
            if not Path(self.checkpoint).is_file():
                raise FileNotFoundError(
                    'RemoteCLIP ViT backbone needs the existing full checkpoint: '
                    '{}.'.format(self.checkpoint))
            print('Loading shared frozen RemoteCLIP ViT-L/14 backbone: {}'.format(
                self.checkpoint), flush=True)
            model = load_remoteclip_visual(self.checkpoint)
            model.to(device=device, dtype=dtype).eval().requires_grad_(False)
            _SHARED_VISUAL_MODELS[key] = model
        return _SHARED_VISUAL_MODELS[key]


@MODELS.register_module()
class RemoteCLIPViTL14Backbone(BaseModule):
    """Return RemoteCLIP ViT layers 8/16/24 as YOLO P3/P4/P5 inputs.

    Input must be RGB in [0, 1].  DIORv2.17's data preprocessor already
    performs BGR-to-RGB conversion and division by 255.  Position embeddings
    are bicubically interpolated, so the frozen ViT can process the existing
    640/800-pixel training and validation images without crop classification.
    """

    def __init__(self, checkpoint='pretrained_models/RemoteCLIP-ViT-L-14.pt',
                 out_layers=(8, 16, 24), out_channels=(256, 512, 512),
                 output_strides=(8, 16, 32), precision='fp16',
                 visual_batch_size=1, adapter_norm=True, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.out_layers = tuple(int(value) for value in out_layers)
        self.output_strides = tuple(int(value) for value in output_strides)
        out_channels = tuple(int(value) for value in out_channels)
        if (len(self.out_layers) != 3 or tuple(sorted(set(self.out_layers))) != self.out_layers
                or self.out_layers[0] < 1 or self.out_layers[-1] > 24):
            raise ValueError('out_layers must be three increasing values in [1, 24].')
        if (len(out_channels) != 3 or len(self.output_strides) != 3
                or any(value <= 0 for value in out_channels + self.output_strides)):
            raise ValueError('out_channels/output_strides need three positive values.')
        if precision not in ('fp16', 'fp32') or int(visual_batch_size) < 1:
            raise ValueError('precision must be fp16/fp32 and visual_batch_size positive.')
        self.visual_batch_size = int(visual_batch_size)
        self._visual_cache = _SharedVisualCache(checkpoint, precision)
        adapters = []
        for channels in out_channels:
            layers = [nn.Conv2d(1024, int(channels), 1, bias=not adapter_norm)]
            if adapter_norm:
                layers.append(nn.BatchNorm2d(int(channels), eps=1e-3, momentum=.03))
            adapters.append(nn.Sequential(*layers))
        self.adapters = nn.ModuleList(adapters)
        self.register_buffer('clip_mean', torch.tensor(CLIP_MEAN).view(1, 3, 1, 1),
                             persistent=False)
        self.register_buffer('clip_std', torch.tensor(CLIP_STD).view(1, 3, 1, 1),
                             persistent=False)

    @staticmethod
    def _position_embedding(embeddings, grid_height, grid_width, dtype, device):
        weight = embeddings.position_embedding.weight.to(device=device, dtype=dtype)
        class_position = weight[:1]
        spatial = weight[1:]
        base = int(round(math.sqrt(len(spatial))))
        if base * base != len(spatial):
            raise ValueError('RemoteCLIP ViT positional embedding is not a square grid.')
        spatial = spatial.reshape(1, base, base, -1).permute(0, 3, 1, 2)
        if (grid_height, grid_width) != (base, base):
            spatial = F.interpolate(spatial.float(), (grid_height, grid_width),
                                    mode='bicubic', align_corners=False).to(dtype)
        channels = spatial.shape[1]
        spatial = spatial.permute(0, 2, 3, 1).reshape(-1, channels)
        return torch.cat((class_position, spatial), dim=0).unsqueeze(0)

    def _encode_chunk(self, pixels, output_sizes):
        model = self._visual_cache.get(pixels.device)
        dtype = next(model.parameters()).dtype
        pixels = pixels.to(dtype=dtype)
        vision = model.vision_model
        embeddings = vision.embeddings
        patches = embeddings.patch_embedding(pixels)
        batch_size, hidden_size, grid_height, grid_width = patches.shape
        patches = patches.flatten(2).transpose(1, 2)
        class_token = embeddings.class_embedding.to(patches).reshape(1, 1, -1)
        class_token = class_token.expand(batch_size, -1, -1)
        hidden = torch.cat((class_token, patches), dim=1)
        hidden = hidden + self._position_embedding(
            embeddings, grid_height, grid_width, hidden.dtype, hidden.device)
        hidden = vision.pre_layrnorm(hidden)
        outputs = []
        wanted = set(self.out_layers)
        for index, layer in enumerate(vision.encoder.layers, start=1):
            hidden = layer(hidden, attention_mask=None,
                           causal_attention_mask=None,
                           output_attentions=False)[0]
            if index in wanted:
                tokens = vision.post_layernorm(hidden[:, 1:]).float()
                feature = tokens.reshape(
                    batch_size, grid_height, grid_width, hidden_size).permute(0, 3, 1, 2)
                level = self.out_layers.index(index)
                outputs.append(F.interpolate(
                    feature, output_sizes[level], mode='bilinear', align_corners=False))
        if len(outputs) != 3:
            raise RuntimeError('Failed to capture all configured RemoteCLIP layers.')
        return outputs

    def forward(self, image) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError('RemoteCLIPViTL14Backbone expects [B, 3, H, W] RGB input.')
        height, width = image.shape[-2:]
        output_sizes = [
            (max(1, math.ceil(height / stride)),
             max(1, math.ceil(width / stride)))
            for stride in self.output_strides
        ]
        pixels = image.float()
        pixels = ((pixels - self.clip_mean.to(pixels)) /
                  self.clip_std.to(pixels))
        per_level = [[], [], []]
        # A 640/800 image produces many ViT tokens.  Sequential frozen chunks
        # cap attention memory while the unchanged detector still trains in its
        # original batch size.
        with torch.no_grad():
            for start in range(0, len(pixels), self.visual_batch_size):
                outputs = self._encode_chunk(
                    pixels[start:start + self.visual_batch_size], output_sizes)
                for level, feature in enumerate(outputs):
                    per_level[level].append(feature)
        features = [torch.cat(parts, dim=0) for parts in per_level]
        return tuple(adapter(feature) for adapter, feature in zip(self.adapters, features))

"""Frozen, training-only RemoteCLIP ViT-L/14 visual teacher."""

import inspect
from pathlib import Path

import torch
from transformers import CLIPVisionConfig, CLIPVisionModelWithProjection


def load_remoteclip_visual(checkpoint):
    """Map the original OpenCLIP checkpoint into a strictly loaded HF tower.

    Matches the official RemoteCLIP ViT-L-14 GELU inference recipe. No text
    weights, model downloads, or detector weights are used by this teacher.
    """
    if not Path(checkpoint).is_file():
        raise FileNotFoundError(
            f'RemoteCLIP visual teacher needs the original full .pt: {checkpoint}. '
            'The text-only export and detector initialization are insufficient.')
    options = dict(map_location='cpu')
    if 'weights_only' in inspect.signature(torch.load).parameters:
        options['weights_only'] = True
    source = torch.load(checkpoint, **options)
    source = source.get('state_dict', source)
    if all(key.startswith('module.') for key in source):
        source = {key[7:]: value for key, value in source.items()}
    vision = {key[7:]: value for key, value in source.items()
              if key.startswith('visual.')}
    del source
    state = {}

    def move(old, new):
        state['vision_model.' + new] = vision.pop(old).float().contiguous()

    for old, new in (
            ('class_embedding', 'embeddings.class_embedding'),
            ('positional_embedding', 'embeddings.position_embedding.weight'),
            ('conv1.weight', 'embeddings.patch_embedding.weight'),
            ('ln_pre.weight', 'pre_layrnorm.weight'),
            ('ln_pre.bias', 'pre_layrnorm.bias'),
            ('ln_post.weight', 'post_layernorm.weight'),
            ('ln_post.bias', 'post_layernorm.bias')):
        move(old, new)
    state['visual_projection.weight'] = vision.pop('proj').t().float().contiguous()
    for index in range(24):
        old = f'transformer.resblocks.{index}.'
        new = f'encoder.layers.{index}.'
        for suffix in ('weight', 'bias'):
            qkv = vision.pop(old + 'attn.in_proj_' + suffix)
            if qkv.shape[0] != 3072:
                raise ValueError('Expected RemoteCLIP ViT-L/14 visual QKV width 3072.')
            for name, tensor in zip(('q', 'k', 'v'), qkv.chunk(3, dim=0)):
                state['vision_model.' + new + f'self_attn.{name}_proj.{suffix}'] = (
                    tensor.float().contiguous())
            for src, dst in (
                    ('attn.out_proj', 'self_attn.out_proj'),
                    ('ln_1', 'layer_norm1'), ('ln_2', 'layer_norm2'),
                    ('mlp.c_fc', 'mlp.fc1'), ('mlp.c_proj', 'mlp.fc2')):
                move(old + src + '.' + suffix, new + dst + '.' + suffix)
    if vision:
        raise ValueError(f'Unmapped visual weights: {list(vision)}')
    config = CLIPVisionConfig(
        hidden_size=1024, intermediate_size=4096, num_hidden_layers=24,
        num_attention_heads=16, image_size=224, patch_size=14,
        projection_dim=768, hidden_act='gelu', layer_norm_eps=1e-5)
    # Teacher initialization must not change the student's dropout/sampling RNG.
    with torch.random.fork_rng(devices=[]):
        model = CLIPVisionModelWithProjection(config)
    model.load_state_dict(state, strict=True)
    return model.eval().requires_grad_(False)


class RemoteCLIPTeacherCache:
    """Runtime resource, deliberately outside nn.Module/state_dict/optimizer.

    EMA's deepcopy gets an empty cache even if constructed after the first loss.
    Evaluation never calls get(), so no teacher file is needed for inference.
    """
    def __init__(self, checkpoint, precision='fp16'):
        self.checkpoint = checkpoint
        self.precision = precision
        self.model = None

    def __deepcopy__(self, memo):
        result = type(self)(self.checkpoint, self.precision)
        memo[id(self)] = result
        return result

    def get(self, device):
        if self.model is None:
            print(f'Loading frozen RemoteCLIP visual teacher: {self.checkpoint}',
                  flush=True)
            self.model = load_remoteclip_visual(self.checkpoint)
        dtype = (torch.float16 if self.precision == 'fp16' and device.type == 'cuda'
                 else torch.float32)
        self.model.to(device=device, dtype=dtype).eval().requires_grad_(False)
        return self.model

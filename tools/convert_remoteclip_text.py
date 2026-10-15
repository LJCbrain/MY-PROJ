"""Prepare the frozen RemoteCLIP text model used by DIORv2.18.

Run from the repository root with the training Python environment. All inputs
are local; the original checkpoints are left intact. RemoteCLIP's official
OpenCLIP ViT-L-14 inference uses GELU (not the HF OpenAI model's QuickGELU):
https://github.com/ChenDelong1999/RemoteCLIP#load-remoteclip
"""

import argparse
from pathlib import Path

import torch
from transformers import (AutoTokenizer, CLIPTextConfig,
                          CLIPTextModelWithProjection)


def convert_text_state_dict(source):
    """Map OpenCLIP ViT-L/14 text weights into HF, checking every text key."""
    source = dict(source)
    target = {}

    def move(old, new):
        target[new] = source.pop(old).float().contiguous()

    move('token_embedding.weight',
         'text_model.embeddings.token_embedding.weight')
    move('positional_embedding',
         'text_model.embeddings.position_embedding.weight')
    move('ln_final.weight', 'text_model.final_layer_norm.weight')
    move('ln_final.bias', 'text_model.final_layer_norm.bias')
    target['text_projection.weight'] = source.pop(
        'text_projection').t().float().contiguous()
    for index in range(12):
        old = f'transformer.resblocks.{index}.'
        new = f'text_model.encoder.layers.{index}.'
        for suffix in ('weight', 'bias'):
            qkv = source.pop(old + 'attn.in_proj_' + suffix)
            if qkv.shape[0] != 3 * 768:
                raise ValueError(f'Unexpected QKV shape: {qkv.shape}')
            for name, tensor in zip(('q', 'k', 'v'), qkv.chunk(3, dim=0)):
                target[new + f'self_attn.{name}_proj.{suffix}'] = (
                    tensor.float().contiguous())
            for src, dst in (
                    ('attn.out_proj', 'self_attn.out_proj'),
                    ('ln_1', 'layer_norm1'), ('ln_2', 'layer_norm2'),
                    ('mlp.c_fc', 'mlp.fc1'), ('mlp.c_proj', 'mlp.fc2')):
                move(old + src + '.' + suffix, new + dst + '.' + suffix)
    unexpected = [key for key in source
                  if not key.startswith('visual.') and key != 'logit_scale']
    if unexpected:
        raise ValueError(f'Unmapped RemoteCLIP keys: {unexpected}')
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',
                        default='pretrained_models/RemoteCLIP-ViT-L-14.pt')
    parser.add_argument('--tokenizer',
                        default='pretrained_models/clip-vit-large-patch14-336')
    parser.add_argument('--output',
                        default='pretrained_models/remoteclip-vit-large-patch14-text')
    parser.add_argument(
        '--detector-checkpoint',
        default='weights/yolo_world_v2_l_clip_large_o365v1_goldg_pretrain_800ft-9df82e55.pth')
    parser.add_argument('--detector-output',
                        default='weights/yolo_world_v2_l_remoteclip_text_init.pth')
    args = parser.parse_args()
    if Path(args.detector_checkpoint).resolve() == Path(args.detector_output).resolve():
        raise ValueError('Detector output must not overwrite the original checkpoint.')

    source = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    state = convert_text_state_dict(source)
    del source
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    config = CLIPTextConfig(
        vocab_size=49408, hidden_size=768, intermediate_size=3072,
        projection_dim=768, num_hidden_layers=12, num_attention_heads=12,
        max_position_embeddings=77, hidden_act='gelu', layer_norm_eps=1e-5,
        attention_dropout=0.0, bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
    model = CLIPTextModelWithProjection(config).eval()
    model.load_state_dict(state, strict=True)
    inputs = tokenizer(['airplane', 'ship', 'storage tank', ''],
                       padding=True, return_tensors='pt')
    with torch.no_grad():
        embeddings = model(**inputs).text_embeds
    if embeddings.shape != (4, 768) or not torch.isfinite(embeddings).all():
        raise ValueError('RemoteCLIP text embedding sanity check failed.')

    # Runner loads this AFTER model construction. Replace its text parameters
    # as well, while preserving every image/neck/head parameter exactly.
    detector = torch.load(args.detector_checkpoint, map_location='cpu',
                          weights_only=False)
    prefix = 'backbone.text_model.model.'
    text_keys = {key for key in detector['state_dict'] if key.startswith(prefix)}
    expected_keys = {prefix + key for key in state}
    if text_keys != expected_keys:
        raise ValueError('Detector text keys differ from the converted text model: '
                         f'{sorted(text_keys ^ expected_keys)}')
    detector['state_dict'].update({prefix + key: value for key, value in state.items()})
    # This is an initialization checkpoint, never an optimizer/EMA resume file.
    detector = dict(meta=dict(
        remoteclip_text_source=str(args.checkpoint),
        detector_source=str(args.detector_checkpoint)),
        state_dict=detector['state_dict'])
    model.save_pretrained(args.output, safe_serialization=True)
    tokenizer.save_pretrained(args.output)
    Path(args.detector_output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(detector, args.detector_output)
    print(f'HF text model: {args.output}')
    print(f'Detector initialization: {args.detector_output}')
    print(f'Validated {len(state)} text tensors; embeddings: {tuple(embeddings.shape)}')


if __name__ == '__main__':
    main()

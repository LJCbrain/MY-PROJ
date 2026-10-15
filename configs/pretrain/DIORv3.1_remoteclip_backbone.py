"""DIORv3.1: DIORv2.17 with only its image backbone replaced.

The already available RemoteCLIP ViT-L/14 is frozen. Layers 8/16/24 are
converted through three 1x1 channel adapters to the exact [256, 512, 512]
interface consumed by the existing 2.17 neck. No new quality branch, Novel
branch, loss, crop-based inference, text-model change, candidate rule, or NMS
rule is used.

The original 2.17 initialization and complete training schedule are inherited.
Compatible pretrained neck/head/text weights transfer; the old YOLO image
backbone keys are skipped and the three new adapters start from initialization.
"""

_base_ = './DIORv2.17best.py'

model = dict(
    backbone=dict(
        image_model=dict(
            _delete_=True,
            type='RemoteCLIPViTL14Backbone',
            checkpoint='pretrained_models/RemoteCLIP-ViT-L-14.pt',
            out_layers=(8, 16, 24),
            out_channels=(256, 512, 512),
            output_strides=(8, 16, 32),
            precision='fp16',
            visual_batch_size=1,
            adapter_norm=True,
        )))

# Keep all data, initialization, losses, OVC/O branch, prompt, neck, head,
# schedule, and inference settings inherited from DIORv2.17best.py.
randomness = dict(seed=179871018)
work_dir = 'work_dirs/DIORv3.1_remoteclip_backbone'

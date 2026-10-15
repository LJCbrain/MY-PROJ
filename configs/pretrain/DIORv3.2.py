"""DIORv3.2: full semantic projection + description relations over DIORv2.17.

Only the opt-in text subclass, full projection and auxiliary loss are new. Keep
2.17's YOLO image backbone, frozen CLIP, prompt bank, O/OVC, WSR, Base-16
image-disjoint split and pretrained initialization. No residual or RNR.

The fixed teacher is loaded from the precomputed matrix on the first TRAINING
forward. With matrix_path=None, encode the manifest once instead. Inference
keeps the projection, but needs neither teacher file. Disabling BOTH the
projection and description loss recovers the original 2.17 computation.
"""

_base_ = './DIORv2.17best.py'

custom_imports = dict(
    allow_failed_imports=False,
    imports=['yolo_world', 'yolo_world.evaluation',
             'yolo_world.models.backbones.description_clip'])

model = dict(
    backbone=dict(
        text_model=dict(
            type='DescriptionRegularizedCLIPLanguageBackbone',
            prompt_bank_cfg=dict(text_topology_weight=0.0),
            semantic_adapter_cfg=dict(enabled=True, bias=True),
            description_regularization_cfg=dict(
                description_path='configs/pretrain/dior_v3_2/aerial_descriptions.json',
                matrix_path='assets/dior_v3_2_description_matrix_v2.pt',
                loss_weight=0.01,
                temperature=0.03,
                topk=3,
                margin_scale=1.0,
                warmup_iters=1000))))

# The inherited text-model group has lr_mult=0.01. The new full projection
# learns at the detector's base LR (0.001), not the frozen-CLIP group's LR.
optim_wrapper = dict(paramwise_cfg=dict(custom_keys={
    'backbone.text_model.semantic_adapter': dict(lr_mult=1.0)}))

# Match the actual DIORv2.17 run: work_dirs/DIORv2.17/20260917_151954.log.
# Runtime seed=179871018; best HM=0.14296774 (14.3%) at iter 25968.
# Keep the baseline RNG/cuDNN policy; a fixed seed is not an AP guarantee.
randomness = dict(seed=179871018)
work_dir = 'work_dirs/DIORv3.2_semantic_full'

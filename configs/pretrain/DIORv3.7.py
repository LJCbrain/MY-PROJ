# DIOR 3.7: Mona joins the FIRST remote-sensing training run.
# Image backbone + 18 C2f Mona adapters + neck/head + added visual modules
# train together. CLIP stays frozen; the new Prompt Bank is trainable.
_base_ = './DIORv2.17best.py'

work_dir = 'work_dirs/DIORv3.7_joint'
load_from = 'weights/yolo_world_v2_l_clip_large_o365v1_goldg_pretrain_800ft-9df82e55.pth'
resume = False

# Match DIORv2.17best.py; keep its other randomness defaults unchanged.
randomness = dict(seed=179871018)

model = dict(
    backbone=dict(
        type='MonaMultiModalYOLOBackbone',
        frozen_stages=-1,
        freeze_text_model=False,
        image_model=dict(
            type='MonaYOLOv8CSPDarknet', freeze_original=False,
            frozen_stages=-1, norm_eval=False,
            mona_cfg=dict(enabled=True, bottleneck_dim=64, dropout=0.1,
                          zero_init=True))))

# Starting hyperparameters for joint training, not measured optimum values.
# Original image backbone/neck/head: 1e-4; Mona and new visual modules: 3e-4;
# Prompt Bank: 5e-5 (keys: 2e-5). CLIP has requires_grad=False.
base_lr = 1e-4
_mona_custom_keys = {
    'neck.visual_adapter': dict(lr_mult=3.0),
    'neck.dual_stream': dict(lr_mult=3.0),
    'neck.wsr': dict(lr_mult=3.0),
    'backbone.text_model.prompt_bank': dict(lr_mult=0.5),
    'backbone.text_model.prompt_bank.keys': dict(lr_mult=0.2, decay_mult=0.0),
    'backbone.text_model.prompt_bank.prompts': dict(lr_mult=0.5, decay_mult=0.0),
    'logit_scale': dict(decay_mult=0.0),
}
for _stage, _num_blocks in enumerate((3, 6, 6, 3), start=1):
    for _block in range(_num_blocks):
        _prefix = f'backbone.image_model.stage{_stage}.1.mona_adapters.{_block}'
        _mona_custom_keys[_prefix] = dict(lr_mult=3.0)
        for _name in ('norm', 'gamma', 'gammax'):
            _mona_custom_keys[f'{_prefix}.{_name}'] = dict(
                lr_mult=3.0, decay_mult=0.0)
optim_wrapper = dict(
    constructor='OVCFineTuneOptimizerConstructor',
    optimizer=dict(lr=base_lr),
    paramwise_cfg=dict(
        _delete_=True, norm_decay_mult=0.0, bias_decay_mult=0.0,
        custom_keys=_mona_custom_keys))

# Use the complete 2.17 first-stage data/loss protocol and actual 800px
# Mosaic pipeline. Retain its pipeline switch at 49772 and its HM selection.
# Warm all groups proportionally, avoiding YOLO's independent bias LR spike.
max_epochs = 50
max_iters = 54100
param_scheduler = [
    dict(type='LinearLR', begin=0, end=1000, by_epoch=False,
         start_factor=0.1, end_factor=1.0),
    dict(type='CosineAnnealingLR', begin=1000, end=max_iters,
         by_epoch=False, T_max=max_iters - 1000, eta_min_ratio=0.01),
]
default_hooks = dict(
    param_scheduler=dict(_delete_=True, type='ParamSchedulerHook'))

custom_hooks = [
    dict(type='MonaJointTrainingHook'),
    dict(type='EMAHook', ema_type='IntegerSafeExpMomentumEMA',
         momentum=0.0001, update_buffers=True, strict_load=False, priority=49),
    dict(type='OVCDiagnosticsHook', interval=50),
    _base_.custom_hooks[2],
]

# New first-stage run (AMP wrapper is already inherited):
# python tools/train.py --config configs/pretrain/DIORv3.7.py --work-dir work_dirs/DIORv3.7_joint
# Do not add --resume or --amp on the first start. Resume only a checkpoint
# produced by this joint-training config. The previous frozen experiment is
# preserved as configs/pretrain/DIORv3.7_frozen.py.

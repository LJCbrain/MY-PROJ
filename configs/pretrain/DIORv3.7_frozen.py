# DIOR 3.7: block-wise Mona inside a completely frozen original YOLO backbone.
# CNN adaptation: one Mona after each C2f internal bottleneck, not the paper's
# MSA/MLP insertion into Swin. With 2.17's depth/width: 3+6+6+3 = 18 adapters.
_base_ = './DIORv3.6.py'

work_dir = 'work_dirs/DIORv3.7_B'
resume = False

model = dict(
    backbone=dict(
        type='MonaMultiModalYOLOBackbone',
        frozen_stages=4,
        freeze_text_model=True,  # existing CLIP and Prompt Bank
        image_model=dict(
            type='MonaYOLOv8CSPDarknet', frozen_stages=4, norm_eval=True,
            mona_cfg=dict(enabled=True, bottleneck_dim=64, dropout=0.1,
                          zero_init=True))),
    bbox_head=dict(head_module=dict(mona_cfg=dict(enabled=False))))

# Smaller updates than 3.6: whole-backbone adapters affect localization and
# all feature scales. These are starting hyperparameters, not a paper recipe
# or evidence that Novel will improve. All original backbone weights are
# frozen; neck/head use 1e-6, existing visual additions use 3e-6, Mona 1e-4.
base_lr = 1e-6
_mona_custom_keys = {
    'neck.visual_adapter': dict(lr_mult=3.0),
    'neck.visual_adapter.adapter.1': dict(lr_mult=3.0, decay_mult=0.0),
    'neck.dual_stream': dict(lr_mult=3.0),
    'neck.wsr': dict(lr_mult=3.0),
    'logit_scale': dict(decay_mult=0.0),
}
for _stage, _num_blocks in enumerate((3, 6, 6, 3), start=1):
    for _block in range(_num_blocks):
        _prefix = f'backbone.image_model.stage{_stage}.1.mona_adapters.{_block}'
        _mona_custom_keys[_prefix] = dict(lr_mult=100.0)
        for _name in ('norm', 'gamma', 'gammax'):
            _mona_custom_keys[f'{_prefix}.{_name}'] = dict(
                lr_mult=100.0, decay_mult=0.0)
optim_wrapper = dict(
    optimizer=dict(lr=base_lr),
    paramwise_cfg=dict(
        _delete_=True, norm_decay_mult=0.0, bias_decay_mult=0.0,
        custom_keys=_mona_custom_keys))

# Retain the same base16 data, 800px close-mosaic pipeline, loss settings,
# 500-step warmup, 10820-step budget, 1082-step validation and HM selection
# as 3.6. Frozen Prompt Bank uses its existing inference behavior; its
# semantic-anchor loss and prompt counter updates are inactive. Routing
# regularizers/statistics may still be logged from detached visual context.
custom_hooks = [
    dict(type='MonaBackboneFineTuneHook', source_iter=25968, freeze_bn=True),
    dict(type='EMAHook', ema_type='IntegerSafeExpMomentumEMA',
         momentum=0.0001, update_buffers=True, strict_load=False, priority=49),
    dict(type='OVCDiagnosticsHook', interval=50),
]

# B: python tools/train.py --config configs/pretrain/DIORv3.7_frozen.py --work-dir work_dirs/DIORv3.7_B
# A: same frozen backbone/LRs/schedule with no Mona:
# python tools/train.py --config configs/pretrain/DIORv3.7_frozen.py --work-dir work_dirs/DIORv3.7_A --cfg-options model.backbone.image_model.mona_cfg.enabled=False
# Initial loading is from the inherited 2.17 best; do not add --resume.

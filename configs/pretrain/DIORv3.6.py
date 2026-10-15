# DIOR 3.6: plan B by default; plan A uses the same trained starting point,
# frozen stages, low learning rates, schedule, data and losses without Mona.
# The supplied .pth path is a checkpoint, rather than a Python config file.
_base_ = './DIORv2.17best.py'

work_dir = 'work_dirs/DIORv3.6_B'
load_from = ('/home/dingyujie/ljc/YOLO-World/work_dirs/DIORv2.17/'
             'best_coco_bbox_mAP_50_HM_iter_25968.pth')
resume = False

# Only P3 classification sees Mona, after the existing dual stream and WSR.
# Regression and PromptBank context continue to read the existing neck output.
model = dict(
    backbone=dict(
        frozen_stages=3,  # stem + stage1/2/3; stage4 remains trainable
        image_model=dict(norm_eval=True)),
    bbox_head=dict(
        head_module=dict(
            mona_cfg=dict(
                enabled=True, active_levels=(0,), bottleneck_dim=64,
                dropout=0.1, zero_init=True))))

# Peak learning rates: Mona 3e-4, trained additions 3e-5, other trainable
# detector weights 1e-5. CLIP remains frozen through the inherited config.
base_lr = 1e-5
optim_wrapper = dict(
    constructor='OVCFineTuneOptimizerConstructor',
    optimizer=dict(lr=base_lr),
    paramwise_cfg=dict(
        _delete_=True,
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
        custom_keys={
            'neck.visual_adapter': dict(lr_mult=3.0),
            'neck.visual_adapter.adapter.1': dict(lr_mult=3.0, decay_mult=0.0),
            'neck.dual_stream': dict(lr_mult=3.0),
            'neck.wsr': dict(lr_mult=3.0),
            'backbone.text_model.prompt_bank': dict(lr_mult=3.0),
            'backbone.text_model.prompt_bank.keys': dict(lr_mult=1.0, decay_mult=0.0),
            'backbone.text_model.prompt_bank.prompts': dict(lr_mult=3.0, decay_mult=0.0),
            'bbox_head.head_module.cls_mona': dict(lr_mult=30.0),
            'bbox_head.head_module.cls_mona.0.norm': dict(lr_mult=30.0, decay_mult=0.0),
            'bbox_head.head_module.cls_mona.0.gamma': dict(lr_mult=30.0, decay_mult=0.0),
            'bbox_head.head_module.cls_mona.0.gammax': dict(lr_mult=30.0, decay_mult=0.0),
            'logit_scale': dict(decay_mult=0.0),
        }))

# Replace the inherited 50-epoch YOLO LR hook, including its large bias warmup.
# All groups preserve their LR ratios during the new 500-step warmup/cosine.
max_epochs = 10
max_iters = 10820  # 10 virtual epochs, each containing 1082 iterations
param_scheduler = [
    dict(type='LinearLR', begin=0, end=500, by_epoch=False,
         start_factor=0.1, end_factor=1.0),
    dict(type='CosineAnnealingLR', begin=500, end=max_iters,
         by_epoch=False, T_max=max_iters - 500, eta_min_ratio=0.01),
]
default_hooks = dict(
    param_scheduler=dict(_delete_=True, type='ParamSchedulerHook'),
    checkpoint=dict(interval=1082, max_keep_ckpts=2))
train_cfg = dict(
    _delete_=True, type='IterBasedTrainLoop', max_iters=max_iters,
    val_interval=1082)

# Start directly with 2.17's 800px pipeline without Mosaic/MixUp.
# Use the actual switch pipeline (including text loading), rather than the
# unused 640px train_pipeline_stage2 alias in the original flattened config.
train_pipeline = _base_.custom_hooks[2].switch_pipeline
train_pipeline_close_mosaic = train_pipeline
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))

# Replace the inherited pipeline switch at iteration 49772.
# EMA starts from checkpoint['state_dict'] (the saved evaluated EMA weights);
# the old optimizer and old EMA trajectory are not resumed.
custom_hooks = [
    dict(type='MonaFineTuneHook', source_iter=25968, freeze_bn=True),
    dict(type='EMAHook', ema_type='IntegerSafeExpMomentumEMA',
         momentum=0.0001, update_buffers=True, strict_load=False, priority=49),
    dict(type='OVCDiagnosticsHook', interval=50),
]

# B (default):
# python tools/train.py --config configs/pretrain/DIORv3.6.py --work-dir work_dirs/DIORv3.6_B
# A (same low-LR adaptation, no Mona):
# python tools/train.py --config configs/pretrain/DIORv3.6.py --work-dir work_dirs/DIORv3.6_A --cfg-options model.bbox_head.head_module.mona_cfg.enabled=False
# Do not add --resume for the first start from 2.17. The inherited AMP wrapper
# is already enabled; tools/train.py's --amp is incompatible with this wrapper.

_base_ = './DIORv2.5.py'

# A longer DIOR fine-tuning schedule needs a clean-image adaptation phase
# before validation and checkpoint selection.  Eight epochs is the midpoint of
# the requested 5--10 epoch window.
max_epochs = 50
close_mosaic_epochs = 4
train_batch_size_per_gpu = 4

# Pure Regional Background Prior control.  DIORv2.5 does not explicitly pass
# `use_caqr`, while the current gate defaults it to True; override the default
# here so no CAQR map is computed or fused.  All Prompt Bank, WSR, data, and
# optimisation settings remain exactly those inherited from DIORv2.5.
model = dict(
    neck=dict(
        fgbg_cfg=dict(
            use_caqr=False)))

# Values in a base config are evaluated before this child overrides its module
# variables, so explicitly rebuild every schedule-dependent entry here.
default_hooks = dict(
    param_scheduler=dict(max_epochs=max_epochs))
custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='IntegerSafeExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        strict_load=False,
        priority=49),
    dict(
        type='mmdet.PipelineSwitchHook',
        switch_epoch=max_epochs - close_mosaic_epochs,
        switch_pipeline=_base_.train_pipeline_close_mosaic),
]
train_cfg = dict(max_epochs=max_epochs)

# This must be overridden in both places: the dataloader controls the actual
# batch, while MMYOLO's optimiser constructor uses batch_size_per_gpu when it
# determines gradient accumulation.
train_dataloader = dict(batch_size=train_batch_size_per_gpu)
optim_wrapper = dict(
    optimizer=dict(batch_size_per_gpu=train_batch_size_per_gpu))

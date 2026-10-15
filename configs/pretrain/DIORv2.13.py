"""V1 semantic reliability + zero-start learnable dual stream + original WSR.

No Regional gate or foreground auxiliary losses. Raw A3 routes P3/C3;
per-image spatial normalization of A3/A4/A5 supplies detached WSR evidence.
This also changes the learning rate relative to 2.12, so it is a combined
experiment rather than an isolated architectural ablation.
"""

_base_ = './DIORv2.12.py'

base_lr = 1e-3
train_batch_size_per_gpu = 4
max_epochs = 50
close_mosaic_epochs = 4

model = dict(
    loss_fgbg_mask_bce_weight=0.0,
    loss_fgbg_mask_dice_weight=0.0,
    loss_fgbg_consistency_weight=0.0,
    neck=dict(
        type='SemanticDualStreamYOLOWorldWSRPAFPN',
        fgbg_cfg=None,
        dual_stream_cfg=dict(
            _delete_=True,
            hidden_channels=64,
            norm_groups=8,
            learnable_scale=True,
            # lambda = 0.1 * tanh(alpha), alpha starts at zero.
            # out_proj has nonzero initialization, so alpha can learn at step 1.
            residual_scale=0.1),
        # S = sigmoid((A - spatial_mean) / max(spatial_std, 1e-3)).
        # R = 1 - (1-S3)(1-0.5*up(S4))(1-0.25*up(S5)).
        # R is a relative support prior, not a foreground probability.
        semantic_reliability_cfg=dict(
            temperature=1.0,
            std_floor=1e-3,
            fusion_weights=(0.5, 0.25))))

train_dataloader = dict(batch_size=train_batch_size_per_gpu)
optim_wrapper = dict(
    accumulative_counts=1,
    optimizer=dict(lr=base_lr, batch_size_per_gpu=train_batch_size_per_gpu),
    paramwise_cfg=dict(custom_keys={
        'neck.dual_stream.residual_alpha': dict(lr_mult=1.0, decay_mult=0.0)}))

randomness = dict(seed=179871018)
train_cfg = dict(
    max_epochs=max_epochs,
    val_interval=4,
    dynamic_intervals=[(max_epochs - close_mosaic_epochs, 1)])
custom_hooks = [
    *_base_.custom_hooks,
    dict(type='SemanticWSRDiagnosticsHook', interval=50),
]

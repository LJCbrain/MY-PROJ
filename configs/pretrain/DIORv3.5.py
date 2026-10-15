# DIOR 3.5: complete 2.17 + a single residual LSK on the P5 classification input.
# P5_cls = P5 + gamma * LSK(P5); regression still reads the original P5.
# Level indices 0/1/2 refer to P3/P4/P5, with channels 256/512/512.
_base_ = './DIORv2.17best.py'

work_dir = 'work_dirs/DIORv3.5'

model = dict(
    bbox_head=dict(
        head_module=dict(
            lsk_cfg=dict(active_levels=(2,), init_scale=1e-2))))

# Only the new per-channel residual scale is exempted from weight decay.
# New LSK convolutions use the inherited AdamW learning rate and decay.
optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys={
            'bbox_head.head_module.cls_lsk.2.residual_scale': dict(decay_mult=0.0),
        }))

# Same pretrained checkpoint, data split, seed (179871018), AMP wrapper,
# 54100 training iterations and HM checkpoint selection as 2.17.
# This is an independent run, not a continuation of the 2.17 best checkpoint.
resume = False

# tools/train.py has a default work-dir that overrides the config: set it explicitly.
# python tools/train.py --config configs/pretrain/DIORv3.5.py --work-dir work_dirs/DIORv3.5
# Do not add --amp: the inherited OVCTrackedAmpOptimWrapper already enables AMP.
# Original-head ablation: --cfg-options model.bbox_head.head_module.lsk_cfg=None
# Use a separate work-dir for that ablation.

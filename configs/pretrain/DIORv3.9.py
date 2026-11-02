# DIOR 3.9, scheme C: base-16 semantic graph before every C2fAttention.
# Train the classifier on Base-16 only; GZSD/ZSD heads use 20/4 classes.
# The neck uses only base names; GZSD/ZSD classifier vocabularies stay 20/4.
_base_ = './DIORv2.17best.py'

work_dir = 'work_dirs/DIORv3.9'
resume = False

model = dict(
    neck_guide_texts=_base_.DIOR_BASE_CLASSES,
    neck=dict(
        dsg_cfg=dict(
            type='DynamicSemanticGraph',
            text_channels=768,
            hidden_channels=256,
            similarity_threshold=0.5,
            similarity_quantile=0.8,
            max_residual_scale=0.1,
            init_scale=0.0,
            novel_indices=())))

optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys={'neck.dsg.gamma': dict(decay_mult=0.0)}))

# Keep 2.17's data, pretrained initialization, schedule, seed and existing
# GuardPrompt/OVC objectives. DSG adds no direction/topology/distillation loss.
# Its self-loops, degree normalization and bounded residual are adaptations
# for C2fAttention; frozen CLIP means the base-16 graph topology is fixed.
# A threshold of 0.5 alone makes the actual pretrained CLIP bank fully
# connected. Use max(0.5, off-diagonal similarity 80th percentile), based
# exclusively on the frozen base-16 text bank, to retain selective edges.
# OVCDiagnosticsHook records the effective threshold and base graph degrees.
#
# Vocabulary-only control B: model.neck.dsg_cfg=None
# Original control A: also set model.neck_guide_texts=None
#
# Use a fresh directory for the corrected training vocabulary:
# python tools/train.py --config configs/pretrain/DIORv3.9.py \
#   --work-dir work_dirs/DIORv3.9_base16

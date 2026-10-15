# DIOR 3.4: complete 2.17 + final text-to-image residual cross-attention.
# F -> F + alpha * Attention(Q=F, K=T0, V=T0) at P3/P4/P5.
# PromptBank continues to consume detached pre-attention neck features.
_base_ = './DIORv2.17best.py'

work_dir = 'work_dirs/DIORv3.4'

model = dict(
    neck=dict(
        text_cross_attn_cfg=dict(
            type='MultiScaleTextCrossAttention',
            in_channels=[256, 512, 512],
            text_channels=768,
            embed_channels=256,
            num_heads=8,
            active_levels=(0, 1, 2),
            init_scale=1e-3,
            attention_dropout=0.0)))

optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys={
            'neck.text_cross_attn.residual_scales': dict(decay_mult=0.0),
        }))

# Inherit 2.17's pretrained initialization, data, losses and training schedule.
# Inherit randomness unchanged: the 2.17 seed is 179871018.
# This does not resume a trained 2.17 checkpoint or its optimizer state.
resume = False

# Ablations:
#   model.neck.text_cross_attn_cfg=None                  -> original 2.17
#   model.neck.text_cross_attn_cfg.active_levels=(0,)    -> P3 only

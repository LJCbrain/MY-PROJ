_base_ = './DIORv2.5.py'

# DIOR-only Prompt-Bank ablation.  Keep the complete visual-prior path from
# DIORv2.5: Regional background prior is the primary foreground estimate,
# CAQR is a bounded auxiliary correction, and WSR consumes the fused prior.
# The frozen CLIP text embedding is sent directly to C2fAttention and the
# detection head; no learned prompt keys or prompt tokens are constructed.

# Regional 主导 + CAQR 辅助 + WSR，无 Prompt Bank
model = dict(
    backbone=dict(
        text_model=dict(prompt_bank_cfg=None)),
    neck=dict(
        # The detector no longer requests post-neck Prompt-Bank context.
        prompt_bank_compatible=False,
        fgbg_cfg=dict(
            # Regional prior remains the full-strength primary estimate.
            # CAQR enters through
            #   1 - (1 - regional) * (1 - beta * caqr)
            # and beta is capped at 0.3 after warm-up.
            use_caqr=True,
            caqr_beta_max=0.3,
            caqr_beta_warmup_iters=1000,
            direct_feature_boost=True)))

# Remove stale Prompt-Bank optimizer rules while preserving the original
# optimizer, bias/norm decay policy, frozen CLIP treatment, and logit scale.
optim_wrapper = dict(
    paramwise_cfg=dict(
        _delete_=True,
        bias_decay_mult=0.0,
        norm_decay_mult=0.0,
        custom_keys={
            'backbone.text_model': dict(lr_mult=0.01),
            'logit_scale': dict(weight_decay=0.0),
        }))

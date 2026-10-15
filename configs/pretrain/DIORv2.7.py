_base_ = './DIORv2.5.py'

# DIOR-only parallel visual-prior experiment.  It inherits from DIORv2.5 so
# the visual-prior-free DIORv2.6 ablation cannot change this experiment's
# losses, WSR, or GuardPrompt settings.  Datasets, evaluators, and the original
# YOLO-World text-aware PAFPN remain inherited.
# No RemoteCLIP teacher, proposal distillation, or additional supervision is
# introduced.
# CAQR 主导 + Regional 辅助的并行结构
model = dict(
    backbone=dict(
        text_model=dict(
            prompt_bank_cfg=dict(
                # Geometry metrics are logging-only.  Their GPU eigvalsh
                # calculation can fail on a nearly degenerate key Gram matrix;
                # disabling it does not alter Prompt Bank routing or losses.
                collect_key_stats=False))),
    neck=dict(
        type='ParallelVisualPriorYOLOWorldWSRPAFPN',
        # Disable the inherited post-PAFPN gate.  The same CAQR-primary and
        # Regional-auxiliary logic now runs on reduced backbone features before
        # any text-guided C2fAttention block.
        fgbg_cfg=None,
        visual_prior_cfg=dict(
            type='CAQRPrimaryRegionalAuxGate',
            in_channels=[256, 512, 512],
            region_grid_size=3,
            background_keep_ratio=0.75,
            attention_heads=[4, 8, 8],
            hidden_ratio=0.25,
            min_hidden=16,
            caqr_init_prob=0.5,
            regional_aux_strength=0.2,
            regional_aux_warmup_iters=1000,
            fusion_weights=(0.5, 0.25),
            # The prior branch only predicts objectness.  Feature enhancement
            # is performed once by the isolated residual fusion below.
            direct_feature_boost=False,
            return_parts=True),
        visual_residual_cfg=dict(
            visual_channels=[256, 512, 512],
            output_channels=[256, 512, 512],
            max_scale=0.1,
            init_scale=0.05,
            warmup_iters=1000,
            # Detection classification cannot reshape the class-agnostic map.
            detach_prior=True)))

# Preserve the validation cadence that this experiment previously inherited
# from DIORv2.6CAQR before that file became the visual-prior-free ablation.
train_cfg = dict(val_interval=3)

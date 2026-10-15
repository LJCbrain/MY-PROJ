# DIOR 3.3 additive revision: complete 2.17 + localization quality.
# Preserve O, its two-view BCE, O-weighted dual stream, and every WSR source.
_base_ = './DIORv2.17best.py'

work_dir = 'work_dirs/DIORv3.3_additive'

model = dict(
    type='YOLOWorldQualityOVCDetector',
    neck=dict(
        type='QualityOVCYOLOWorldWSRPAFPN',
        retain_objectness=True,
        # objectness_reweight=True and fixed dual scale=0.02 are inherited.
        # Rank the existing 32 visual WSR slots by O*(1-S3)*q0_support.
        # 96 semantic + 32 visual + 64 detail + 64 grid remain unchanged.
        # q0_support is a spatial prior; q1 below evaluates an actual box.
        quality_wsr_enabled=True,
        quality_wsr_start_iters=1000,
        quality_wsr_warmup_iters=1000,
        quality_head_cfg=dict(
            type='ClassAgnosticLocalizationHead',
            hidden_channels=64,
            norm_groups=8,
            reg_max=16,
            featmap_strides=(8, 16, 32),
            match_topk=10,
            center_radius=2.5,
            target_scale=4.0)),
    # All original OVC settings, including objectness_weight=0.1, are inherited.
    quality_training_cfg=dict(
        bbox_weight=1.0,
        dfl_weight=0.25,
        q0_weight=0.1,
        q1_weight=0.1,
        warmup_iters=1000),
    # Preserve the original detection score by default. Same-checkpoint
    # evaluation can enable p_YOLO*q1 for both groups without retraining.
    quality_score_enabled=False,
    quality_score_power=1.0)

optim_wrapper = dict(constructor='QualityOVCOptimizerConstructor')

# Use 2.17's original initialization/data/schedule; retrain this revision.
# No changes to backbone/text/prompt bank, main TAL/head, input resolution,
# Base-only split, augmentations, batch size, learning rate, or HM checkpointing.
# New losses train VA + localization head, with main features/boxes detached.
# Independent controls:
#   model.neck.quality_wsr_enabled=False   -> original O WSR ranking
#   model.quality_score_enabled=True      -> final score p_YOLO*q1
# With WSR quality off, score off, and all four localization loss weights
# zero, the original 2.17 computations and losses are recovered.
# proposal_instances preserves original B1/O recall; auxiliary B0/q0 proposals
# are attached separately as quality_proposal_instances (no metric replacement).

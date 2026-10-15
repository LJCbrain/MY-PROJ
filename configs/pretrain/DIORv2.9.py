_base_ = './DIORv2.5.py'

# Prompt-matching-strength ablation.  Keep the DIORv2.5 visual branch and all
# other Guard Prompt settings unchanged so that only matching supervision is
# tested.
model = dict(
    backbone=dict(
        text_model=dict(
            prompt_bank_cfg=dict(
                # Start visual-text prompt matching immediately, but ramp it
                # over the first epoch instead of applying the final weight to
                # strongly augmented early samples.
                loss_match_weight=0.2,
                loss_match_start_iter=0,
                loss_match_warmup_iters=1000))))

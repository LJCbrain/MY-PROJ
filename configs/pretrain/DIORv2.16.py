"""Image-disjoint DIORv2.15 with WSR and text topology enabled.

The inherited DIORv2.17 training split removes every image containing a
Novel-4 instance. The existing fixed 0.02 OVC fusion is intentionally kept so
this experiment measures text topology while retaining the original WSR.
"""

_base_ = './DIORv2.17best.py'

model = dict(
    backbone=dict(
        text_model=dict(
            prompt_bank_cfg=dict(
                text_topology_temperature=0.1,
                text_topology_weight=0.02))),
    neck=dict(wsr_enabled=True))

# Match the DIORv2.17 reference run (20260917_151954.log) exactly.
randomness = dict(seed=179871018)

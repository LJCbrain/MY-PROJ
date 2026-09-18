"""DIORv2.15 ablation with weak support retrieval (WSR) removed."""

_base_ = './DIORv2.17.py'

model = dict(neck=dict(wsr_enabled=False))

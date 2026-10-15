"""DIOR 2.10 control versus E2 dual stream before reliability-guided WSR.

Keep the original Regional feature boost, reliability maps and auxiliary
losses. Fuse post-Regional P3 with pre-text backbone C3, then feed the result
to the original WSR with the detached, pre-fusion Regional reliability map.
WSR retains 128 reliable + 64 detail + 64 grid candidates and its 0.5 log
prior weight. A / (1-A) route only the dual streams, not WSR reliability.
All other model, data and training settings are inherited from 2.10.
"""

_base_ = './DIORv2.10.py'

model = dict(
    neck=dict(
        type='DualStreamYOLOWorldWSRPAFPN',
        dual_stream_cfg=dict(
            hidden_channels=64,
            norm_groups=8,
            residual_scale=0.1)))

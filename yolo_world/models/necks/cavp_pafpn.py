"""Text-independent visual pyramid for CAVP-Full."""

import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from mmyolo.models.necks import YOLOv8PAFPN
from mmyolo.registry import MODELS


@MODELS.register_module()
class CAVPVisualPAFPN(BaseModule):
    def __init__(self, in_channels=(256, 512, 512),
                 out_channels=(128, 256, 256), num_csp_blocks=1,
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg=dict(type='SiLU', inplace=True), init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        if len(in_channels) != 3 or len(out_channels) != 3:
            raise ValueError('CAVP requires three image pyramid levels.')
        self.projections = nn.ModuleList([
            ConvModule(cin, cout, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)
            for cin, cout in zip(in_channels, out_channels)
        ])
        self.pan = YOLOv8PAFPN(
            in_channels=list(out_channels), out_channels=list(out_channels),
            num_csp_blocks=num_csp_blocks, norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, image_features):
        if len(image_features) != len(self.projections):
            raise ValueError('CAVP image pyramid levels do not match.')
        # Enforce the gradient boundary for all callers, not only the detector.
        projected = tuple(proj(feat.detach()) for proj, feat in
                          zip(self.projections, image_features))
        return self.pan(projected)

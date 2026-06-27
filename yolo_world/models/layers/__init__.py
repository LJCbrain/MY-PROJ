# Copyright (c) Tencent Inc. All rights reserved.
# Basic brick modules for PAFPN based on CSPLayers

from .yolo_bricks import (
    CSPLayerWithTwoConv,
    GuideAgnosticCSPLayerWithTwoConv,
    MaxSigmoidAttnBlock,
    MaxSigmoidCSPLayerWithTwoConv,
    ImagePoolingAttentionModule,
    RepConvMaxSigmoidCSPLayerWithTwoConv,
    RepMaxSigmoidCSPLayerWithTwoConv,
    ClassAgnosticForegroundGate,
    WeakSupportRetriever,
)

__all__ = [
    'CSPLayerWithTwoConv',
    'GuideAgnosticCSPLayerWithTwoConv',
    'MaxSigmoidAttnBlock',
    'MaxSigmoidCSPLayerWithTwoConv',
    'RepConvMaxSigmoidCSPLayerWithTwoConv',
    'RepMaxSigmoidCSPLayerWithTwoConv',
    'ImagePoolingAttentionModule',
    'ClassAgnosticForegroundGate',
    'WeakSupportRetriever',
]

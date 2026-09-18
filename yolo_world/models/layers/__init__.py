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
    CAQRPrimaryRegionalAuxGate,
    RegionalBackgroundPriorGate,
    RegionalTextResidual,
    WeakSupportRetriever,
)
from .ema import IntegerSafeExpMomentumEMA

__all__ = [
    'CSPLayerWithTwoConv',
    'GuideAgnosticCSPLayerWithTwoConv',
    'MaxSigmoidAttnBlock',
    'MaxSigmoidCSPLayerWithTwoConv',
    'RepConvMaxSigmoidCSPLayerWithTwoConv',
    'RepMaxSigmoidCSPLayerWithTwoConv',
    'ImagePoolingAttentionModule',
    'ClassAgnosticForegroundGate',
    'CAQRPrimaryRegionalAuxGate',
    'RegionalBackgroundPriorGate',
    'RegionalTextResidual',
    'WeakSupportRetriever',
    'IntegerSafeExpMomentumEMA',
]

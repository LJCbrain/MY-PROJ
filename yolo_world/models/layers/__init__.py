# Copyright (c) Tencent Inc. All rights reserved.
# Basic brick modules for PAFPN based on CSPLayers

from .yolo_bricks import (
    CSPLayerWithTwoConv,
    MaxSigmoidAttnBlock,
    MaxSigmoidCSPLayerWithTwoConv,
    ImagePoolingAttentionModule,
    RepConvMaxSigmoidCSPLayerWithTwoConv,
    RepMaxSigmoidCSPLayerWithTwoConv,
    ClassAgnosticForegroundGate,  # 新增代码：导入类别无关前景门控模块

    )

__all__ = ['CSPLayerWithTwoConv',
           'MaxSigmoidAttnBlock',
           'MaxSigmoidCSPLayerWithTwoConv',
           'RepConvMaxSigmoidCSPLayerWithTwoConv',
           'RepMaxSigmoidCSPLayerWithTwoConv',
           'ImagePoolingAttentionModule',
           'ClassAgnosticForegroundGate',  # 新增代码：导出类别无关前景门控模块
           ]

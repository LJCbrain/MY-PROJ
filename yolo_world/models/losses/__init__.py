# Copyright (c) Tencent Inc. All rights reserved.
from .dynamic_loss import CoVMSELoss  # 新增代码：移除当前未实现的 MultiScaleFGBGLoss 残留导入

__all__ = ['CoVMSELoss']

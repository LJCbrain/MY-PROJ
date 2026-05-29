# Copyright (c) Tencent Inc. All rights reserved.
from .yolo_world import YOLOWorldDetector, SimpleYOLOWorldDetector  # 新增代码：移除不存在 detector 的残留引用
from .yolo_world_image import YOLOWorldImageDetector

__all__ = ['YOLOWorldDetector', 'SimpleYOLOWorldDetector', 'YOLOWorldImageDetector']  # 新增代码：只导出当前真实存在的 detector

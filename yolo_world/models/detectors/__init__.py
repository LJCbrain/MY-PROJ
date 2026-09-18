# Copyright (c) Tencent Inc. All rights reserved.
from .yolo_world import YOLOWorldDetector, SimpleYOLOWorldDetector  # 新增代码：移除不存在 detector 的残留引用
from .yolo_world_image import YOLOWorldImageDetector
from .yolo_world_cavp import YOLOWorldCAVPDetector
from .yolo_world_ovc import YOLOWorldOVCDetector, YOLOWorldOVCPartialDetector

__all__ = ['YOLOWorldDetector', 'SimpleYOLOWorldDetector', 'YOLOWorldImageDetector',
           'YOLOWorldCAVPDetector', 'YOLOWorldOVCDetector', 'YOLOWorldOVCPartialDetector']

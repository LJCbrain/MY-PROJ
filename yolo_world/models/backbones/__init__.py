# Copyright (c) Tencent Inc. All rights reserved.
# YOLO Multi-Modal Backbone (Vision Language)
# Vision: YOLOv8 CSPDarknet
# Language: CLIP Text Encoder (12-layer transformer)
from .mm_backbone import (
    MultiModalYOLOBackbone,
    HuggingVisionBackbone,
    HuggingCLIPLanguageBackbone,
    MultiScalePromptBank,
    PseudoLanguageBackbone,
    SemanticallyGuardedClassConditionedPrompting)
from .remoteclip_vit import RemoteCLIPViTL14Backbone
from .mona_backbone import MonaMultiModalYOLOBackbone, MonaYOLOv8CSPDarknet

__all__ = [
    'MultiModalYOLOBackbone',
    'HuggingVisionBackbone',
    'HuggingCLIPLanguageBackbone',
    'MultiScalePromptBank',
    'PseudoLanguageBackbone',
    'SemanticallyGuardedClassConditionedPrompting',
    'RemoteCLIPViTL14Backbone',
    'MonaMultiModalYOLOBackbone',
    'MonaYOLOv8CSPDarknet',
]

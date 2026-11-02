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
from .remoteclip_rn50 import RemoteCLIPRN50Backbone
from .remoteclip_rn50_text import RemoteCLIPRN50LanguageBackbone
from .remoteclip_prompt_text import RemoteCLIPPromptLanguageBackbone
from .mona_backbone import MonaMultiModalYOLOBackbone, MonaYOLOv8CSPDarknet

__all__ = [
    'MultiModalYOLOBackbone',
    'HuggingVisionBackbone',
    'HuggingCLIPLanguageBackbone',
    'MultiScalePromptBank',
    'PseudoLanguageBackbone',
    'SemanticallyGuardedClassConditionedPrompting',
    'RemoteCLIPViTL14Backbone',
    'RemoteCLIPRN50Backbone',
    'RemoteCLIPRN50LanguageBackbone',
    'RemoteCLIPPromptLanguageBackbone',
    'MonaMultiModalYOLOBackbone',
    'MonaYOLOv8CSPDarknet',
]

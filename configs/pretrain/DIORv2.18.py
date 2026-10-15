"""DIORv2.15 with the frozen RemoteCLIP ViT-L/14 text encoder.

All training images containing any Novel-4 instance are excluded. Validation
and test settings are inherited unchanged from DIORv2.15.

Prepare the text model and detector initialization once before training:
    python tools/convert_remoteclip_text.py
"""

_base_ = './DIORv2.15.py'

text_model_name = 'pretrained_models/remoteclip-vit-large-patch14-text'
model = dict(
    backbone=dict(
        text_model=dict(
            model_name=text_model_name,
            frozen_modules=['all'])))

# The original detector checkpoint also contains CLIP text weights. Use the
# converted initialization so Runner.load_from cannot overwrite RemoteCLIP.
load_from = 'weights/yolo_world_v2_l_remoteclip_text_init.pth'

train_dataloader = dict(
    dataset=dict(
        dataset=dict(
            ann_file=(
                'Annotations2/'
                'train_base_16cls_image_disjoint_yoloworld.json'))))

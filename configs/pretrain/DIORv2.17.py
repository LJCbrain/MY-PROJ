"""DIORv2.15 trained on the image-disjoint Base-16 DIOR split.

All training images containing any Novel-4 instance are excluded. Validation
and test settings are inherited unchanged from DIORv2.15.
"""

_base_ = './DIORv2.15.py'

train_dataloader = dict(
    dataset=dict(
        dataset=dict(
            ann_file=(
                'Annotations2/'
                'train_base_16cls_image_disjoint_yoloworld.json'))))

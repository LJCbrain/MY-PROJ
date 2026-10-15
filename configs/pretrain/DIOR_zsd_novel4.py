_base_ = './DIOR_DOTAbaseline.py'

# Annotation labels stay in DIOR's original spelling.  The separate text file
# uses natural-language prompts so CLIP receives "basketball court" rather
# than the compact COCO category name "basketballcourt".
DIOR_ZSD_CLASSES = (
    'airport',
    'basketballcourt',
    'groundtrackfield',
    'windmill',
)

data_root = 'data/DIOR'
zsd_ann_file = 'Annotations/test_novel_4cls_yoloworld.json'
zsd_text_path = 'data/DIOR/classes_novel_4cls.json'
img_scale = (800, 800)

model = dict(num_test_classes=4)

zsd_test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='YOLOv5KeepRatioResize', scale=img_scale),
    dict(
        type='LetterResize',
        scale=img_scale,
        allow_scale_up=False,
        pad_val=dict(img=114)),
    dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
    dict(type='LoadText'),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'pad_param', 'texts')),
]

zsd_test_dataset = dict(
    _delete_=True,
    type='MultiModalDataset',
    dataset=dict(
        type='YOLOv5CocoDataset',
        metainfo=dict(classes=DIOR_ZSD_CLASSES),
        data_root=data_root,
        test_mode=True,
        ann_file=zsd_ann_file,
        data_prefix=dict(img='images/test/'),
        batch_shapes_cfg=None),
    class_text_path=zsd_text_path,
    pipeline=zsd_test_pipeline)

test_dataloader = dict(dataset=zsd_test_dataset)
test_evaluator = dict(
    type='DIORCocoMetric',
    ann_file=f'{data_root}/{zsd_ann_file}',
    metric='bbox',
    base_classes=(),
    novel_classes=DIOR_ZSD_CLASSES)

"""DIORv2.17 model and training recipe on DOTA-v1.0.

Only dataset-dependent settings are replaced. Training uses the strict
patch-image-disjoint Base-11 split (8,622 non-empty patches with no Novel-4
ground truth). Validation and testing use the 3,236-patch Full-15 validation
set; the official DOTA test set is not used.
"""

_base_ = './DIORv2.17best.py'

DOTA_BASE_CLASSES = (
    'plane', 'ship', 'storage-tank', 'baseball-diamond',
    'basketball-court', 'ground-track-field', 'harbor', 'bridge',
    'large-vehicle', 'small-vehicle', 'roundabout')
DOTA_NOVEL_CLASSES = (
    'tennis-court', 'helicopter', 'soccer-ball-field', 'swimming-pool')
DOTA_FULL_CLASSES = DOTA_BASE_CLASSES + DOTA_NOVEL_CLASSES

num_classes = len(DOTA_FULL_CLASSES)
num_test_classes = num_classes
num_training_classes = len(DOTA_BASE_CLASSES)

dota_data_root = 'data/DOTA800'
dota_base_class_text_path = 'data/DOTA800/classes_base_11cls.json'
dota_full_class_text_path = 'data/DOTA800/classes_full_15cls.json'

# Preserve every model component and loss from DIORv2.17. These dimensions
# and semantic anchors are dataset-dependent and therefore follow DOTA.
model = dict(
    num_train_classes=num_training_classes,
    num_test_classes=num_test_classes,
    backbone=dict(
        text_model=dict(
            prompt_bank_cfg=dict(
                semantic_anchor_texts=DOTA_FULL_CLASSES))),
    bbox_head=dict(
        head_module=dict(num_classes=num_training_classes)),
    train_cfg=dict(assigner=dict(num_classes=num_training_classes)))

text_transform = [
    dict(
        type='RandomLoadText',
        num_neg_samples=(num_training_classes, num_training_classes),
        max_num_samples=num_training_classes,
        padding_to_max=True,
        padding_value=''),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                   'flip_direction', 'texts', 'partial_labels'))
]

train_pipeline = [
    *_base_.pre_transform,
    dict(
        type='MultiModalMosaic',
        img_scale=_base_.img_scale,
        pad_val=114.0,
        pre_transform=_base_.pre_transform),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(1 - _base_.affine_scale,
                             1 + _base_.affine_scale),
        max_aspect_ratio=_base_.max_aspect_ratio,
        border=(-_base_.img_scale[0] // 2, -_base_.img_scale[1] // 2),
        border_val=(114, 114, 114)),
    *_base_.last_transform[:-1],
    *text_transform,
]

train_pipeline_close_mosaic = [
    *_base_.pre_transform,
    dict(type='YOLOv5KeepRatioResize', scale=_base_.img_scale),
    dict(
        type='LetterResize',
        scale=_base_.img_scale,
        allow_scale_up=True,
        pad_val=dict(img=114.0)),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(1 - _base_.affine_scale,
                             1 + _base_.affine_scale),
        max_aspect_ratio=_base_.max_aspect_ratio,
        border_val=(114, 114, 114)),
    *_base_.last_transform[:-1],
    *text_transform,
]

dota_train_dataset = dict(
    type='MultiModalDataset',
    dataset=dict(
        type='PartialLabelYOLOv5CocoDataset',
        metainfo=dict(classes=DOTA_BASE_CLASSES),
        data_root=dota_data_root,
        ann_file='annotations/train_base_11cls_patch_disjoint_hbb.json',
        data_prefix=dict(img='train/images/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32)),
    class_text_path=dota_base_class_text_path,
    pipeline=train_pipeline)

train_dataloader = dict(
    dataset=dict(_delete_=True, **dota_train_dataset))

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='YOLOv5KeepRatioResize', scale=_base_.img_scale),
    dict(
        type='LetterResize',
        scale=_base_.img_scale,
        allow_scale_up=False,
        pad_val=dict(img=114)),
    dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
    dict(type='LoadText'),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'pad_param', 'texts'))
]

dota_gzsd_val_dataset = dict(
    type='MultiModalDataset',
    dataset=dict(
        type='YOLOv5CocoDataset',
        metainfo=dict(classes=DOTA_FULL_CLASSES),
        data_root=dota_data_root,
        test_mode=True,
        ann_file='annotations/val_full_15cls_hbb.json',
        data_prefix=dict(img='val/images/'),
        batch_shapes_cfg=None),
    class_text_path=dota_full_class_text_path,
    pipeline=test_pipeline)

val_dataloader = dict(
    dataset=dict(_delete_=True, **dota_gzsd_val_dataset))
test_dataloader = dict(
    dataset=dict(_delete_=True, **dota_gzsd_val_dataset))

dota_gzsd_evaluator = dict(
    type='OVCDIORCocoMetric',
    ann_file='data/DOTA800/annotations/val_full_15cls_hbb.json',
    metric='bbox',
    base_classes=DOTA_BASE_CLASSES,
    novel_classes=DOTA_NOVEL_CLASSES,
    proposal_nums=(100, 1, 10))

dota_proposal_evaluator = dict(
    type='PriorProposalRecallMetric',
    ann_file='data/DOTA800/annotations/val_full_15cls_hbb.json',
    base_classes=DOTA_BASE_CLASSES,
    novel_classes=DOTA_NOVEL_CLASSES,
    iou_thresholds=(0.3, 0.5),
    prefix='proposal_O',
    proposal_nums=(100, 300),
    report_classwise=True)

val_evaluator = [dota_gzsd_evaluator, dota_proposal_evaluator]
test_evaluator = [dota_gzsd_evaluator, dota_proposal_evaluator]

# Keep DIORv2.17's hooks and iteration schedule. Only the switched pipeline's
# text vocabulary changes from 16 DIOR Base classes to 11 DOTA Base classes.
custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='IntegerSafeExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        strict_load=False,
        priority=49),
    dict(type='OVCDiagnosticsHook', interval=50),
    dict(
        type='OVCIterationPipelineSwitchHook',
        switch_iter=49772,
        switch_pipeline=train_pipeline_close_mosaic),
]

work_dir = 'work_dirs/DOTAv1'

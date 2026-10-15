"""Regional-prior/WSR/RSSP model on the RT-OVAD xView Base-48 split.

This configuration inherits every non-dataset model module and training
hyper-parameter from ``DIOR_DOTAv1.py``.  Only xView-dependent class counts,
semantic-anchor vocabulary, data paths, pipelines and evaluators are replaced.
"""

_base_ = './DIOR_DOTAv1.py'

XVIEW_BASE_CLASSES = (
    'building', 'small-car', 'truck',
    'truck-tractor-with-flatbed-trailer', 'construction-site', 'trailer',
    'utility-truck', 'cargo-truck', 'vehicle-lot', 'hut-or-tent',
    'damaged-building', 'container-crane', 'cargo-car', 'storage-tank',
    'facility', 'dump-truck', 'front-loader-or-bulldozer',
    'ground-grader', 'shed', 'passenger-vehicle', 'tower-crane', 'sailboat',
    'fishing-vessel', 'shipping-container', 'tower', 'cargo-plane',
    'aircraft-hangar', 'haul-truck', 'yacht', 'oil-tanker', 'ferry',
    'tugboat', 'truck-tractor', 'passenger-car', 'helipad', 'crane-truck',
    'engineering-vehicle', 'small-aircraft', 'pylon', 'container-ship',
    'locomotive', 'truck-tractor-with-liquid-tank', 'flat-car', 'tank-car',
    'straddle-carrier', 'cement-mixer', 'fixed-wing-aircraft',
    'railway-vehicle')

XVIEW_NOVEL_CLASSES = (
    'helicopter', 'bus', 'pickup-truck',
    'truck-tractor-with-box-trailer', 'maritime-vessel', 'motorboat',
    'barge', 'reach-stacker', 'mobile-crane', 'scraper-or-tractor',
    'excavator', 'shipping-container-lot')

XVIEW_FULL_CLASSES = XVIEW_BASE_CLASSES + XVIEW_NOVEL_CLASSES

num_classes = 60
num_test_classes = num_classes
num_training_classes = 48
img_scale = (800, 800)

# Preserve DIOR_DOTAv1.py's detector, C2F, Regional Prior, WSR and RSSP.
# Only dataset-dependent output sizes and semantic-anchor vocabulary change.
model = dict(
    num_train_classes=num_training_classes,
    num_test_classes=num_test_classes,
    backbone=dict(
        text_model=dict(
            prompt_bank_cfg=dict(
                semantic_anchor_texts=XVIEW_FULL_CLASSES,
                semantic_anchor_embed_path=(
                    'data/xView800/clip_vit_large_patch14_336_'
                    'xview60_text_embeddings.pt')))),
    bbox_head=dict(
        head_module=dict(num_classes=num_training_classes)),
    train_cfg=dict(assigner=dict(num_classes=num_training_classes)))

xview_data_root = 'data/xView800'
xview_base_class_text_path = 'data/xView800/classes_base_48cls.json'
xview_full_class_text_path = 'data/xView800/classes_full_60cls.json'
xview_novel_class_text_path = 'data/xView800/classes_novel_12cls.json'

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
                   'flip_direction', 'texts'))
]

train_pipeline = [
    *_base_.pre_transform,
    dict(
        type='MultiModalMosaic',
        img_scale=img_scale,
        pad_val=114.0,
        pre_transform=_base_.pre_transform),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(1 - _base_.affine_scale,
                             1 + _base_.affine_scale),
        max_aspect_ratio=_base_.max_aspect_ratio,
        border=(-img_scale[0] // 2, -img_scale[1] // 2),
        border_val=(114, 114, 114)),
    *_base_.last_transform[:-1],
    *text_transform,
]

train_pipeline_close_mosaic = [
    *_base_.pre_transform,
    dict(type='YOLOv5KeepRatioResize', scale=img_scale),
    dict(
        type='LetterResize',
        scale=img_scale,
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
    *text_transform
]

xview_train_dataset = dict(
    type='MultiModalDataset',
    dataset=dict(
        type='YOLOv5CocoDataset',
        metainfo=dict(classes=XVIEW_BASE_CLASSES),
        data_root=xview_data_root,
        ann_file='annotations/train_base_48cls_patch_disjoint_hbb.json',
        data_prefix=dict(img='train/images/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32)),
    class_text_path=xview_base_class_text_path,
    pipeline=train_pipeline)

train_dataloader = dict(
    batch_size=4,
    collate_fn=dict(type='yolow_collate'),
    dataset=dict(_delete_=True, **xview_train_dataset))

test_pipeline = [
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
                   'scale_factor', 'pad_param', 'texts'))
]

xview_gzsd_val_dataset = dict(
    type='MultiModalDataset',
    dataset=dict(
        type='YOLOv5CocoDataset',
        metainfo=dict(classes=XVIEW_FULL_CLASSES),
        data_root=xview_data_root,
        test_mode=True,
        ann_file='annotations/val_full_60cls_hbb.json',
        data_prefix=dict(img='val/images/'),
        batch_shapes_cfg=None),
    class_text_path=xview_full_class_text_path,
    pipeline=test_pipeline)

xview_zsd_val_dataset = dict(
    type='MultiModalDataset',
    dataset=dict(
        type='YOLOv5CocoDataset',
        metainfo=dict(classes=XVIEW_NOVEL_CLASSES),
        data_root=xview_data_root,
        test_mode=True,
        ann_file='annotations/val_novel_12cls_hbb.json',
        data_prefix=dict(img='val/images/'),
        batch_shapes_cfg=None),
    class_text_path=xview_novel_class_text_path,
    pipeline=test_pipeline)

xview_gzsd_val_evaluator = dict(
    type='AerialTableIIICocoMetric',
    ann_file='data/xView800/annotations/val_full_60cls_hbb.json',
    metric='bbox',
    base_classes=XVIEW_BASE_CLASSES,
    novel_classes=XVIEW_NOVEL_CLASSES,
    recall_max_dets=100)

xview_zsd_val_evaluator = dict(
    type='AerialTableIIICocoMetric',
    ann_file='data/xView800/annotations/val_novel_12cls_hbb.json',
    metric='bbox',
    base_classes=(),
    novel_classes=XVIEW_NOVEL_CLASSES,
    recall_max_dets=100)

val_dataloader = dict(
    dataset=dict(_delete_=True, **xview_gzsd_val_dataset))
val_evaluator = dict(
    type='DIORCocoMetric',
    ann_file='data/xView800/annotations/val_full_60cls_hbb.json',
    metric='bbox',
    base_classes=XVIEW_BASE_CLASSES,
    novel_classes=XVIEW_NOVEL_CLASSES)

test_dataloader = dict(
    dataset=dict(_delete_=True, **xview_gzsd_val_dataset))
test_evaluator = [
    xview_gzsd_val_evaluator,
    dict(
        type='PriorProposalRecallMetric',
        ann_file='data/xView800/annotations/val_full_60cls_hbb.json',
        base_classes=XVIEW_BASE_CLASSES,
        novel_classes=XVIEW_NOVEL_CLASSES,
        iou_thresholds=(0.3, 0.5),
        proposal_num=300)
]

# Same hooks as DIOR_DOTAv1.py; only the switched pipeline carries xView's
# 48-slot text sampling.
custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        strict_load=False,
        priority=49),
    dict(
        type='MultiDatasetPipelineSwitchHook',
        switch_epoch=_base_.max_epochs - _base_.close_mosaic_epochs,
        switch_pipeline=train_pipeline_close_mosaic)
]

_base_ = ('../../third_party/mmyolo/configs/yolov8/'
          'yolov8_l_syncbn_fast_8xb16-500e_coco.py')
custom_imports = dict(
    imports=['yolo_world', 'yolo_world.evaluation'],
    allow_failed_imports=False)

# DIOR zero-shot split
DIOR_BASE_CLASSES = (
    'airplane',
    'baseballfield',
    'bridge',
    'chimney',
    'dam',
    'Expressway-Service-area',
    'Expressway-toll-station',
    'golffield',
    'harbor',
    'overpass',
    'ship',
    'stadium',
    'storagetank',
    'tenniscourt',
    'trainstation',
    'vehicle',
)
DIOR_NOVEL_CLASSES = (
    'airport',
    'basketballcourt',
    'groundtrackfield',
    'windmill',
)
DIOR_FULL_CLASSES = (
    'airplane',
    'airport',
    'baseballfield',
    'basketballcourt',
    'bridge',
    'chimney',
    'dam',
    'Expressway-Service-area',
    'Expressway-toll-station',
    'golffield',
    'groundtrackfield',
    'harbor',
    'overpass',
    'ship',
    'stadium',
    'storagetank',
    'tenniscourt',
    'trainstation',
    'vehicle',
    'windmill',
)

# RT-OVAD Table III upper block: DOTA-v1.0 non-asterisk split.
DOTA_BASE_CLASSES = (
    'plane', 'ship', 'storage-tank', 'baseball-diamond',
    'basketball-court', 'ground-track-field', 'harbor', 'bridge',
    'large-vehicle', 'small-vehicle', 'roundabout',
)
DOTA_NOVEL_CLASSES = (
    'tennis-court', 'helicopter', 'soccer-ball-field', 'swimming-pool',
)
DOTA_FULL_CLASSES = DOTA_BASE_CLASSES + DOTA_NOVEL_CLASSES

# hyper-parameters
num_classes = 20
num_test_classes = num_classes
num_training_classes = 16
max_epochs = 50
close_mosaic_epochs = 2
save_epoch_intervals = 2
text_channels = 768
neck_embed_channels = [128, 256, _base_.last_stage_out_channels // 2]
neck_num_heads = [4, 8, _base_.last_stage_out_channels // 2 // 32]
base_lr = 2e-3
weight_decay = 0.0125
train_batch_size_per_gpu = 4
text_model_name = r'pretrained_models/clip-vit-large-patch14-336'
img_scale = (800, 800)
load_from = 'weights/yolo_world_v2_l_clip_large_o365v1_goldg_pretrain_800ft-9df82e55.pth'

# model settings
model = dict(
    type='YOLOWorldDetector',
    mm_neck=True,
    num_train_classes=num_training_classes,
    num_test_classes=num_test_classes,
    data_preprocessor=dict(type='YOLOWDetDataPreprocessor'),
    backbone=dict(
        _delete_=True,
        type='MultiModalYOLOBackbone',
        image_model={{_base_.model.backbone}},
        text_model=dict(
            type='HuggingCLIPLanguageBackbone',
            model_name=text_model_name,
            frozen_modules=['all'])),
    neck=dict(
        type='YOLOWorldDualPAFPN',
        guide_channels=text_channels,
        embed_channels=neck_embed_channels,
        num_heads=neck_num_heads,
        block_cfg=dict(type='MaxSigmoidCSPLayerWithTwoConv'),
        text_enhancder=dict(
            type='ImagePoolingAttentionModule',
            embed_channels=256,
            num_heads=8,
            pool_size=3)),
    bbox_head=dict(
        type='YOLOWorldHead',
        head_module=dict(
            type='YOLOWorldHeadModule',
            use_bn_head=True,
            embed_dims=text_channels,
            num_classes=num_training_classes)),
    train_cfg=dict(assigner=dict(num_classes=num_training_classes)))

# dataset settings
dior_data_root = 'data/DIOR'
dota_data_root = 'data/DOTA800'
dior_base_class_text_path = 'data/DIOR/classes_base_16cls.json'
dior_full_class_text_path = 'data/DIOR/classes_full_20cls.json'
dior_novel_class_text_path = 'data/DIOR/classes_novel_4cls.json'
dota_base_class_text_path = 'data/DOTA800/classes_base_11cls.json'
dota_full_class_text_path = 'data/DOTA800/classes_full_15cls.json'
dota_novel_class_text_path = 'data/DOTA800/classes_novel_4cls.json'

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

train_pipeline_stage2 = [
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

dior_train_dataset = dict(
    type='MultiModalDataset',
    dataset=dict(
        type='YOLOv5CocoDataset',
        metainfo=dict(classes=DIOR_BASE_CLASSES),
        data_root=dior_data_root,
        ann_file='Annotations2/train_base_16cls_image_disjoint_yoloworld.json',
        data_prefix=dict(img='JPEGImages/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32)),
    class_text_path=dior_base_class_text_path,
    pipeline=train_pipeline)

dota_train_dataset = dict(
    type='MultiModalDataset',
    dataset=dict(
        type='YOLOv5CocoDataset',
        metainfo=dict(classes=DOTA_BASE_CLASSES),
        data_root=dota_data_root,
        ann_file='annotations/train_base_11cls_patch_disjoint_hbb.json',
        data_prefix=dict(img='train/images/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32)),
    class_text_path=dota_base_class_text_path,
    pipeline=train_pipeline)

# DOTA's 11 Base prompts are padded by RandomLoadText to DIOR's common
# 16-slot training format; labels remain local to each child dataset.
train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    collate_fn=dict(type='yolow_collate'),
    dataset=dict(
        _delete_=True,
        type='ConcatDataset',
        datasets=[dior_train_dataset, dota_train_dataset],
        ignore_keys=['classes', 'palette']))

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

dior_gzsd_val_dataset = dict(
    type='MultiModalDataset',
    dataset=dict(
        type='YOLOv5CocoDataset',
        metainfo=dict(classes=DIOR_FULL_CLASSES),
        data_root=dior_data_root,
        test_mode=True,
        ann_file='Annotations2/val.json',
        data_prefix=dict(img='JPEGImages/'),
        batch_shapes_cfg=None),
    class_text_path=dior_full_class_text_path,
    pipeline=test_pipeline)

dior_zsd_val_dataset = dict(
    type='MultiModalDataset',
    dataset=dict(
        type='YOLOv5CocoDataset',
        metainfo=dict(classes=DIOR_NOVEL_CLASSES),
        data_root=dior_data_root,
        test_mode=True,
        ann_file='Annotations2/val_novel_4cls.json',
        data_prefix=dict(img='JPEGImages/'),
        batch_shapes_cfg=None),
    class_text_path=dior_novel_class_text_path,
    pipeline=test_pipeline)

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

dota_zsd_val_dataset = dict(
    type='MultiModalDataset',
    dataset=dict(
        type='YOLOv5CocoDataset',
        metainfo=dict(classes=DOTA_NOVEL_CLASSES),
        data_root=dota_data_root,
        test_mode=True,
        ann_file='annotations/val_novel_4cls_hbb.json',
        data_prefix=dict(img='val/images/'),
        batch_shapes_cfg=None),
    class_text_path=dota_novel_class_text_path,
    pipeline=test_pipeline)

dior_gzsd_val_evaluator = dict(
    type='AerialTableIIICocoMetric',
    ann_file='data/DIOR/Annotations2/val.json',
    metric='bbox',
    base_classes=DIOR_BASE_CLASSES,
    novel_classes=DIOR_NOVEL_CLASSES,
    recall_max_dets=300)
dior_zsd_val_evaluator = dict(
    type='AerialTableIIICocoMetric',
    ann_file='data/DIOR/Annotations2/val_novel_4cls.json',
    metric='bbox',
    base_classes=(),
    novel_classes=DIOR_NOVEL_CLASSES,
    recall_max_dets=300)
dota_gzsd_val_evaluator = dict(
    type='AerialTableIIICocoMetric',
    ann_file='data/DOTA800/annotations/val_full_15cls_hbb.json',
    metric='bbox',
    base_classes=DOTA_BASE_CLASSES,
    novel_classes=DOTA_NOVEL_CLASSES,
    recall_max_dets=300)
dota_zsd_val_evaluator = dict(
    type='AerialTableIIICocoMetric',
    ann_file='data/DOTA800/annotations/val_novel_4cls_hbb.json',
    metric='bbox',
    base_classes=(),
    novel_classes=DOTA_NOVEL_CLASSES,
    recall_max_dets=300)

# Training validates DIOR GZSD only; this preserves the same best-HM
# checkpoint selection protocol used by DIOR_DOTAv1.py.
val_dataloader = dict(
    dataset=dict(_delete_=True, **dior_gzsd_val_dataset))
val_evaluator = dict(
    type='DIORCocoMetric',
    ann_file='data/DIOR/Annotations2/val.json',
    metric='bbox',
    base_classes=DIOR_BASE_CLASSES,
    novel_classes=DIOR_NOVEL_CLASSES)

# The Table III script swaps this dataset four times for DIOR/DOTA GZSD/ZSD.
test_dataloader = dict(
    dataset=dict(_delete_=True, **dior_gzsd_val_dataset))
test_evaluator = dior_gzsd_val_evaluator

# training settings
default_hooks = dict(
    param_scheduler=dict(max_epochs=max_epochs),
    checkpoint=dict(
        interval=save_epoch_intervals,
        save_best='coco/bbox_mAP_50_HM',
        rule='greater'))
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
        switch_epoch=max_epochs - close_mosaic_epochs,
        switch_pipeline=train_pipeline_stage2)
]
train_cfg = dict(
    max_epochs=max_epochs,
    val_interval=5,
    dynamic_intervals=[((max_epochs - close_mosaic_epochs),
                        _base_.val_interval_stage2)])

visualizer = dict(
    type='mmdet.DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend')
    ],
    name='visualizer')

optim_wrapper = dict(
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=base_lr,
        weight_decay=weight_decay,
        batch_size_per_gpu=train_batch_size_per_gpu),
    paramwise_cfg=dict(
        bias_decay_mult=0.0,
        norm_decay_mult=0.0,
        custom_keys={
            'backbone.text_model': dict(lr_mult=0.01),
            'logit_scale': dict(weight_decay=0.0)
        }),
    constructor='YOLOWv5OptimizerConstructor')

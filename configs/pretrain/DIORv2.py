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

# hyper-parameters
num_classes = 20
num_test_classes = num_classes
num_training_classes = 16
max_epochs = 30
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

# model settings
model = dict(
    type='YOLOWorldDetector',
    mm_neck=True,
    num_train_classes=num_training_classes,
    num_test_classes=num_test_classes,
    loss_fgbg_mask_bce_weight=0.03,
    loss_fgbg_mask_dice_weight=0.03,
    loss_fgbg_quality_weight=0.01,
    loss_fgbg_consistency_weight=0.003,
    loss_fgbg_consistency_mode='l1',
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
        type='YOLOWorldWSRPAFPN',
        guide_channels=text_channels,
        embed_channels=neck_embed_channels,
        num_heads=neck_num_heads,
        block_cfg=dict(type='GuideAgnosticCSPLayerWithTwoConv'),
        fgbg_cfg=dict(
            type='ClassAgnosticForegroundGate',
            in_channels=[256, 512, _base_.last_stage_out_channels],
            hidden_ratio=0.25,
            min_hidden=16,
            max_gamma=0.15,
            gamma_init_bias=-0.3,
            gamma_warmup_iters=1000,
            return_parts=True,
        ),
        wsr_cfg=dict(
            type='WeakSupportRetriever',
            in_channels=[256, 512, _base_.last_stage_out_channels],
            embed_channels=64,
            num_reliable_tokens=128,
            num_detail_tokens=64,
            grid_size=8,
            reliability_nms_kernel=7,
            detail_kernel=5,
            exclusion_radius=1,
            reliability_log_weight=0.5,
            support_dropout=0.2,
            residual_scale=0.1,
            detach_reliability=True,
            enhance_level=0,
        )),
    bbox_head=dict(
        type='YOLOWorldHead',
        head_module=dict(
            type='YOLOWorldHeadModule',
            use_bn_head=True,
            embed_dims=text_channels,
            num_classes=num_training_classes)),
    train_cfg=dict(assigner=dict(num_classes=num_training_classes)))

# dataset settings
data_root = 'data/DIOR'
dior_base_class_text_path = 'data/DIOR/classes_base_16cls.json'
dior_full_class_text_path = 'data/DIOR/classes_full_20cls.json'

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

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    collate_fn=dict(type='yolow_collate'),
    dataset=dict(
        _delete_=True,
        type='MultiModalDataset',
        dataset=dict(
            type='YOLOv5CocoDataset',
            metainfo=dict(classes=DIOR_BASE_CLASSES),
            data_root=data_root,
            ann_file='Annotations/train_base_16cls_yoloworld.json',
            data_prefix=dict(img='images/train/'),
            filter_cfg=dict(filter_empty_gt=False, min_size=32)),
        class_text_path=dior_base_class_text_path,
        pipeline=train_pipeline))

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

dior_val_dataset = dict(
    _delete_=True,
    type='MultiModalDataset',
    dataset=dict(
        type='YOLOv5CocoDataset',
        metainfo=dict(classes=DIOR_FULL_CLASSES),
        data_root=data_root,
        test_mode=True,
        ann_file='Annotations/val_yoloworld.json',
        data_prefix=dict(img='images/val/'),
        batch_shapes_cfg=None),
    class_text_path=dior_full_class_text_path,
    pipeline=test_pipeline)

dior_test_dataset = dict(
    _delete_=True,
    type='MultiModalDataset',
    dataset=dict(
        type='YOLOv5CocoDataset',
        metainfo=dict(classes=DIOR_FULL_CLASSES),
        data_root=data_root,
        test_mode=True,
        ann_file='Annotations/test_yoloworld.json',
        data_prefix=dict(img='images/test/'),
        batch_shapes_cfg=None),
    class_text_path=dior_full_class_text_path,
    pipeline=test_pipeline)

val_dataloader = dict(dataset=dior_val_dataset)
test_dataloader = dict(dataset=dior_test_dataset)

val_evaluator = dict(
    type='DIORCocoMetric',
    ann_file='data/DIOR/Annotations/val_yoloworld.json',
    metric='bbox',
    base_classes=DIOR_BASE_CLASSES,
    novel_classes=DIOR_NOVEL_CLASSES)
test_evaluator = dict(
    type='DIORCocoMetric',
    ann_file='data/DIOR/Annotations/test_yoloworld.json',
    metric='bbox',
    base_classes=DIOR_BASE_CLASSES,
    novel_classes=DIOR_NOVEL_CLASSES)

# training settings
default_hooks = dict(
    param_scheduler=dict(max_epochs=max_epochs),
    checkpoint=dict(interval=save_epoch_intervals, rule='greater'))
custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        strict_load=False,
        priority=49),
    dict(
        type='mmdet.PipelineSwitchHook',
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

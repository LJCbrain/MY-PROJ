# DIOR 2.17: standalone configuration; no config inheritance.
# Image-disjoint Base-16 split; images containing Novel-4 instances excluded.
# Partial Base labels, normalized regularization, fixed 0.02 dual stream, WSR.
# AMP is enabled: OVCTrackedAmpOptimWrapper uses dynamic loss scaling.

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
DIOR_NOVEL_CLASSES = (
    'airport',
    'basketballcourt',
    'groundtrackfield',
    'windmill',
)
GUARD_PROMPT_ANCHOR_CLASSES = (
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
_backend_args = None
_multiscale_resize_transforms = [
    dict(
        transforms=[
            dict(scale=(
                640,
                640,
            ), type='YOLOv5KeepRatioResize'),
            dict(
                allow_scale_up=False,
                pad_val=dict(img=114),
                scale=(
                    640,
                    640,
                ),
                type='LetterResize'),
        ],
        type='Compose'),
    dict(
        transforms=[
            dict(scale=(
                320,
                320,
            ), type='YOLOv5KeepRatioResize'),
            dict(
                allow_scale_up=False,
                pad_val=dict(img=114),
                scale=(
                    320,
                    320,
                ),
                type='LetterResize'),
        ],
        type='Compose'),
    dict(
        transforms=[
            dict(scale=(
                960,
                960,
            ), type='YOLOv5KeepRatioResize'),
            dict(
                allow_scale_up=False,
                pad_val=dict(img=114),
                scale=(
                    960,
                    960,
                ),
                type='LetterResize'),
        ],
        type='Compose'),
]
affine_scale = 0.9
albu_train_transforms = [
    dict(p=0.01, type='Blur'),
    dict(p=0.01, type='MedianBlur'),
    dict(p=0.01, type='ToGray'),
    dict(p=0.01, type='CLAHE'),
]
backend_args = None
base_lr = 0.001
batch_shapes_cfg = None
close_mosaic_epochs = 4
custom_hooks = [
    dict(
        ema_type='IntegerSafeExpMomentumEMA',
        momentum=0.0001,
        priority=49,
        strict_load=False,
        type='EMAHook',
        update_buffers=True),
    dict(interval=50, type='OVCDiagnosticsHook'),
    dict(
        switch_iter=49772,
        switch_pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(scale=(
                800,
                800,
            ), type='YOLOv5KeepRatioResize'),
            dict(
                allow_scale_up=True,
                pad_val=dict(img=114.0),
                scale=(
                    800,
                    800,
                ),
                type='LetterResize'),
            dict(
                border_val=(
                    114,
                    114,
                    114,
                ),
                max_aspect_ratio=100,
                max_rotate_degree=0.0,
                max_shear_degree=0.0,
                scaling_ratio_range=(
                    0.09999999999999998,
                    1.9,
                ),
                type='YOLOv5RandomAffine'),
            dict(
                bbox_params=dict(
                    format='pascal_voc',
                    label_fields=[
                        'gt_bboxes_labels',
                        'gt_ignore_flags',
                    ],
                    type='BboxParams'),
                keymap=dict(gt_bboxes='bboxes', img='image'),
                transforms=[
                    dict(p=0.01, type='Blur'),
                    dict(p=0.01, type='MedianBlur'),
                    dict(p=0.01, type='ToGray'),
                    dict(p=0.01, type='CLAHE'),
                ],
                type='mmdet.Albu'),
            dict(type='YOLOv5HSVRandomAug'),
            dict(prob=0.5, type='mmdet.RandomFlip'),
            dict(
                max_num_samples=16,
                num_neg_samples=(
                    16,
                    16,
                ),
                padding_to_max=True,
                padding_value='',
                type='RandomLoadText'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'flip',
                    'flip_direction',
                    'texts',
                    'partial_labels',
                ),
                type='mmdet.PackDetInputs'),
        ],
        type='OVCIterationPipelineSwitchHook'),
]
custom_imports = dict(
    allow_failed_imports=False,
    imports=[
        'yolo_world',
        'yolo_world.evaluation',
    ])
data_root = 'data/coco/'
dataset_type = 'YOLOv5CocoDataset'
deepen_factor = 1.0
default_hooks = dict(
    checkpoint=dict(
        by_epoch=False,
        interval=4328,
        max_keep_ckpts=2,
        rule='greater',
        save_best='coco/bbox_mAP_50_HM',
        save_last=True,
        type='CheckpointHook'),
    logger=dict(interval=50, log_metric_by_epoch=False, type='LoggerHook'),
    param_scheduler=dict(
        lr_factor=0.01,
        max_epochs=50,
        scheduler_type='linear',
        type='OVCIterationParamSchedulerHook',
        virtual_epoch_iters=1082),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='mmdet.DetVisualizationHook'))
default_scope = 'mmyolo'
dior_base_class_text_path = 'data/DIOR/classes_base_16cls.json'
dior_data_root = 'data/DIOR'
dior_full_class_text_path = 'data/DIOR/classes_full_20cls.json'
dior_gzsd_val_dataset = dict(
    class_text_path='data/DIOR/classes_full_20cls.json',
    dataset=dict(
        ann_file='Annotations2/val.json',
        batch_shapes_cfg=None,
        data_prefix=dict(img='JPEGImages/'),
        data_root='data/DIOR',
        metainfo=dict(
            classes=(
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
            )),
        test_mode=True,
        type='YOLOv5CocoDataset'),
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(scale=(
            800,
            800,
        ), type='YOLOv5KeepRatioResize'),
        dict(
            allow_scale_up=False,
            pad_val=dict(img=114),
            scale=(
                800,
                800,
            ),
            type='LetterResize'),
        dict(_scope_='mmdet', type='LoadAnnotations', with_bbox=True),
        dict(type='LoadText'),
        dict(
            meta_keys=(
                'img_id',
                'img_path',
                'ori_shape',
                'img_shape',
                'scale_factor',
                'pad_param',
                'texts',
            ),
            type='mmdet.PackDetInputs'),
    ],
    type='MultiModalDataset')
dior_gzsd_val_evaluator = dict(
    ann_file='data/DIOR/Annotations2/val.json',
    base_classes=(
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
    ),
    metric='bbox',
    novel_classes=(
        'airport',
        'basketballcourt',
        'groundtrackfield',
        'windmill',
    ),
    recall_max_dets=100,
    type='AerialTableIIICocoMetric')
dior_novel_class_text_path = 'data/DIOR/classes_novel_4cls.json'
dior_train_dataset = dict(
    class_text_path='data/DIOR/classes_base_16cls.json',
    dataset=dict(
        ann_file='Annotations2/train_base_16cls_image_disjoint_yoloworld.json',
        data_prefix=dict(img='JPEGImages/'),
        data_root='data/DIOR',
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        metainfo=dict(
            classes=(
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
            )),
        type='YOLOv5CocoDataset'),
    pipeline=[
        dict(backend_args=None, type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(
            img_scale=(
                800,
                800,
            ),
            pad_val=114.0,
            pre_transform=[
                dict(backend_args=None, type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
            ],
            type='MultiModalMosaic'),
        dict(
            border=(
                -400,
                -400,
            ),
            border_val=(
                114,
                114,
                114,
            ),
            max_aspect_ratio=100,
            max_rotate_degree=0.0,
            max_shear_degree=0.0,
            scaling_ratio_range=(
                0.09999999999999998,
                1.9,
            ),
            type='YOLOv5RandomAffine'),
        dict(
            bbox_params=dict(
                format='pascal_voc',
                label_fields=[
                    'gt_bboxes_labels',
                    'gt_ignore_flags',
                ],
                type='BboxParams'),
            keymap=dict(gt_bboxes='bboxes', img='image'),
            transforms=[
                dict(p=0.01, type='Blur'),
                dict(p=0.01, type='MedianBlur'),
                dict(p=0.01, type='ToGray'),
                dict(p=0.01, type='CLAHE'),
            ],
            type='mmdet.Albu'),
        dict(type='YOLOv5HSVRandomAug'),
        dict(prob=0.5, type='mmdet.RandomFlip'),
        dict(
            max_num_samples=16,
            num_neg_samples=(
                16,
                16,
            ),
            padding_to_max=True,
            padding_value='',
            type='RandomLoadText'),
        dict(
            meta_keys=(
                'img_id',
                'img_path',
                'ori_shape',
                'img_shape',
                'flip',
                'flip_direction',
                'texts',
            ),
            type='mmdet.PackDetInputs'),
    ],
    type='MultiModalDataset')
dior_zsd_val_dataset = dict(
    class_text_path='data/DIOR/classes_novel_4cls.json',
    dataset=dict(
        ann_file='Annotations2/val_novel_4cls.json',
        batch_shapes_cfg=None,
        data_prefix=dict(img='JPEGImages/'),
        data_root='data/DIOR',
        metainfo=dict(
            classes=(
                'airport',
                'basketballcourt',
                'groundtrackfield',
                'windmill',
            )),
        test_mode=True,
        type='YOLOv5CocoDataset'),
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(scale=(
            800,
            800,
        ), type='YOLOv5KeepRatioResize'),
        dict(
            allow_scale_up=False,
            pad_val=dict(img=114),
            scale=(
                800,
                800,
            ),
            type='LetterResize'),
        dict(_scope_='mmdet', type='LoadAnnotations', with_bbox=True),
        dict(type='LoadText'),
        dict(
            meta_keys=(
                'img_id',
                'img_path',
                'ori_shape',
                'img_shape',
                'scale_factor',
                'pad_param',
                'texts',
            ),
            type='mmdet.PackDetInputs'),
    ],
    type='MultiModalDataset')
dior_zsd_val_evaluator = dict(
    ann_file='data/DIOR/Annotations2/val_novel_4cls.json',
    base_classes=(),
    metric='bbox',
    novel_classes=(
        'airport',
        'basketballcourt',
        'groundtrackfield',
        'windmill',
    ),
    recall_max_dets=100,
    type='AerialTableIIICocoMetric')
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
img_scale = (
    800,
    800,
)
img_scales = [
    (
        640,
        640,
    ),
    (
        320,
        320,
    ),
    (
        960,
        960,
    ),
]
last_stage_out_channels = 512
last_transform = [
    dict(
        bbox_params=dict(
            format='pascal_voc',
            label_fields=[
                'gt_bboxes_labels',
                'gt_ignore_flags',
            ],
            type='BboxParams'),
        keymap=dict(gt_bboxes='bboxes', img='image'),
        transforms=[
            dict(p=0.01, type='Blur'),
            dict(p=0.01, type='MedianBlur'),
            dict(p=0.01, type='ToGray'),
            dict(p=0.01, type='CLAHE'),
        ],
        type='mmdet.Albu'),
    dict(type='YOLOv5HSVRandomAug'),
    dict(prob=0.5, type='mmdet.RandomFlip'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'flip',
            'flip_direction',
        ),
        type='mmdet.PackDetInputs'),
]
launcher = 'none'
load_from = 'weights/yolo_world_v2_l_clip_large_o365v1_goldg_pretrain_800ft-9df82e55.pth'
log_level = 'INFO'
log_processor = dict(
    by_epoch=False, num_digits=8, type='LogProcessor', window_size=50)
loss_bbox_weight = 7.5
loss_cls_weight = 0.5
loss_dfl_weight = 0.375
lr_factor = 0.01
max_aspect_ratio = 100
max_epochs = 50
max_keep_ckpts = 2
mixup_prob = 0.15
model = dict(
    backbone=dict(
        image_model=dict(
            act_cfg=dict(inplace=True, type='SiLU'),
            arch='P5',
            deepen_factor=1.0,
            last_stage_out_channels=512,
            norm_cfg=dict(eps=0.001, momentum=0.03, type='BN'),
            type='YOLOv8CSPDarknet',
            widen_factor=1.0),
        text_model=dict(
            add_mask=True,
            frozen_modules=[
                'all',
            ],
            model_name='pretrained_models/clip-vit-large-patch14-336',
            pad_value='',
            prompt_bank_cfg=dict(
                balance_temperature=0.5,
                balance_use_all_text=True,
                class_chunk_size=64,
                class_semantic_residual_weight=1.0,
                collect_key_stats=True,
                image_channels=[
                    256,
                    512,
                    512,
                ],
                key_dim=768,
                key_orthogonalization='qr',
                loss_balance_warmup_iters=1000,
                loss_balance_warmup_weight=0.1,
                loss_balance_weight=0.05,
                loss_match_start_iter=1000,
                loss_match_warmup_iters=0,
                loss_match_weight=0.05,
                loss_orthogonal_weight=0.3,
                normalize_match_by_num_scales=True,
                num_prompts=10,
                num_scales=3,
                orthogonal_key_init=True,
                per_scale_balance=True,
                prompt_length=12,
                routing_temperature=0.5,
                semantic_anchor_embed_path=None,
                semantic_anchor_sample_size=0,
                semantic_anchor_texts=(
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
                ),
                semantic_anchor_warmup_iters=1000,
                semantic_anchor_weight=0.05,
                spatial_temperature=0.1,
                use_reliability=False,
                use_reliability_prior=False),
            type='HuggingCLIPLanguageBackbone'),
        type='MultiModalYOLOBackbone'),
    bbox_head=dict(
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        head_module=dict(
            act_cfg=dict(inplace=True, type='SiLU'),
            embed_dims=768,
            featmap_strides=[
                8,
                16,
                32,
            ],
            in_channels=[
                256,
                512,
                512,
            ],
            norm_cfg=dict(eps=0.001, momentum=0.03, type='BN'),
            num_classes=16,
            reg_max=16,
            type='YOLOWorldHeadModule',
            use_bn_head=True,
            widen_factor=1.0),
        loss_bbox=dict(
            bbox_format='xyxy',
            iou_mode='ciou',
            loss_weight=7.5,
            reduction='sum',
            return_iou=False,
            type='IoULoss'),
        loss_cls=dict(
            loss_weight=0.5,
            reduction='none',
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True),
        loss_dfl=dict(
            loss_weight=0.375,
            reduction='mean',
            type='mmdet.DistributionFocalLoss'),
        prior_generator=dict(
            offset=0.5, strides=[
                8,
                16,
                32,
            ], type='mmdet.MlvlPointGenerator'),
        proposal_cfg=dict(
            max_per_img=300, nms_iou_threshold=0.7, pre_nms_topk=3000),
        type='PartialLabelYOLOWorldHead'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            0.0,
            0.0,
            0.0,
        ],
        std=[
            255.0,
            255.0,
            255.0,
        ],
        type='PartialLabelYOLOWDetDataPreprocessor'),
    loss_fgbg_consistency_weight=0.0,
    loss_fgbg_mask_bce_weight=0.0,
    loss_fgbg_mask_dice_weight=0.0,
    mm_neck=True,
    neck=dict(
        act_cfg=dict(inplace=True, type='SiLU'),
        adapter_enabled=True,
        block_cfg=dict(type='MaxSigmoidCSPLayerWithTwoConv'),
        context_floor=0.25,
        deepen_factor=1.0,
        dual_stream_cfg=dict(
            fixed_scale=0.02,
            hidden_channels=64,
            learnable_scale=True,
            norm_groups=8,
            residual_scale=0.1,
            scale_mode='fixed',
            scale_warmup_iters=1000),
        embed_channels=[
            128,
            256,
            256,
        ],
        fgbg_cfg=None,
        guide_channels=768,
        in_channels=[
            256,
            512,
            512,
        ],
        ipooling_cfg=None,
        norm_cfg=dict(eps=0.001, momentum=0.03, type='BN'),
        num_csp_blocks=3,
        num_heads=[
            4,
            8,
            8,
        ],
        objectness_reweight=True,
        out_channels=[
            256,
            512,
            512,
        ],
        ovc_enabled=True,
        prompt_bank_compatible=True,
        semantic_reliability_cfg=dict(
            fusion_weights=(
                0.5,
                0.25,
            ), std_floor=0.001, temperature=1.0),
        type='OVCYOLOWorldWSRPAFPN',
        widen_factor=1.0,
        wsr_cfg=dict(
            detach_reliability=True,
            detail_kernel=5,
            embed_channels=64,
            enhance_level=0,
            exclusion_radius=1,
            grid_size=8,
            in_channels=[
                256,
                512,
                512,
            ],
            num_detail_tokens=64,
            num_reliable_tokens=128,
            num_visual_tokens=32,
            reliability_log_weight=0.0,
            reliability_nms_kernel=7,
            residual_scale=0.1,
            support_dropout=0.2,
            type='DualSourceWeakSupportRetriever')),
    num_test_classes=20,
    num_train_classes=16,
    ovc_training_cfg=dict(
        color_amount=0.2,
        consistency_enabled=True,
        consistency_weight=0.05,
        covariance_weight=0.001,
        grayscale_prob=0.1,
        hue=0.02,
        max_rois_per_image=32,
        normalized_regularization=True,
        objectness_weight=0.1,
        variance_weight=0.01,
        warmup_iters=1000),
    test_cfg=dict(
        max_per_img=300,
        multi_label=True,
        nms=dict(iou_threshold=0.7, type='nms'),
        nms_pre=30000,
        score_thr=0.001),
    train_cfg=dict(
        assigner=dict(
            alpha=0.5,
            beta=6.0,
            eps=1e-09,
            num_classes=16,
            topk=10,
            type='BatchTaskAlignedAssigner',
            use_ciou=True)),
    type='YOLOWorldOVCPartialDetector')
model_test_cfg = dict(
    max_per_img=300,
    multi_label=True,
    nms=dict(iou_threshold=0.7, type='nms'),
    nms_pre=30000,
    score_thr=0.001)
mosaic_affine_transform = [
    dict(
        img_scale=(
            640,
            640,
        ),
        pad_val=114.0,
        pre_transform=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
        ],
        type='Mosaic'),
    dict(
        border=(
            -320,
            -320,
        ),
        border_val=(
            114,
            114,
            114,
        ),
        max_aspect_ratio=100,
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(
            0.09999999999999998,
            1.9,
        ),
        type='YOLOv5RandomAffine'),
]
neck_embed_channels = [
    128,
    256,
    256,
]
neck_num_heads = [
    4,
    8,
    8,
]
norm_cfg = dict(eps=0.001, momentum=0.03, type='BN')
num_classes = 20
num_det_layers = 3
num_test_classes = 20
num_training_classes = 16
optim_wrapper = dict(
    accumulative_counts=1,
    clip_grad=dict(max_norm=10.0),
    constructor='OVCOptimizerConstructor',
    loss_scale='dynamic',
    optimizer=dict(
        batch_size_per_gpu=4, lr=0.001, type='AdamW', weight_decay=0.0125),
    paramwise_cfg=dict(
        bias_decay_mult=0.0,
        custom_keys=dict({
            'backbone.text_model':
            dict(lr_mult=0.01),
            'backbone.text_model.prompt_bank':
            dict(lr_mult=0.5),
            'backbone.text_model.prompt_bank.class_score_projs':
            dict(lr_mult=0.5),
            'backbone.text_model.prompt_bank.image_projs':
            dict(lr_mult=0.5),
            'backbone.text_model.prompt_bank.keys':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.text_model.prompt_bank.prompts':
            dict(decay_mult=0.0, lr_mult=0.5),
            'logit_scale':
            dict(weight_decay=0.0)
        }),
        norm_decay_mult=0.0),
    type='OVCTrackedAmpOptimWrapper')
param_scheduler = None
persistent_workers = True
pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(scale=(
        800,
        800,
    ), type='YOLOv5KeepRatioResize'),
    dict(
        allow_scale_up=True,
        pad_val=dict(img=114.0),
        scale=(
            800,
            800,
        ),
        type='LetterResize'),
    dict(
        border_val=(
            114,
            114,
            114,
        ),
        max_aspect_ratio=100,
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(
            0.09999999999999998,
            1.9,
        ),
        type='YOLOv5RandomAffine'),
    dict(
        bbox_params=dict(
            format='pascal_voc',
            label_fields=[
                'gt_bboxes_labels',
                'gt_ignore_flags',
            ],
            type='BboxParams'),
        keymap=dict(gt_bboxes='bboxes', img='image'),
        transforms=[
            dict(p=0.01, type='Blur'),
            dict(p=0.01, type='MedianBlur'),
            dict(p=0.01, type='ToGray'),
            dict(p=0.01, type='CLAHE'),
        ],
        type='mmdet.Albu'),
    dict(type='YOLOv5HSVRandomAug'),
    dict(prob=0.5, type='mmdet.RandomFlip'),
    dict(
        max_num_samples=16,
        num_neg_samples=(
            16,
            16,
        ),
        padding_to_max=True,
        padding_value='',
        type='RandomLoadText'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'flip',
            'flip_direction',
            'texts',
            'partial_labels',
        ),
        type='mmdet.PackDetInputs'),
]
pre_transform = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
]
randomness = dict(seed=179871018)
resume = False
save_epoch_intervals = 2
strides = [
    8,
    16,
    32,
]
tal_alpha = 0.5
tal_beta = 6.0
tal_topk = 10
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        class_text_path='data/DIOR/classes_full_20cls.json',
        dataset=dict(
            ann_file='Annotations2/val.json',
            batch_shapes_cfg=None,
            data_prefix=dict(img='JPEGImages/'),
            data_root='data/DIOR',
            metainfo=dict(
                classes=(
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
                )),
            test_mode=True,
            type='YOLOv5CocoDataset'),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(scale=(
                800,
                800,
            ), type='YOLOv5KeepRatioResize'),
            dict(
                allow_scale_up=False,
                pad_val=dict(img=114),
                scale=(
                    800,
                    800,
                ),
                type='LetterResize'),
            dict(_scope_='mmdet', type='LoadAnnotations', with_bbox=True),
            dict(type='LoadText'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                    'pad_param',
                    'texts',
                ),
                type='mmdet.PackDetInputs'),
        ],
        type='MultiModalDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = [
    dict(
        ann_file='data/DIOR/Annotations2/val.json',
        base_classes=(
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
        ),
        metric='bbox',
        novel_classes=(
            'airport',
            'basketballcourt',
            'groundtrackfield',
            'windmill',
        ),
        proposal_nums=(
            100,
            1,
            10,
        ),
        type='OVCDIORCocoMetric'),
    dict(
        ann_file='data/DIOR/Annotations2/val.json',
        base_classes=(
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
        ),
        iou_thresholds=(
            0.3,
            0.5,
        ),
        novel_classes=(
            'airport',
            'basketballcourt',
            'groundtrackfield',
            'windmill',
        ),
        prefix='proposal_O',
        proposal_nums=(
            100,
            300,
        ),
        report_classwise=True,
        type='PriorProposalRecallMetric'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(scale=(
        800,
        800,
    ), type='YOLOv5KeepRatioResize'),
    dict(
        allow_scale_up=False,
        pad_val=dict(img=114),
        scale=(
            800,
            800,
        ),
        type='LetterResize'),
    dict(_scope_='mmdet', type='LoadAnnotations', with_bbox=True),
    dict(type='LoadText'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
            'pad_param',
            'texts',
        ),
        type='mmdet.PackDetInputs'),
]
text_channels = 768
text_model_name = 'pretrained_models/clip-vit-large-patch14-336'
text_transform = [
    dict(
        max_num_samples=16,
        num_neg_samples=(
            16,
            16,
        ),
        padding_to_max=True,
        padding_value='',
        type='RandomLoadText'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'flip',
            'flip_direction',
            'texts',
        ),
        type='mmdet.PackDetInputs'),
]
train_ann_file = 'annotations/instances_train2017.json'
train_batch_size_per_gpu = 4
train_cfg = dict(
    dynamic_intervals=[
        (
            49772,
            1082,
        ),
    ],
    max_iters=54100,
    type='IterBasedTrainLoop',
    val_interval=4328)
train_data_prefix = 'train2017/'
train_dataloader = dict(
    batch_size=4,
    collate_fn=dict(type='yolow_partial_collate'),
    dataset=dict(
        class_text_path='data/DIOR/classes_base_16cls.json',
        dataset=dict(
            ann_file='Annotations2/train_base_16cls_image_disjoint_yoloworld.json',
            data_prefix=dict(img='JPEGImages/'),
            data_root='data/DIOR',
            filter_cfg=dict(filter_empty_gt=False, min_size=32),
            metainfo=dict(
                classes=(
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
                )),
            type='PartialLabelYOLOv5CocoDataset'),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                img_scale=(
                    800,
                    800,
                ),
                pad_val=114.0,
                pre_transform=[
                    dict(backend_args=None, type='LoadImageFromFile'),
                    dict(type='LoadAnnotations', with_bbox=True),
                ],
                type='MultiModalMosaic'),
            dict(
                border=(
                    -400,
                    -400,
                ),
                border_val=(
                    114,
                    114,
                    114,
                ),
                max_aspect_ratio=100,
                max_rotate_degree=0.0,
                max_shear_degree=0.0,
                scaling_ratio_range=(
                    0.09999999999999998,
                    1.9,
                ),
                type='YOLOv5RandomAffine'),
            dict(
                bbox_params=dict(
                    format='pascal_voc',
                    label_fields=[
                        'gt_bboxes_labels',
                        'gt_ignore_flags',
                    ],
                    type='BboxParams'),
                keymap=dict(gt_bboxes='bboxes', img='image'),
                transforms=[
                    dict(p=0.01, type='Blur'),
                    dict(p=0.01, type='MedianBlur'),
                    dict(p=0.01, type='ToGray'),
                    dict(p=0.01, type='CLAHE'),
                ],
                type='mmdet.Albu'),
            dict(type='YOLOv5HSVRandomAug'),
            dict(prob=0.5, type='mmdet.RandomFlip'),
            dict(
                max_num_samples=16,
                num_neg_samples=(
                    16,
                    16,
                ),
                padding_to_max=True,
                padding_value='',
                type='RandomLoadText'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'flip',
                    'flip_direction',
                    'texts',
                    'partial_labels',
                ),
                type='mmdet.PackDetInputs'),
        ],
        type='MultiModalDataset'),
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=True, type='OVCInfiniteSampler'))
train_num_workers = 8
train_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        img_scale=(
            800,
            800,
        ),
        pad_val=114.0,
        pre_transform=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
        ],
        type='MultiModalMosaic'),
    dict(
        border=(
            -400,
            -400,
        ),
        border_val=(
            114,
            114,
            114,
        ),
        max_aspect_ratio=100,
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(
            0.09999999999999998,
            1.9,
        ),
        type='YOLOv5RandomAffine'),
    dict(
        bbox_params=dict(
            format='pascal_voc',
            label_fields=[
                'gt_bboxes_labels',
                'gt_ignore_flags',
            ],
            type='BboxParams'),
        keymap=dict(gt_bboxes='bboxes', img='image'),
        transforms=[
            dict(p=0.01, type='Blur'),
            dict(p=0.01, type='MedianBlur'),
            dict(p=0.01, type='ToGray'),
            dict(p=0.01, type='CLAHE'),
        ],
        type='mmdet.Albu'),
    dict(type='YOLOv5HSVRandomAug'),
    dict(prob=0.5, type='mmdet.RandomFlip'),
    dict(
        max_num_samples=16,
        num_neg_samples=(
            16,
            16,
        ),
        padding_to_max=True,
        padding_value='',
        type='RandomLoadText'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'flip',
            'flip_direction',
            'texts',
            'partial_labels',
        ),
        type='mmdet.PackDetInputs'),
]
train_pipeline_close_mosaic = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(scale=(
        800,
        800,
    ), type='YOLOv5KeepRatioResize'),
    dict(
        allow_scale_up=True,
        pad_val=dict(img=114.0),
        scale=(
            800,
            800,
        ),
        type='LetterResize'),
    dict(
        border_val=(
            114,
            114,
            114,
        ),
        max_aspect_ratio=100,
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(
            0.09999999999999998,
            1.9,
        ),
        type='YOLOv5RandomAffine'),
    dict(
        bbox_params=dict(
            format='pascal_voc',
            label_fields=[
                'gt_bboxes_labels',
                'gt_ignore_flags',
            ],
            type='BboxParams'),
        keymap=dict(gt_bboxes='bboxes', img='image'),
        transforms=[
            dict(p=0.01, type='Blur'),
            dict(p=0.01, type='MedianBlur'),
            dict(p=0.01, type='ToGray'),
            dict(p=0.01, type='CLAHE'),
        ],
        type='mmdet.Albu'),
    dict(type='YOLOv5HSVRandomAug'),
    dict(prob=0.5, type='mmdet.RandomFlip'),
    dict(
        max_num_samples=16,
        num_neg_samples=(
            16,
            16,
        ),
        padding_to_max=True,
        padding_value='',
        type='RandomLoadText'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'flip',
            'flip_direction',
            'texts',
            'partial_labels',
        ),
        type='mmdet.PackDetInputs'),
]
train_pipeline_stage2 = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(scale=(
        640,
        640,
    ), type='YOLOv5KeepRatioResize'),
    dict(
        allow_scale_up=True,
        pad_val=dict(img=114.0),
        scale=(
            640,
            640,
        ),
        type='LetterResize'),
    dict(
        border_val=(
            114,
            114,
            114,
        ),
        max_aspect_ratio=100,
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(
            0.09999999999999998,
            1.9,
        ),
        type='YOLOv5RandomAffine'),
    dict(
        bbox_params=dict(
            format='pascal_voc',
            label_fields=[
                'gt_bboxes_labels',
                'gt_ignore_flags',
            ],
            type='BboxParams'),
        keymap=dict(gt_bboxes='bboxes', img='image'),
        transforms=[
            dict(p=0.01, type='Blur'),
            dict(p=0.01, type='MedianBlur'),
            dict(p=0.01, type='ToGray'),
            dict(p=0.01, type='CLAHE'),
        ],
        type='mmdet.Albu'),
    dict(type='YOLOv5HSVRandomAug'),
    dict(prob=0.5, type='mmdet.RandomFlip'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'flip',
            'flip_direction',
        ),
        type='mmdet.PackDetInputs'),
]
transform = dict(
    meta_keys=(
        'img_id',
        'img_path',
        'ori_shape',
        'img_shape',
        'flip',
        'flip_direction',
        'texts',
        'partial_labels',
    ),
    type='mmdet.PackDetInputs')
tta_model = dict(
    tta_cfg=dict(max_per_img=300, nms=dict(iou_threshold=0.65, type='nms')),
    type='mmdet.DetTTAModel')
tta_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(
        transforms=[
            [
                dict(
                    transforms=[
                        dict(scale=(
                            640,
                            640,
                        ), type='YOLOv5KeepRatioResize'),
                        dict(
                            allow_scale_up=False,
                            pad_val=dict(img=114),
                            scale=(
                                640,
                                640,
                            ),
                            type='LetterResize'),
                    ],
                    type='Compose'),
                dict(
                    transforms=[
                        dict(scale=(
                            320,
                            320,
                        ), type='YOLOv5KeepRatioResize'),
                        dict(
                            allow_scale_up=False,
                            pad_val=dict(img=114),
                            scale=(
                                320,
                                320,
                            ),
                            type='LetterResize'),
                    ],
                    type='Compose'),
                dict(
                    transforms=[
                        dict(scale=(
                            960,
                            960,
                        ), type='YOLOv5KeepRatioResize'),
                        dict(
                            allow_scale_up=False,
                            pad_val=dict(img=114),
                            scale=(
                                960,
                                960,
                            ),
                            type='LetterResize'),
                    ],
                    type='Compose'),
            ],
            [
                dict(prob=1.0, type='mmdet.RandomFlip'),
                dict(prob=0.0, type='mmdet.RandomFlip'),
            ],
            [
                dict(type='mmdet.LoadAnnotations', with_bbox=True),
            ],
            [
                dict(
                    meta_keys=(
                        'img_id',
                        'img_path',
                        'ori_shape',
                        'img_shape',
                        'scale_factor',
                        'pad_param',
                        'flip',
                        'flip_direction',
                    ),
                    type='mmdet.PackDetInputs'),
            ],
        ],
        type='TestTimeAug'),
]
val_ann_file = 'annotations/instances_val2017.json'
val_batch_size_per_gpu = 1
val_cfg = dict(type='ValLoop')
val_data_prefix = 'train2017/'
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        class_text_path='data/DIOR/classes_full_20cls.json',
        dataset=dict(
            ann_file='Annotations2/val.json',
            batch_shapes_cfg=None,
            data_prefix=dict(img='JPEGImages/'),
            data_root='data/DIOR',
            metainfo=dict(
                classes=(
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
                )),
            test_mode=True,
            type='YOLOv5CocoDataset'),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(scale=(
                800,
                800,
            ), type='YOLOv5KeepRatioResize'),
            dict(
                allow_scale_up=False,
                pad_val=dict(img=114),
                scale=(
                    800,
                    800,
                ),
                type='LetterResize'),
            dict(_scope_='mmdet', type='LoadAnnotations', with_bbox=True),
            dict(type='LoadText'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                    'pad_param',
                    'texts',
                ),
                type='mmdet.PackDetInputs'),
        ],
        type='MultiModalDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = [
    dict(
        ann_file='data/DIOR/Annotations2/val.json',
        base_classes=(
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
        ),
        metric='bbox',
        novel_classes=(
            'airport',
            'basketballcourt',
            'groundtrackfield',
            'windmill',
        ),
        proposal_nums=(
            100,
            1,
            10,
        ),
        type='OVCDIORCocoMetric'),
    dict(
        ann_file='data/DIOR/Annotations2/val.json',
        base_classes=(
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
        ),
        iou_thresholds=(
            0.3,
            0.5,
        ),
        novel_classes=(
            'airport',
            'basketballcourt',
            'groundtrackfield',
            'windmill',
        ),
        prefix='proposal_O',
        proposal_nums=(
            100,
            300,
        ),
        report_classwise=True,
        type='PriorProposalRecallMetric'),
]
val_interval_stage2 = 1
val_num_workers = 2
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='mmdet.DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
    ])
weight_decay = 0.0125
widen_factor = 1.0
work_dir = 'work_dirs/DIORv2.17'

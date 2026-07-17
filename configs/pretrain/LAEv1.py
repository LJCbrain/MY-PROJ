import json
import pathlib

_base_ = ('../../third_party/mmyolo/configs/yolov8/'
          'yolov8_l_syncbn_fast_8xb16-500e_coco.py')
OUT_CHANNELS_ = [128, 256, _base_.last_stage_out_channels // 2]
custom_imports = dict(imports=['yolo_world'], allow_failed_imports=False)


def _load_class_texts(path):
    """Read one LAE class-text table as the matching COCO class tuple."""
    source = pathlib.Path(path)
    with source.open('r', encoding='utf-8') as file:
        entries = json.load(file)
    texts = []
    for index, entry in enumerate(entries):
        if isinstance(entry, str):
            text = entry
        elif (isinstance(entry, list) and len(entry) == 1
              and isinstance(entry[0], str)):
            text = entry[0]
        else:
            raise TypeError(
                f'Invalid class-text entry {index} in {source}; expected '
                'a string or a one-item string list.')
        if not text:
            raise ValueError(f'Empty class text {index} in {source}.')
        texts.append(text)
    return tuple(texts)


# LAE-80C is the sole validation/test vocabulary.  Training data remains in
# the original per-source local label spaces and is remapped per image by
# RandomLoadText to the fixed 80 prompt slots.
data_root = 'data/LAE-1M/'
num_classes = 80
num_training_classes = 80
num_test_classes = 80

max_epochs = 32
close_mosaic_epochs = 2
save_epoch_intervals = 2
text_channels = 768
neck_embed_channels = OUT_CHANNELS_
neck_num_heads = [4, 8, _base_.last_stage_out_channels // 2 // 32]
base_lr = 2e-3
weight_decay = 0.05 / 2
train_batch_size_per_gpu =4
img_scale = (800, 800)
text_model_name = r'pretrained_models/clip-vit-large-patch14-336'
load_from = (
    'weights/yolo_world_v2_l_clip_large_o365v1_goldg_pretrain_800ft-'
    '9df82e55.pth')

semantic_anchor_path = (
    'data/LAE-1M/semantic_anchors/lae_1m_anchor_96.json')
semantic_anchor_embed_path = (
    'data/LAE-1M/semantic_anchors/lae_1m_anchor_96_clip_vit_large.pt')
semantic_anchor_texts = _load_class_texts(semantic_anchor_path)
if len(semantic_anchor_texts) != 96:
    raise ValueError(
        f'Expected exactly 96 LAE semantic anchors, got '
        f'{len(semantic_anchor_texts)} from {semantic_anchor_path}.')


model = dict(
    type='YOLOWorldDetector',
    mm_neck=True,
    num_train_classes=num_training_classes,
    num_test_classes=num_test_classes,
    loss_fgbg_mask_bce_weight=0.03,
    loss_fgbg_mask_dice_weight=0.03,
    loss_fgbg_quality_weight=0.01,
    loss_fgbg_consistency_weight=0.003,
    data_preprocessor=dict(type='YOLOWDetDataPreprocessor'),
    backbone=dict(
        _delete_=True,
        type='MultiModalYOLOBackbone',
        image_model={{_base_.model.backbone}},
        text_model=dict(
            type='HuggingCLIPLanguageBackbone',
            model_name=text_model_name,
            frozen_modules=['all'],
            prompt_bank_cfg=dict(
                image_channels=[256, 512, _base_.last_stage_out_channels],
                num_prompts=10,
                prompt_length=12,
                num_scales=3,
                key_dim=text_channels,
                use_reliability=True,
                loss_match_weight=0.7,
                loss_orthogonal_weight=0.3,
                semantic_anchor_texts=semantic_anchor_texts,
                semantic_anchor_embed_path=semantic_anchor_embed_path,
                semantic_anchor_weight=0.05,
                semantic_anchor_warmup_iters=1000))),
    neck=dict(
        type='YOLOWorldWSRPAFPN',
        guide_channels=text_channels,
        embed_channels=neck_embed_channels,
        num_heads=neck_num_heads,
        prompt_bank_compatible=True,
        block_cfg=dict(type='MaxSigmoidCSPLayerWithTwoConv'),
        fgbg_cfg=dict(
            type='ClassAgnosticForegroundGate',
            in_channels=[256, 512, _base_.last_stage_out_channels],
            hidden_ratio=0.25,
            min_hidden=16,
            max_gamma=0.15,
            gamma_init_bias=-0.3,
            gamma_warmup_iters=1000,
            return_parts=True),
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
            enhance_level=0)),
    bbox_head=dict(
        type='YOLOWorldHead',
        head_module=dict(
            type='YOLOWorldHeadModule',
            use_bn_head=True,
            embed_dims=text_channels,
            num_classes=num_training_classes)),
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
    *text_transform,
]


# Full LAE-1M training sources.  category_id_offset=-1 is required only for
# xView, whose COCO category definitions are 0..59 while annotations are 1..60.
train_sources = [
    dict(
        name='AID',
        img='LAE-COD/AID/images/images',
        ann='LAE-COD/AID/AID.json',
        texts='LAE-COD/AID/AID_categories.json'),
    dict(
        name='EMS',
        img='LAE-COD/EMS/images/images',
        ann='LAE-COD/EMS/EMS.json',
        texts='LAE-COD/EMS/EMS_categories.json'),
    dict(
        name='NWPU-RESISC45',
        img='LAE-COD/NWPU-RESISC45/images/images',
        ann='LAE-COD/NWPU-RESISC45/NWPURESISC45.json',
        texts='LAE-COD/NWPU-RESISC45/NWPU-RESISC45_categories.json'),
    dict(
        name='SLM',
        img='LAE-COD/SLM/images/images',
        ann='LAE-COD/SLM/SLM.json',
        texts='LAE-COD/SLM/SLM_categories.json'),
    dict(
        name='DIOR',
        img='LAE-FOD/DIOR/trainval_images/JPEGImages-trainval',
        ann='LAE-FOD/DIOR/processed_LAE-1M_DIOR_train.json',
        texts='LAE-FOD/DIOR/DIOR_categories.json'),
    dict(
        name='DOTAv2',
        img='LAE-FOD/DOTAv2/images/images',
        ann='LAE-FOD/DOTAv2/processed_LAE-1M_DOTAv2_train.json',
        texts='LAE-FOD/DOTAv2/DOTAv2_categories.json'),
    dict(
        name='FAIR1M',
        img='LAE-FOD/FAIR1M/images/images',
        ann='LAE-FOD/FAIR1M/processed_LAE-1M_FAIR1M_train.json',
        texts='LAE-FOD/FAIR1M/FAIR1M_categories.json'),
    dict(
        name='HRSC2016',
        img='LAE-FOD/HRSC2016/images/images',
        ann='LAE-FOD/HRSC2016/processed_LAE-1M_HRSC2016_train.json',
        texts='LAE-FOD/HRSC2016/HRSC2016_categories.json'),
    dict(
        name='NWPU-VHR-10',
        img='LAE-FOD/NWPU VHR-10/images/images',
        ann='LAE-FOD/NWPU VHR-10/processed_LAE-1M_NWPU-VHR-10_train.json',
        texts='LAE-FOD/NWPU VHR-10/NWPU VHR-10_categories.json'),
    dict(
        name='Power-Plant',
        img='LAE-FOD/Power-Plant/Images/Images',
        ann='LAE-FOD/Power-Plant/processed_LAE-1M_Power-Plant_train.json',
        texts='LAE-FOD/Power-Plant/Power-Plant_categories.json'),
    dict(
        name='RSOD',
        img='LAE-FOD/RSOD/images/images',
        ann='LAE-FOD/RSOD/processed_LAE-1M_RSOD_train.json',
        texts='LAE-FOD/RSOD/RSOD_categories.json'),
    dict(
        name='xView',
        img='LAE-FOD/xview/train_images_1024_05/train_images_1024_05',
        ann='LAE-FOD/xview/processed_LAE-1M_Xview_train_1024_05.json',
        texts='LAE-FOD/xview/xview_categories.json',
        annotation_category_id_offset=-1),
]


def _build_lae_train_dataset(source):
    class_text_path = f"{data_root}{source['texts']}"
    return dict(
        type='MultiModalDataset',
        dataset=dict(
            type='LAEYOLOv5CocoDataset',
            data_root=data_root,
            ann_file=source['ann'],
            data_prefix=dict(img=source['img']),
            metainfo=dict(classes=_load_class_texts(class_text_path)),
            class_text_path=class_text_path,
            annotation_category_id_offset=source.get(
                'annotation_category_id_offset', 0),
            strict_category_validation=True,
            filter_cfg=dict(filter_empty_gt=False, min_size=32)),
        class_text_path=class_text_path,
        pipeline=train_pipeline)


train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    collate_fn=dict(type='yolow_collate'),
    dataset=dict(
        _delete_=True,
        type='ConcatDataset',
        datasets=[_build_lae_train_dataset(source) for source in train_sources],
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

lae_80c_class_text_path = (
    'data/LAE-1M/LAE-80C/LAE-80C-benchmark_categories.json')
lae_80c_dataset = dict(
    _delete_=True,
    type='MultiModalDataset',
    dataset=dict(
        type='LAEYOLOv5CocoDataset',
        data_root=data_root,
        test_mode=True,
        ann_file='LAE-80C/LAE-80C-benchmark.json',
        data_prefix=dict(img='LAE-80C/images/images/'),
        metainfo=dict(classes=_load_class_texts(lae_80c_class_text_path)),
        class_text_path=lae_80c_class_text_path,
        strict_category_validation=True,
        batch_shapes_cfg=None),
    class_text_path=lae_80c_class_text_path,
    pipeline=test_pipeline)

val_dataloader = dict(dataset=lae_80c_dataset)
test_dataloader = val_dataloader

val_evaluator = dict(
    type='mmdet.CocoMetric',
    ann_file='data/LAE-1M/LAE-80C/LAE-80C-benchmark.json',
    metric='bbox')
test_evaluator = val_evaluator


default_hooks = dict(
    param_scheduler=dict(max_epochs=max_epochs),
    checkpoint=dict(
        interval=save_epoch_intervals,
        save_best='coco/bbox_mAP',
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
        type='mmdet.PipelineSwitchHook',
        switch_epoch=max_epochs - close_mosaic_epochs,
        switch_pipeline=train_pipeline_stage2)
]
train_cfg = dict(
    max_epochs=max_epochs,
    val_interval=3,
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
    # PyTorch 1.11 CUDA kernels used by this neck do not support BF16
    # nearest-neighbor upsampling.  FP16 AMP remains compatible here.
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    dtype='float16',
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
            'backbone.text_model.prompt_bank.prompts': dict(
                lr_mult=0.5, decay_mult=0.0),
            'backbone.text_model.prompt_bank.keys': dict(
                lr_mult=0.5, decay_mult=0.0),
            'backbone.text_model.prompt_bank.image_projs': dict(lr_mult=0.5),
            'backbone.text_model.prompt_bank': dict(lr_mult=0.5),
            'backbone.text_model': dict(lr_mult=0.01),
            'logit_scale': dict(weight_decay=0.0)
        }),
    constructor='YOLOWv5OptimizerConstructor')

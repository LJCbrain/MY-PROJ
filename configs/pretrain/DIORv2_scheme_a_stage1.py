"""Scheme A Stage 1: image-disjoint Base-16 supervised training."""

_base_ = './DIORv2.py'

custom_imports = dict(
    imports=[
        'yolo_world',
        'yolo_world.evaluation',
        'yolo_world.models.detectors.yolo_world_scheme_a',
    ],
    allow_failed_imports=False)

max_epochs = 50
close_mosaic_epochs = 2
save_epoch_intervals = 2

model = dict(
    type='SchemeAYOLOWorldDetector',
    freeze_prompt_bank=False,
    clean_only_fgbg_loss=False,
    prompt_bank_inference_mode='prompted',
    prompt_bank_base_classes=_base_.DIOR_BASE_CLASSES,
    neck=dict(
        fgbg_cfg=dict(
            _delete_=True,
            type='RegionalBackgroundPriorGate',
            in_channels=[256, 512, _base_.last_stage_out_channels],
            region_grid_size=3,
            background_keep_ratio=0.75,
            attention_heads=[4, 8, 8],
            hidden_ratio=0.25,
            min_hidden=16,
            quality_init_prob=0.01,
            max_gamma=0.15,
            gamma_init_bias=-0.3,
            gamma_warmup_iters=1000,
            fusion_weights=(0.5, 0.25),
            return_parts=True)),
    bbox_head=dict(
        proposal_cfg=dict(
            pre_nms_topk=3000,
            nms_iou_threshold=0.7,
            max_per_img=300)))

val_evaluator = [
    dict(
        type='DIORCocoMetric',
        ann_file='data/DIOR/Annotations2/val.json',
        metric='bbox',
        base_classes=_base_.DIOR_BASE_CLASSES,
        novel_classes=_base_.DIOR_NOVEL_CLASSES),
    dict(
        type='PriorProposalRecallMetric',
        ann_file='data/DIOR/Annotations2/val.json',
        base_classes=_base_.DIOR_BASE_CLASSES,
        novel_classes=_base_.DIOR_NOVEL_CLASSES,
        iou_thresholds=(0.3, 0.5),
        proposal_num=300),
]
test_evaluator = [
    dict(
        type='DIORCocoMetric',
        ann_file='data/DIOR/Annotations2/test.json',
        metric='bbox',
        base_classes=_base_.DIOR_BASE_CLASSES,
        novel_classes=_base_.DIOR_NOVEL_CLASSES),
    dict(
        type='PriorProposalRecallMetric',
        ann_file='data/DIOR/Annotations2/test.json',
        base_classes=_base_.DIOR_BASE_CLASSES,
        novel_classes=_base_.DIOR_NOVEL_CLASSES,
        iou_thresholds=(0.3, 0.5),
        proposal_num=300),
]

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
        type='mmdet.PipelineSwitchHook',
        switch_epoch=max_epochs - close_mosaic_epochs,
        switch_pipeline=_base_.train_pipeline_stage2),
]
train_cfg = dict(
    max_epochs=max_epochs,
    val_interval=5,
    dynamic_intervals=[(20, 2)])

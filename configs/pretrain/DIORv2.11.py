_base_ = './DIORv2.10.py'

# CAVP-Full: preserve the v2.10 semantic path and its normal training.
# WSR still consumes Regional reliability. CAQR is not instantiated here.
model = dict(
    type='YOLOWorldCAVPDetector',
    neck=dict(fgbg_cfg=dict(use_caqr=False, build_caqr_modules=False)),
    cavp_neck=dict(
        type='CAVPVisualPAFPN',
        in_channels=[256, 512, 512], out_channels=[128, 256, 256],
        num_csp_blocks=1),
    cavp_head=dict(
        type='CAVPHead',
        head_module=dict(
            type='YOLOv8HeadModule', num_classes=1,
            in_channels=[128, 256, 256], reg_max=16,
            featmap_strides=[8, 16, 32]),
        prior_generator=dict(
            type='mmdet.MlvlPointGenerator', offset=0.5, strides=[8, 16, 32]),
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        loss_cls=dict(type='mmdet.CrossEntropyLoss', use_sigmoid=True,
                      reduction='none', loss_weight=0.5),
        loss_bbox=dict(type='IoULoss', iou_mode='ciou', bbox_format='xyxy',
                       reduction='sum', loss_weight=7.5, return_iou=False),
        loss_dfl=dict(type='mmdet.DistributionFocalLoss', reduction='mean',
                      loss_weight=1.5 / 4),
        train_cfg=dict(assigner=dict(
            type='BatchTaskAlignedAssigner', num_classes=1, use_ciou=False,
            topk=10, alpha=0.5, beta=6.0, eps=1e-9))),
    cavp_cfg=dict(
        enabled=True, fusion_enabled=False, loss_weight=1.0,
        pre_nms_topk=3000, proposal_nms_iou=0.7, proposal_max_per_img=300,
        semantic_readout='source_center', source_weight=0.5,
        semantic_score_thr=0.001, semantic_margin=0.0,
        score_temperature=1.0, score_bias=0.0,
        duplicate_iou=0.7, max_per_img=300))

# Preserve AP/HM and checkpoint selection; report visual proposal coverage
# separately. These fixed-IoU recalls are not COCO-style averaged AR.
cavp_evaluator = dict(
    type='PriorProposalRecallMetric',
    ann_file='data/DIOR/Annotations2/val.json',
    base_classes=_base_.DIOR_BASE_CLASSES,
    novel_classes=_base_.DIOR_NOVEL_CLASSES,
    proposal_nums=(100, 300), iou_thresholds=(0.3, 0.5),
    report_classwise=True, prefix='cavp_proposal')
val_evaluator = [_base_.val_evaluator, cavp_evaluator]
test_evaluator = [_base_.test_evaluator, cavp_evaluator]

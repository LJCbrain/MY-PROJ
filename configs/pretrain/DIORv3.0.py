"""RemoteCLIP region distillation into the existing YOLO-World classifier.

Baseline: DIORv2.18 (2.17 protocol + frozen RemoteCLIP text), image-disjoint
Base-16 training. Teacher uses ONLY transformed training GT regions.
Inference does not load the teacher or require its original .pt file.

Required for training:
  pretrained_models/RemoteCLIP-ViT-L-14.pt
  pretrained_models/remoteclip-vit-large-patch14-text/
  weights/yolo_world_v2_l_remoteclip_text_init.pth

Disable just distillation with:
  --cfg-options model.remoteclip_distill_cfg.loss_weight=0
"""

_base_ = './DIORv2.18.py'

custom_imports = dict(
    allow_failed_imports=False,
    imports=['yolo_world', 'yolo_world.evaluation',
             'yolo_world.models.detectors.yolo_world_remoteclip_distill'])

model = dict(
    type='YOLOWorldRemoteCLIPDistillDetector',
    train_cfg=dict(assigner=dict(type='DistillBatchTaskAlignedAssigner')),
    remoteclip_distill_cfg=dict(
        checkpoint='pretrained_models/RemoteCLIP-ViT-L-14.pt',
        loss_weight=0.1,
        warmup_iters=2000,
        max_regions_per_image=8,
        teacher_batch_size=8,
        context=1.0,
        min_box_size=2.0,
        precision='fp16',
        seed=300))

work_dir = 'work_dirs/DIORv3.0'

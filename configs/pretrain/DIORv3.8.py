# DIOR 3.8: evaluation-only Base score gate on the DIOR 2.17 best weights.
# No retraining: retain the original model, Novel score threshold and multi-label
# postprocessing. Gate Base pairs BEFORE global top-k, NMS and the image limit.
_base_ = './DIORv2.17best.py'

work_dir = 'work_dirs/DIORv3.8_base_thr002'
load_from = 'work_dirs/DIORv2.17/best_coco_bbox_mAP_50_HM_iter_25968.pth'
resume = False

# These are 0-based columns in the full 20-class inference text bank.
base_class_ids = [
    _base_.DIOR_FULL_CLASSES.index(name)
    for name in _base_.DIOR_BASE_CLASSES
]
base_score_thr = 0.02

model = dict(test_cfg=dict(
    base_class_ids=base_class_ids,
    base_score_thr=base_score_thr))
model_test_cfg = dict(
    _base_.model_test_cfg,
    base_class_ids=base_class_ids,
    base_score_thr=base_score_thr)

# tools/test.py overrides load_from, so always supply --checkpoint explicitly.
# python tools/test.py --config configs/pretrain/DIORv3.8.py \
#   --checkpoint work_dirs/DIORv2.17/best_coco_bbox_mAP_50_HM_iter_25968.pth \
#   --work-dir work_dirs/DIORv3.8_base_thr002 \
#   --out work_dirs/DIORv3.8_base_thr002/predictions.pkl

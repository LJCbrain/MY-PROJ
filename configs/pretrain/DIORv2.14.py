"""Single-model two-view Adapter + objectness + consistency + visual WSR.

No teacher, class holdout, CAVP or Regional. EMA is only for evaluation.
Inherits DIOR 2.13 data, 800px, batch=4, lr=1e-3 and 50-epoch schedule.
"""
_base_ = './DIORv2.13.py'

model = dict(
    type='YOLOWorldOVCDetector',
    ovc_training_cfg=dict(
        consistency_enabled=True, objectness_weight=0.1,
        consistency_weight=0.05, variance_weight=0.01, warmup_iters=1000,
        max_rois_per_image=32, color_amount=0.2, hue=0.02, grayscale_prob=0.1),
    neck=dict(
        type='OVCYOLOWorldWSRPAFPN', ovc_enabled=True, adapter_enabled=True,
        objectness_reweight=True, context_floor=0.25,
        wsr_cfg=dict(type='DualSourceWeakSupportRetriever',
                     num_reliable_tokens=128, num_visual_tokens=32,
                     reliability_log_weight=0.0)))

optim_wrapper = dict(
    type='OVCGroupedAmpOptimWrapper', constructor='OVCOptimizerConstructor')

val_evaluator = dict(type='OVCDIORCocoMetric')
test_evaluator = {**_base_.val_evaluator, '_delete_': True,
                  'type': 'OVCDIORCocoMetric'}

custom_hooks = [
    *[hook for hook in _base_.custom_hooks
      if hook['type'] != 'SemanticWSRDiagnosticsHook'],
    dict(type='OVCDiagnosticsHook', interval=50),
]

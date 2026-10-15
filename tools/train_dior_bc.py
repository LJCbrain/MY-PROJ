"""Short B/C fine-tuning from the DIOR 2.17 best weights (fresh optimizer).

Read the original configuration embedded in the checkpoint, rather than a
possibly edited config file. Keep the architecture/data and replace only the
classification position weights and the explicitly shared short-run schedule.
"""
import argparse
import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def build_config(source, checkpoint, case, work_dir, steps=8656, lr=1e-4):
    from mmengine.config import Config

    cfg = Config.fromstring(source, '.py')
    if cfg.model.type != 'YOLOWorldOVCPartialDetector':
        raise ValueError('Expected the DIOR 2.17 OVC checkpoint, without distillation.')
    head = cfg.model.bbox_head
    if head.type != 'PartialLabelYOLOWorldHead':
        raise ValueError('Checkpoint must use PartialLabelYOLOWorldHead.')
    head.cls_foreground_weight = 1.0 if case == 'B' else 0.25
    head.cls_background_weight = 0.25 if case == 'B' else 1.0
    cfg.load_from = str(checkpoint)
    cfg.resume = False
    cfg.work_dir = str(work_dir)
    cfg.launcher = 'none'
    cfg.optim_wrapper.optimizer.lr = lr
    cfg.base_lr = lr
    cfg.train_cfg = dict(type='IterBasedTrainLoop', max_iters=steps,
                         val_interval=min(4328, steps), dynamic_intervals=[])
    cfg.default_hooks.param_scheduler = dict(type='ParamSchedulerHook')
    cfg.param_scheduler = [dict(type='ConstantLR', factor=1.0,
                               begin=0, end=steps, by_epoch=False)]
    # The source checkpoint was saved before the clean-image stage. Retain
    # that pipeline throughout this short experiment, without a new switch.
    cfg.custom_hooks = [h for h in cfg.get('custom_hooks', [])
                        if h['type'] != 'OVCIterationPipelineSwitchHook']
    hook = cfg.default_hooks.checkpoint
    hook.update(by_epoch=False, interval=min(4328, steps), max_keep_ckpts=2)
    hook.pop('out_dir', None)
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', required=True, choices=['B', 'C'])
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--work-dir', required=True)
    parser.add_argument('--steps', type=int, default=8656)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()
    if args.steps < 1 or not 0 < args.lr < float('inf'):
        parser.error('--steps and --lr must be positive and finite')
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    work_dir = Path(args.work_dir).expanduser().resolve()
    if not checkpoint.is_file():
        parser.error('Checkpoint does not exist: {}'.format(checkpoint))
    if work_dir.exists() and any(work_dir.iterdir()):
        parser.error('Use a new or empty work directory: {}'.format(work_dir))

    import torch
    from mmengine.runner import Runner
    from mmyolo.utils import register_all_modules

    load_kwargs = dict(map_location='cpu')
    if 'weights_only' in inspect.signature(torch.load).parameters:
        load_kwargs['weights_only'] = False
    state = torch.load(str(checkpoint), **load_kwargs)
    source = state.get('meta', {}).get('cfg')
    if not isinstance(source, str):
        raise ValueError('Checkpoint lacks meta.cfg; cannot reconstruct the original run safely.')
    source_iter = state.get('meta', {}).get('iter')
    del state
    cfg = build_config(source, checkpoint, args.case, work_dir,
                       steps=args.steps, lr=args.lr)
    register_all_modules()
    # Import registrations before Runner builds custom models and hooks.
    import yolo_world  # noqa: F401
    import yolo_world.evaluation  # noqa: F401
    print('Case {}: fg={}, bg={}; {} NEW steps, constant lr={}; fresh optimizer; '
          'source checkpoint iteration={}'.format(
              args.case, cfg.model.bbox_head.cls_foreground_weight,
              cfg.model.bbox_head.cls_background_weight, args.steps,
              args.lr, source_iter), flush=True)
    Runner.from_cfg(cfg).train()


if __name__ == '__main__':
    main()

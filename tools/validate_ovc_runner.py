"""Bounded CPU integration check of real Runner, EMA, data and pipeline switch."""
import os
import sys
from pathlib import Path

os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import argparse
    import torch
    from mmengine.config import Config
    from mmengine.runner import Runner
    from mmyolo.utils import register_all_modules
    import yolo_world  # noqa
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=12)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--work-dir', default='.codex_tmp/ovc215_runner_smoke')
    args = parser.parse_args()
    register_all_modules()
    torch.set_num_threads(4)
    cfg = Config.fromfile(str(ROOT / 'configs/pretrain/DIORv2.15.py'))
    cfg.work_dir = args.work_dir
    cfg.model.data_preprocessor.setdefault('non_blocking', False)
    cfg.optim_wrapper.type = 'OVCTrackedOptimWrapper'
    cfg.optim_wrapper.pop('loss_scale', None)
    cfg.train_cfg.max_iters = args.steps
    cfg.train_dataloader.num_workers = args.workers
    cfg.train_dataloader.persistent_workers = args.workers > 0
    cfg.val_dataloader = cfg.val_evaluator = cfg.val_cfg = None
    cfg.default_hooks.logger.interval = 1
    cfg.default_hooks.checkpoint = dict(type='CheckpointHook', by_epoch=False,
                                       interval=args.steps, save_last=True, max_keep_ckpts=1)
    for hook in cfg.custom_hooks:
        if hook.type == 'OVCIterationPipelineSwitchHook':
            hook.switch_iter = 1
        if hook.type == 'OVCDiagnosticsHook':
            hook.interval = 1
    # Runner chooses CUDA whenever available. Override only device placement
    # for this explicitly CPU check; do not alter CUDA operators or model code.
    class CPURunner(Runner):
        def wrap_model(self, model_wrapper_cfg, model):
            return model.cpu()
    runner = CPURunner.from_cfg(cfg)
    runner.train()
    assert runner.iter == args.steps
    assert int(runner.model._ovc_iteration) == args.steps
    assert runner.model.neck.dual_stream.scale_mode == 'fixed'
    switches = [h for h in runner.hooks if h.__class__.__name__ == 'OVCIterationPipelineSwitchHook']
    assert len(switches) == 1 and switches[0]._switched
    assert not any(t.__class__.__name__ == 'MultiModalMosaic'
                   for t in runner.train_dataloader.dataset.pipeline.transforms)
    print('RUNNER SMOKE PASSED: real data, EMA, iteration hooks, checkpoint and clean pipeline.')


if __name__ == '__main__':
    main()

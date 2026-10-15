"""Real DIOR images + complete pretrained detector; bounded training smoke test.

This does not measure AP. --device cpu uses FP32. When a DIORv3.2 text
adapter is present, also verify its real optimizer update and inference use.
"""
import argparse
import contextlib
import copy
import json
import os
from pathlib import Path
import sys
import time
import traceback
import faulthandler

os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/pretrain/DIORv2.14.py')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--image-size', type=int, default=800)
    parser.add_argument('--steps', type=int, default=6)
    parser.add_argument('--save-checkpoint', action='store_true')
    parser.add_argument('--output-dir', default='.codex_tmp/ovc_smoke')
    args = parser.parse_args()
    from mmengine.config import Config
    from mmengine.optim import build_optim_wrapper
    from mmengine.runner import load_checkpoint
    from mmyolo.registry import MODELS
    from mmyolo.utils import register_all_modules
    from mmdet.structures import DetDataSample
    import yolo_world  # noqa: F401

    register_all_modules()
    torch.set_num_threads(4)
    torch.manual_seed(179871018)
    cfg = Config.fromfile(str(ROOT / args.config))
    output = ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    stack_log = (output / 'stacks.log').open('w')
    faulthandler.dump_traceback_later(120, repeat=True, file=stack_log)
    dataset_cfg = cfg.train_dataloader.dataset.dataset
    payload = json.loads((ROOT / dataset_cfg.data_root / dataset_cfg.ann_file).read_text())
    by_image = {}
    for ann in payload['annotations']:
        by_image.setdefault(ann['image_id'], []).append(ann)
    candidates = [im for im in payload['images'] if 1 <= len(by_image.get(im['id'], [])) <= 10]
    if cfg.model.get('ovc_training_cfg', {}).get('normalized_regularization', False):
        candidates = [im for im in candidates if sum(
            a['bbox'][2] * args.image_size / im['width'] >= 16 and
            a['bbox'][3] * args.image_size / im['height'] >= 16
            for a in by_image[im['id']]) >= 3]
    if any(im.get('partial_labels', False) for im in candidates):
        mixed = [im for im in candidates if im.get('partial_labels', False)]
        complete = [im for im in candidates if not im.get('partial_labels', False)]
        candidates = [im for pair in zip(complete, mixed) for im in pair]
    chosen = candidates[:args.batch_size]
    if len(chosen) != args.batch_size:
        raise ValueError('Not enough legal training images.')
    images, rows = [], []
    for i, im in enumerate(chosen):
        with Image.open(ROOT / 'data/DIOR/JPEGImages' / im['file_name']) as source:
            rgb = np.asarray(source.convert('RGB').resize((args.image_size, args.image_size))).copy()
        images.append(torch.from_numpy(rgb[:, :, ::-1].copy()).permute(2, 0, 1))
        sx, sy = args.image_size / im['width'], args.image_size / im['height']
        for ann in by_image[im['id']]:
            x, y, w, h = ann['bbox']
            rows.append([i, ann['category_id'], x * sx, y * sy, (x + w) * sx, (y + h) * sy])
    batch = dict(inputs=torch.stack(images), data_samples=dict(
        bboxes_labels=torch.tensor(rows, dtype=torch.float32),
        texts=[list(cfg.DIOR_BASE_CLASSES) for _ in chosen]))
    if any('partial_labels' in im for im in chosen):
        batch['data_samples']['partial_labels'] = torch.tensor(
            [im['partial_labels'] for im in chosen], dtype=torch.bool)
    report = dict(config=args.config, device=args.device, batch_size=args.batch_size,
                  image_size=args.image_size, images=[im['file_name'] for im in chosen],
                  steps=[], status='running', torch_version=torch.__version__,
                  cuda_runtime=torch.version.cuda,
                  amp_enabled=args.device.startswith('cuda'),
                  amp_skipped_steps=0, nonfinite_loss_steps=0)
    path = output / 'report.json'
    path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    try:
        print('Building full detector and loading local pretrained weights.', flush=True)
        with (output / 'build.log').open('w', encoding='utf-8') as log:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                model = MODELS.build(cfg.model)
                model.init_weights()
                load_checkpoint(model, str(ROOT / cfg.load_from), map_location='cpu', strict=False)
                model.to(args.device)
                opt = copy.deepcopy(cfg.optim_wrapper)
                if args.device == 'cpu':
                    opt.type = ('OVCTrackedOptimWrapper' if opt.type == 'OVCTrackedAmpOptimWrapper'
                                else 'OVCGroupedOptimWrapper')
                    opt.pop('loss_scale', None)
                wrapper = build_optim_wrapper(model, opt)
        semantic_adapter = getattr(model.backbone.text_model, 'semantic_adapter', None)
        semantic_grad_norms = []
        semantic_grad_handle = None
        if semantic_adapter is not None:
            before_semantic = semantic_adapter.weight.detach().cpu().clone()
            semantic_ids = {id(p) for p in semantic_adapter.parameters()}
            report['semantic_adapter_learning_rates'] = [
                group['lr'] for group in wrapper.optimizer.param_groups
                if any(id(p) in semantic_ids for p in group['params'])]
            if not all(lr > 0 for lr in report['semantic_adapter_learning_rates']):
                raise RuntimeError('Semantic adapter has a zero learning rate.')
            semantic_grad_handle = semantic_adapter.weight.register_hook(
                lambda grad: semantic_grad_norms.append(float(grad.float().norm())))
        model.train()
        phase_handles = []
        for name, module in [('image_backbone', model.backbone.image_model),
                             ('neck', model.neck), ('head', model.bbox_head)]:
            phase_handles.append(module.register_forward_pre_hook(
                lambda m, x, name=name: print('Starting ' + name, flush=True)))
            phase_handles.append(module.register_forward_hook(
                lambda m, x, out, name=name: print('Finished ' + name, flush=True)))
        model.neck.collect_diagnostics = True
        model.neck.dual_stream.collect_diagnostics = True
        main_parameter = model.bbox_head.head_module.reg_preds[0][-1].weight
        auxiliary_parameter = model.neck.visual_adapter.adapter[0].weight
        before_main = main_parameter.detach().cpu().clone()
        before_aux = auxiliary_parameter.detach().cpu().clone()
        if args.device.startswith('cuda'):
            torch.cuda.reset_peak_memory_stats()
            report['gpu'] = torch.cuda.get_device_name()
        for step in range(args.steps):
            if args.device.startswith('cuda'):
                torch.cuda.synchronize()
            start = time.perf_counter()
            loss_values = model.train_step(copy.deepcopy(batch), wrapper)
            if args.device.startswith('cuda'):
                torch.cuda.synchronize()
            item = dict(step=step + 1, seconds=time.perf_counter() - start,
                        losses={k: float(v) for k, v in loss_values.items()},
                        diagnostics={k: float(v) for k, v in model.last_aux_diagnostics.items()})
            for name in ('ovc_main_grad_norm', 'ovc_aux_grad_norm', 'ovc_amp_skipped'):
                try:
                    item[name] = float(wrapper.message_hub.get_scalar('train/' + name).current())
                except KeyError:
                    pass
            if hasattr(wrapper, 'loss_scaler'):
                item['amp_scale'] = wrapper.loss_scaler.get_scale()
            report['amp_skipped_steps'] += int(item.get('ovc_amp_skipped', 0))
            report['steps'].append(item)
            print(json.dumps(item), flush=True)
            if not all(np.isfinite(v) for v in item['losses'].values()):
                report['nonfinite_loss_steps'] += 1
                raise RuntimeError('Non-finite loss in full-model smoke test.')
            if not item.get('ovc_amp_skipped', 0) and any(
                    not np.isfinite(item.get(name, 0.))
                    for name in ('ovc_main_grad_norm', 'ovc_aux_grad_norm')):
                raise RuntimeError('Non-finite gradient without an AMP-skipped step.')
            path.write_text(json.dumps(report, indent=2), encoding='utf-8')
        report['main_parameter_changed'] = not torch.equal(before_main, main_parameter.detach().cpu())
        report['adapter_parameter_changed'] = not torch.equal(before_aux, auxiliary_parameter.detach().cpu())
        if semantic_adapter is not None:
            report['semantic_adapter_parameter_changed'] = not torch.equal(
                before_semantic, semantic_adapter.weight.detach().cpu())
            report['semantic_adapter_grad_norms'] = semantic_grad_norms
            semantic_grad_handle.remove()
            if (not report['semantic_adapter_parameter_changed'] or
                    not any(np.isfinite(norm) and norm > 0 for norm in semantic_grad_norms)):
                raise RuntimeError('Semantic adapter did not receive a valid update.')
        if not report['main_parameter_changed'] or not report['adapter_parameter_changed']:
            raise RuntimeError('Smoke test did not update both main and auxiliary parameters.')
        if args.device.startswith('cuda'):
            report['peak_allocated_mib'] = torch.cuda.max_memory_allocated() / 2**20
            report['peak_reserved_mib'] = torch.cuda.max_memory_reserved() / 2**20
        for handle in phase_handles:
            handle.remove()
        if args.save_checkpoint:
            torch.save(dict(state_dict=model.state_dict()), output / 'smoke_only.pth')
        # Check ordinary inference: only one image Backbone pass, full 20 texts.
        model.eval()
        inputs = model.data_preprocessor(copy.deepcopy(batch), training=True)['inputs']
        samples = []
        for i in range(args.batch_size):
            sample = DetDataSample(metainfo=dict(
                img_id=chosen[i]['id'], ori_shape=(args.image_size, args.image_size),
                img_shape=(args.image_size, args.image_size), scale_factor=(1., 1.),
                batch_input_shape=(args.image_size, args.image_size), pad_param=(0, 0, 0, 0)))
            sample.texts = list(cfg.DIOR_FULL_CLASSES)
            samples.append(sample)
        calls = []
        handle = model.backbone.image_model.register_forward_hook(lambda *_: calls.append(1))
        semantic_calls = []
        if semantic_adapter is not None:
            semantic_handle = semantic_adapter.register_forward_hook(
                lambda module, inputs, result: semantic_calls.append(tuple(result.shape)))
        with torch.no_grad():
            predictions = model.predict(inputs, samples)
        handle.remove()
        if semantic_adapter is not None:
            semantic_handle.remove()
            assert semantic_calls == [(args.batch_size, len(cfg.DIOR_FULL_CLASSES),
                                       semantic_adapter.out_features)]
            report['inference_semantic_adapter_shapes'] = semantic_calls
        assert len(calls) == 1
        report['inference_backbone_calls'] = len(calls)
        report['detections'] = [len(s.pred_instances) for s in predictions]
        report['status'] = 'passed'
    except Exception:
        report['status'] = 'failed'
        report['error'] = traceback.format_exc()
        raise
    finally:
        faulthandler.cancel_dump_traceback_later()
        stack_log.close()
        if args.device.startswith('cuda') and torch.cuda.is_initialized():
            report['peak_allocated_mib'] = torch.cuda.max_memory_allocated() / 2**20
            report['peak_reserved_mib'] = torch.cuda.max_memory_reserved() / 2**20
        path.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print('Report:', path, flush=True)


if __name__ == '__main__':
    main()

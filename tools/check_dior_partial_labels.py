"""Read-only DIOR partial-label audit; no checkpoint or full model required.

Run from the repository root. Exit 0 means all checks passed, 1 means a check
failed. Real data stages and isolated loss-gradient stages are reported separately.
"""
import argparse
import copy
import hashlib
import inspect
import json
import os
import random
import sys
import traceback
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def fingerprint(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def flag(sample):
    if 'data_samples' in sample:
        return sample['data_samples'].metainfo.get('partial_labels')
    return sample.get('partial_labels')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/pretrain/DIORv2.15.py')
    parser.add_argument('--output-dir', default='work_dirs/partial_labels_check')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--batches', type=int, default=8)
    args = parser.parse_args()
    require(args.workers >= 0 and args.batches > 0, 'Invalid workers/batches')
    os.chdir(ROOT)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = dict(config=fingerprint(args.config), checks=[], limitations=[
        'CPU only; no full Backbone, training run, AMP or checkpoint evaluation.',
        'Worker audit uses forced partial/complete indices, not a prevalence estimate.',
        'Gradient audit uses synthetic tensors with the real loss/masking code.'])

    def check(name, function):
        entry = dict(name=name)
        report['checks'].append(entry)
        print('\n[CHECK] ' + name, flush=True)
        try:
            function(entry)
            entry['status'] = 'PASS'
        except Exception as error:
            entry.update(status='FAIL', error=str(error), traceback=traceback.format_exc())
            print(entry['traceback'], flush=True)
        print('[{}] {}'.format(entry['status'], name), flush=True)
        (output / 'report.json').write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')

    state = {}

    def setup(entry):
        import numpy as np
        import torch
        import mmengine
        import mmdet
        import mmyolo
        from mmengine.config import Config
        from mmyolo.utils import register_all_modules
        import yolo_world
        register_all_modules()
        torch.set_num_threads(2)
        random.seed(215)
        np.random.seed(215)
        torch.manual_seed(215)
        cfg = Config.fromfile(args.config)
        state['cfg'] = cfg
        entry['versions'] = {m.__name__: getattr(m, '__version__', '?')
                             for m in (torch, mmengine, mmdet, mmyolo)}
        entry['python'] = sys.executable
        entry['sources'] = [fingerprint(ROOT / p) for p in (
            'yolo_world/datasets/partial_labels.py',
            'yolo_world/datasets/mm_dataset.py',
            'yolo_world/datasets/transformers/mm_mix_img_transforms.py',
            'yolo_world/models/data_preprocessors/data_preprocessor.py',
            'yolo_world/models/dense_heads/yolo_world_head.py',
            'yolo_world/models/dense_heads/yolo_world_partial_head.py',
            'yolo_world/models/detectors/yolo_world_ovc.py')]
        entry['package_paths'] = {m.__name__: str(Path(m.__file__).resolve())
                                  for m in (torch, mmengine, mmdet, mmyolo, yolo_world)}
        (output / 'resolved_config.py').write_text(cfg.pretty_text, encoding='utf-8')
        print('Python:', sys.executable, '\nVersions:', entry['versions'], flush=True)

    def raw_json(entry):
        cfg = state['cfg'].train_dataloader.dataset
        while 'dataset' in cfg:
            cfg = cfg.dataset
        path = Path(cfg.ann_file)
        if not path.is_absolute():
            path = Path(cfg.get('data_root', '')) / path
        entry['annotation'] = fingerprint(path)
        data = json.loads(path.read_text(encoding='utf-8'))
        images = data['images']
        missing = [im['id'] for im in images if 'partial_labels' not in im]
        invalid = [im['id'] for im in images if type(im.get('partial_labels')) is not bool]
        partial = [im['id'] for im in images if im.get('partial_labels') is True]
        entry.update(images=len(images), annotations=len(data['annotations']),
                     partial_images=len(partial), complete_images=len(images)-len(partial),
                     missing_flag_count=len(missing), invalid_flag_count=len(invalid),
                     missing_flag_examples=missing[:20], partial_id_examples=partial[:20])
        state['raw_flags'] = {im['id']: im.get('partial_labels') for im in images}
        print(json.dumps(entry, ensure_ascii=False), flush=True)
        require(not missing and not invalid, 'JSON partial_labels missing or not boolean')
        require(len(images) == 4933 and len(partial) == 607,
                'Unexpected image counts; expected 4933 total / 607 partial. Check annotation version.')
        allowed = set(cfg.metainfo.classes)
        cats = {cat['id']: cat['name'] for cat in data['categories']}
        valid_ids = set()
        import math
        for ann in data['annotations']:
            require(cats.get(ann['category_id']) in allowed, 'Non-Base annotation: ' + str(ann['id']))
            box = ann['bbox']
            require(len(box) == 4 and all(math.isfinite(v) for v in box)
                    and box[2] > 0 and box[3] > 0, 'Invalid bbox: ' + str(ann['id']))
            valid_ids.add(ann['image_id'])
        require(all(im['id'] in valid_ids for im in images), 'Image without valid Base annotation')

    def dataset_check(entry):
        from mmyolo.registry import DATASETS
        ds = DATASETS.build(copy.deepcopy(state['cfg'].train_dataloader.dataset))
        state['dataset'] = ds
        partial, complete, mismatches = [], [], []
        for index in range(len(ds)):
            info = ds.get_data_info(index)
            value = info.get('partial_labels')
            if type(value) is not bool or value != state['raw_flags'].get(info['img_id']):
                mismatches.append(dict(index=index, img_id=info['img_id'], parsed=value))
            (partial if value is True else complete).append(index)
        entry.update(images=len(ds), partial=len(partial), complete=len(complete),
                     mismatches=mismatches[:20], mismatch_count=len(mismatches))
        state.update(partial=partial, complete=complete)
        print(entry, flush=True)
        require(not mismatches, 'JSON -> Dataset flag mismatch; inspect parser')
        require(len(ds) == 4933 and len(partial) == 607, 'Dataset count/filter mismatch')

    def trace_pipeline(entry):
        ds = state['dataset']
        require(state['partial'] and state['complete'], 'Need both partial and complete images')
        entry['traces'] = []
        # Force Mosaic to test all-complete and complete + partial-source cases.
        for start_partial, mixed_partial in ((False, False), (False, True), (True, False)):
            index = state['partial' if start_partial else 'complete'][0]
            result = ds.get_data_info(index)
            result['dataset'] = ds
            expected = start_partial
            trace = dict(img_id=result['img_id'], force_partial_source=mixed_partial, stages=[])
            entry['traces'].append(trace)
            for transform in ds.pipeline.transforms:
                name = type(transform).__name__
                before = expected
                if name == 'MultiModalMosaic':
                    ids = [state['complete'][0]] * 3
                    if mixed_partial:
                        ids[0] = state['partial'][0]
                    with patch.object(transform, 'get_indexes', return_value=ids), patch.object(transform, 'prob', 1.0):
                        result = transform(result)
                    expected = expected or mixed_partial
                else:
                    result = transform(result)
                require(result is not None, name + ' returned None')
                actual = flag(result)
                trace['stages'].append(dict(transform=name, before=before, expected=expected, actual=actual))
                print('{} img={} expected={} actual={}'.format(name, trace['img_id'], expected, actual), flush=True)
                require(actual is expected, 'Flag first diverged at ' + name)

    def worker_pipeline(entry):
        from functools import partial
        from torch.utils.data import DataLoader
        from mmengine.dataset import COLLATE_FUNCTIONS
        from mmyolo.registry import MODELS
        cfg = state['cfg']
        ds = state['dataset']
        require(state['partial'] and state['complete'], 'Need partial and complete images')
        collate_cfg = dict(cfg.train_dataloader.collate_fn)
        collate_type = collate_cfg.pop('type')
        collate = COLLATE_FUNCTIONS.get(collate_type) if isinstance(collate_type, str) else collate_type
        entry['collate_source'] = fingerprint(inspect.getfile(collate))
        preprocessor = MODELS.build(copy.deepcopy(cfg.model.data_preprocessor)).cpu()
        indices = []
        for i in range(args.batches * 4):
            group = state['partial' if i % 2 else 'complete']
            indices.append(group[(i // 2) % len(group)])
        loader = DataLoader(ds, batch_size=4, sampler=indices, num_workers=args.workers,
                            persistent_workers=args.workers > 0,
                            collate_fn=partial(collate, **collate_cfg))
        iterator = iter(loader)
        entry['batches'] = []
        try:
            for i in range(args.batches):
                batch = next(iterator)
                incoming = batch['data_samples'].get('partial_labels')
                require(incoming is not None, 'Collate dropped partial_labels')
                before = incoming.tolist()
                require(before[1] is True and before[3] is True,
                        'Known partial source lost its flag in worker pipeline')
                processed = preprocessor(batch, training=True)['data_samples']
                after = processed['partial_labels'].tolist()
                metas = [meta.get('partial_labels') for meta in processed['img_metas']]
                entry['batches'].append(dict(batch=i, collated=before, preprocessed=after, img_metas=metas))
                print('batch={} collate={} preprocessor={} img_metas={}'.format(i, before, after, metas), flush=True)
                require(before == after == metas, 'Preprocessor or img_metas lost flags')
        finally:
            if hasattr(iterator, '_shutdown_workers'):
                iterator._shutdown_workers()

    def gradients(entry):
        import torch
        from torch import nn
        from mmyolo.registry import MODELS
        from yolo_world.models.detectors.yolo_world import YOLOWorldDetector
        from yolo_world.models.layers.ovc_training import box_trimaps
        cfg = state['cfg']
        head_cfg = copy.deepcopy(cfg.model.bbox_head)
        head_cfg.train_cfg = copy.deepcopy(cfg.model.train_cfg)
        head_cfg.head_module.in_channels = [8, 16, 32]
        head = MODELS.build(head_cfg).cpu()
        classes = head.num_classes
        flags = [False, True, False, True]
        metas = [dict(batch_input_shape=(32, 32), partial_labels=v) for v in flags]
        gt = torch.tensor([[i, 0, 0, 0, 24, 24] for i in range(3)], dtype=torch.float32)
        captured = {}
        def capture(module, inputs, result):
            captured['foreground'] = result['fg_mask_pre_prior'].detach().bool()
        handle = head.assigner.register_forward_hook(capture)
        try:
            for masked in (False, True):
                logits = [torch.zeros(4, classes, s, s, requires_grad=True) for s in (4, 2, 1)]
                boxes = [torch.full((4, 4, s, s), 2., requires_grad=True) for s in (4, 2, 1)]
                dists = [torch.zeros(4, head.head_module.reg_max*4, s, s, requires_grad=True) for s in (4, 2, 1)]
                losses = head.loss_by_feat(logits, boxes, dists,
                    torch.ones(4, classes) if masked else None, gt, metas)
                losses['loss_cls'].backward()
                grad = torch.cat([x.grad.permute(0, 2, 3, 1).reshape(4, -1, classes) for x in logits], dim=1)
                fg = captured['foreground']
                require(fg[1].any() and not fg[3].any(), 'Synthetic assigner must match partial GT and leave empty partial unmatched')
                require(torch.isfinite(grad).all(), 'Non-finite classification gradient')
                require(grad[1][~fg[1]].abs().sum() == 0 and grad[3].abs().sum() == 0,
                        'Partial unmatched/empty image has classification gradient')
                require(grad[0][~fg[0]].abs().sum() > 0 and grad[1][fg[1]].abs().sum() > 0,
                        'Complete negatives or partial positives accidentally disabled')
                entry['classification_text_mask_' + str(masked)] = dict(
                    partial_unmatched_grad=float(grad[1][~fg[1]].abs().sum()),
                    empty_partial_grad=float(grad[3].abs().sum()),
                    complete_negative_grad=float(grad[0][~fg[0]].abs().sum()),
                    partial_positive_grad=float(grad[1][fg[1]].abs().sum()))
        finally:
            handle.remove()
        # Execute the configured detector's real loss method (including its
        # objectness mask). Stub only expensive feature extraction/detection.
        detector_class = MODELS.get(cfg.model.type)
        model = detector_class.__new__(detector_class)
        nn.Module.__init__(model)
        model.register_buffer('_ovc_iteration', torch.tensor(1000))
        oa = torch.zeros(4, 1, 4, 4, requires_grad=True)
        ob = torch.zeros_like(oa, requires_grad=True)
        visual = torch.zeros(4, 64, 4, 4)
        model.neck = SimpleNamespace(adapter_enabled=True, dual_stream=SimpleNamespace(),
            main_visual=visual, main_objectness_logits=oa, collect_diagnostics=False,
            clear_auxiliary_cache=lambda: None)
        model.bbox_head = head
        model.data_preprocessor = MODELS.build(copy.deepcopy(cfg.model.data_preprocessor)).cpu()
        model.ovc_training_cfg = dict(color_amount=.2, hue=.02, grayscale_prob=.1,
            consistency_enabled=False, objectness_weight=.1, warmup_iters=1000)
        model.auxiliary_visual_forward = MethodType(lambda self, inputs: (visual, ob), model)
        # Leave definite background cells outside the expanded objectness box.
        objectness_gt = gt.clone()
        objectness_gt[:, 2:] = torch.tensor([2., 2., 14., 14.])
        samples = dict(bboxes_labels=objectness_gt, img_metas=metas, partial_labels=torch.tensor(flags))
        with patch.object(YOLOWorldDetector, 'loss', return_value={}):
            losses = model.loss(torch.zeros(4, 3, 32, 32), samples)
        losses['loss_ovc_objectness'].backward()
        target, _, negative = box_trimaps(objectness_gt[:, [0, 2, 3, 4, 5]], 4, (4, 4), (32, 32))
        for name, values in (('main', oa), ('auxiliary', ob)):
            grad = values.grad
            require(grad is not None and torch.isfinite(grad).all(), 'Missing/non-finite objectness gradient')
            require(grad[1][target[1] == 0].abs().sum() == 0 and grad[3].abs().sum() == 0,
                    'Detector objectness loss failed to ignore partial background/empty GT')
            require(grad[1][target[1] > 0].abs().sum() > 0 and grad[0][negative[0]].abs().sum() > 0,
                    'Objectness positives or complete negatives disabled')
            entry['objectness_' + name] = dict(partial_background_grad=float(grad[1][target[1] == 0].abs().sum()),
                empty_partial_grad=float(grad[3].abs().sum()), complete_negative_grad=float(grad[0][negative[0]].abs().sum()))
        print(json.dumps(entry, indent=2), flush=True)

    check('environment_and_source_versions', setup)
    check('raw_annotation_flags', raw_json)
    check('dataset_parsed_flags', dataset_check)
    check('forced_mosaic_and_transform_trace', trace_pipeline)
    check('real_worker_collate_preprocessor', worker_pipeline)
    check('classification_and_detector_objectness_gradients', gradients)
    report['passed'] = all(x['status'] == 'PASS' for x in report['checks'])
    (output / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print('\n{}: {}'.format('ALL CHECKS PASSED' if report['passed'] else 'CHECK FAILED', output / 'report.json'), flush=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())

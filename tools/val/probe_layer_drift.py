"""Locate Best-to-Final representation drift on fixed DIOR validation ROIs.

Run ``extract`` once for Best and once for Final, then run ``compare``. The
comparison uses class centroids built only from Best and reports same-ROI
cosine drift plus the Novel-vs-Base reference-margin change for every tensor.
Validation labels are used only after inference for this diagnostic.
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DIOR_CLASSES = (
    'airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge',
    'chimney', 'dam', 'Expressway-Service-area',
    'Expressway-toll-station', 'golffield', 'groundtrackfield', 'harbor',
    'overpass', 'ship', 'stadium', 'storagetank', 'tenniscourt',
    'trainstation', 'vehicle', 'windmill')
DIOR_NOVEL_CLASSES = (
    'airport', 'basketballcourt', 'groundtrackfield', 'windmill')
DEFAULT_HOOKS = (
    'backbone.image_model.stage2',
    'backbone.image_model.stage3',
    'backbone.image_model.stage4',
    'neck.visual_adapter',
    'neck.dual_stream',
    'neck.wsr',
    'neck',
    'bbox_head.head_module',
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='command', required=True)
    extract_parser = subparsers.add_parser('extract')
    extract_parser.add_argument('config')
    extract_parser.add_argument('checkpoint')
    extract_parser.add_argument('--output', required=True)
    extract_parser.add_argument('--device', default='cuda:0')
    extract_parser.add_argument('--max-images', type=int, default=0,
                                help='0 means the complete validation set.')
    extract_parser.add_argument(
        '--hooks', default=','.join(DEFAULT_HOOKS),
        help='Comma-separated exact module names.')
    extract_parser.add_argument('--list-modules', action='store_true')
    compare_parser = subparsers.add_parser('compare')
    compare_parser.add_argument('--best', required=True)
    compare_parser.add_argument('--final', required=True)
    compare_parser.add_argument('--output', required=True)
    compare_parser.add_argument('--csv-output', default='')
    return parser.parse_args()


def collect_4d_tensors(obj, path=''):
    """Return stable (nested path, tensor) pairs for every 4-D tensor."""
    import torch
    found = []
    if torch.is_tensor(obj):
        if obj.ndim == 4:
            found.append((path or 'tensor', obj))
        return found
    if isinstance(obj, (list, tuple)):
        for index, value in enumerate(obj):
            child = str(index) if not path else '{}.{}'.format(path, index)
            found.extend(collect_4d_tensors(value, child))
        return found
    if isinstance(obj, dict):
        for key in sorted(obj):
            child = str(key) if not path else '{}.{}'.format(path, key)
            found.extend(collect_4d_tensors(obj[key], child))
    return found


def print_interesting_modules(model):
    tokens = ('backbone', 'neck', 'dual', 'wsr', 'adapter', 'support',
              'retriev', 'bbox_head')
    print('\n===== candidate modules =====')
    for name, module in model.named_modules():
        target = '{} {}'.format(name.lower(), module.__class__.__name__.lower())
        if any(token in target for token in tokens):
            print('{:80s} {}'.format(name, module.__class__.__name__))


def get_tensor_boxes(sample):
    boxes = sample.gt_instances.bboxes
    if hasattr(boxes, 'tensor'):
        boxes = boxes.tensor
    return boxes, sample.gt_instances.labels


def get_img_id(sample, fallback):
    meta = getattr(sample, 'metainfo', {})
    for key in ('img_id', 'image_id', 'img_path', 'ori_filename'):
        if key in meta:
            return str(meta[key])
    return str(fallback)


def get_input_hw(sample, inputs):
    meta = getattr(sample, 'metainfo', {})
    for key in ('batch_input_shape', 'pad_shape', 'img_shape'):
        value = meta.get(key)
        if value is not None and len(value) >= 2:
            return int(value[0]), int(value[1])
    return int(inputs.shape[-2]), int(inputs.shape[-1])


def build_model_loader(config_path, checkpoint, device):
    import torch
    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmengine.runner import Runner
    from mmengine.runner.checkpoint import load_checkpoint
    from mmyolo.registry import MODELS
    from mmyolo.utils import register_all_modules
    import yolo_world  # noqa: F401

    config_file = Path(config_path).expanduser()
    checkpoint_file = Path(checkpoint).expanduser()
    if not config_path or not config_file.is_file():
        raise FileNotFoundError(
            'Invalid config path received by probe_layer_drift.py: {!r}. '
            'Set DIAG_CONFIG to an existing .py config.'.format(config_path))
    if not checkpoint or not checkpoint_file.is_file():
        raise FileNotFoundError(
            'Invalid checkpoint path received by probe_layer_drift.py: {!r}.'
            .format(checkpoint))
    register_all_modules()
    cfg = Config.fromfile(str(config_file))
    missing_keys = [key for key in ('model', 'val_dataloader')
                    if key not in cfg]
    if missing_keys:
        raise KeyError(
            'Config {!r} is missing required keys {}. Loaded top-level keys: '
            '{}'.format(str(config_file), missing_keys, sorted(cfg.keys())))
    # The training snapshot may contain the later semantic-fusion interface
    # while the checked-out language backbone predates that constructor
    # argument.  DIOR 2.15 stored it disabled, so removing it exactly preserves
    # inference behavior.  Never silently remove an enabled component.
    text_model_cfg = cfg.model.get('backbone', {}).get('text_model', {})
    semantic_fusion_cfg = text_model_cfg.get('semantic_fusion_cfg', None)
    if semantic_fusion_cfg is not None:
        if semantic_fusion_cfg.get('enabled', False):
            raise RuntimeError(
                'The saved config enables semantic_fusion_cfg, but the '
                'current HuggingCLIPLanguageBackbone does not implement it.')
        text_model_cfg.pop('semantic_fusion_cfg')
        print('Removed disabled semantic_fusion_cfg for code compatibility.',
              flush=True)
    init_default_scope(cfg.get('default_scope', 'mmyolo'))
    cfg.load_from = None
    cfg.resume = False
    print('Building model from: {}'.format(config_file), flush=True)
    try:
        model = MODELS.build(cfg.model)
    except BaseException as error:
        print('MODEL_BUILD_FAILED: {}: {!r}'.format(
            type(error).__name__, error), file=sys.stderr, flush=True)
        raise
    print('Model construction complete.', flush=True)
    model.init_weights()
    load_checkpoint(model, str(checkpoint_file), map_location='cpu',
                    strict=False)
    model.to(torch.device(device))
    model.eval()
    loader_cfg = cfg.val_dataloader.copy()
    loader_cfg.update(batch_size=1, num_workers=0, persistent_workers=False)
    dataloader = Runner.build_dataloader(loader_cfg, seed=0)
    return model, dataloader


def atomic_save_npz(output, values):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + '.tmp')
    try:
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, **values)
        os.replace(str(temporary), str(output))
    finally:
        if temporary.exists():
            temporary.unlink()


def extract(args):
    import torch
    from torchvision.ops import roi_align

    model, dataloader = build_model_loader(
        args.config, args.checkpoint, args.device)
    if args.list_modules:
        print_interesting_modules(model)
        return
    modules = dict(model.named_modules())
    requested = tuple(name.strip() for name in args.hooks.split(',')
                      if name.strip())
    explicit_hooks = args.hooks != ','.join(DEFAULT_HOOKS)
    missing = [name for name in requested if name not in modules]
    if missing and explicit_hooks:
        raise KeyError('Modules do not exist: {}'.format(', '.join(missing)))
    if missing:
        print('Skipping unavailable default hooks: {}'.format(', '.join(missing)))
    hook_names = [name for name in requested if name in modules]
    if not hook_names:
        raise RuntimeError('No valid hook modules were selected.')

    print('\n===== hooks =====')
    for name in hook_names:
        print(name)
    cache = {}
    handles = []

    def make_pre_hook(name):
        def hook(_module, inputs):
            cache[name + '::in'] = [
                (path, tensor.detach())
                for path, tensor in collect_4d_tensors(inputs)]
        return hook

    def make_post_hook(name):
        def hook(_module, _inputs, output):
            cache[name + '::out'] = [
                (path, tensor.detach())
                for path, tensor in collect_4d_tensors(output)]
        return hook

    for name in hook_names:
        handles.append(modules[name].register_forward_pre_hook(
            make_pre_hook(name)))
        handles.append(modules[name].register_forward_hook(
            make_post_hook(name)))

    records = defaultdict(lambda: {'keys': [], 'labels': [], 'features': []})
    image_counter = 0
    try:
        with torch.no_grad():
            for batch in dataloader:
                if args.max_images > 0 and image_counter >= args.max_images:
                    break
                cache.clear()
                data = model.data_preprocessor(batch, training=False)
                inputs = data['inputs']
                samples = data.get('data_samples')
                if not isinstance(samples, (list, tuple)):
                    raise RuntimeError(
                        'Validation data_samples must be a list or tuple, got '
                        '{}'.format(type(samples)))
                model._run_forward(data, mode='predict')
                for batch_index, sample in enumerate(samples):
                    if args.max_images > 0 and image_counter >= args.max_images:
                        break
                    boxes, labels = get_tensor_boxes(sample)
                    image_id = get_img_id(sample, image_counter)
                    input_h, input_w = get_input_hw(sample, inputs)
                    boxes = boxes.detach().float()
                    labels = labels.detach().cpu().numpy().astype(np.int64)
                    valid = ((boxes[:, 2] > boxes[:, 0]) &
                             (boxes[:, 3] > boxes[:, 1]))
                    valid_indices = valid.nonzero(as_tuple=False).flatten()
                    if valid_indices.numel() == 0:
                        image_counter += 1
                        continue
                    for location, tensors in cache.items():
                        for tensor_path, feature in tensors:
                            if batch_index >= feature.shape[0]:
                                continue
                            # A one-channel objectness/reliability map has no
                            # meaningful class-centroid geometry. Its behavior
                            # is covered by the detection-path diagnostic.
                            if feature.shape[1] < 2:
                                continue
                            feature_h, feature_w = feature.shape[-2:]
                            scaled = boxes[valid_indices].clone()
                            scaled[:, 0::2] *= feature_w / float(input_w)
                            scaled[:, 1::2] *= feature_h / float(input_h)
                            pooled = roi_align(
                                feature[batch_index:batch_index + 1].float(),
                                [scaled], output_size=(1, 1),
                                spatial_scale=1.0, sampling_ratio=2,
                                aligned=True)[:, :, 0, 0]
                            layer = '{}[{}]'.format(location, tensor_path)
                            target = records[layer]
                            for local_index, gt_index in enumerate(
                                    valid_indices.tolist()):
                                target['keys'].append(
                                    '{}::{}'.format(image_id, gt_index))
                                target['labels'].append(int(labels[gt_index]))
                                target['features'].append(
                                    pooled[local_index].cpu().half().numpy())
                    image_counter += 1
                    if image_counter % 100 == 0:
                        print('processed images: {}'.format(image_counter),
                              flush=True)
                if hasattr(model, 'neck') and hasattr(
                        model.neck, 'clear_auxiliary_cache'):
                    model.neck.clear_auxiliary_cache()
    finally:
        for handle in handles:
            handle.remove()

    layer_names = list(records)
    if not layer_names:
        raise RuntimeError('No 4-D hook outputs were captured.')
    save_values = {
        'format_version': np.asarray([2], dtype=np.int64),
        'config': np.asarray([args.config]),
        'checkpoint': np.asarray([args.checkpoint]),
        'num_images': np.asarray([image_counter], dtype=np.int64),
        'classes': np.asarray(DIOR_CLASSES),
        'layer_names': np.asarray(layer_names),
    }
    print('\n===== captured tensors =====')
    for index, layer in enumerate(layer_names):
        record = records[layer]
        features = np.stack(record['features']).astype(np.float16)
        save_values['keys_{}'.format(index)] = np.asarray(record['keys'])
        save_values['labels_{}'.format(index)] = np.asarray(
            record['labels'], dtype=np.int64)
        save_values['feats_{}'.format(index)] = features
        print('{}: n={}, dim={}'.format(layer, len(features), features.shape[1]))
    atomic_save_npz(args.output, save_values)
    print('\nSaved atomically: {}'.format(args.output))


def normalize_np(values):
    values = values.astype(np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.clip(norms, 1e-12, None)


def build_reference_centroids(features, labels):
    normalized = normalize_np(features)
    centers = {}
    for class_id in range(len(DIOR_CLASSES)):
        mask = labels == class_id
        if not np.any(mask):
            continue
        center = normalized[mask].mean(axis=0)
        center /= max(float(np.linalg.norm(center)), 1e-12)
        centers[class_id] = center
    return centers


def reference_margins(features, labels, centers):
    normalized = normalize_np(features)
    novel_ids = {DIOR_CLASSES.index(name) for name in DIOR_NOVEL_CLASSES}
    base_ids = [index for index in range(len(DIOR_CLASSES))
                if index not in novel_ids]
    values = []
    per_class = defaultdict(list)
    for row, label in zip(normalized, labels):
        label = int(label)
        if label not in novel_ids or label not in centers:
            continue
        base_scores = [float(row @ centers[index]) for index in base_ids
                       if index in centers]
        if not base_scores:
            continue
        margin = float(row @ centers[label]) - max(base_scores)
        values.append(margin)
        per_class[DIOR_CLASSES[label]].append(margin)
    return {
        'overall': float(np.mean(values)) if values else None,
        'per_class': {
            name: float(np.mean(class_values))
            for name, class_values in per_class.items()
        },
    }


def load_layer_archive(path):
    archive = np.load(path, allow_pickle=False)
    required = {'layer_names', 'classes'}
    missing = required - set(archive.files)
    if missing:
        raise ValueError('{} is missing {}'.format(path, sorted(missing)))
    classes = tuple(str(value) for value in archive['classes'].tolist())
    if classes != DIOR_CLASSES:
        raise ValueError('Unexpected class order in {}: {}'.format(
            path, classes))
    return archive


def compare(args):
    best = load_layer_archive(args.best)
    final = load_layer_archive(args.final)
    best_layers = [str(value) for value in best['layer_names'].tolist()]
    final_layers = [str(value) for value in final['layer_names'].tolist()]
    common_layers = [name for name in best_layers if name in final_layers]
    if not common_layers:
        raise RuntimeError('Best and Final archives have no common layers.')
    novel_ids = {DIOR_CLASSES.index(name) for name in DIOR_NOVEL_CLASSES}
    result = {}
    flat_rows = []
    for layer_index, layer in enumerate(common_layers):
        best_index = best_layers.index(layer)
        final_index = final_layers.index(layer)
        best_keys = best['keys_{}'.format(best_index)].astype(str)
        final_keys = final['keys_{}'.format(final_index)].astype(str)
        best_labels = best['labels_{}'.format(best_index)]
        final_labels = final['labels_{}'.format(final_index)]
        best_features = best['feats_{}'.format(best_index)].astype(np.float32)
        final_features = final['feats_{}'.format(final_index)].astype(np.float32)
        best_map = {key: index for index, key in enumerate(best_keys)}
        final_map = {key: index for index, key in enumerate(final_keys)}
        common_keys = sorted(set(best_map) & set(final_map))
        if not common_keys:
            continue
        best_rows = np.asarray([best_map[key] for key in common_keys])
        final_rows = np.asarray([final_map[key] for key in common_keys])
        labels = best_labels[best_rows]
        if not np.array_equal(labels, final_labels[final_rows]):
            raise ValueError('GT labels differ for layer {}'.format(layer))
        best_values = best_features[best_rows]
        final_values = final_features[final_rows]
        if best_values.shape[1] != final_values.shape[1]:
            raise ValueError('Feature dimensions differ for {}'.format(layer))
        cosine = np.sum(
            normalize_np(best_values) * normalize_np(final_values), axis=1)
        novel_mask = np.asarray([int(label) in novel_ids for label in labels])
        base_mask = ~novel_mask
        centers = build_reference_centroids(best_values, labels)
        best_margin = reference_margins(best_values, labels, centers)
        final_margin = reference_margins(final_values, labels, centers)
        per_class_cosine = {}
        for class_name in DIOR_NOVEL_CLASSES:
            class_id = DIOR_CLASSES.index(class_name)
            mask = labels == class_id
            if np.any(mask):
                per_class_cosine[class_name] = float(np.mean(cosine[mask]))
        best_overall = best_margin['overall']
        final_overall = final_margin['overall']
        delta = (None if best_overall is None or final_overall is None else
                 final_overall - best_overall)
        layer_result = {
            'order': layer_index,
            'num_common_roi': len(common_keys),
            'best_final_cos_all': float(np.mean(cosine)),
            'best_final_cos_base': (
                float(np.mean(cosine[base_mask])) if base_mask.any() else None),
            'best_final_cos_novel': (
                float(np.mean(cosine[novel_mask]))
                if novel_mask.any() else None),
            'best_ref_margin': best_margin,
            'final_ref_margin': final_margin,
            'delta_ref_margin': delta,
            'best_final_cos_per_novel_class': per_class_cosine,
        }
        result[layer] = layer_result
        flat_rows.append({
            'order': layer_index,
            'layer': layer,
            'num_common_roi': len(common_keys),
            'cosine_all': layer_result['best_final_cos_all'],
            'cosine_base': layer_result['best_final_cos_base'],
            'cosine_novel': layer_result['best_final_cos_novel'],
            'best_ref_margin': best_overall,
            'final_ref_margin': final_overall,
            'delta_ref_margin': delta,
        })
    if not flat_rows:
        raise RuntimeError('No aligned layer records could be compared.')
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'best_archive': str(args.best),
        'final_archive': str(args.final),
        'interpretation': (
            'More negative delta_ref_margin means Final novel ROIs moved '
            'toward Best base-class visual centroids at that tensor.'),
        'layers': result,
    }
    output.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    csv_output = (Path(args.csv_output) if args.csv_output else
                  output.with_suffix('.csv'))
    with csv_output.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
    print('\n===== LAYER DRIFT =====')
    for row in flat_rows:
        print('{order:02d} {layer}\n'
              '   novel_cos={cosine_novel}  margin: {best_ref_margin} -> '
              '{final_ref_margin}  delta={delta_ref_margin}'.format(**row))
    print('\nSaved: {}'.format(output))
    print('Saved: {}'.format(csv_output))


def main():
    args = parse_args()
    if args.command == 'extract':
        extract(args)
    else:
        compare(args)


if __name__ == '__main__':
    main()

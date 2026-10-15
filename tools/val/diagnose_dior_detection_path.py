"""Diagnose where Novel detections disappear between dense output and NMS.

For every DIOR GT instance this script records three stages from one forward:

1. decoded dense boxes before score filtering;
2. true-class candidates after score threshold and global ``nms_pre`` top-k;
3. final post-NMS detections.

Run ``extract`` for Best and Final, then ``compare``.  GT is used only for
evaluation after inference and never changes model predictions.
"""

import argparse
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DIOR_NOVEL_CLASSES = (
    'airport', 'basketballcourt', 'groundtrackfield', 'windmill')
RATE_FIELDS = (
    'dense_localized', 'pre_nms_true_candidate', 'post_nms_any_match',
    'post_nms_correct', 'post_nms_base_confusion')
MEAN_FIELDS = (
    'dense_max_iou', 'localized_true_score', 'localized_best_base_score',
    'localized_true_base_margin', 'localized_true_rank',
    'pre_nms_true_score', 'post_nms_correct_score')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='command', required=True)
    extract_parser = subparsers.add_parser('extract')
    extract_parser.add_argument('config')
    extract_parser.add_argument('checkpoint')
    extract_parser.add_argument('--output-dir', required=True)
    extract_parser.add_argument('--device', default='cuda:0')
    extract_parser.add_argument('--max-images', type=int, default=0,
                                help='0 means the complete validation set.')
    extract_parser.add_argument('--iou-threshold', type=float, default=0.5)
    compare_parser = subparsers.add_parser('compare')
    compare_parser.add_argument('--best', required=True,
                                help='Best gt_detection_path.csv')
    compare_parser.add_argument('--final', required=True,
                                help='Final gt_detection_path.csv')
    compare_parser.add_argument('--output', required=True)
    return parser.parse_args()


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
            'Invalid config path received by '
            'diagnose_dior_detection_path.py: {!r}. Set DIAG_CONFIG to an '
            'existing .py config.'.format(config_path))
    if not checkpoint or not checkpoint_file.is_file():
        raise FileNotFoundError(
            'Invalid checkpoint path received by '
            'diagnose_dior_detection_path.py: {!r}.'.format(checkpoint))
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
    # argument. DIOR 2.15 stored it disabled, so dropping it preserves the
    # trained inference graph. Refuse an enabled setting instead of changing
    # model behavior silently.
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
    return cfg, model, dataloader


def tensor_boxes(instances):
    boxes = instances.bboxes
    return boxes.tensor if hasattr(boxes, 'tensor') else boxes


def image_id(sample, fallback):
    meta = getattr(sample, 'metainfo', {})
    for key in ('img_id', 'image_id', 'img_path', 'ori_filename'):
        if key in meta:
            return str(meta[key])
    return str(fallback)


def boxes_to_original(boxes, meta):
    """Invert validation resize and letterbox for transformed GT/dense boxes."""
    boxes = boxes.clone().float()
    pad = meta.get('pad_param')
    if pad is not None:
        boxes -= boxes.new_tensor([pad[2], pad[0], pad[2], pad[0]])
    scale = boxes.new_tensor(meta.get('scale_factor', (1.0, 1.0))).flatten()
    if scale.numel() == 1:
        scale = scale.repeat(2)
    boxes /= scale[:2].repeat(2)
    ori_h, ori_w = meta['ori_shape'][:2]
    boxes[:, 0::2].clamp_(0, ori_w)
    boxes[:, 1::2].clamp_(0, ori_h)
    return boxes


def box_iou_matrix(boxes1, boxes2):
    import torch
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((len(boxes1), len(boxes2)))
    top_left = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    bottom_right = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    intersection = (bottom_right - top_left).clamp(min=0).prod(dim=-1)
    area1 = (boxes1[:, 2:] - boxes1[:, :2]).clamp(min=0).prod(dim=-1)
    area2 = (boxes2[:, 2:] - boxes2[:, :2]).clamp(min=0).prod(dim=-1)
    return intersection / (area1[:, None] + area2[None, :] - intersection).clamp(min=1e-9)


def decode_dense(head, cls_scores, bbox_preds, meta):
    """Match YOLOWorldHead.predict_by_feat before filtering."""
    import torch
    feature_sizes = [score.shape[2:] for score in cls_scores]
    priors = head.prior_generator.grid_priors(
        feature_sizes, dtype=cls_scores[0].dtype,
        device=cls_scores[0].device)
    flat_priors = torch.cat(priors)
    strides = torch.cat([
        flat_priors.new_full((size.numel() * head.num_base_priors,), stride)
        for size, stride in zip(feature_sizes, head.featmap_strides)
    ])
    num_classes = head.num_classes
    scores = torch.cat([
        value.permute(0, 2, 3, 1).reshape(1, -1, num_classes)
        for value in cls_scores
    ], dim=1).sigmoid()[0].float()
    distances = torch.cat([
        value.permute(0, 2, 3, 1).reshape(1, -1, 4)
        for value in bbox_preds
    ], dim=1)
    boxes = head.bbox_coder.decode(
        flat_priors[None], distances, strides)[0].float()
    return boxes_to_original(boxes, meta), scores


def select_pre_nms_pairs(scores, score_threshold, nms_pre):
    """Return location/class pairs after the head's threshold and top-k."""
    import torch
    flat = scores.flatten()
    valid = torch.nonzero(flat > score_threshold, as_tuple=False).flatten()
    if nms_pre > 0 and valid.numel() > nms_pre:
        _, order = flat[valid].topk(nms_pre)
        valid = valid[order]
    num_classes = scores.shape[1]
    return valid // num_classes, valid % num_classes, flat[valid]


def safe_float(value):
    if value is None:
        return None
    return float(value)


def analyze_image(sample, dense_boxes, dense_scores, pre_locations,
                  pre_labels, pre_scores, post, class_names, novel_names,
                  iou_threshold):
    import torch
    gt_boxes = boxes_to_original(
        tensor_boxes(sample.gt_instances), sample.metainfo).to(
            dense_boxes.device)
    gt_labels = sample.gt_instances.labels.to(dense_boxes.device).long()
    post_boxes = tensor_boxes(post).float()
    post_labels = post.labels.long()
    post_scores = post.scores.float()
    dense_ious = box_iou_matrix(gt_boxes, dense_boxes)
    post_ious = box_iou_matrix(gt_boxes, post_boxes)
    novel_ids = {index for index, name in enumerate(class_names)
                 if name in novel_names}
    base_ids = [index for index in range(len(class_names))
                if index not in novel_ids]
    rows = []
    current_image_id = image_id(sample, 'unknown')
    for gt_index, label_tensor in enumerate(gt_labels):
        label = int(label_tensor.item())
        dense_iou = dense_ious[gt_index]
        dense_max_iou = float(dense_iou.max().item()) if dense_iou.numel() else 0.0
        localized = torch.nonzero(
            dense_iou >= iou_threshold, as_tuple=False).flatten()
        dense_localized = localized.numel() > 0
        true_score = best_base_score = margin = rank = None
        local_top_label = local_top_score = None
        if dense_localized:
            localized_true = dense_scores[localized, label]
            local_choice = int(localized[localized_true.argmax()].item())
            score_vector = dense_scores[local_choice]
            true_score = float(score_vector[label].item())
            if base_ids:
                best_base_score = float(score_vector[base_ids].max().item())
                margin = true_score - best_base_score
            rank = int((score_vector > score_vector[label]).sum().item()) + 1
            local_top_score_tensor, local_top_label_tensor = score_vector.max(0)
            local_top_label = int(local_top_label_tensor.item())
            local_top_score = float(local_top_score_tensor.item())

        pre_true = pre_labels == label
        pre_localized = (dense_iou[pre_locations] >= iou_threshold
                         if pre_locations.numel() else pre_true)
        pre_keep = pre_true & pre_localized
        pre_nms_true_candidate = bool(pre_keep.any().item())
        pre_true_score = (float(pre_scores[pre_keep].max().item())
                          if pre_nms_true_candidate else None)

        post_iou = post_ious[gt_index]
        post_match = post_iou >= iou_threshold
        post_correct_mask = post_match & (post_labels == label)
        post_nms_any_match = bool(post_match.any().item())
        post_nms_correct = bool(post_correct_mask.any().item())
        post_correct_score = (float(post_scores[post_correct_mask].max().item())
                              if post_nms_correct else None)
        post_best_iou = (float(post_iou.max().item())
                         if post_iou.numel() else 0.0)
        matched_label = matched_score = None
        post_nms_base_confusion = False
        if post_nms_any_match:
            matched_indices = torch.nonzero(
                post_match, as_tuple=False).flatten()
            matched_choice = int(
                matched_indices[post_scores[matched_indices].argmax()].item())
            matched_label = int(post_labels[matched_choice].item())
            matched_score = float(post_scores[matched_choice].item())
            post_nms_base_confusion = (
                not post_nms_correct and matched_label in base_ids)

        if post_nms_correct:
            failure = 'detected_correctly'
        elif not dense_localized:
            failure = 'no_dense_box_iou50'
        elif true_score is None or true_score <= 0:
            failure = 'no_true_class_score'
        elif not pre_nms_true_candidate:
            failure = 'threshold_or_nms_pre_pruned'
        elif post_nms_base_confusion:
            failure = 'base_class_confusion_after_nms'
        else:
            failure = 'nms_or_max_per_img_removed'

        rows.append({
            'record_id': '{}::{}'.format(current_image_id, gt_index),
            'image_id': current_image_id,
            'gt_index': gt_index,
            'gt_label': label,
            'class_name': class_names[label],
            'group': 'novel' if label in novel_ids else 'base',
            'dense_max_iou': dense_max_iou,
            'dense_localized': int(dense_localized),
            'localized_true_score': safe_float(true_score),
            'localized_best_base_score': safe_float(best_base_score),
            'localized_true_base_margin': safe_float(margin),
            'localized_true_rank': rank,
            'localized_top_label': local_top_label,
            'localized_top_class': (
                class_names[local_top_label]
                if local_top_label is not None else None),
            'localized_top_score': safe_float(local_top_score),
            'pre_nms_true_candidate': int(pre_nms_true_candidate),
            'pre_nms_true_score': safe_float(pre_true_score),
            'post_nms_any_match': int(post_nms_any_match),
            'post_nms_best_iou': post_best_iou,
            'post_nms_correct': int(post_nms_correct),
            'post_nms_correct_score': safe_float(post_correct_score),
            'post_nms_matched_label': matched_label,
            'post_nms_matched_class': (
                class_names[matched_label]
                if matched_label is not None else None),
            'post_nms_matched_score': safe_float(matched_score),
            'post_nms_base_confusion': int(post_nms_base_confusion),
            'failure_stage': failure,
        })
    return rows


def group_summary(rows):
    summary = {'count': len(rows)}
    for field in RATE_FIELDS:
        summary[field + '_rate'] = (
            float(np.mean([float(row[field]) for row in rows]))
            if rows else None)
    for field in MEAN_FIELDS:
        values = [float(row[field]) for row in rows
                  if row.get(field) not in (None, '')]
        summary[field + '_mean'] = float(np.mean(values)) if values else None
    summary['failure_counts'] = dict(Counter(
        row['failure_stage'] for row in rows))
    return summary


def summarize(rows, checkpoint=None, images=None, iou_threshold=0.5):
    novel_rows = [row for row in rows if row['group'] == 'novel']
    groups = {
        'all': group_summary(rows),
        'base': group_summary([row for row in rows if row['group'] == 'base']),
        'novel': group_summary(novel_rows),
    }
    for class_name in DIOR_NOVEL_CLASSES:
        groups[class_name] = group_summary([
            row for row in novel_rows if row['class_name'] == class_name])
    return {
        'checkpoint': checkpoint,
        'images': images,
        'iou_threshold': iou_threshold,
        'stages': ['dense_decoded', 'threshold_and_nms_pre', 'post_nms'],
        'groups': groups,
    }


def atomic_write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    try:
        temporary.write_text(text, encoding='utf-8')
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    try:
        with temporary.open('w', newline='', encoding='utf-8-sig') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def extract(args):
    import torch
    cfg, model, dataloader = build_model_loader(
        args.config, args.checkpoint, args.device)
    class_names = tuple(dataloader.dataset.metainfo['classes'])
    novel_names = tuple(cfg.get('DIOR_NOVEL_CLASSES', DIOR_NOVEL_CLASSES))
    if not set(novel_names).issubset(set(class_names)):
        raise ValueError('Novel classes are missing from validation metadata.')
    rows = []
    image_counter = 0
    with torch.no_grad():
        for batch in dataloader:
            if args.max_images > 0 and image_counter >= args.max_images:
                break
            data = model.data_preprocessor(batch, training=False)
            inputs = data['inputs']
            samples = data['data_samples']
            if len(samples) != 1:
                raise RuntimeError('Diagnostic loader must use batch_size=1.')
            sample = samples[0]
            img_feats, txt_feats, txt_masks = model.extract_feat(inputs, samples)
            num_classes = (txt_feats.shape[1] if txt_feats.dim() == 3
                           else txt_feats.shape[0])
            if num_classes != len(class_names):
                raise RuntimeError(
                    'Text/class mismatch: {} text slots, {} metadata classes.'
                    .format(num_classes, len(class_names)))
            model.bbox_head.num_classes = num_classes
            cls_scores, bbox_preds = model.bbox_head(
                img_feats, txt_feats, txt_masks)
            dense_boxes, dense_scores = decode_dense(
                model.bbox_head, cls_scores, bbox_preds, sample.metainfo)
            test_cfg = model.bbox_head.test_cfg
            score_threshold = float(test_cfg.get('score_thr', -1))
            nms_pre = int(test_cfg.get('nms_pre', 100000))
            pre_locations, pre_labels, pre_scores = select_pre_nms_pairs(
                dense_scores, score_threshold, nms_pre)
            post = model.bbox_head.predict_by_feat(
                cls_scores, bbox_preds,
                batch_img_metas=[sample.metainfo],
                cfg=test_cfg, rescale=True, with_nms=True)[0]
            rows.extend(analyze_image(
                sample, dense_boxes, dense_scores, pre_locations,
                pre_labels, pre_scores, post, class_names, novel_names,
                args.iou_threshold))
            image_counter += 1
            if image_counter % 100 == 0:
                print('processed images: {}'.format(image_counter), flush=True)
            if hasattr(model, 'neck') and hasattr(
                    model.neck, 'clear_auxiliary_cache'):
                model.neck.clear_auxiliary_cache()
    if not rows:
        raise RuntimeError('No GT rows were collected.')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / 'gt_detection_path.csv'
    summary_path = output_dir / 'summary.json'
    write_csv(csv_path, rows)
    summary = summarize(rows, args.checkpoint, image_counter,
                        args.iou_threshold)
    atomic_write_text(summary_path, json.dumps(summary, indent=2))
    print(json.dumps(summary['groups']['novel'], indent=2))
    print('\nSaved: {}'.format(csv_path))
    print('Saved: {}'.format(summary_path))


def read_csv(path):
    with Path(path).open('r', newline='', encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def compare(args):
    best_rows = read_csv(args.best)
    final_rows = read_csv(args.final)
    best = {row['record_id']: row for row in best_rows}
    final = {row['record_id']: row for row in final_rows}
    common = sorted(set(best) & set(final))
    if not common:
        raise RuntimeError('Best and Final CSV files have no common GT rows.')
    aligned_best = [best[key] for key in common]
    aligned_final = [final[key] for key in common]
    for best_row, final_row in zip(aligned_best, aligned_final):
        if best_row['class_name'] != final_row['class_name']:
            raise ValueError('Class mismatch for {}'.format(
                best_row['record_id']))
    best_summary = summarize(aligned_best)
    final_summary = summarize(aligned_final)
    comparison = {}
    for group in ('novel',) + DIOR_NOVEL_CLASSES:
        best_group = best_summary['groups'][group]
        final_group = final_summary['groups'][group]
        metrics = {}
        for field in tuple(name + '_rate' for name in RATE_FIELDS) + tuple(
                name + '_mean' for name in MEAN_FIELDS):
            best_value = best_group[field]
            final_value = final_group[field]
            metrics[field] = {
                'best': best_value,
                'final': final_value,
                'delta': (None if best_value is None or final_value is None
                          else final_value - best_value),
            }
        transitions = Counter(
            '{} -> {}'.format(best[key]['failure_stage'],
                              final[key]['failure_stage'])
            for key in common
            if best[key]['class_name'] == group or
            (group == 'novel' and best[key]['group'] == 'novel'))
        comparison[group] = {
            'count': best_group['count'],
            'metrics': metrics,
            'failure_transitions': dict(transitions),
        }
    payload = {
        'best_csv': str(args.best),
        'final_csv': str(args.final),
        'num_common_gt': len(common),
        'interpretation': {
            'dense_localized': 'Decoded box geometry can reach IoU threshold.',
            'pre_nms_true_candidate': (
                'A correct-class candidate survives score_thr and nms_pre.'),
            'post_nms_correct': (
                'A correct-class prediction survives NMS/max_per_img.'),
            'localized_true_base_margin': (
                'Positive is better; negative means a Base class outscores '
                'the true Novel class at the selected localized position.'),
        },
        'groups': comparison,
    }
    output = Path(args.output)
    atomic_write_text(output, json.dumps(payload, indent=2))
    print('\n===== DETECTION PATH: BEST -> FINAL =====')
    for group in ('novel',) + DIOR_NOVEL_CLASSES:
        metrics = comparison[group]['metrics']
        print('\n{} (n={})'.format(group, comparison[group]['count']))
        for field in ('dense_localized_rate', 'pre_nms_true_candidate_rate',
                      'post_nms_correct_rate',
                      'localized_true_base_margin_mean'):
            values = metrics[field]
            print('  {}: {} -> {}  delta={}'.format(
                field, values['best'], values['final'], values['delta']))
    print('\nSaved: {}'.format(output))


def main():
    args = parse_args()
    if args.command == 'extract':
        extract(args)
    else:
        compare(args)


if __name__ == '__main__':
    main()

"""Evaluate DIOR OVC detections and export source coverage / fixed GT readouts.

GT is used only for evaluation, never support selection or predictions.
Use --max-images 0 for the complete validation set; subsets are marked as such.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F


def ranked_points(score, count=128):
    from yolo_world.models.layers.ovc_layers import _stable_descending_indices
    flat = score.detach().float().flatten()
    peak = score == F.max_pool2d(score, 7, stride=1, padding=3)
    order = _stable_descending_indices(flat)
    peaks = order[peak.flatten()[order]][:count]
    used = torch.zeros_like(flat, dtype=torch.bool)
    used[peaks] = True
    return torch.cat((peaks, order[~used[order]][:max(count-len(peaks), 0)]))


def box_class_readout(scores, boxes, image_index, strides):
    """Fixed 3x3 bin centers, max across samples/scales; a diagnostic, not AP."""
    if not len(boxes):
        return boxes.new_empty((0, scores[0].shape[1]))
    fractions = (torch.arange(3, device=boxes.device).float() + .5) / 3
    yy, xx = torch.meshgrid(fractions, fractions, indexing='ij')
    coords = boxes[:, None, :2] + (boxes[:, None, 2:] - boxes[:, None, :2]) * torch.stack((xx.flatten(), yy.flatten()), -1)
    values = []
    for logits, stride in zip(scores, strides):
        h, w = logits.shape[-2:]
        grid = 2 * coords / coords.new_tensor([stride * w, stride * h]) - 1
        sampled = F.grid_sample(logits[image_index:image_index + 1].float(), grid[None],
                                align_corners=False, padding_mode='border')[0]
        values.append(sampled.amax(-1).T)
    return torch.stack(values).amax(0)


def support_hits(boxes, indices, shape, input_shape):
    h, w = shape
    points = torch.stack(((indices % w + .5) * input_shape[1] / w,
                          (indices // w + .5) * input_shape[0] / h), -1)
    return ((points[None, :, 0] >= boxes[:, None, 0])
            & (points[None, :, 0] <= boxes[:, None, 2])
            & (points[None, :, 1] >= boxes[:, None, 1])
            & (points[None, :, 1] <= boxes[:, None, 3])).any(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--max-images', type=int, default=0)
    parser.add_argument('--visualize', type=int, default=8)
    parser.add_argument('--support-k', type=int, default=128)
    parser.add_argument('--output-dir', default='work_dirs/ovc_diagnostics')
    args = parser.parse_args()
    from mmengine.config import Config
    from mmengine.evaluator import Evaluator
    from mmengine.runner import Runner, load_checkpoint
    from mmyolo.registry import MODELS
    from mmyolo.utils import register_all_modules
    from yolo_world.models.layers.ovc_training import (
        augment_view, box_trimaps, region_consistency)
    import yolo_world  # noqa: F401
    register_all_modules()
    torch.set_num_threads(4)
    torch.manual_seed(214)
    cfg = Config.fromfile(args.config)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model = MODELS.build(cfg.model)
    model.init_weights()
    load_checkpoint(model, args.checkpoint, map_location='cpu', strict=False)
    model.to(args.device)
    model.eval()
    if not getattr(model.neck, 'adapter_enabled', False):
        raise ValueError('This diagnostic requires a checkpoint with an OVC Adapter.')
    loader_cfg = cfg.val_dataloader.copy()
    loader_cfg.update(batch_size=1, num_workers=0, persistent_workers=False)
    loader = Runner.build_dataloader(loader_cfg)
    evaluator = Evaluator(cfg.val_evaluator)
    evaluator.dataset_meta = loader.dataset.metainfo
    capture = {}
    handle = model.bbox_head.register_forward_hook(lambda m, x, out: capture.update(scores=out[0]))
    rows, image_stats = [], []
    names = list(loader.dataset.metainfo['classes'])
    novel = set(cfg.DIOR_NOVEL_CLASSES)
    try:
        for i, batch in enumerate(loader):
            if args.max_images and i >= args.max_images:
                break
            with torch.no_grad():
                data = model.data_preprocessor(batch, training=False)
                inputs, samples = data['inputs'], data['data_samples']
                result = model.predict(inputs, samples)
                evaluator.process(data_samples=result, data_batch=batch)
                sample = result[0]
                neck = model.neck
                shape, input_shape = neck.last_reliability.shape[-2:], inputs.shape[-2:]
                boxes = sample.gt_instances.bboxes
                boxes = boxes.tensor if hasattr(boxes, 'tensor') else boxes
                boxes = boxes.to(inputs.device).float()
                scale = boxes.new_tensor(sample.metainfo.get('scale_factor', (1., 1.)))
                if scale.numel() == 2:
                    scale = scale.repeat(2)
                pad = sample.metainfo.get('pad_param', (0, 0, 0, 0))
                boxes = boxes * scale + boxes.new_tensor([pad[2], pad[0], pad[2], pad[0]])
                labels = sample.gt_instances.labels.to(inputs.device)
                indices = neck.wsr.last_support_indices[0].clone()
                sources = neck.wsr.last_support_sources[0].clone()
                semantic_hits = support_hits(boxes, indices[sources == 0], shape, input_shape)
                visual_hits = support_hits(boxes, indices[sources == 1], shape, input_shape)
                all_hits = support_hits(boxes, indices, shape, input_shape)
                # Fixed-budget semantic-only counterfactual on identical features.
                count = neck.wsr.num_visual_tokens
                try:
                    neck.wsr.num_visual_tokens = 0
                    base_indices = neck.wsr._select_support_indices(
                        neck.last_support_features, neck.last_reliability)[0].clone()
                finally:
                    neck.wsr.num_visual_tokens = count
                    neck.wsr.last_support_indices = indices[None]
                    neck.wsr.last_support_sources = sources[None]
                baseline_hits = support_hits(boxes, base_indices, shape, input_shape)
                scores = box_class_readout(capture['scores'], boxes, 0,
                                           model.bbox_head.featmap_strides)
                order = scores.argsort(1, descending=True)
                rois = torch.cat((boxes.new_zeros(len(boxes), 1), boxes), 1)
                targets = box_trimaps(rois, 1, shape, input_shape)
                objectness = neck.main_objectness_logits.float().sigmoid()
                point_sets = {name: ranked_points(value, args.support_k) for name, value in
                              [('S', neck.last_reliability), ('O', objectness),
                               ('complement', objectness * (1-neck.last_reliability))]}
                point_hits = {name: support_hits(boxes, points, shape, input_shape)
                              for name, points in point_sets.items()}
                ys = (torch.arange(shape[0], device=inputs.device) + .5) * input_shape[0] / shape[0]
                xs = (torch.arange(shape[1], device=inputs.device) + .5) * input_shape[1] / shape[1]
                yy, xx = torch.meshgrid(ys, xs, indexing='ij')
                occupancy = torch.zeros(shape, device=inputs.device, dtype=torch.bool)
                for j, (box, label) in enumerate(zip(boxes, labels)):
                    mask = (xx >= box[0]) & (xx <= box[2]) & (yy >= box[1]) & (yy <= box[3])
                    occupancy |= mask
                    rank = int((order[j] == label).nonzero()[0, 0]) + 1
                    rows.append(dict(
                        image_id=sample.img_id, class_name=names[int(label)],
                        group='novel' if names[int(label)] in novel else 'base',
                        semantic_hit=int(semantic_hits[j]), visual_hit=int(visual_hits[j]),
                        visual_unique=int(visual_hits[j] & ~semantic_hits[j]),
                        all_support_hit=int(all_hits[j]), semantic_only_256_hit=int(baseline_hits[j]),
                        novel_added_over_256=int(all_hits[j] & ~baseline_hits[j]),
                        **{name+'_at_k_hit': int(hit[j]) for name, hit in point_hits.items()},
                        O_only_at_k=int(point_hits['O'][j] & ~point_hits['S'][j]),
                        complement_only_at_k=int(point_hits['complement'][j] & ~point_hits['S'][j]),
                        text_rank=rank, top1=int(rank == 1), top5=int(rank <= 5),
                        S_box_mean=float(neck.last_reliability[0, 0][mask].mean()) if mask.any() else None,
                        O_box_mean=float(objectness[0, 0][mask].mean()) if mask.any() else None))
                va = neck.main_visual
                vb, _ = model.auxiliary_visual_forward(augment_view(inputs, model.data_preprocessor))
                vc, var, stats = region_consistency(
                    va, vb, rois, input_shape,
                    normalized_regularization=model.ovc_training_cfg.get('normalized_regularization', False),
                    collect_spectrum=True)
                image_stats.append(dict(
                    image_id=sample.img_id, consistency=float(vc), variance=float(var),
                    supports_total=indices.numel(), supports_unique=indices.unique().numel(),
                    support_sources={str(s): int((sources == s).sum()) for s in range(5)},
                    **{k: float(v) for k, v in stats.items()},
                    **{name+'_background_point_fraction': float((~occupancy.flatten()[points]).float().mean())
                       for name, points in point_sets.items()},
                    wsr_background_point_fraction=float((~occupancy.flatten()[indices]).float().mean()),
                    O_outside_boxes=float(objectness[0, 0][~occupancy].mean()) if (~occupancy).any() else None))
                if i < args.visualize:
                    import matplotlib
                    matplotlib.use('Agg')
                    import matplotlib.pyplot as plt
                    from matplotlib.patches import Rectangle
                    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
                    image = inputs[0].float()
                    image = image * model.data_preprocessor.std + model.data_preprocessor.mean
                    axes[0].imshow((image / 255).clamp(0, 1).permute(1, 2, 0).cpu())
                    axes[0].set_title('Input / GT (evaluation only)')
                    for box in boxes.cpu():
                        axes[0].add_patch(Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1],
                                                   fill=False, edgecolor='lime', linewidth=.6))
                    tri = torch.full_like(targets[0][0, 0], .5)
                    tri[targets[2][0, 0]] = 0
                    tri[targets[0][0, 0].bool()] = 1
                    for ax, value, title in zip(axes[1:], [neck.last_reliability[0, 0], objectness[0, 0], tri],
                                                 ['S3 / supports', 'O / weak box scores', 'Weak target: 0 / ignore / 1']):
                        ax.imshow(value.cpu(), vmin=0, vmax=1, cmap='viridis')
                        ax.set_title(title)
                    for source, color in [(0, 'white'), (1, 'red'), (2, 'cyan'), (3, 'orange')]:
                        selected = indices[sources == source].cpu()
                        axes[1].scatter(selected % shape[1], selected // shape[1], s=4, c=color)
                    for ax in axes:
                        ax.axis('off')
                    fig.tight_layout()
                    fig.savefig(output / f'image_{i:04d}.png', dpi=130)
                    plt.close(fig)
                neck.clear_auxiliary_cache()
                capture.clear()
    finally:
        handle.remove()
    # COCO's default imgIds cover the entire annotation file, even for a subset.
    for metric in evaluator.metrics:
        if hasattr(metric, 'img_ids'):
            metric.img_ids = [row['image_id'] for row in image_stats]
    metrics = evaluator.evaluate(len(image_stats))
    summary = dict(checkpoint=args.checkpoint, images=len(image_stats),
                   full_validation=len(image_stats) == len(loader.dataset), metrics=metrics,
                   image_ids=[r['image_id'] for r in image_stats], support_k=args.support_k,
                   readout='GT 3x3 max across all levels, diagnostic only', groups={})
    for group in ['base', 'novel', *sorted(novel)]:
        subset = [r for r in rows if r['group'] == group or r['class_name'] == group]
        summary['groups'][group] = dict(count=len(subset), **{
            key: float(np.mean([r[key] for r in subset])) if subset else None
            for key in ('semantic_hit', 'visual_hit', 'visual_unique', 'all_support_hit',
                        'semantic_only_256_hit', 'novel_added_over_256', 'top1', 'top5',
                        'S_at_k_hit', 'O_at_k_hit', 'complement_at_k_hit',
                        'O_only_at_k', 'complement_only_at_k')})
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (output / 'image_statistics.json').write_text(json.dumps(image_stats, indent=2), encoding='utf-8')
    if rows:
        with (output / 'gt_diagnostics.csv').open('w', newline='', encoding='utf-8-sig') as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

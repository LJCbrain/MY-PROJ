"""Stage-one unknown point mining: calibration and evaluation, never training.

Uses the checkpoint's standard state_dict, matching normal checkpoint evaluation.
Calibration uses a fixed Base-only validation subset. Evaluation excludes that
subset. Novel boxes are only used for post-selection point coverage statistics.
No AP, pseudo labels, optimizer, teacher, or WSR selection changes are produced.
"""
import argparse
import csv
import json
import math
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')

import numpy as np
import torch
import torch.nn.functional as F

METHODS = ('original', 'two_objectness', 'U')


def grids(shape, input_shape, device):
    h, w = shape
    y = (torch.arange(h, device=device).float() + .5) * input_shape[0] / h
    x = (torch.arange(w, device=device).float() + .5) * input_shape[1] / w
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    return torch.stack((xx, yy), -1)


def in_boxes(points, boxes):
    return ((points[None, :, 0] >= boxes[:, None, 0]) &
            (points[None, :, 0] <= boxes[:, None, 2]) &
            (points[None, :, 1] >= boxes[:, None, 1]) &
            (points[None, :, 1] <= boxes[:, None, 3]))


def valid_grid(shape, input_shape, content_box, base_boxes, expansion=1.2):
    """Exclude padding, a patch border, and expanded Base boxes (>=1 cell margin)."""
    xy = grids(shape, input_shape, base_boxes.device)
    sx, sy = input_shape[1]/shape[1], input_shape[0]/shape[0]
    left, top, right, bottom = content_box
    valid = ((xy[..., 0] >= left+sx) & (xy[..., 0] <= right-sx) &
             (xy[..., 1] >= top+sy) & (xy[..., 1] <= bottom-sy))
    excluded = torch.zeros_like(valid)
    if len(base_boxes):
        center = (base_boxes[:, :2] + base_boxes[:, 2:]) / 2
        half = (base_boxes[:, 2:] - base_boxes[:, :2]) / 2
        margin = torch.maximum(half * (expansion-1), half.new_tensor([sx, sy]))
        expanded = torch.cat((center-half-margin, center+half+margin), -1)
        excluded = in_boxes(xy.reshape(-1, 2), expanded).any(0).reshape(shape)
    return valid & ~excluded, excluded, xy


@torch.no_grad()
def score_maps(va, vb, oa, ob, semantic):
    a = F.unfold(va.float(), kernel_size=3, padding=1)
    b = F.unfold(vb.float(), kernel_size=3, padding=1)
    # A zero patch supplies no evidence of agreement.
    nonzero = (a.norm(dim=1) > 1e-6) & (b.norm(dim=1) > 1e-6)
    consistency = ((F.cosine_similarity(a, b, dim=1, eps=1e-6)+1)/2)
    consistency = (consistency * nonzero).reshape(semantic.shape).clamp(0, 1)
    joint = (oa.float()*ob.float()).clamp_min(0).sqrt()
    maps = dict(original=oa*(1-semantic), two_objectness=joint*(1-semantic),
                U=joint*consistency*(1-semantic))
    return {k: v[0, 0].detach().cpu() for k, v in maps.items()}, consistency[0, 0].cpu()


def peaks(score, valid, count):
    """Stable 7x7 local peaks; no dense filling, no duplicates, zero is allowed."""
    masked = score.float().masked_fill(~valid, -float('inf'))
    local = F.max_pool2d(masked[None, None], 7, stride=1, padding=3)[0, 0]
    candidates = (valid & torch.isfinite(masked) & (masked > 0) & (masked == local)).flatten().nonzero().flatten()
    if not len(candidates):
        return candidates
    # NumPy stable sort is compatible with the server's PyTorch 1.11.
    order = np.argsort(-masked.flatten()[candidates].numpy(), kind='mergesort')
    selected = []
    width = score.shape[1]
    for index in candidates[torch.from_numpy(order)].tolist():
        y, x = divmod(index, width)
        # Deterministically suppress adjacent equal-score plateau peaks.
        if all(max(abs(y-j//width), abs(x-j % width)) > 3 for j in selected):
            selected.append(index)
            if len(selected) == count:
                break
    return torch.tensor(selected, dtype=torch.long)


def calibrate_threshold(values, image_count, target):
    """Largest admissible candidate budget, with strict score > threshold ties."""
    values = sorted(values, reverse=True)
    allowed = math.floor(image_count * target)
    if not values:
        return 1.0
    if len(values) <= allowed:
        return 0.0
    return float(values[allowed])


def boxes_in_input(sample):
    boxes = sample.gt_instances.bboxes
    boxes = (boxes.tensor if hasattr(boxes, 'tensor') else boxes).detach().cpu().float()
    # Standard DIOR validation loads annotations AFTER geometric transforms.
    scale = torch.tensor(sample.metainfo.get('scale_factor', (1., 1.))).float()
    if len(scale) == 2:
        scale = scale.repeat(2)
    pad = sample.metainfo.get('pad_param', (0, 0, 0, 0))
    return boxes*scale + boxes.new_tensor([pad[2], pad[0], pad[2], pad[0]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('checkpoint', nargs='?', default='', help='Fill in the corrected 2.15 checkpoint path; no default weights')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--calibration-images', type=int, default=128)
    parser.add_argument('--max-images', type=int, default=256, help='0: all remaining validation images')
    parser.add_argument('--seed', type=int, default=215)
    parser.add_argument('--view-seed', type=int, default=215, help='Change to repeat photometric perturbations with the SAME image split')
    parser.add_argument('--objectness-threshold', type=float, default=.5)
    parser.add_argument('--consistency-threshold', type=float, default=.9)
    parser.add_argument('--target-fp', type=float, default=.5)
    parser.add_argument('--visualize', type=int, default=8)
    parser.add_argument('--output-dir', default='work_dirs/unknown_mining')
    args = parser.parse_args()
    if not args.checkpoint.strip():
        parser.error('Checkpoint path is empty. Fill in a checkpoint from the corrected DIOR 2.15 run.')
    if args.calibration_images < 1 or args.max_images < 0 or args.target_fp < 0:
        parser.error('Invalid image counts / target-fp')
    if not all(0 <= v <= 1 for v in (args.objectness_threshold, args.consistency_threshold)):
        parser.error('Thresholds must be in [0,1]')
    from mmengine.config import Config
    from mmengine.dataset import pseudo_collate
    from mmengine.runner import Runner, load_checkpoint
    from mmyolo.registry import MODELS
    from mmyolo.utils import register_all_modules
    from yolo_world.models.layers.ovc_training import augment_view
    import yolo_world  # noqa
    register_all_modules()
    torch.set_num_threads(4)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cfg = Config.fromfile(args.config)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    # Establish the coordinate contract before interpreting validation GT.
    pipeline = cfg.val_dataloader.dataset.pipeline
    types = [x['type'].split('.')[-1] for x in pipeline]
    annotations = types.index('LoadAnnotations')
    if any(('Resize' in name or 'Letter' in name or 'Flip' in name) for name in types[annotations+1:]):
        raise ValueError('Unsupported validation geometry after LoadAnnotations; GT coordinates need an explicit adapter.')
    loader_cfg = cfg.val_dataloader.copy()
    loader_cfg.update(batch_size=1, num_workers=0, persistent_workers=False)
    dataset = Runner.build_dataloader(loader_cfg).dataset
    names = list(dataset.metainfo['classes'])
    base_names, novel_names = set(cfg.DIOR_BASE_CLASSES), set(cfg.DIOR_NOVEL_CLASSES)
    complete = []
    for index in range(len(dataset)):
        instances = dataset.get_data_info(index)['instances']
        if instances and all(names[x['bbox_label']] in base_names for x in instances):
            complete.append(index)
    rng = random.Random(args.seed)
    rng.shuffle(complete)
    if len(complete) < args.calibration_images:
        raise ValueError('Not enough Base-only validation images for requested calibration size')
    calibration = complete[:args.calibration_images]
    held_out = set(calibration)
    evaluation = [i for i in range(len(dataset)) if i not in held_out]
    rng.shuffle(evaluation)
    if args.max_images:
        evaluation = evaluation[:args.max_images]
    if not evaluation:
        raise ValueError('Empty evaluation split')
    protocol = dict(arguments=vars(args), calibration_ids=[dataset.get_data_info(i)['img_id'] for i in calibration],
                    evaluation_ids=[dataset.get_data_info(i)['img_id'] for i in evaluation],
                    base_texts=list(cfg.DIOR_BASE_CLASSES), method_formulas={
                        'original': 'Oa*(1-S_base)', 'two_objectness': 'sqrt(Oa*Ob)*(1-S_base)',
                        'U': 'sqrt(Oa*Ob)*C*(1-S_base)'},
                    note='No AP. Calibration and evaluation are disjoint parts of validation, not a final test set. Base GT exclusion is diagnostic assistance, not deployable GT-free inference.')
    (output/'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    model = MODELS.build(cfg.model)
    model.init_weights()
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu', strict=False)
    weights = checkpoint.get('state_dict', checkpoint)
    weight_names = {key.removeprefix('module.') if hasattr(key, 'removeprefix') else key[7:] if key.startswith('module.') else key for key in weights}
    missing = [key for key, _ in model.named_parameters() if key not in weight_names]
    if missing:
        raise ValueError('Checkpoint lacks model parameters; refusing random diagnostic weights: '+str(missing[:12]))
    del checkpoint, weights
    model.to(args.device).eval()
    if not getattr(model.neck, 'adapter_enabled', False):
        raise ValueError('Requires OVC Adapter')
    versions = {key: p._version for key, p in model.named_parameters()}
    bn = {key: buf.clone() for key, buf in model.named_buffers() if 'running_mean' in key or 'running_var' in key}
    texts = json.loads(Path('data/DIOR/classes_base_16cls.json').read_text(encoding='utf-8'))
    if len(texts) != 16 or [x[0] for x in texts] != list(cfg.DIOR_BASE_CLASSES):
        raise ValueError('Base text file/category order mismatch')
    with torch.no_grad():
        guide, text_mask = model.backbone.forward_text([[x[0] for x in texts]])

    def extract(index):
        with torch.no_grad():
            data = model.data_preprocessor(pseudo_collate([dataset[index]]), training=False)
            inputs, sample = data['inputs'], data['data_samples'][0]
            image_id = int(sample.img_id)
            torch.manual_seed((args.view_seed + image_id*1009) % (2**31))
            # Same unprompted Base text guidance as the training neck. No head
            # is needed for point discovery, and no Novel text/GT enters here.
            backbone = model.backbone.forward_image(inputs)
            model.neck(backbone, guide, txt_masks=text_mask)
            va, logits = model.neck.main_visual, model.neck.main_objectness_logits
            semantic = model.neck.last_reliability.detach().float()
            supports = model.neck.wsr.last_support_indices.clone()
            view_cfg = model.ovc_training_cfg
            auxiliary = augment_view(inputs, model.data_preprocessor,
                view_cfg['color_amount'], view_cfg['hue'], view_cfg['grayscale_prob'])
            vb, lb = model.auxiliary_visual_forward(auxiliary)
            if model.neck.main_visual is not va or not torch.equal(supports, model.neck.wsr.last_support_indices):
                raise RuntimeError('Auxiliary forward modified main-view cache/WSR supports')
            oa, ob = logits.float().sigmoid(), lb.float().sigmoid()
            scores, consistency = score_maps(va, vb, oa, ob, semantic)
            boxes = boxes_in_input(sample)
            labels = sample.gt_instances.labels.cpu()
            base = torch.tensor([names[int(label)] in base_names for label in labels], dtype=torch.bool)
            shape, input_shape = semantic.shape[-2:], inputs.shape[-2:]
            pad = sample.metainfo.get('pad_param', (0, 0, 0, 0))
            resized_shape = sample.metainfo['img_shape']
            content = (float(pad[2]), float(pad[0]), float(resized_shape[1]-pad[3]), float(resized_shape[0]-pad[1]))
            valid, excluded, xy = valid_grid(shape, input_shape, content, boxes[base])
            gates = dict(original=oa[0, 0].cpu() > args.objectness_threshold,
                         two_objectness=(oa[0, 0].cpu() > args.objectness_threshold) & (ob[0, 0].cpu() > args.objectness_threshold))
            gates['U'] = gates['two_objectness'] & (consistency > args.consistency_threshold)
            all_peaks = {name: peaks(scores[name], valid, valid.numel()) for name in METHODS}
            raw = {name: ids[:32] for name, ids in all_peaks.items()}
            # Gate the SAME local peaks rather than turning rejected neighbors
            # into new local maxima. Thresholds cannot create extra candidates.
            gated = {name: ids[gates[name].flatten()[ids]][:8] for name, ids in all_peaks.items()}
            pixels = inputs[0].float()
            if model.data_preprocessor._enable_normalize:
                pixels = pixels*model.data_preprocessor.std[0]+model.data_preprocessor.mean[0]
            if not model.data_preprocessor._channel_conversion:
                pixels = pixels[[2, 1, 0]]
            result = dict(image_id=image_id, boxes=boxes, labels=labels, xy=xy.reshape(-1, 2),
                          scores=scores, raw=raw, gated=gated, valid=valid, excluded=excluded,
                          pixels=pixels.clamp(0,255).byte().permute(1,2,0).cpu().numpy())
            model.neck.clear_auxiliary_cache()
            return result

    calibration_values = {name: [] for name in METHODS}
    for j, index in enumerate(calibration):
        record = extract(index)
        for name in METHODS:
            calibration_values[name].extend(record['scores'][name].flatten()[record['gated'][name]].tolist())
        if (j+1) % 10 == 0 or j+1 == len(calibration):
            print('Calibration {}/{}'.format(j+1, len(calibration)), flush=True)
    thresholds = {name: calibrate_threshold(values, len(calibration), args.target_fp)
                  for name, values in calibration_values.items()}
    calibration_result = dict(thresholds=thresholds, observed_fp_per_image={
        name: sum(v > thresholds[name] for v in values)/len(calibration)
        for name, values in calibration_values.items()}, candidates_before_threshold={
        name: len(values) for name, values in calibration_values.items()})
    (output/'calibration.json').write_text(json.dumps(calibration_result, indent=2), encoding='utf-8')
    image_rows, gt_rows, point_rows = [], [], []
    from PIL import Image, ImageDraw
    for j, index in enumerate(evaluation):
        record = extract(index)
        boxes, labels, xy = record['boxes'], record['labels'], record['xy']
        novel = torch.tensor([names[int(label)] in novel_names for label in labels], dtype=torch.bool)
        raw = record['raw']
        strict = {name: ids[record['scores'][name].flatten()[ids] > thresholds[name]] for name, ids in record['gated'].items()}
        equal_count = min(len(ids) for ids in strict.values())
        sets = dict(rank32=raw, strict8=strict,
                    matched_strict={name: ids[:equal_count] for name, ids in strict.items()})
        for mode, methods in sets.items():
            baseline_hits = in_boxes(xy[methods['original']], boxes).any(1)
            for name, ids in methods.items():
                incidence = in_boxes(xy[ids], boxes)
                hits = incidence.any(1)
                novel_points = incidence[novel].any(0)
                base_points = incidence[~novel].any(0)
                background = ~incidence.any(0)
                image_rows.append(dict(image_id=record['image_id'], mode=mode, method=name,
                    candidates=len(ids), novel_candidates=int(novel_points.sum()), background_candidates=int(background.sum()),
                    base_candidates=int(base_points.sum()), novel_gt=int(novel.sum()), novel_covered=int(hits[novel].sum()),
                    novel_unique_over_original=int((hits & ~baseline_hits & novel).sum())))
                for k, (box, label) in enumerate(zip(boxes, labels)):
                    region = in_boxes(xy, box[None])[0]
                    gt_rows.append(dict(image_id=record['image_id'], gt_index=k, class_name=names[int(label)],
                        mode=mode, method=name, hit=int(hits[k]),
                        base_excluded_cell_fraction=float(record['excluded'].flatten()[region].float().mean()) if region.any() else None,
                        has_eligible_grid_cell=bool((region & record['valid'].flatten()).any())))
                if mode != 'matched_strict':
                    for k, cell in enumerate(ids.tolist()):
                        point_rows.append(dict(image_id=record['image_id'], mode=mode, method=name, cell=cell,
                            input_x=float(xy[cell,0]), input_y=float(xy[cell,1]), score=float(record['scores'][name].flatten()[cell]),
                            novel_hit=bool(novel_points[k]), background=bool(background[k])))
        if j < args.visualize:
            panels = []
            for name in METHODS:
                im = Image.fromarray(record['pixels']).convert('RGB')
                draw = ImageDraw.Draw(im)
                for box, is_novel in zip(boxes.tolist(), novel.tolist()):
                    draw.rectangle(box, outline='red' if is_novel else 'lime', width=2)
                for point in xy[raw[name]].tolist():
                    x,y=point; draw.ellipse((x-3,y-3,x+3,y+3), fill='yellow')
                for point in xy[strict[name]].tolist():
                    x,y=point; draw.ellipse((x-5,y-5,x+5,y+5), outline='cyan', width=2)
                draw.text((8,8), '{} yellow=rank32 cyan=strict8'.format(name), fill='white', stroke_width=1, stroke_fill='black')
                panels.append(im)
            canvas = Image.new('RGB', (sum(im.width for im in panels), panels[0].height))
            offset=0
            for im in panels:
                canvas.paste(im,(offset,0));offset+=im.width
            canvas.save(output/('points_{}.png'.format(record['image_id'])))
        if (j+1) % 10 == 0 or j+1 == len(evaluation):
            print('Evaluation {}/{}'.format(j+1,len(evaluation)), flush=True)
    metrics = {}
    for mode in ('rank32','strict8','matched_strict'):
        metrics[mode] = {}
        for name in METHODS:
            rows=[r for r in image_rows if r['mode']==mode and r['method']==name]
            total=lambda key:sum(r[key] for r in rows)
            ratio=lambda a,b: a/b if b else None
            metrics[mode][name]=dict(candidates=total('candidates'), candidates_per_image=total('candidates')/len(rows),
                zero_candidate_images=sum(r['candidates']==0 for r in rows),
                novel_candidate_precision=ratio(total('novel_candidates'),total('candidates')),
                background_fraction=ratio(total('background_candidates'),total('candidates')),
                base_overlap_fraction=ratio(total('base_candidates'),total('candidates')),
                novel_gt=total('novel_gt'), novel_instance_coverage=ratio(total('novel_covered'),total('novel_gt')),
                unique_novel_over_original=total('novel_unique_over_original'))
            complete_rows=[r for r in rows if r['novel_gt']==0]
            metrics[mode][name]['heldout_complete_candidates_per_image']=ratio(sum(r['candidates'] for r in complete_rows),len(complete_rows))
            metrics[mode][name]['per_novel_class']={}
            for cls in cfg.DIOR_NOVEL_CLASSES:
                selected=[r for r in gt_rows if r['mode']==mode and r['method']==name and r['class_name']==cls]
                metrics[mode][name]['per_novel_class'][cls]=dict(gt=len(selected), coverage=ratio(sum(r['hit'] for r in selected),len(selected)))
    for filename, rows in [('images.csv',image_rows),('gt_coverage.csv',gt_rows),('points.csv',point_rows)]:
        with (output/filename).open('w',newline='',encoding='utf-8') as stream:
            if rows:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    unchanged=all(p._version==versions[key] and p.grad is None for key,p in model.named_parameters())
    unchanged &= all(torch.equal(buf,bn[key]) for key,buf in model.named_buffers() if key in bn)
    summary=dict(images=len(evaluation), calibration_images=len(calibration), metrics=metrics,
        calibration=calibration_result, weights_and_bn_unchanged=unchanged,
        full_validation_AP=False, checkpoint=str(Path(args.checkpoint).resolve()),
        warnings=['Point coverage is not proposal recall or AP.',
                  'No automatic promotion: inspect background, coverage, per-class counts and a second view seed.',
                  'matched_strict can become empty when any method selects zero; inspect its counts.',
                  'Threshold .5 objectness/.9 consistency are initial settings, not calibrated probabilities.'])
    (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    if not unchanged:
        raise RuntimeError('Unexpected parameter or BN change during diagnostic')
    print('DONE: '+str(output/'summary.json'),flush=True)


if __name__ == '__main__':
    main()

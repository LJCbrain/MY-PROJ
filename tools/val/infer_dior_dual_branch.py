"""Frozen DIOR inference: extract -> score (CLIP/RemoteCLIP) -> evaluate.

No GT boxes select proposals, crops, labels, or scores. GT is used only by
COCOeval. Per-image caches permit interruption and resume with identical settings.
"""
import argparse
import contextlib
import csv
import hashlib
import inspect
import io
import json
import os
import platform
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
# Share candidate selection/fusion/NMS with the detector without importing its registry.
import importlib.util
_roi_spec = importlib.util.spec_from_file_location(
    '_yolo_world_clip_roi_infer', ROOT / 'yolo_world/models/layers/clip_roi.py')
_roi = importlib.util.module_from_spec(_roi_spec)
_roi_spec.loader.exec_module(_roi)
nms_indices = _roi.nms_indices
candidate_indices = _roi.candidate_indices
candidate_subset = _roi.candidate_subset
geometric_fusion = _roi.geometric_fusion
postprocess = _roi.postprocess

CLASSES = ('airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge',
           'chimney', 'dam', 'Expressway-Service-area', 'Expressway-toll-station',
           'golffield', 'groundtrackfield', 'harbor', 'overpass', 'ship', 'stadium',
           'storagetank', 'tenniscourt', 'trainstation', 'vehicle', 'windmill')
NOVEL_IDS = (1, 3, 10, 19)
# User-frozen deployment protocol, 2026-10-23. Historical sweeps require
# explicit CLI overrides; new experiments use this single setting.
FIXED_FUSION = dict(ranking='yolo_max', k=100, multiplier='none', novel_lambda=.85)
FUSION_MODELS = ('clip', 'remoteclip', 'clip_roi', 'skyclip_roi', 'remoteclip_roi', 'rn50_roi', 'remoteclip_rn50_roi')


def log(message):
    print(time.strftime('[%H:%M:%S] ') + message, flush=True)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               allow_nan=False), encoding='utf-8')
    os.replace(str(temp), str(path))


def save_npz(path, **arrays):
    temp = Path(str(path) + '.tmp')
    with temp.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(str(temp), str(path))


def check_protocol(path, value):
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != value:
            raise ValueError('Cache settings changed; use a new directory: ' + str(path))
    else:
        write_json(path, value)


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))






def read_manifest(cache):
    manifest = load_json(cache / 'manifest.json')
    if not manifest['images']:
        raise ValueError('Empty extraction manifest')
    return manifest


def extract(args):
    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmengine.runner import Runner
    from mmyolo.registry import MODELS
    from mmyolo.utils import register_all_modules
    import yolo_world  # noqa: F401
    import yolo_world.evaluation  # noqa: F401
    from tools.val.diagnose_dior_detection_path import decode_dense

    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    options = dict(map_location='cpu')
    if 'weights_only' in inspect.signature(torch.load).parameters:
        options['weights_only'] = False
    checkpoint = Path(args.checkpoint).resolve()
    log('Reading detector checkpoint and original configuration.')
    state = torch.load(str(checkpoint), **options)
    cfg = Config.fromstring(state['meta']['cfg'], '.py')
    if cfg.model.type != 'YOLOWorldOVCPartialDetector':
        raise ValueError('Expected a YOLOWorldOVCPartialDetector checkpoint (e.g. DIOR 2.17 or 3.12).')
    text_cfg = cfg.model.backbone.get('text_model', {})
    if ('semantic_fusion_cfg' in text_cfg
            and not text_cfg['semantic_fusion_cfg'].get('enabled', False)):
        text_cfg.pop('semantic_fusion_cfg')
    evaluators = cfg.val_evaluator
    if isinstance(evaluators, dict):
        evaluators = [evaluators]
    metric = next(v for v in evaluators if v.get('metric') == 'bbox')
    ann_file = str(Path(metric['ann_file']).resolve())
    protocol = dict(version=1, checkpoint_sha256=digest(checkpoint),
                    annotation_sha256=digest(ann_file), classes=list(CLASSES),
                    max_images=args.max_images, o_candidates=args.o_candidates,
                    semantic_candidates=args.semantic_candidates,
                    construction='union of clipped-box O-ranked and max-YOLO-score-ranked NMS candidates',
                    selection_uses_gt=False)
    check_protocol(cache / 'extraction_protocol.json', protocol)
    cfg.dump(str(cache / 'source_config.py'))
    register_all_modules()
    init_default_scope(cfg.get('default_scope', 'mmyolo'))
    model = MODELS.build(cfg.model)
    model.init_weights()
    # EMA deployment weights live in state_dict. Do not load ema_state_dict.
    model.load_state_dict(state['state_dict'], strict=True)
    del state
    model.to(args.device).eval().requires_grad_(False)
    loader_cfg = cfg.val_dataloader.copy()
    loader_cfg.update(batch_size=1, num_workers=0, persistent_workers=False)
    loader = Runner.build_dataloader(loader_cfg, seed=0)
    if tuple(loader.dataset.metainfo['classes']) != CLASSES:
        raise ValueError('Detector category order does not match DIOR 20 classes.')
    test_cfg = model.bbox_head.test_cfg
    if not test_cfg.multi_label or test_cfg.get('yolox_style', False):
        raise ValueError('This experiment expects the original multi-label DIOR postprocessing.')
    manifest = dict(protocol=protocol, ann_file=ann_file,
                    max_dets=list(metric.get('proposal_nums', [100, 300, 1000])),
                    score_thr=float(test_cfg.score_thr), nms_pre=int(test_cfg.nms_pre),
                    nms_iou=float(test_cfg.nms.iou_threshold),
                    max_per_img=int(test_cfg.max_per_img), images=[])
    folder = cache / 'candidates'
    folder.mkdir(exist_ok=True)
    start = time.monotonic()
    with torch.no_grad():
        for number, batch in enumerate(loader):
            if args.max_images and number >= args.max_images:
                break
            raw_meta = batch['data_samples'][0].metainfo
            img_id = int(raw_meta['img_id'])
            path = folder / '{}.npz'.format(img_id)
            entry = dict(id=img_id, path=str(Path(raw_meta['img_path']).resolve()),
                         height=int(raw_meta['ori_shape'][0]), width=int(raw_meta['ori_shape'][1]))
            if not path.exists():
                data = model.data_preprocessor(batch, training=False)
                samples = data['data_samples']
                feats, texts, masks = model.extract_feat(data['inputs'], samples)
                model.bbox_head.num_classes = texts.shape[-2]
                cls, reg = model.bbox_head(feats, texts, masks)
                boxes, scores = decode_dense(model.bbox_head, cls, reg, samples[0].metainfo)
                post = model.bbox_head.predict_by_feat(
                    cls, reg, batch_img_metas=[samples[0].metainfo], rescale=True)[0]
                o_map = model.neck.main_objectness_logits.detach().float().sigmoid()
                o_scores = torch.cat([F.interpolate(o_map, size=x.shape[-2:], mode='bilinear',
                                                    align_corners=False).flatten() for x in reg])
                keep = candidate_indices(boxes, scores, o_scores,
                                         args.o_candidates, args.semantic_candidates)
                arrays = dict(boxes=boxes[keep], yolo_scores=scores[keep],
                              objectness=o_scores[keep], dense_indices=keep,
                              original_boxes=post.bboxes, original_scores=post.scores,
                              original_labels=post.labels)
                save_npz(path, **{k: v.cpu().numpy() for k, v in arrays.items()})
                model.neck.clear_auxiliary_cache()
            manifest['images'].append(entry)
            if number == 0 or (number + 1) % 100 == 0:
                log('Extracted {}/{} images; {:.0f}s'.format(
                    number + 1, min(len(loader), args.max_images or len(loader)),
                    time.monotonic() - start))
    write_json(cache / 'manifest.json', manifest)
    log('Candidate cache complete. Evaluating original detector and candidate controls.')
    summaries = evaluate_cache(cache, ['yolo'], [1.0])
    if args.expected_hm is not None:
        actual = 100 * summaries['original']['hm']
        if abs(actual - args.expected_hm) > args.hm_tolerance:
            raise ValueError('Original HM {:.4f}% differs from expected {:.4f}%; '
                             'check checkpoint/data/code before scoring VLM crops.'.format(
                                 actual, args.expected_hm))


def score(args):
    from transformers import AutoTokenizer, CLIPModel
    from tools.val.compare_clip_gt_crops import CropDataset, crop_bounds, build_prompts, load_remoteclip

    cache = Path(args.cache_dir)
    manifest = read_manifest(cache)
    folder = cache / args.model
    folder.mkdir(exist_ok=True)
    class_texts = Path(args.class_texts)
    model_path = Path(args.clip_model if args.model == 'clip' else args.remote_checkpoint)
    # Model identity includes complete local weights; prevent stale score reuse.
    weight_files = ([p for p in sorted(model_path.iterdir())
                     if p.suffix in ('.bin', '.safetensors', '.json', '.txt')]
                    if model_path.is_dir() else [model_path])
    identity = {p.name: digest(p) for p in weight_files}
    tokenizer_identity = {p.name: digest(p) for p in sorted(Path(args.clip_model).iterdir())
                          if p.suffix in ('.json', '.txt')}
    protocol = dict(extraction=manifest['protocol'], model=args.model,
                    weights=identity, tokenizer=tokenizer_identity,
                    context=args.context, precision='FP32',
                    class_text_sha256=digest(class_texts), template=args.template,
                    crop_policy='clipped region, mean-color square pad, native-resolution bicubic')
    check_protocol(folder / 'protocol.json', protocol)
    prompts = build_prompts(class_texts, [args.template])
    tokenizer = AutoTokenizer.from_pretrained(args.clip_model, local_files_only=True)
    log('Loading full {} (image + text), local weights only.'.format(args.model))
    if args.model == 'clip':
        model, info = CLIPModel.from_pretrained(args.clip_model, local_files_only=True,
                                               output_loading_info=True)
        if info['missing_keys'] or info.get('mismatched_keys'):
            raise ValueError('Incomplete CLIP weights: {}'.format(info))
    else:
        model = load_remoteclip(args.remote_checkpoint, tokenizer)
    model.to(args.device).eval().requires_grad_(False)
    size = model.config.vision_config.image_size
    with torch.no_grad():
        vectors = []
        for names in prompts:
            tokens = tokenizer(names, padding=True, return_tensors='pt')
            tokens = {k: v.to(args.device) for k, v in tokens.items()}
            text = F.normalize(model.get_text_features(**tokens).float(), dim=-1)
            vectors.append(F.normalize(text.mean(0), dim=-1))
        text = torch.stack(vectors)
        logit_scale = float(model.logit_scale.exp().cpu())
        if not np.isfinite(logit_scale) or logit_scale <= 0:
            raise ValueError('Invalid VLM logit scale')
        write_json(folder / 'model_info.json', dict(logit_scale=logit_scale, image_size=size))
        start = time.monotonic()
        completed_crops = 0
        for index, entry in enumerate(manifest['images']):
            source = cache / 'candidates' / '{}.npz'.format(entry['id'])
            target = folder / source.name
            if target.exists():
                continue
            with np.load(source) as data:
                boxes = data['boxes']
            records = []
            for x1, y1, x2, y2 in boxes:
                bounds = crop_bounds((x1, y1, x2-x1, y2-y1), entry['width'], entry['height'], args.context)
                if bounds is None:
                    raise ValueError('Invalid cached candidate for image {}'.format(entry['id']))
                records.append(dict(file_name=entry['path'], width=entry['width'],
                                    height=entry['height'], crop_xyxy=bounds))
            dataset = CropDataset(records, '.', size)
            outputs = []
            for offset in range(0, len(dataset), args.batch_size):
                pixels = torch.stack([dataset[j] for j in range(offset, min(offset+args.batch_size, len(dataset)))])
                visual = F.normalize(model.get_image_features(pixel_values=pixels.to(args.device)).float(), dim=-1)
                outputs.append((visual @ text.T).cpu().numpy())
            cosine = np.concatenate(outputs) if outputs else np.empty((0, 20), np.float32)
            if not np.isfinite(cosine).all():
                raise ValueError('Nonfinite similarities')
            save_npz(target, cosine=cosine, candidate_sha256=np.array(digest(source)))
            completed_crops += len(records)
            if index == 0 or (index+1) % 10 == 0 or index+1 == len(manifest['images']):
                elapsed = time.monotonic()-start
                log('{}: {}/{} images, {} new crops, {:.1f} crops/s'.format(
                    args.model, index+1, len(manifest['images']), completed_crops,
                    completed_crops/max(elapsed, 1e-9)))
    log('Scores cached. This step does not select labels or use GT.')


def score_matrix(data, cosine=None, logit_scale=1.0, multiplier='objectness'):
    if cosine is None:
        return torch.from_numpy(data['yolo_scores']).float()
    # All 20 classes compete. These are initial scores, not calibrated
    # foreground probabilities. Keep raw cosine values for later experiments.
    values = torch.from_numpy(cosine).float()
    scores = (values * logit_scale).softmax(-1)
    if multiplier == 'objectness':
        scores *= torch.from_numpy(data['objectness']).float()[:, None]
    elif multiplier == 'yolo_max':
        scores *= torch.from_numpy(data['yolo_scores']).float().max(-1).values[:, None]
    elif multiplier != 'none':
        raise ValueError('Unknown VLM multiplier: ' + multiplier)
    return scores






def predictions(data, matrix, manifest, split=False, scale=1.0):
    boxes = torch.from_numpy(data['boxes']).float()
    if split:
        matrix = matrix[:, list(NOVEL_IDS)] * scale
        locations, columns = torch.where(matrix > manifest['score_thr'])
        labels = torch.tensor(NOVEL_IDS)[columns]
    else:
        locations, labels = torch.where(matrix > manifest['score_thr'])
        columns = labels
    scores = matrix[locations, columns]
    boxes = boxes[locations]
    if split:
        original_labels = torch.from_numpy(data['original_labels']).long()
        base = ~torch.isin(original_labels, torch.tensor(NOVEL_IDS))
        boxes = torch.cat((torch.from_numpy(data['original_boxes']).float()[base], boxes))
        scores = torch.cat((torch.from_numpy(data['original_scores']).float()[base], scores))
        labels = torch.cat((original_labels[base], labels))
    return postprocess(boxes, scores, labels, manifest)


def metric_values(ann_file, image_ids, records, max_dets):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(ann_file)
        if records:
            dt = gt.loadRes(records)
        else:
            dt = COCO()
            dt.dataset = dict(images=gt.dataset['images'], categories=gt.dataset['categories'], annotations=[])
            dt.createIndex()
        evaluator = COCOeval(gt, dt, 'bbox')
        evaluator.params.imgIds = image_ids
        evaluator.params.maxDets = list(max_dets)
        evaluator.params.iouThrs = np.array([.5])
        evaluator.evaluate()
        evaluator.accumulate()
    precision = evaluator.eval['precision'][0, :, :, 0, -1]
    values = {}
    for column, cat_id in enumerate(evaluator.params.catIds):
        p = precision[:, column]
        valid = p[p >= 0]
        values[gt.cats[cat_id]['name']] = float(valid.mean()) if len(valid) else 0.0
    def mean(names):
        numbers = [values[n] for n in names if values[n] is not None]
        return float(np.mean(numbers)) if numbers else 0.0
    base = mean([c for i, c in enumerate(CLASSES) if i not in NOVEL_IDS])
    novel = mean([CLASSES[i] for i in NOVEL_IDS])
    return dict(base_ap50=base, novel_ap50=novel, hm=2*base*novel/(base+novel) if base+novel else 0.,
                per_class_ap50=values, predictions=len(records), images=len(image_ids))


def evaluate_cache(cache, models, scales):
    manifest = read_manifest(cache)
    if digest(manifest['ann_file']) != manifest['protocol']['annotation_sha256']:
        raise ValueError('Annotation file changed since extraction')
    cats = load_json(manifest['ann_file'])['categories']
    name_ids = {c['name']: c['id'] for c in cats}
    if set(name_ids) != set(CLASSES):
        raise ValueError('Annotation class names differ')
    variants = [('original', 'yolo', False, 1.)]
    for name in models:
        variants.append((name + '_candidates', name, False, 1.))
        for scale in ([1.] if name == 'yolo' else scales):
            variants.append(('split_{}_scale{:g}'.format(name, scale), name, True, scale))
    out = cache / 'evaluation'
    out.mkdir(exist_ok=True)
    summaries = {}
    for variant, name, split, scale in variants:
        records = []
        logit_scale = 1. if name == 'yolo' else load_json(cache / name / 'model_info.json')['logit_scale']
        for entry in manifest['images']:
            source = cache / 'candidates' / '{}.npz'.format(entry['id'])
            with np.load(source) as data:
                if variant == 'original':
                    result = tuple(torch.from_numpy(data['original_'+key]) for key in ('boxes', 'scores', 'labels'))
                else:
                    cosine = None
                    if name != 'yolo':
                        with np.load(cache / name / source.name) as scored:
                            if str(scored['candidate_sha256']) != digest(source):
                                raise ValueError('Candidate/score cache mismatch')
                            cosine = scored['cosine']
                        if cosine.shape != data['yolo_scores'].shape:
                            raise ValueError('Wrong score shape')
                    result = predictions(data, score_matrix(data, cosine, logit_scale), manifest, split, scale)
                boxes, scores, labels = (v.numpy() for v in result)
            for box, score_value, label in zip(boxes, scores, labels):
                x1, y1, x2, y2 = map(float, box)
                records.append(dict(image_id=entry['id'], category_id=name_ids[CLASSES[int(label)]],
                                    bbox=[x1, y1, x2-x1, y2-y1], score=float(score_value)))
        result = metric_values(manifest['ann_file'], [e['id'] for e in manifest['images']], records, manifest['max_dets'])
        summaries[variant] = result
        write_json(out / (variant + '.json'), result)
        log('{}: Base={:.2f}, Novel={:.2f}, HM={:.2f}'.format(
            variant, 100*result['base_ap50'], 100*result['novel_ap50'], 100*result['hm']))
    write_json(out / ('summary_' + '_'.join(models) + '.json'), summaries)
    return summaries




def two_folds(image_ids, seed):
    if len(image_ids) < 2 or len(set(image_ids)) != len(image_ids):
        raise ValueError('Two folds require at least two unique images.')
    shuffled = np.random.RandomState(seed).permutation(sorted(image_ids)).tolist()
    return [sorted(shuffled[::2]), sorted(shuffled[1::2])]


def ap_summary(per_class):
    base = float(np.mean([per_class[c] for i, c in enumerate(CLASSES) if i not in NOVEL_IDS]))
    novel = float(np.mean([per_class[CLASSES[i]] for i in NOVEL_IDS]))
    return dict(base_ap50=base, novel_ap50=novel,
                hm=2*base*novel/(base+novel) if base+novel else 0., per_class_ap50=per_class)


def select_twofold(rows, summaries):
    """Choose only on the training fold; ties favor smaller external weight."""
    groups = {}
    controls = {}
    for row in rows:
        key = (row['ranking'], row['requested_k'])
        if row['model'] == 'yolo':
            controls[key] = row
        else:
            groups.setdefault((row['model'], row['multiplier']) + key, []).append(row)
    selections = []
    # Per-setting ablations plus a fully nested selection across the whole grid.
    jobs = [(str(key), group + [controls[key[2:]]]) for key, group in groups.items()]
    jobs.append(('all_settings_selected_on_training_fold', rows))
    for group_name, group in jobs:
        for train_fold in (0, 1):
            test_fold = 1 - train_fold
            chosen = min(group, key=lambda r: (
                -summaries[r['variant']]['folds'][train_fold]['novel_ap50'],
                r['novel_lambda'], r['variant']))
            control = controls[(chosen['ranking'], chosen['requested_k'])]
            train = summaries[chosen['variant']]['folds'][train_fold]
            heldout = summaries[chosen['variant']]['folds'][test_fold]
            baseline = summaries[control['variant']]['folds'][test_fold]
            record = dict(group=group_name, train_fold=train_fold, heldout_fold=test_fold,
                          selected_variant=chosen['variant'], selected_lambda=chosen['novel_lambda'],
                          model=chosen['model'], multiplier=chosen['multiplier'],
                          ranking=chosen['ranking'], requested_k=chosen['requested_k'],
                          train_Novel_AP50=100*train['novel_ap50'],
                          Novel_AP50=100*heldout['novel_ap50'], Base_AP50=100*heldout['base_ap50'],
                          HM=100*heldout['hm'], control_Novel_AP50=100*baseline['novel_ap50'],
                          delta_vs_control_pp=100*(heldout['novel_ap50']-baseline['novel_ap50']))
            record.update({c: 100*heldout['per_class_ap50'][c] for c in CLASSES})
            record['mean_actual_k'] = chosen.get('mean_actual_k', chosen['requested_k'])
            record['cached_processing_ms_per_image'] = chosen.get('selection_fusion_nms_ms_per_image', 0.)
            selections.append(record)
    means = []
    for group_name, _ in jobs:
        pair = [r for r in selections if r['group'] == group_name]
        mean = dict(group=group_name, metric='unweighted mean of two held-out fold APs; not pooled AP')
        family = next((key for key in groups if str(key) == group_name), None)
        mean.update(model=family[0] if family else 'joint_selection',
                    multiplier=family[1] if family else 'selected',
                    ranking=family[2] if family else 'selected',
                    requested_k=family[3] if family else None)
        for key in ('Novel_AP50', 'Base_AP50', 'HM', 'control_Novel_AP50', 'delta_vs_control_pp',
                    'mean_actual_k', 'cached_processing_ms_per_image') + CLASSES:
            mean[key] = float(np.mean([r[key] for r in pair]))
        mean['both_folds_improve_and_airport_windmill_positive'] = all(
            r['delta_vs_control_pp'] > 0 and r['airport'] > 0 and r['windmill'] > 0 for r in pair)
        means.append(mean)
    return dict(selections=selections, means=means)


def write_csv(path, rows):
    with Path(path).open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_k_curves(out, means):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        log('Matplotlib unavailable; K curve data are in twofold_means.csv.')
        return
    for model in sorted({r['model'] for r in means} - {'joint_selection'}):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        groups = sorted({(r['multiplier'], r['ranking']) for r in means if r['model'] == model})
        for multiplier, ranking in groups:
            points = sorted((r for r in means if r['model'] == model and r['multiplier'] == multiplier
                             and r['ranking'] == ranking and r['requested_k'] is not None),
                            key=lambda r: r['requested_k'])
            if not points:
                continue
            for ax, metric in zip(axes, ['Novel_AP50', 'cached_processing_ms_per_image']):
                ax.plot([r['requested_k'] for r in points], [r[metric] for r in points],
                        marker='o', label=multiplier + ' / ' + ranking)
                ax.set_xlabel('Requested K (limited by cached candidate count)')
                ax.grid(alpha=.25)
        axes[0].set_ylabel('Mean held-out Novel AP50 (%)')
        axes[1].set_ylabel('Cached selection + fusion + NMS (ms/image)')
        axes[0].legend(fontsize=7)
        fig.suptitle(model + ': two-fold lambda selection; encoder forward time excluded')
        fig.tight_layout()
        fig.savefig(out / ('k_curve_' + model + '.png'), dpi=160)
        plt.close(fig)


def evaluate_fusion_cache(cache, models, lambdas, multipliers, output_dir=None,
                          ann_file=None, reference_novel=8.39, allow_partial=False,
                          cross_validate=False, candidate_ks=None, rankings=None, fold_seed=0,
                          max_per_img=1000):
    """No forwards: D_B postprocessing on the *retained* candidate universe.

    Original post-NMS predictions cannot reproduce dense D_B. Never merge them
    into this control: all variants use identical cached boxes for all classes.
    Predictions are spooled one variant at a time and evaluated one class at a
    time, bounding RAM independently of the number of sweep combinations.
    """
    from pycocotools.coco import COCO
    from tools.val.eval_dior_candidate_protection import evaluate_category

    started = time.perf_counter()
    if isinstance(max_per_img, bool) or not isinstance(max_per_img, int) or max_per_img < 1:
        raise ValueError('max_per_img must be a positive integer; AP remains eval@100.')
    cache = Path(cache).resolve()
    candidate_ks = list(dict.fromkeys(candidate_ks or [None]))
    rankings = list(dict.fromkeys(rankings or ['objectness']))
    if any(k is not None and (not isinstance(k, int) or k < 1) for k in candidate_ks):
        raise ValueError('Candidate K must be a positive integer.')
    if any(r not in ('objectness', 'yolo_max') for r in rankings):
        raise ValueError('Invalid candidate ranking.')
    models = list(dict.fromkeys(models))
    lambdas = list(dict.fromkeys(lambdas))
    multipliers = list(dict.fromkeys(multipliers))
    if not models or any(m not in FUSION_MODELS for m in models):
        raise ValueError('Select supported cached VLM scores.')
    if not lambdas or any(not np.isfinite(x) or not 0 <= x <= 1 for x in lambdas):
        raise ValueError('Novel lambdas must be finite and in [0, 1].')
    if not multipliers or any(m not in ('objectness', 'yolo_max', 'none') for m in multipliers):
        raise ValueError('Unknown VLM multiplier.')
    if not np.isfinite(reference_novel) or not 0 <= reference_novel <= 100:
        raise ValueError('Reference Novel AP50 must be a percentage in [0, 100].')
    manifest = read_manifest(cache)
    extraction = manifest['protocol']
    if extraction.get('classes') != list(CLASSES) or extraction.get('selection_uses_gt') is not False:
        raise ValueError('Cache must declare DIOR class order and GT-free candidate selection.')
    annotation = Path(ann_file or manifest['ann_file']).resolve()
    if digest(annotation) != extraction['annotation_sha256']:
        raise ValueError('Annotation file changed since extraction.')
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(annotation))
    name_ids = {c['name']: c['id'] for c in gt.cats.values()}
    if len(gt.cats) != len(CLASSES) or set(name_ids) != set(CLASSES):
        raise ValueError('Annotation class names differ.')
    image_ids = [entry['id'] for entry in manifest['images']]
    if len(set(image_ids)) != len(image_ids) or not set(image_ids).issubset(gt.imgs):
        raise ValueError('Duplicate or unknown image IDs in cache.')
    complete = set(image_ids) == set(gt.imgs)
    if not complete and not allow_partial:
        raise ValueError('Incomplete split cache; use --allow-partial for smoke tests only.')
    folds = two_folds(image_ids, fold_seed) if cross_validate else []
    fold_counts = []
    for ids in folds:
        id_set = set(ids)
        counts = {c: sum(1 for a in gt.anns.values() if a['image_id'] in id_set
                        and a['category_id'] == name_ids[c] and not a.get('iscrowd', 0)) for c in CLASSES}
        if any(counts[CLASSES[i]] == 0 for i in NOVEL_IDS):
            raise ValueError('A fold lacks Novel ground truth; use a larger cache. Do not tune seeds on AP.')
        fold_counts.append(counts)
    model_meta = {}
    for name in models:
        provenance = load_json(cache / name / 'protocol.json')
        if provenance.get('extraction') != extraction or provenance.get('model') != name:
            raise ValueError('VLM extraction protocol mismatch: ' + name)
        info = load_json(cache / name / 'model_info.json')
        scale = info['logit_scale']
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError('Invalid VLM logit scale: ' + name)
        model_meta[name] = dict(protocol=provenance, model_info=info)

    log('Checking complete candidate/cosine caches; no models will be loaded.')
    # Hash all input arrays once, and retain a compact identity of this run.
    cache_identity = hashlib.sha256()
    for entry in manifest['images']:
        source = cache / 'candidates' / '{}.npz'.format(entry['id'])
        source_hash = digest(source)
        cache_identity.update((str(entry['id']) + source_hash).encode())
        with np.load(source) as data:
            boxes, yolo, objects = data['boxes'], data['yolo_scores'], data['objectness']
            n = len(boxes)
            if (boxes.shape != (n, 4) or yolo.shape != (n, len(CLASSES))
                    or objects.shape != (n,) or not np.isfinite(boxes).all()
                    or not (boxes[:, 2:] > boxes[:, :2]).all()):
                raise ValueError('Invalid candidate shapes/boxes: ' + str(source))
            if any(not np.isfinite(a).all() or ((a < 0) | (a > 1)).any()
                   for a in (yolo, objects)):
                raise ValueError('Candidate probabilities must be finite in [0, 1].')
            for name in models:
                scored_path = cache / name / source.name
                with np.load(scored_path) as scored:
                    if str(scored['candidate_sha256']) != source_hash:
                        raise ValueError('Candidate/score cache mismatch: ' + str(scored_path))
                    if scored['cosine'].shape != yolo.shape or not np.isfinite(scored['cosine']).all():
                        raise ValueError('Invalid cosine matrix: ' + str(scored_path))
                cache_identity.update((name + digest(scored_path)).encode())

    out = Path(output_dir) if output_dir else cache / (
        'fusion_D_B_eval100_' + time.strftime('%Y%m%d_%H%M%S') + '_' + str(os.getpid()))
    out = out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Use a new or empty output directory: ' + str(out))
    out.mkdir(parents=True, exist_ok=True)
    settings = dict(score_thr=1e-4, nms_pre=2_147_483_647, nms_iou=.7, max_per_img=max_per_img)
    # Import/initialize CPU NMS before timing any of the configurations.
    postprocess(torch.tensor([[0., 0., 1., 1.]]), torch.ones(1), torch.zeros(1, dtype=torch.long), settings)
    protocol = dict(
        version=2, experiment='cached_candidate_fusion', cache_dir=str(cache),
        cache_sha256=cache_identity.hexdigest(), extraction=extraction, models=model_meta,
        ann_file=str(annotation), image_ids=image_ids, full_split=complete,
        postprocessing='D_B', settings=settings, eval_max_dets=[1, 10, 100],
        candidate_scope='retained O/semantic union; not the full dense D_B candidate set',
        formula='Novel: YOLO**(1-lambda) * (softmax(logit_scale*cosine)*multiplier)**lambda',
        base_scores='unmodified YOLO on the same cached candidates',
        lambdas=lambdas, multipliers=multipliers,
        reference_novel_ap50_percent=reference_novel,
        reference_verified=False, selection_uses_gt=False,
        evaluation_uses_gt=True, tuning_on_evaluation_split=not cross_validate,
        cross_validate=cross_validate, fold_seed=fold_seed, folds=folds,
        fold_class_counts=fold_counts, fold_method='sorted image IDs, seeded shuffle, alternating halves',
        split_limit='image-disjoint only; geographic/source-scene independence is not established',
        candidate_ks=candidate_ks, rankings=rankings,
        timing_scope='CPU cached replay and COCO evaluation; excludes detector/crop encoder forward',
        timing_environment=dict(platform=platform.platform(), cpu=platform.processor(),
                                torch_version=torch.__version__, cpu_threads=torch.get_num_threads()),
        preflight_seconds=time.perf_counter()-started)
    write_json(out / 'protocol.json', protocol)
    variants = []
    for ranking in rankings:
        for k in candidate_ks:
            prefix = '' if candidate_ks == [None] and len(rankings) == 1 else '{}_K{}_'.format(ranking, k or 'all')
            variants.append((prefix + 'yolo_candidates', None, 'none', 0., ranking, k, prefix))
            variants += [(prefix + '{}_{}_lambda{}'.format(m, mult, repr(float(lam))),
                          m, mult, lam, ranking, k, prefix)
                         for m in models for mult in multipliers for lam in lambdas if lam != 0]
    summaries, rows = {}, []
    log('Planned {} variants; max_per_img={}; timing includes no encoder forward.'.format(
        len(variants), max_per_img))
    for variant, name, multiplier, lam, ranking, k, prefix in variants:
        log('Evaluating ' + variant)
        count = 0
        actual_ks = []
        timing = dict(cache_read_s=0., selection_fusion_nms_s=0., prediction_write_s=0., coco_eval_s=0.)
        variant_started = time.perf_counter()
        # Temporary prediction files are removed after each variant's metrics.
        with tempfile.TemporaryDirectory(prefix='predictions_', dir=out) as temp:
            temp = Path(temp)
            with contextlib.ExitStack() as stack:
                writers = {c: stack.enter_context((temp / (str(name_ids[c]) + '.jsonl')).open(
                    'w', encoding='utf-8')) for c in CLASSES}
                for index, entry in enumerate(manifest['images']):
                    tick = time.perf_counter()
                    source = cache / 'candidates' / '{}.npz'.format(entry['id'])
                    with np.load(source) as arrays:
                        data = {key: arrays[key] for key in ('boxes', 'yolo_scores', 'objectness')}
                    cosine = None
                    if name is not None:
                        with np.load(cache / name / source.name) as scored:
                            cosine = scored['cosine']
                    timing['cache_read_s'] += time.perf_counter()-tick
                    tick = time.perf_counter()
                    keep = candidate_subset(data, ranking, k)
                    actual_ks.append(len(keep))
                    data = {key: value[keep] for key, value in data.items()}
                    matrix = score_matrix(data)
                    if name is not None:
                        vlm = score_matrix(data, cosine[keep],
                                           model_meta[name]['model_info']['logit_scale'], multiplier)
                        matrix = geometric_fusion(matrix, vlm, lam)
                    boxes, scores, labels = (v.numpy() for v in predictions(data, matrix, settings))
                    timing['selection_fusion_nms_s'] += time.perf_counter()-tick
                    tick = time.perf_counter()
                    count += len(scores)
                    boxes = boxes.copy()
                    boxes[:, 2:] -= boxes[:, :2]
                    for label in np.unique(labels):
                        selected = labels == label
                        detections = np.column_stack((boxes[selected], scores[selected])).tolist()
                        writers[CLASSES[int(label)]].write(json.dumps(
                            dict(image_id=entry['id'], detections=detections)) + '\n')
                    timing['prediction_write_s'] += time.perf_counter()-tick
                    if (index + 1) % 500 == 0:
                        log('{}: {}/{} images'.format(variant, index + 1, len(image_ids)))
            tick = time.perf_counter()
            per_class = {c: evaluate_category(gt, temp / (str(name_ids[c]) + '.jsonl'),
                                              image_ids, name_ids[c], [100])[100]['ap50']
                         for c in CLASSES}
            fold_metrics = [ap_summary({c: evaluate_category(
                gt, temp / (str(name_ids[c]) + '.jsonl'), ids, name_ids[c], [100])[100]['ap50']
                for c in CLASSES}) for ids in folds]
            timing['coco_eval_s'] = time.perf_counter()-tick
        timing['variant_wall_s'] = time.perf_counter()-variant_started
        timing['cached_replay_s'] = sum(timing[key] for key in
                                       ('cache_read_s', 'selection_fusion_nms_s', 'prediction_write_s'))
        timing['selection_fusion_nms_ms_per_image'] = 1000*timing['selection_fusion_nms_s']/len(image_ids)
        result = ap_summary(per_class)
        base, novel = result['base_ap50'], result['novel_ap50']
        result.update(predictions=count, images=len(image_ids), folds=fold_metrics, timing=timing)
        summaries[variant] = result
        write_json(out / (variant + '.json'), result)
        control = summaries[prefix + 'yolo_candidates']['novel_ap50']
        row = dict(variant=variant, model=name or 'yolo', multiplier=multiplier,
                   ranking=ranking, requested_k=k, max_per_img=max_per_img,
                   mean_actual_k=float(np.mean(actual_ks)),
                   min_actual_k=int(min(actual_ks)), max_actual_k=int(max(actual_ks)),
                   images_below_requested_k=sum(v < k for v in actual_ks) if k is not None else 0,
                   novel_lambda=float(lam), Base_AP50=100*base, Novel_AP50=100*novel,
                   HM=100*result['hm'], delta_vs_candidates_pp=100*(novel-control),
                   delta_vs_reference_pp=100*novel-reference_novel,
                   airport_windmill_positive=per_class['airport'] > 0 and per_class['windmill'] > 0,
                   predictions=count, images=len(image_ids), full_split=complete)
        row['passes_reference_and_nonzero'] = (
            row['Novel_AP50'] > reference_novel and row['airport_windmill_positive'])
        row.update({c: 100*per_class[c] for c in CLASSES})
        row.update(timing)
        rows.append(row)
        with (out / 'summary.csv').open('w', newline='', encoding='utf-8-sig') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        write_json(out / 'summary.json', dict(protocol=protocol, rows=rows, variants=summaries,
                                              completed=len(rows), planned=len(variants)))
        log('{}: Base={:.4f}, Novel={:.4f}, HM={:.4f}, airport={:.6f}, windmill={:.6f}'.format(
            variant, 100*base, 100*novel, 100*result['hm'],
            100*per_class['airport'], 100*per_class['windmill']))
        log('Cached replay {:.2f}s; COCO {:.2f}s; completed {}/{}.'.format(
            timing['cached_replay_s'], timing['coco_eval_s'], len(rows), len(variants)))
    stability = {}
    for name in models:
        for multiplier in multipliers:
            for ranking in rankings:
                for k in candidate_ks:
                    group = sorted((row for row in rows if row['model'] == name
                                    and row['multiplier'] == multiplier and row['ranking'] == ranking
                                    and row['requested_k'] == k), key=lambda row: row['novel_lambda'])
                    passing = [row['novel_lambda'] for row in group if row['passes_reference_and_nonzero']]
                    adjacent = any(group[i]['passes_reference_and_nonzero']
                                   and group[i + 1]['passes_reference_and_nonzero']
                                   for i in range(len(group) - 1))
                    stability['{}/{}/{}/K{}'.format(name, multiplier, ranking, k)] = dict(
                        passing_lambdas=passing, adjacent_passing_grid_points=adjacent,
                        interpretation='exploratory full-split grid only; use twofold results for selection')
    cv = select_twofold(rows, summaries) if cross_validate else None
    if cv:
        for row in cv['selections'] + cv['means']:
            row['max_per_img'] = max_per_img
        write_json(out / 'twofold.json', dict(protocol=protocol, **cv))
        write_csv(out / 'twofold_selections.csv', cv['selections'])
        write_csv(out / 'twofold_means.csv', cv['means'])
        plot_k_curves(out, cv['means'])
    timings = dict(preflight_s=protocol['preflight_seconds'], total_wall_s=time.perf_counter()-started,
                   scope=protocol['timing_scope'], environment=protocol['timing_environment'],
                   cached_replay_s=sum(row['cached_replay_s'] for row in rows),
                   coco_eval_s=sum(row['coco_eval_s'] for row in rows))
    write_json(out / 'timing.json', timings)
    write_csv(out / 'timing.csv', [{key: row[key] for key in (
        'variant', 'ranking', 'requested_k', 'max_per_img', 'mean_actual_k', 'images', 'cache_read_s',
        'selection_fusion_nms_s', 'prediction_write_s', 'cached_replay_s', 'coco_eval_s',
        'variant_wall_s', 'selection_fusion_nms_ms_per_image')} for row in rows])
    write_json(out / 'summary.json', dict(protocol=protocol, rows=rows, variants=summaries,
                                          completed=len(rows), planned=len(variants), stability=stability,
                                          twofold=cv, timing=timings))
    with (out / 'summary.txt').open('w', encoding='utf-8') as stream:
        stream.write(('FULL VAL' if complete else 'PARTIAL SPLIT / SMOKE') +
                     '; cached candidates; D_B postprocessing; AP50 eval@100; '
                     'all AP values are percentages.\n')
        stream.write('Global post-NMS max_per_img={}.\n'.format(max_per_img))
        stream.write('External reference {} is unverified; inspect the same-candidate YOLO control.\n'.format(reference_novel))
        if cv:
            stream.write('Selection: see twofold_means.csv and twofold_selections.csv. '
                         'Held-out fold means differ from full-val pooled AP. '
                         'Full-val scores below are exploratory, not selection criteria.\n')
        for row in rows:
            stream.write('{}: Novel={:.4f} Base={:.4f} HM={:.4f} airport={:.6f} '
                         'windmill={:.6f} pass={}\n'.format(
                             row['variant'], row['Novel_AP50'], row['Base_AP50'],
                             row['HM'], row['airport'], row['windmill'],
                             row['passes_reference_and_nonzero']))
        for key, value in stability.items():
            stream.write('{}: adjacent passing lambdas={}\n'.format(
                key, value['adjacent_passing_grid_points']))
    log('Fusion sweep saved to ' + str(out))
    log('Cached candidates only; the external reference is unverified. '
        'Validation sweep gains need independent confirmation.')
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    extract_parser = sub.add_parser('extract')
    extract_parser.add_argument('--checkpoint', required=True)
    extract_parser.add_argument('--cache-dir', required=True)
    extract_parser.add_argument('--device', default='cuda:0')
    extract_parser.add_argument('--max-images', type=int, default=0)
    extract_parser.add_argument('--o-candidates', type=int, default=300)
    extract_parser.add_argument('--semantic-candidates', type=int, default=100)
    extract_parser.add_argument('--expected-hm', type=float, default=None,
                                help='Expected original HM in percent; full validation only')
    extract_parser.add_argument('--hm-tolerance', type=float, default=.1,
                                help='Allowed difference in percentage points')
    score_parser = sub.add_parser('score')
    score_parser.add_argument('--cache-dir', required=True)
    score_parser.add_argument('--model', required=True, choices=['clip', 'remoteclip'])
    score_parser.add_argument('--device', default='cuda:0')
    score_parser.add_argument('--batch-size', type=int, default=8)
    score_parser.add_argument('--context', type=float, default=1.0)
    score_parser.add_argument('--clip-model', default='pretrained_models/clip-vit-large-patch14-336')
    score_parser.add_argument('--remote-checkpoint', default='pretrained_models/RemoteCLIP-ViT-L-14.pt')
    score_parser.add_argument('--class-texts', default='data/DIOR/classes_full_20cls.json')
    score_parser.add_argument('--template', default='{name}')
    eval_parser = sub.add_parser('evaluate')
    eval_parser.add_argument('--cache-dir', required=True)
    eval_parser.add_argument('--models', nargs='+', default=['yolo', 'clip', 'remoteclip'],
                             choices=['yolo', 'clip', 'remoteclip'])
    eval_parser.add_argument('--novel-scales', nargs='+', type=float, default=[1.0, .1, .01])
    fusion_parser = sub.add_parser('evaluate-fusion', aliases=['evaluate-fusion-cv'],
                                   help='Cached fusion; evaluate-fusion-cv adds two folds and K sweep')
    fusion_parser.add_argument('--cache-dir', required=True)
    fusion_parser.add_argument('--models', nargs='+', choices=FUSION_MODELS, default=['clip'])
    fusion_parser.add_argument('--lambdas', nargs='+', type=float, default=[FIXED_FUSION['novel_lambda']],
                               help='External weight for Novel; Base stays YOLO. Zero is the shared control.')
    fusion_parser.add_argument('--multipliers', nargs='+', choices=['objectness', 'yolo_max', 'none'],
                               default=[FIXED_FUSION['multiplier']])
    fusion_parser.add_argument('--output-dir', help='New/empty directory; defaults to timestamped cache subdirectory')
    fusion_parser.add_argument('--ann-file', help='Relocated annotation file; original SHA256 must match')
    fusion_parser.add_argument('--reference-novel', type=float, default=8.39,
                               help='Unverified external Novel AP50 reference in percent, for reporting only')
    fusion_parser.add_argument('--allow-partial', action='store_true', help='Allow incomplete split for smoke tests only')
    fusion_parser.add_argument('--cpu-threads', type=int, default=4)
    fusion_parser.add_argument('--candidate-ks', nargs='+', type=int, default=[FIXED_FUSION['k']],
                               help='Top K cached candidates; fixed default: 100')
    fusion_parser.add_argument('--rankings', nargs='+', choices=['objectness', 'yolo_max'],
                               default=[FIXED_FUSION['ranking']], help='Fixed default: yolo_max')
    fusion_parser.add_argument('--fold-seed', type=int, default=0)
    fusion_parser.add_argument('--max-per-img', type=int, default=1000,
                               help='Global post-NMS detection cap; does not change K or AP eval@100.')
    args = parser.parse_args()
    if args.command == 'extract':
        if args.max_images < 0 or args.o_candidates < 1 or args.semantic_candidates < 1:
            parser.error('Invalid image/proposal count')
        if args.expected_hm is not None and args.max_images:
            parser.error('--expected-hm requires full validation (--max-images 0)')
        extract(args)
    elif args.command == 'score':
        if args.batch_size < 1 or not 0 < args.context < float('inf'):
            parser.error('Invalid batch size or crop context')
        score(args)
    elif args.command in ('evaluate-fusion', 'evaluate-fusion-cv'):
        if args.cpu_threads < 1:
            parser.error('cpu-threads must be positive.')
        torch.set_num_threads(args.cpu_threads)
        cv = args.command == 'evaluate-fusion-cv'
        evaluate_fusion_cache(args.cache_dir, args.models, args.lambdas, args.multipliers,
                              args.output_dir, args.ann_file, args.reference_novel, args.allow_partial,
                              cross_validate=cv, candidate_ks=args.candidate_ks,
                              rankings=args.rankings,
                              fold_seed=args.fold_seed, max_per_img=args.max_per_img)
    else:
        if any(not 0 < x < float('inf') for x in args.novel_scales):
            parser.error('Novel scales must be positive and finite')
        evaluate_cache(Path(args.cache_dir), args.models, args.novel_scales)


if __name__ == '__main__':
    main()

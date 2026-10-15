"""Frozen DIOR inference: extract -> score (CLIP/RemoteCLIP) -> evaluate.

No GT boxes select proposals, crops, labels, or scores. GT is used only by
COCOeval. Per-image caches permit interruption and resume with identical settings.
"""
import argparse
import contextlib
import hashlib
import inspect
import io
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
CLASSES = ('airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge',
           'chimney', 'dam', 'Expressway-Service-area', 'Expressway-toll-station',
           'golffield', 'groundtrackfield', 'harbor', 'overpass', 'ship', 'stadium',
           'storagetank', 'tenniscourt', 'trainstation', 'vehicle', 'windmill')
NOVEL_IDS = (1, 3, 10, 19)


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


def nms_indices(boxes, scores, topk, limit, threshold):
    from mmcv.ops import nms
    if len(boxes) == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)
    order = scores.argsort(descending=True)[:topk]
    _, keep = nms(boxes[order].contiguous(), scores[order].contiguous(), threshold)
    return order[keep[:limit]]


def candidate_indices(boxes, scores, objectness, o_limit=300, semantic_limit=100):
    valid = torch.isfinite(boxes).all(1) & ((boxes[:, 2:] - boxes[:, :2]) > 0).all(1)
    indices = torch.where(valid)[0]
    b = boxes[indices]
    first = nms_indices(b, objectness[indices], 3000, o_limit, .7)
    second = nms_indices(b, scores[indices].max(1).values, 3000, semantic_limit, .7)
    return indices[torch.unique(torch.cat((first, second)), sorted=True)]


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
        raise ValueError('Use the original DIOR 2.17 checkpoint, not a distillation model.')
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


def score_matrix(data, cosine=None, logit_scale=1.0):
    if cosine is None:
        return torch.from_numpy(data['yolo_scores']).float()
    # All 20 classes compete. These are initial scores, not calibrated
    # foreground probabilities. Keep raw cosine values for later experiments.
    values = torch.from_numpy(cosine).float()
    return (values * logit_scale).softmax(-1) * torch.from_numpy(data['objectness']).float()[:, None]


def postprocess(boxes, scores, labels, manifest):
    from mmcv.ops import batched_nms
    keep = (scores > manifest['score_thr']) & torch.isfinite(scores)
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
    if not len(scores):
        return boxes, scores, labels
    order = scores.argsort(descending=True)
    if manifest['nms_pre'] > 0:
        order = order[:manifest['nms_pre']]
    boxes, scores, labels = boxes[order], scores[order], labels[order]
    _, keep = batched_nms(boxes.contiguous(), scores.contiguous(), labels,
                         dict(type='nms', iou_threshold=manifest['nms_iou']), class_agnostic=False)
    keep = keep[:manifest['max_per_img']]
    return boxes[keep], scores[keep], labels[keep]


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
    else:
        if any(not 0 < x < float('inf') for x in args.novel_scales):
            parser.error('Novel scales must be positive and finite')
        evaluate_cache(Path(args.cache_dir), args.models, args.novel_scales)


if __name__ == '__main__':
    main()

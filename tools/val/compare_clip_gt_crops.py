"""Compare full CLIP and RemoteCLIP on identical DIOR validation GT crops.

This measures 20-way crop classification, NOT detection AP. No training, NMS,
or detector checkpoint is involved. Uses local torch/transformers/Pillow only.
RemoteCLIP's original full .pt is REQUIRED (the text-only export is insufficient).
Official RemoteCLIP inference: https://github.com/ChenDelong1999/RemoteCLIP

Example (from repository root):
    python tools/val/compare_clip_gt_crops.py --device cuda:0
    python tools/val/compare_clip_gt_crops.py --max-per-class 5 --batch-size 4
"""

import argparse
import csv
import gc
import inspect
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, CLIPConfig, CLIPModel

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.convert_remoteclip_text import convert_text_state_dict

CLASSES = (
    'airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge',
    'chimney', 'dam', 'Expressway-Service-area', 'Expressway-toll-station',
    'golffield', 'groundtrackfield', 'harbor', 'overpass', 'ship', 'stadium',
    'storagetank', 'tenniscourt', 'trainstation', 'vehicle', 'windmill')
NOVEL = ('airport', 'basketballcourt', 'groundtrackfield', 'windmill')
MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)


def log(message):
    print(time.strftime('[%H:%M:%S] ') + message, flush=True)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False), encoding='utf-8')


def write_csv(path, rows, fields):
    with Path(path).open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_remoteclip(path, tokenizer):
    """Strict in-memory OpenCLIP -> HF conversion, including the visual tower.

    GELU matches the official create_model_and_transforms('ViT-L-14') recipe.
    QKV is split in Q/K/V order; both projection matrices are transposed.
    """
    options = dict(map_location='cpu')
    # PyTorch 1.11 forwards unknown keywords to pickle; newer versions expose
    # weights_only explicitly. The old-version fallback is for local weights.
    if 'weights_only' in inspect.signature(torch.load).parameters:
        options['weights_only'] = True
    source = torch.load(str(path), **options)
    if 'state_dict' in source:
        source = source['state_dict']
    if all(key.startswith('module.') for key in source):
        source = {key[7:]: value for key, value in source.items()}
    state = convert_text_state_dict(source)
    vision = {key[7:]: value for key, value in source.items()
              if key.startswith('visual.')}

    def move(old, new):
        state['vision_model.' + new] = vision.pop(old).float().contiguous()

    for old, new in (
            ('class_embedding', 'embeddings.class_embedding'),
            ('positional_embedding', 'embeddings.position_embedding.weight'),
            ('conv1.weight', 'embeddings.patch_embedding.weight'),
            ('ln_pre.weight', 'pre_layrnorm.weight'),
            ('ln_pre.bias', 'pre_layrnorm.bias'),
            ('ln_post.weight', 'post_layernorm.weight'),
            ('ln_post.bias', 'post_layernorm.bias')):
        move(old, new)
    state['visual_projection.weight'] = vision.pop('proj').t().contiguous()
    for index in range(24):
        old = f'transformer.resblocks.{index}.'
        new = f'encoder.layers.{index}.'
        for suffix in ('weight', 'bias'):
            qkv = vision.pop(old + 'attn.in_proj_' + suffix)
            if qkv.shape[0] != 3072:
                raise ValueError('Expected RemoteCLIP ViT-L/14 QKV width 3072.')
            for name, tensor in zip(('q', 'k', 'v'), qkv.chunk(3, dim=0)):
                state['vision_model.' + new + f'self_attn.{name}_proj.{suffix}'] = tensor.contiguous()
            for src, dst in (
                    ('attn.out_proj', 'self_attn.out_proj'),
                    ('ln_1', 'layer_norm1'), ('ln_2', 'layer_norm2'),
                    ('mlp.c_fc', 'mlp.fc1'), ('mlp.c_proj', 'mlp.fc2')):
                move(old + src + '.' + suffix, new + dst + '.' + suffix)
    if vision:
        raise ValueError(f'Unmapped visual weights: {list(vision)}')
    state['logit_scale'] = source['logit_scale']
    config = CLIPConfig(
        projection_dim=768,
        text_config=dict(
            vocab_size=49408, hidden_size=768, intermediate_size=3072,
            num_hidden_layers=12, num_attention_heads=12,
            max_position_embeddings=77, hidden_act='gelu',
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id),
        vision_config=dict(
            hidden_size=1024, intermediate_size=4096, num_hidden_layers=24,
            num_attention_heads=16, image_size=224, patch_size=14,
            hidden_act='gelu'))
    model = CLIPModel(config)
    model.load_state_dict(state, strict=True)
    return model


def crop_bounds(bbox, width, height, context):
    x, y, w, h = map(float, bbox)
    if not all(math.isfinite(v) for v in (x, y, w, h)) or w <= 0 or h <= 0:
        return None
    # Discard boxes outside the image before expanding for context.
    if min(width, x + w) <= max(0, x) or min(height, y + h) <= max(0, y):
        return None
    cx, cy = x + w / 2, y + h / 2
    bounds = (max(0, math.floor(cx - w * context / 2)),
              max(0, math.floor(cy - h * context / 2)),
              min(width, math.ceil(cx + w * context / 2)),
              min(height, math.ceil(cy + h * context / 2)))
    return bounds


def load_records(annotation_path, context=1.0, max_per_class=0, seed=0):
    data = json.loads(Path(annotation_path).read_text(encoding='utf-8'))
    categories = {c['id']: c['name'] for c in data['categories']}
    if len(categories) != 20 or set(categories.values()) != set(CLASSES):
        raise ValueError('Annotation categories must match the full DIOR 20 classes.')
    images = {im['id']: im for im in data['images']}
    records, skipped = [], Counter()
    seen = set()
    for ann in data['annotations']:
        if ann['id'] in seen:
            raise ValueError(f"Duplicate annotation id: {ann['id']}")
        seen.add(ann['id'])
        if ann.get('iscrowd', 0) or ann.get('ignore', 0):
            skipped['crowd_or_ignore'] += 1
            continue
        im = images[ann['image_id']]
        bounds = crop_bounds(ann['bbox'], im['width'], im['height'], context)
        if bounds is None:
            skipped['invalid_box'] += 1
            continue
        records.append(dict(
            annotation_id=ann['id'], image_id=im['id'], file_name=im['file_name'],
            width=im['width'], height=im['height'], bbox=ann['bbox'],
            crop_xyxy=bounds, label=CLASSES.index(categories[ann['category_id']])))
    if max_per_class:
        rng = np.random.RandomState(seed)
        subset = []
        for label in range(20):
            group = [r for r in records if r['label'] == label]
            indices = rng.permutation(len(group))[:max_per_class]
            subset.extend(group[int(i)] for i in indices)
        records = subset
    records.sort(key=lambda r: (r['image_id'], r['annotation_id']))
    if not records:
        raise ValueError('No valid GT crops to evaluate.')
    return records, dict(skipped)


class CropDataset(Dataset):
    def __init__(self, records, image_root, size):
        self.records, self.image_root, self.size = records, Path(image_root), size
        self.cached_path, self.cached_image = None, None

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        path = self.image_root / record['file_name']
        if path != self.cached_path:
            with Image.open(path) as image:
                rgb = image.convert('RGB')
            if rgb.size != (record['width'], record['height']):
                raise ValueError(f'Image/annotation dimensions differ: {path}')
            self.cached_path, self.cached_image = path, rgb
        crop = self.cached_image.crop(record['crop_xyxy'])
        # Preserve the entire box: square padding avoids the usual center-crop
        # removing the ends of bridges, runways, and elongated objects.
        side = max(crop.size)
        square = Image.new('RGB', (side, side), tuple(round(x * 255) for x in MEAN))
        square.paste(crop, ((side - crop.width) // 2, (side - crop.height) // 2))
        resample = getattr(Image, 'Resampling', Image).BICUBIC
        array = np.asarray(square.resize((self.size, self.size), resample),
                           dtype=np.float32) / 255.0
        array = (array - np.asarray(MEAN, dtype=np.float32)) / np.asarray(STD, dtype=np.float32)
        return torch.from_numpy(array.transpose(2, 0, 1).copy())


def build_prompts(class_text_path, templates):
    aliases = json.loads(Path(class_text_path).read_text(encoding='utf-8'))
    if not isinstance(aliases, list) or len(aliases) != 20:
        raise ValueError('Class text JSON must have 20 entries in CLASSES order.')
    prompts = []
    for entry in aliases:
        names = [entry] if isinstance(entry, str) else entry
        if not names or any(not isinstance(n, str) or not n.strip() for n in names):
            raise ValueError('Each class needs at least one non-empty text alias.')
        prompts.append([template.format(name=name) for name in names
                        for template in templates])
    return prompts


@torch.no_grad()
def evaluate(model, tokenizer, prompts, records, args, name):
    model.eval().requires_grad_(False).to(args.device)
    size = model.config.vision_config.image_size
    text_vectors = []
    for texts in prompts:
        tokens = tokenizer(texts, padding=True, return_tensors='pt')
        if tokens['input_ids'].shape[1] > model.config.text_config.max_position_embeddings:
            raise ValueError('Prompt exceeds model context length; shorten it.')
        tokens = {key: value.to(args.device) for key, value in tokens.items()}
        vectors = F.normalize(model.get_text_features(**tokens).float(), dim=-1)
        text_vectors.append(F.normalize(vectors.mean(dim=0), dim=-1))
    text_vectors = torch.stack(text_vectors)
    loader = DataLoader(CropDataset(records, args.image_root, size),
                        batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers,
                        pin_memory=str(args.device).startswith('cuda'))
    outputs, processed = [], 0
    log(f'{name}: {len(records)} crops; input={size}px; FP32; starting inference')
    start = time.monotonic()
    for batch_index, pixels in enumerate(loader):
        images = F.normalize(model.get_image_features(
            pixel_values=pixels.to(args.device, non_blocking=True)).float(), dim=-1)
        scores = images @ text_vectors.t()
        if not torch.isfinite(scores).all():
            raise ValueError(f'Nonfinite cosine scores in {name}, batch {batch_index}')
        outputs.append(scores.cpu().numpy())
        processed += len(pixels)
        if batch_index == 0 or (batch_index + 1) % args.log_interval == 0 or processed == len(records):
            elapsed = time.monotonic() - start
            eta = elapsed / processed * (len(records) - processed)
            log(f'{name}: {processed}/{len(records)}; elapsed={elapsed:.0f}s; ETA={eta:.0f}s')
    return np.concatenate(outputs), dict(image_size=size,
                                        hidden_act=model.config.vision_config.hidden_act)


def summarize(scores, labels):
    # Stable ordering makes ties deterministic, with canonical class index first.
    order = np.argsort(-scores, axis=1, kind='stable')
    predictions = order[:, 0]
    ranks = (order == labels[:, None]).argmax(axis=1) + 1
    matrix = np.zeros((20, 20), dtype=np.int64)
    np.add.at(matrix, (labels, predictions), 1)
    per_class = []
    for label, name in enumerate(CLASSES):
        mask = labels == label
        count = int(mask.sum())
        per_class.append(dict(
            class_name=name, group='novel' if name in NOVEL else 'base', count=count,
            top1_accuracy=float((ranks[mask] == 1).mean()) if count else None,
            top5_accuracy=float((ranks[mask] <= 5).mean()) if count else None,
            mean_true_rank=float(ranks[mask].mean()) if count else None))
    groups = {}
    for group in ('all', 'base', 'novel'):
        indices = [i for i, name in enumerate(CLASSES)
                   if group == 'all' or (name in NOVEL) == (group == 'novel')]
        mask = np.isin(labels, indices)
        supported = [per_class[i]['top1_accuracy'] for i in indices
                     if per_class[i]['count']]
        groups[group] = dict(
            count=int(mask.sum()), classes_with_samples=len(supported),
            micro_top1=float((ranks[mask] == 1).mean()) if mask.any() else None,
            macro_top1=float(np.mean(supported)) if supported else None,
            micro_top5=float((ranks[mask] <= 5).mean()) if mask.any() else None)
    novel_mask = np.isin(labels, [CLASSES.index(n) for n in NOVEL])
    pred_base = ~np.isin(predictions, [CLASSES.index(n) for n in NOVEL])
    groups['novel']['predicted_as_base_rate'] = (
        float(pred_base[novel_mask].mean()) if novel_mask.any() else None)
    return dict(groups=groups, per_class=per_class), matrix, order, ranks


def save_results(output, scores, records, protocol):
    labels = np.array([r['label'] for r in records])
    summary, details, class_rows, confusions = {}, {}, [], []
    for name, values in scores.items():
        result, matrix, order, ranks = summarize(values, labels)
        summary[name] = result
        details[name] = (order, ranks)
        class_rows.extend(dict(model=name, **r) for r in result['per_class'])
        write_csv(output / f'confusion_{name}.csv',
                  [dict(true_class=c, **dict(zip(CLASSES, map(int, matrix[i]))))
                   for i, c in enumerate(CLASSES)], ['true_class', *CLASSES])
        for i, true_name in enumerate(CLASSES):
            for j in np.argsort(-matrix[i], kind='stable'):
                if i != j and matrix[i, j]:
                    confusions.append(dict(
                        model=name, true_class=true_name, predicted_class=CLASSES[j],
                        count=int(matrix[i, j]),
                        fraction_of_true_class=float(matrix[i, j] / matrix[i].sum())))
        log(f'{name} group accuracies: {json.dumps(result["groups"])}')
        for novel in NOVEL:
            r = result['per_class'][CLASSES.index(novel)]
            top_errors = [x for x in confusions if x['model'] == name and x['true_class'] == novel][:3]
            log(f'{name} {novel}: n={r["count"]}, top1={r["top1_accuracy"]}; '
                f'confusions={[(x["predicted_class"], x["count"]) for x in top_errors]}')
    paired = {}
    clip_ok = details['clip'][1] == 1
    remote_ok = details['remoteclip'][1] == 1
    for group in ('all', 'base', 'novel'):
        mask = np.array([group == 'all' or (CLASSES[y] in NOVEL) == (group == 'novel') for y in labels])
        paired[group] = dict(
            both_correct=int((mask & clip_ok & remote_ok).sum()),
            clip_only_correct=int((mask & clip_ok & ~remote_ok).sum()),
            remoteclip_only_correct=int((mask & ~clip_ok & remote_ok).sum()),
            both_wrong=int((mask & ~clip_ok & ~remote_ok).sum()))
    instance_rows = []
    for i, record in enumerate(records):
        row = {k: record[k] for k in ('annotation_id', 'image_id', 'file_name')}
        row.update(true_class=CLASSES[labels[i]], bbox_xywh=json.dumps(record['bbox']),
                   crop_xyxy=json.dumps(record['crop_xyxy']))
        for name, values in scores.items():
            order, ranks = details[name]
            wrong = values[i].copy()
            wrong[labels[i]] = -np.inf
            row.update({
                name + '_prediction': CLASSES[order[i, 0]],
                name + '_true_rank': int(ranks[i]),
                name + '_true_cosine': float(values[i, labels[i]]),
                name + '_true_minus_best_wrong': float(values[i, labels[i]] - wrong.max()),
                name + '_top5': '|'.join(CLASSES[j] for j in order[i, :5])})
        instance_rows.append(row)
    write_csv(output / 'per_instance.csv', instance_rows, list(instance_rows[0]))
    write_csv(output / 'per_class.csv', class_rows, list(class_rows[0]))
    write_csv(output / 'top_confusions.csv', confusions,
              ['model', 'true_class', 'predicted_class', 'count', 'fraction_of_true_class'])
    np.savez_compressed(output / 'cosine_scores.npz', labels=labels,
                        annotation_ids=[r['annotation_id'] for r in records],
                        classes=np.array(CLASSES), **scores)
    write_json(output / 'summary.json', dict(protocol=protocol, models=summary, paired=paired))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ann-file', default='data/DIOR/Annotations2/val.json')
    parser.add_argument('--image-root', default='data/DIOR/JPEGImages')
    parser.add_argument('--class-texts', default='data/DIOR/classes_full_20cls.json',
                        help='20 text entries in canonical DIOR order.')
    parser.add_argument('--clip-model', default='pretrained_models/clip-vit-large-patch14-336')
    parser.add_argument('--remote-checkpoint', default='pretrained_models/RemoteCLIP-ViT-L-14.pt')
    parser.add_argument('--output-dir', default='work_dirs/clip_remoteclip_gt_crops')
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--context', type=float, default=1.0,
                        help='Expand bbox width/height by this factor (>=1).')
    parser.add_argument('--max-per-class', type=int, default=0,
                        help='0: all GTs; positive: seeded subset per class for a smoke test.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--template', action='append',
                        help='Repeatable prompt with {name}; default: bare class text.')
    parser.add_argument('--log-interval', type=int, default=20)
    args = parser.parse_args()
    if (args.batch_size < 1 or args.workers < 0 or args.max_per_class < 0
            or args.log_interval < 1 or not math.isfinite(args.context) or args.context < 1):
        parser.error('Invalid batch size, workers, sample cap, log interval or context.')
    args.template = args.template or ['{name}']
    if any('{name}' not in t for t in args.template):
        parser.error('Each --template must contain {name}.')
    return args


def main():
    args = parse_args()
    log('Checking local inputs (no model downloads).')
    for path in (args.ann_file, args.class_texts, args.remote_checkpoint):
        if not Path(path).is_file():
            raise FileNotFoundError(
                f'Missing file: {path}. Full RemoteCLIP .pt is needed for its visual branch; '
                'the converted text folder/detector initialization cannot replace it.')
    if not Path(args.clip_model).is_dir():
        raise FileNotFoundError(f'Missing local full CLIP directory: {args.clip_model}')
    records, skipped = load_records(args.ann_file, args.context, args.max_per_class, args.seed)
    for filename in {r['file_name'] for r in records}:
        if not (Path(args.image_root) / filename).is_file():
            raise FileNotFoundError(str(Path(args.image_root) / filename))
    prompts = build_prompts(args.class_texts, args.template)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    protocol = dict(
        arguments=vars(args), classes=list(CLASSES), novel_classes=list(NOVEL),
        prompts=dict(zip(CLASSES, prompts)), skipped=skipped, crops=len(records),
        crop_policy='clip bbox to image; mean-color square padding; bicubic resize',
        precision='float32', metric='20-way GT-crop classification, not detection AP',
        resolution_note='Each model uses its native resolution (default CLIP 336, RemoteCLIP 224).',
        pretraining_note='RemoteCLIP paper includes DIOR in pretraining; not a strict unseen-data claim.',
        versions=dict(torch=torch.__version__))
    write_json(output / 'protocol.json', protocol)
    log(f'Prepared {len(records)} identical crops for both models; skipped={skipped}')
    tokenizer = AutoTokenizer.from_pretrained(args.clip_model, local_files_only=True)
    scores = {}
    for name in ('clip', 'remoteclip'):
        log(f'Loading full {name} image + text weights...')
        if name == 'clip':
            model, loading = CLIPModel.from_pretrained(
                args.clip_model, local_files_only=True, output_loading_info=True)
            if loading['missing_keys'] or loading.get('mismatched_keys'):
                raise ValueError(f'CLIP needs complete image + text weights: {loading}')
        else:
            model = load_remoteclip(args.remote_checkpoint, tokenizer)
        scores[name], protocol[name] = evaluate(model, tokenizer, prompts, records, args, name)
        np.save(output / f'{name}_cosine.npy', scores[name])
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    save_results(output, scores, records, protocol)
    log(f'Done. Results: {output.resolve()}')


if __name__ == '__main__':
    main()

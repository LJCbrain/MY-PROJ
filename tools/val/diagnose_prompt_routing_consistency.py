"""Measure per-class prompt-routing consistency of the Prompt Bank.

Aggregate routing usage tells you whether the 10 prompts are load-balanced.
It does NOT tell you whether a given class routes to the *same* prompt across
images. A class whose prompt index scatters image-to-image effectively gets an
average of unrelated prompts, which is indistinguishable from having no prompt
bank at all -- and that failure mode is invisible in the training log.

This script runs inference with the full 20-class vocabulary and reports, per
class and per scale:

  consistency  max(hist) / sum(hist)   1.0 = always the same prompt
  entropy      normalized routing entropy over prompts (0 = deterministic)
  dominant     the most frequently selected prompt index

Base and Novel classes are summarised separately. If Novel consistency is much
lower than Base consistency, the Prompt Bank is not doing useful work for the
unseen classes.

Validation labels are never used; only the class *names* enter the model, which
is exactly what happens during normal evaluation.

Example
-------
python tools/val/diagnose_prompt_routing_consistency.py \
    configs/pretrain/DIORv2.17best.py \
    work_dirs/DIORv2.17/best_coco_bbox_mAP_50_HM_epoch_24.pth \
    --max-images 300 --output work_dirs/routing_consistency.json
"""

import argparse
import json
import sys
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


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max-images', type=int, default=300,
                        help='0 means the complete validation set.')
    parser.add_argument('--output', default='')
    parser.add_argument('--class-text-path', default='',
                        help='Override the class-name JSON used for ordering.')
    return parser.parse_args()


def load_class_names(cfg, override):
    """Recover the class-name ordering actually fed to the model."""
    path = override
    if not path:
        node = cfg.get('val_dataloader', {}).get('dataset', {})
        while isinstance(node, dict):
            if 'class_text_path' in node:
                path = node['class_text_path']
                break
            node = node.get('dataset', {})
    if not path:
        return list(DIOR_CLASSES)
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = ROOT / path
    if not resolved.exists():
        print(f'[warn] {resolved} not found, falling back to DIOR_CLASSES')
        return list(DIOR_CLASSES)
    with open(resolved, 'r', encoding='utf-8') as handle:
        raw = json.load(handle)
    names = []
    for item in raw:
        names.append(item[0] if isinstance(item, (list, tuple)) else item)
    return names


def find_prompt_bank(model):
    for name, module in model.named_modules():
        if name.endswith('prompt_bank'):
            return name, module
    raise RuntimeError(
        'No module named *prompt_bank found. Is the Prompt Bank enabled in '
        'this config?')


def summarise(counts, num_prompts):
    """counts: [num_classes, num_prompts] -> consistency, entropy, dominant."""
    total = counts.sum(axis=-1)
    safe = np.maximum(total, 1)[:, None]
    probability = counts / safe
    consistency = counts.max(axis=-1) / np.maximum(total, 1)
    with np.errstate(divide='ignore', invalid='ignore'):
        log_probability = np.where(probability > 0, np.log(probability), 0.0)
    entropy = -(probability * log_probability).sum(axis=-1)
    entropy = entropy / np.log(num_prompts) if num_prompts > 1 else entropy
    dominant = counts.argmax(axis=-1)
    return consistency, entropy, dominant, total


def main():
    args = parse_args()

    import torch
    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmengine.runner import Runner
    from mmengine.runner.checkpoint import load_checkpoint
    from mmyolo.registry import MODELS

    import yolo_world  # noqa: F401  (registers custom modules)

    cfg = Config.fromfile(args.config)
    init_default_scope(cfg.get('default_scope', 'mmyolo'))

    class_names = load_class_names(cfg, args.class_text_path)
    num_classes = len(class_names)
    novel_set = set(DIOR_NOVEL_CLASSES)
    print(f'[info] {num_classes} classes, '
          f'{sum(n in novel_set for n in class_names)} of them Novel')

    model = MODELS.build(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.eval().to(args.device)

    bank_name, prompt_bank = find_prompt_bank(model)
    num_prompts = int(prompt_bank.num_prompts)
    num_scales = int(prompt_bank.num_scales)
    print(f'[info] prompt bank at "{bank_name}": '
          f'{num_prompts} prompts x {num_scales} scales')

    dataloader = Runner.build_dataloader(cfg.val_dataloader)

    counts = np.zeros((num_scales, num_classes, num_prompts), dtype=np.int64)
    seen = 0
    for batch in dataloader:
        with torch.no_grad():
            model.test_step(batch)
        indices = prompt_bank.last_selected_indices
        if indices is None:
            raise RuntimeError(
                'last_selected_indices is None -- the prompt bank did not run '
                'during inference.')
        indices = indices.detach().cpu().numpy()  # [B, num_scales, N]
        if indices.shape[-1] != num_classes:
            raise RuntimeError(
                f'routing covers {indices.shape[-1]} classes but '
                f'{num_classes} class names were loaded; pass '
                f'--class-text-path explicitly.')
        batch_size = indices.shape[0]
        scale_grid, class_grid = np.meshgrid(
            np.arange(num_scales), np.arange(num_classes), indexing='ij')
        for b in range(batch_size):
            np.add.at(counts, (scale_grid, class_grid, indices[b]), 1)
        seen += batch_size
        if args.max_images and seen >= args.max_images:
            break
    print(f'[info] aggregated routing over {seen} images\n')

    report = {'checkpoint': args.checkpoint, 'images': seen,
              'num_prompts': num_prompts, 'scales': {}}
    base_means, novel_means = [], []

    for scale in range(num_scales):
        consistency, entropy, dominant, total = summarise(
            counts[scale], num_prompts)
        print(f'=== scale P{scale + 3} ===')
        print(f'{"class":<26}{"consist":>9}{"entropy":>9}'
              f'{"prompt":>8}   kind')
        order = np.argsort(-consistency)
        for index in order:
            kind = 'NOVEL' if class_names[index] in novel_set else 'base'
            print(f'{class_names[index]:<26}{consistency[index]:>9.3f}'
                  f'{entropy[index]:>9.3f}{dominant[index]:>8d}   {kind}')

        base_idx = [i for i, n in enumerate(class_names) if n not in novel_set]
        novel_idx = [i for i, n in enumerate(class_names) if n in novel_set]
        base_mean = float(consistency[base_idx].mean())
        novel_mean = float(consistency[novel_idx].mean())
        base_means.append(base_mean)
        novel_means.append(novel_mean)
        print(f'\n  base  mean consistency: {base_mean:.3f}')
        print(f'  NOVEL mean consistency: {novel_mean:.3f}')

        collisions = []
        for n in novel_idx:
            shared = [class_names[b] for b in base_idx
                      if dominant[b] == dominant[n]]
            if shared:
                collisions.append(
                    f'{class_names[n]} -> prompt {dominant[n]} shared with '
                    f'{", ".join(shared)}')
        if collisions:
            print('  novel/base prompt collisions:')
            for line in collisions:
                print(f'    {line}')
        print()

        report['scales'][f'P{scale + 3}'] = {
            'consistency': {class_names[i]: float(consistency[i])
                            for i in range(num_classes)},
            'entropy': {class_names[i]: float(entropy[i])
                        for i in range(num_classes)},
            'dominant_prompt': {class_names[i]: int(dominant[i])
                                for i in range(num_classes)},
            'base_mean_consistency': base_mean,
            'novel_mean_consistency': novel_mean,
        }

    overall_base = float(np.mean(base_means))
    overall_novel = float(np.mean(novel_means))
    report['base_mean_consistency'] = overall_base
    report['novel_mean_consistency'] = overall_novel

    print('=== verdict ===')
    print(f'base  mean consistency across scales: {overall_base:.3f}')
    print(f'NOVEL mean consistency across scales: {overall_novel:.3f}')
    ratio = overall_novel / overall_base if overall_base > 0 else 0.0
    print(f'novel / base ratio: {ratio:.3f}')
    if ratio < 0.7:
        print('-> Novel routing is markedly less consistent than Base. The '
              'Prompt Bank is effectively inactive for unseen classes.')
    elif overall_novel > 0.9 and overall_base > 0.9:
        print('-> Routing is near-deterministic per class for both groups. '
              'The bank behaves like a fixed per-class prompt table.')
    else:
        print('-> Novel routing is comparable to Base. The Prompt Bank is not '
              'the bottleneck; look at the text embeddings and the '
              'base/novel decision boundary instead.')

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, 'w', encoding='utf-8') as handle:
            json.dump(report, handle, indent=2)
        print(f'\n[info] written to {out}')


if __name__ == '__main__':
    main()

"""Preflight validator for the full LAE-1M training configuration.

It builds every configured source, exercising the strict annotation/category
checks, then runs representative training samples through RandomLoadText and
verifies the fixed 80-prompt contract.  LAE-80C is checked as the fixed
80-class validation/test vocabulary.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmyolo.registry import DATASETS

import yolo_world  # noqa: F401 - registers datasets and transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/pretrain/LAEv1.py')
    parser.add_argument('--samples-per-source', type=int, default=1)
    return parser.parse_args()


def sample_indices(length: int, count: int):
    if count <= 0:
        return []
    candidates = [0, length // 2, length - 1]
    indices = []
    for index in candidates:
        if index not in indices:
            indices.append(index)
        if len(indices) == count:
            return indices
    return indices


def assert_prompt_sample(dataset, index: int, expected_text_count: int) -> None:
    sample = dataset[index]
    data_sample = sample['data_samples']
    texts = data_sample.texts
    labels = data_sample.gt_instances.labels
    if len(texts) != expected_text_count:
        raise AssertionError(
            f'Expected {expected_text_count} prompts, got {len(texts)} at '
            f'{dataset.dataset.ann_file} index={index}.')
    if labels.numel() and (int(labels.min()) < 0
                           or int(labels.max()) >= expected_text_count):
        raise AssertionError(
            f'Prompt-remapped label range [{int(labels.min())}, '
            f'{int(labels.max())}] is invalid for {expected_text_count} '
            f'prompts at {dataset.dataset.ann_file} index={index}.')


def main() -> None:
    args = parse_args()
    if args.samples_per_source < 0:
        raise ValueError('--samples-per-source must be non-negative.')

    init_default_scope('mmyolo')
    cfg = Config.fromfile(args.config)
    if cfg.num_classes != 80 or cfg.num_training_classes != 80:
        raise AssertionError('LAE pretraining requires num_classes=80 and '
                             'num_training_classes=80.')

    anchor_enabled = bool(cfg.get('semantic_anchor_enabled', True))
    if anchor_enabled:
        anchor_path = Path(cfg.semantic_anchor_path)
        embedding_path = Path(cfg.semantic_anchor_embed_path)
        payload = torch.load(embedding_path, map_location='cpu')
        if tuple(payload.get('texts', ())) != tuple(cfg.semantic_anchor_texts):
            raise AssertionError('Semantic-anchor text JSON and embedding '
                                 'table do not match exactly.')
        embeddings = payload.get('embeddings')
        if (not torch.is_tensor(embeddings)
                or tuple(embeddings.shape) != (96, 768)):
            shape = (None if not torch.is_tensor(embeddings) else
                     tuple(embeddings.shape))
            raise AssertionError(
                'Expected a [96, 768] semantic-anchor embedding table, got '
                f'{shape}.')
        if not anchor_path.is_file():
            raise FileNotFoundError(anchor_path)
    elif (cfg.semantic_anchor_texts or
          cfg.semantic_anchor_embed_path is not None):
        raise AssertionError(
            'Disabled semantic anchors must not load texts or embeddings.')

    for source_cfg in cfg.train_dataloader.dataset.datasets:
        dataset = DATASETS.build(source_cfg)
        class_count = len(dataset.class_texts)
        if class_count != len(dataset.dataset.metainfo['classes']):
            raise AssertionError(
                f'Class/text mismatch in {dataset.dataset.ann_file}.')
        for index in sample_indices(len(dataset), args.samples_per_source):
            assert_prompt_sample(dataset, index, cfg.num_training_classes)
        print(f'OK train {dataset.dataset.ann_file}: '
              f'{len(dataset)} images, {class_count} local classes')

    val_dataset = DATASETS.build(cfg.val_dataloader.dataset)
    if len(val_dataset.class_texts) != 80:
        raise AssertionError('LAE-80C must expose exactly 80 class texts.')
    eval_sample = val_dataset[0]
    eval_texts = eval_sample['data_samples'].texts
    eval_labels = eval_sample['data_samples'].gt_instances.labels
    if len(eval_texts) != 80:
        raise AssertionError(
            f'LAE-80C inference sample has {len(eval_texts)} texts, not 80.')
    if eval_labels.numel() and (int(eval_labels.min()) < 0
                                or int(eval_labels.max()) >= 80):
        raise AssertionError('LAE-80C emitted an out-of-range local label.')
    print(f'OK eval {val_dataset.dataset.ann_file}: '
          f'{len(val_dataset)} images, 80 classes')


if __name__ == '__main__':
    main()

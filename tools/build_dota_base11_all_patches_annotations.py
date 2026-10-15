"""Create an all-patch Base-11 DOTA-v1.0 training annotation file.

The source file keeps every paper-protocol 800x800 crop with all 15 DOTA
classes.  This utility preserves every image record, removes the four Novel
class annotations, and remaps the remaining Base-11 category ids to 0..10.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List


BASE_CLASSES = (
    'plane', 'ship', 'storage-tank', 'baseball-diamond',
    'basketball-court', 'ground-track-field', 'harbor', 'bridge',
    'large-vehicle', 'small-vehicle', 'roundabout')
NOVEL_CLASSES = (
    'tennis-court', 'helicopter', 'soccer-ball-field', 'swimming-pool')
EXPECTED_PATCHES = 18430


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--source',
        default='data/DOTA800/annotations/train_full_15cls_hbb_all_patches.json',
        help='Full-15 annotation file containing every training patch.')
    parser.add_argument(
        '--output',
        default='data/DOTA800/annotations/train_base_11cls_hbb_all_patches.json',
        help='Destination Base-11 annotation file.')
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open('r', encoding='utf-8') as stream:
        payload = json.load(stream)
    for key in ('images', 'annotations', 'categories'):
        if key not in payload or not isinstance(payload[key], list):
            raise ValueError(f'{path}: missing list-valued {key!r}.')
    return payload


def build_payload(source: Dict[str, Any]) -> Dict[str, Any]:
    name_by_id = {category['id']: category['name']
                  for category in source['categories']}
    source_names = set(name_by_id.values())
    missing = set(BASE_CLASSES + NOVEL_CLASSES) - source_names
    if missing:
        raise ValueError(f'Source categories are missing: {sorted(missing)}')

    target_id_by_name = {name: index for index, name in enumerate(BASE_CLASSES)}
    target_annotations: List[Dict[str, Any]] = []
    for annotation in source['annotations']:
        class_name = name_by_id.get(annotation.get('category_id'))
        if class_name not in target_id_by_name:
            continue
        copied = dict(annotation)
        copied['id'] = len(target_annotations)
        copied['category_id'] = target_id_by_name[class_name]
        target_annotations.append(copied)

    images = [dict(image) for image in source['images']]
    if len(images) != EXPECTED_PATCHES:
        raise ValueError(
            f'Expected {EXPECTED_PATCHES} source patches, got {len(images)}.')

    return {
        'info': {
            **source.get('info', {}),
            'description': (
                'DOTA-v1.0 train Base-11: all 18,430 paper-protocol '
                '800x800 patches retained; Novel-4 annotations excluded.'),
            'image_policy': 'all_patches_base_only',
        },
        'licenses': source.get('licenses', []),
        'images': images,
        'annotations': target_annotations,
        'categories': [
            {'id': index, 'name': name, 'supercategory': 'object'}
            for index, name in enumerate(BASE_CLASSES)
        ],
    }


def validate(payload: Dict[str, Any]) -> None:
    if len(payload['images']) != EXPECTED_PATCHES:
        raise AssertionError('Output does not retain every patch.')
    category_names = [category['name'] for category in payload['categories']]
    if category_names != list(BASE_CLASSES):
        raise AssertionError('Output categories are not the expected Base-11 split.')
    valid_ids = set(range(len(BASE_CLASSES)))
    invalid = [ann for ann in payload['annotations']
               if ann['category_id'] not in valid_ids]
    if invalid:
        raise AssertionError(f'Found {len(invalid)} non-Base-11 annotations.')
    image_ids = {image['id'] for image in payload['images']}
    missing_images = [ann for ann in payload['annotations']
                      if ann['image_id'] not in image_ids]
    if missing_images:
        raise AssertionError('An annotation references an absent image.')


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.part')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(payload, stream, ensure_ascii=False, separators=(',', ':'))
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    source_path = Path(args.source)
    output_path = Path(args.output)
    source = load_json(source_path)
    payload = build_payload(source)
    validate(payload)
    write_json(output_path, payload)
    print(f'Wrote: {output_path}')
    print(f'Images: {len(payload["images"])}')
    print(f'Base-11 annotations: {len(payload["annotations"])}')
    print('Novel-4 annotations: 0')


if __name__ == '__main__':
    main()

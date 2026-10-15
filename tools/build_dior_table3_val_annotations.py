"""Create DIOR Novel-4 validation COCO JSON for Table III ZSD evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DIOR_NOVEL_CLASSES = (
    'airport', 'basketballcourt', 'groundtrackfield', 'windmill')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path,
                        default=Path('data/DIOR/Annotations2/val.json'))
    parser.add_argument('--output', type=Path,
                        default=Path('data/DIOR/Annotations2/val_novel_4cls.json'))
    parser.add_argument('--class-text-output', type=Path,
                        default=Path('data/DIOR/classes_novel_4cls.json'))
    args = parser.parse_args()

    source = json.loads(args.input.read_text(encoding='utf-8'))
    source_id_to_name = {
        category['id']: category['name'] for category in source['categories']}
    source_to_target_id = {
        source_id: target_id
        for target_id, name in enumerate(DIOR_NOVEL_CLASSES)
        for source_id, source_name in source_id_to_name.items()
        if source_name == name
    }
    if len(source_to_target_id) != len(DIOR_NOVEL_CLASSES):
        missing = [
            name for name in DIOR_NOVEL_CLASSES
            if name not in source_id_to_name.values()]
        raise ValueError(f'Missing DIOR Novel-4 categories: {missing}')

    annotations = []
    for annotation in source['annotations']:
        if annotation['category_id'] not in source_to_target_id:
            continue
        copied = annotation.copy()
        copied['id'] = len(annotations)
        copied['category_id'] = source_to_target_id[annotation['category_id']]
        annotations.append(copied)

    result = {
        'info': {
            'description': 'DIOR validation annotations for Table III ZSD '
                           '(Novel-4 prompts only).'},
        'licenses': source.get('licenses', []),
        'images': source['images'],
        'annotations': annotations,
        'categories': [
            {'id': index, 'name': name, 'supercategory': 'object'}
            for index, name in enumerate(DIOR_NOVEL_CLASSES)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                           encoding='utf-8')
    args.class_text_output.parent.mkdir(parents=True, exist_ok=True)
    args.class_text_output.write_text(
        json.dumps([[name] for name in DIOR_NOVEL_CLASSES],
                   ensure_ascii=False, indent=2),
        encoding='utf-8')
    print(
        f'Wrote {args.output}: images={len(result["images"])}, '
        f'Novel-4 annotations={len(annotations)}\n'
        f'Wrote {args.class_text_output}')


if __name__ == '__main__':
    main()

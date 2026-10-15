"""Build the DIOR image-disjoint Base-16 training annotation.

Every original training image containing a Novel-4 instance is excluded.  The
script writes only the fully supervised Base-16 JSON; it does not create
partial-label data, pseudo boxes, or any Stage-2 artefact.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path


BASE_CLASSES = (
    'airplane', 'baseballfield', 'bridge', 'chimney', 'dam',
    'Expressway-Service-area', 'Expressway-toll-station', 'golffield',
    'harbor', 'overpass', 'ship', 'stadium', 'storagetank', 'tenniscourt',
    'trainstation', 'vehicle')
NOVEL_CLASSES = ('airport', 'basketballcourt', 'groundtrackfield', 'windmill')

EXPECTED_IMAGE_COUNT = 4326
EXPECTED_ANNOTATION_COUNT = 28330


def _build_annotation(source: dict) -> dict:
    source_names = {cat['id']: cat['name'] for cat in source['categories']}
    base_to_id = {name: index for index, name in enumerate(BASE_CLASSES)}
    novel_image_ids = {
        ann['image_id'] for ann in source['annotations']
        if source_names[ann['category_id']] in NOVEL_CLASSES
    }
    images = [image for image in source['images']
              if image['id'] not in novel_image_ids]
    retained_image_ids = {image['id'] for image in images}

    annotations = []
    for ann in source['annotations']:
        class_name = source_names[ann['category_id']]
        if (ann['image_id'] not in retained_image_ids or
                class_name not in base_to_id):
            continue
        copied = ann.copy()
        copied['id'] = len(annotations)
        copied['category_id'] = base_to_id[class_name]
        annotations.append(copied)

    return dict(
        info=dict(description=(
            'DIOR train split with Base-16 labels. Images containing any '
            'Novel-4 object are excluded.')),
        licenses=source.get('licenses', []),
        images=images,
        annotations=annotations,
        categories=[
            dict(id=index, name=name, supercategory='object')
            for index, name in enumerate(BASE_CLASSES)
        ])


def _validate(source: dict, result: dict, image_root: Path) -> None:
    source_names = {cat['id']: cat['name'] for cat in source['categories']}
    source_annotations = defaultdict(list)
    for ann in source['annotations']:
        source_annotations[ann['image_id']].append(ann)
    novel_image_ids = {
        image_id for image_id, annotations in source_annotations.items()
        if any(source_names[ann['category_id']] in NOVEL_CLASSES
               for ann in annotations)
    }

    output_image_ids = {image['id'] for image in result['images']}
    source_image_ids = {image['id'] for image in source['images']}
    if output_image_ids != source_image_ids - novel_image_ids:
        raise ValueError('Retained images are not image-disjoint from Novel-4.')
    if len(result['images']) != EXPECTED_IMAGE_COUNT:
        raise ValueError(f'Expected {EXPECTED_IMAGE_COUNT} images, got '
                         f'{len(result["images"])}.')
    if len(result['annotations']) != EXPECTED_ANNOTATION_COUNT:
        raise ValueError(f'Expected {EXPECTED_ANNOTATION_COUNT} annotations, '
                         f'got {len(result["annotations"])}.')
    if [cat['name'] for cat in result['categories']] != list(BASE_CLASSES):
        raise ValueError('Categories must be the ordered Base-16 class list.')

    category_names = {cat['id']: cat['name'] for cat in result['categories']}
    annotated_ids = set()
    for ann in result['annotations']:
        if ann['image_id'] not in output_image_ids:
            raise ValueError('An annotation references an excluded image.')
        if category_names.get(ann['category_id']) not in BASE_CLASSES:
            raise ValueError('An annotation is not a Base-16 label.')
        annotated_ids.add(ann['image_id'])
    if annotated_ids != output_image_ids:
        raise ValueError('Every retained image must have a Base-16 annotation.')

    missing = [image['file_name'] for image in result['images']
               if not (image_root / image['file_name']).is_file()]
    if missing:
        raise FileNotFoundError(
            f'{len(missing)} images are missing under {image_root}: '
            f'{", ".join(missing[:10])}')


def main() -> None:
    annotation_root = (Path(__file__).resolve().parents[1] / 'data' / 'DIOR' /
                       'Annotations2')
    source_path = annotation_root / 'train.json'
    output_path = annotation_root / 'train_base_16cls_image_disjoint_yoloworld.json'
    image_root = annotation_root.parent / 'JPEGImages'

    source = json.loads(source_path.read_text(encoding='utf-8'))
    result = _build_annotation(source)
    _validate(source, result, image_root)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    counts = Counter(ann['category_id'] for ann in result['annotations'])
    print(
        f'Wrote {output_path}\n'
        f'images={len(result["images"])} '
        f'annotations={len(result["annotations"])} '
        f'categories={len(BASE_CLASSES)}\n'
        f'base_instance_counts={dict(sorted(counts.items()))}')


if __name__ == '__main__':
    main()

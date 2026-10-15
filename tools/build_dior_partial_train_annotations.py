"""Keep DIOR train images with valid Base GT, dropping all Novel annotations."""
import json
import math
from collections import Counter
from pathlib import Path

from build_dior_base16_train_annotations import BASE_CLASSES, NOVEL_CLASSES


def build_annotation(source):
    names = {c['id']: c['name'] for c in source['categories']}
    images = {im['id']: im for im in source['images']}
    novel_images = {a['image_id'] for a in source['annotations']
                    if names[a['category_id']] in NOVEL_CLASSES}
    annotations, retained, rejected = [], set(), []
    for ann in source['annotations']:
        name = names[ann['category_id']]
        if name not in BASE_CLASSES:
            continue
        im = images[ann['image_id']]
        x, y, w, h = ann['bbox']
        if (not all(math.isfinite(v) for v in (x, y, w, h)) or w <= 0 or h <= 0
                or min(x+w, im['width']) <= max(x, 0)
                or min(y+h, im['height']) <= max(y, 0)
                or ann.get('ignore', False) or ann.get('iscrowd', False)):
            rejected.append(dict(id=ann['id'], image_id=ann['image_id'], bbox=ann['bbox']))
            continue
        # Keep only detection fields, never hidden Novel geometry or labels.
        annotations.append(dict(id=len(annotations), image_id=ann['image_id'],
                                category_id=BASE_CLASSES.index(name), bbox=[x, y, w, h],
                                area=w*h, iscrowd=0))
        retained.add(ann['image_id'])
    result = dict(
        images=[dict(id=im['id'], file_name=im['file_name'], width=im['width'],
                     height=im['height'], partial_labels=im['id'] in novel_images)
                for im in source['images'] if im['id'] in retained],
        annotations=annotations,
        categories=[dict(id=i, name=n, supercategory='object') for i, n in enumerate(BASE_CLASSES)])
    stats = dict(images=len(retained), annotations=len(annotations),
                 mixed=sum(im['partial_labels'] for im in result['images']),
                 complete=sum(not im['partial_labels'] for im in result['images']),
                 excluded_images=len(images)-len(retained), invalid_base_annotations=rejected,
                 base_instances=dict(Counter(BASE_CLASSES[a['category_id']] for a in annotations)))
    return result, stats


def main():
    root = Path(__file__).resolve().parents[1] / 'data/DIOR'
    source = json.loads((root / 'Annotations2/train.json').read_text(encoding='utf-8'))
    result, stats = build_annotation(source)
    missing = [im['file_name'] for im in result['images']
               if not (root / 'JPEGImages' / im['file_name']).is_file()]
    if missing:
        raise FileNotFoundError(f'Missing {len(missing)} images: {missing[:5]}')
    expected = dict(images=4933, annotations=30219, mixed=607, complete=4326)
    stats['expected_differences'] = {k: dict(expected=v, actual=stats[k])
                                     for k, v in expected.items() if v != stats[k]}
    path = root / 'Annotations2/train_base_16cls_partial_yoloworld.json'
    path.write_text(json.dumps(result, ensure_ascii=False), encoding='utf-8')
    path.with_suffix('.report.json').write_text(json.dumps(stats, indent=2), encoding='utf-8')
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()

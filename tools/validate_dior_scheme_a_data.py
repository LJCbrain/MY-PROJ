"""Validate the DIOR image-disjoint Base-16 training annotation."""

import argparse
import json
from pathlib import Path


BASE_CLASSES = {
    'airplane', 'baseballfield', 'bridge', 'chimney', 'dam',
    'Expressway-Service-area', 'Expressway-toll-station', 'golffield',
    'harbor', 'overpass', 'ship', 'stadium', 'storagetank', 'tenniscourt',
    'trainstation', 'vehicle'}
NOVEL_CLASSES = {'airport', 'basketballcourt', 'groundtrackfield', 'windmill'}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='data/DIOR/Annotations2')
    parser.add_argument('--image-root', default='data/DIOR/JPEGImages')
    args = parser.parse_args()

    annotation_path = (Path(args.root) /
                       'train_base_16cls_image_disjoint_yoloworld.json')
    payload = json.loads(annotation_path.read_text(encoding='utf-8'))
    names = {cat['id']: cat['name'] for cat in payload['categories']}

    assert len(payload['images']) == 4326
    assert len(payload['annotations']) == 28330
    assert set(names.values()) == BASE_CLASSES
    assert all(names[ann['category_id']] not in NOVEL_CLASSES
               for ann in payload['annotations'])
    missing = [image['file_name'] for image in payload['images']
               if not (Path(args.image_root) / image['file_name']).is_file()]
    assert not missing, f'Missing image files: {missing[:10]}'
    print('image_disjoint_base16=OK images=4326 annotations=28330 categories=16')


if __name__ == '__main__':
    main()

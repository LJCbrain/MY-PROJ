"""Create a compact four-class DIOR test annotation for ZSD evaluation."""

import argparse
import json
from pathlib import Path


NOVEL_CLASSES = (
    'airport',
    'basketballcourt',
    'groundtrackfield',
    'windmill',
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input',
        default='data/DIOR/Annotations/test_yoloworld.json')
    parser.add_argument(
        '--output',
        default='data/DIOR/Annotations/test_novel_4cls_yoloworld.json')
    args = parser.parse_args()

    source = json.loads(Path(args.input).read_text(encoding='utf-8'))
    source_name_to_id = {
        category['name']: category['id'] for category in source['categories']
    }
    source_to_target_id = {
        source_name_to_id[name]: target_id
        for target_id, name in enumerate(NOVEL_CLASSES)
    }

    result = {
        key: source[key]
        for key in ('info', 'licenses') if key in source
    }
    result['images'] = source['images']
    result['categories'] = [
        dict(id=target_id, name=name, supercategory='')
        for target_id, name in enumerate(NOVEL_CLASSES)
    ]
    result['annotations'] = [
        {**annotation, 'category_id': source_to_target_id[annotation['category_id']]}
        for annotation in source['annotations']
        if annotation['category_id'] in source_to_target_id
    ]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, separators=(',', ':')),
        encoding='utf-8')
    print(f'Wrote {len(result["annotations"])} annotations to {output_path}')


if __name__ == '__main__':
    main()

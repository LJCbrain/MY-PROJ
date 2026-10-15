"""Build image-disjoint DOTA-v1.0 HBB COCO annotations for Table III.

The script never crops, copies, moves, or otherwise changes DOTA images.  It
only reads the official OBB/HBB txt annotations and writes COCO JSON files.
Novel-image filtering is decided from the original OBB ``labelTxt`` labels,
before HBB annotations are written, so an image containing any Novel-4 object
cannot enter the Base-11 training JSON.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image


# Table III upper block, i.e. the original DOTA-v1.0 split without "*".
DOTA_BASE_CLASSES = (
    'plane', 'ship', 'storage-tank', 'baseball-diamond',
    'basketball-court', 'ground-track-field', 'harbor', 'bridge',
    'large-vehicle', 'small-vehicle', 'roundabout')
DOTA_NOVEL_CLASSES = (
    'tennis-court', 'helicopter', 'soccer-ball-field', 'swimming-pool')
DOTA_FULL_CLASSES = DOTA_BASE_CLASSES + DOTA_NOVEL_CLASSES
IMAGE_SUFFIXES = frozenset({'.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff'})
# DOTA labels occasionally use the inclusive image boundary (one pixel beyond
# PIL's width/height). Larger deviations mean the image and label file do not
# describe the same scene and must not enter the training JSON.
MAX_BOUNDARY_OVERRUN = 5.0


@dataclass(frozen=True)
class DotaObject:
    points: tuple[float, float, float, float, float, float, float, float]
    class_name: str
    difficulty: int


def parse_dota_label(path: Path) -> list[DotaObject]:
    """Read either Task1 OBB or Task2 HBB DOTA per-image annotation txt."""
    objects: list[DotaObject] = []
    for line_number, raw_line in enumerate(
            path.read_text(encoding='utf-8').splitlines(), start=1):
        tokens = raw_line.split()
        # Task1 files start with imagesource/gsd headers. Task2 files do not.
        if len(tokens) < 10:
            continue
        try:
            points = tuple(float(value) for value in tokens[:8])
            difficulty = int(float(tokens[9]))
        except ValueError as exc:
            raise ValueError(
                f'Malformed DOTA label in {path}:{line_number}: {raw_line!r}') from exc
        if len(points) != 8:
            raise AssertionError('DOTA annotations must contain four points.')
        objects.append(
            DotaObject(points=points, class_name=tokens[8], difficulty=difficulty))
    return objects


def collect_labels(label_dir: Path) -> dict[str, list[DotaObject]]:
    if not label_dir.is_dir():
        raise FileNotFoundError(f'Missing DOTA label directory: {label_dir}')
    labels = {path.stem: parse_dota_label(path)
              for path in sorted(label_dir.glob('*.txt'))}
    if not labels:
        raise FileNotFoundError(f'No DOTA .txt labels found in {label_dir}')
    return labels


def collect_images(image_dir: Path) -> dict[str, Path]:
    if not image_dir.is_dir():
        raise FileNotFoundError(f'Missing DOTA image directory: {image_dir}')
    # The official archives can unpack either directly into ``images`` or
    # below it (for example ``val/images/part1/images/P0003.png``).  Preserve
    # that layout and record each image path relative to ``image_dir`` in the
    # generated COCO JSON, rather than moving files just to satisfy a loader.
    image_paths = [
        path for path in image_dir.rglob('*')
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    duplicate_stems = [
        stem for stem, count in Counter(path.stem for path in image_paths).items()
        if count > 1
    ]
    if duplicate_stems:
        raise ValueError(
            f'Duplicate DOTA image stems below {image_dir}: '
            f'{sorted(duplicate_stems)[:5]}.')
    images = {path.stem: path for path in image_paths}
    if not images:
        raise FileNotFoundError(
            f'No extracted DOTA images found in {image_dir}. Extract the '
            'official image zip files before building COCO JSON.')
    return images


def ensure_matching_stems(raw_labels: dict[str, list[DotaObject]],
                          hbb_labels: dict[str, list[DotaObject]],
                          images: dict[str, Path], split: str) -> None:
    raw_stems = set(raw_labels)
    hbb_stems = set(hbb_labels)
    image_stems = set(images)
    if raw_stems != hbb_stems:
        raise ValueError(
            f'{split}: Task1/Task2 label files differ; '
            f'only_task1={sorted(raw_stems - hbb_stems)[:5]}, '
            f'only_task2={sorted(hbb_stems - raw_stems)[:5]}.')
    if raw_stems != image_stems:
        raise ValueError(
            f'{split}: image and label counts/names differ; '
            f'only_labels={sorted(raw_stems - image_stems)[:5]}, '
            f'only_images={sorted(image_stems - raw_stems)[:5]}.')


def hbb_from_points(points: Iterable[float]) -> tuple[float, float, float, float]:
    values = tuple(points)
    xs = values[0::2]
    ys = values[1::2]
    return min(xs), min(ys), max(xs), max(ys)


def find_severely_mismatched_images(
        images: dict[str, Path], hbb_labels: dict[str, list[DotaObject]],
        stems: Iterable[str]) -> set[str]:
    """Find labels that cannot describe their paired image.

    A one-pixel overrun is accepted and clipped below because DOTA labels use
    inclusive boundary coordinates.  A large overrun instead signals a wrong
    image/label pair; retaining it would create invalid COCO boxes.
    """
    mismatched = set()
    for stem in stems:
        with Image.open(images[stem]) as image:
            width, height = image.size
        for obj in hbb_labels[stem]:
            xmin, ymin, xmax, ymax = hbb_from_points(obj.points)
            if (xmin < -MAX_BOUNDARY_OVERRUN or
                    ymin < -MAX_BOUNDARY_OVERRUN or
                    xmax > width + MAX_BOUNDARY_OVERRUN or
                    ymax > height + MAX_BOUNDARY_OVERRUN):
                mismatched.add(stem)
                break
    return mismatched


def build_coco(split: str,
               images: dict[str, Path],
               image_root: Path,
               hbb_labels: dict[str, list[DotaObject]],
               retained_stems: set[str],
               classes: tuple[str, ...],
               description: str) -> dict:
    class_to_id = {class_name: index for index, class_name in enumerate(classes)}
    image_records = []
    annotation_records = []

    for image_id, stem in enumerate(sorted(retained_stems)):
        image_path = images[stem]
        with Image.open(image_path) as image:
            width, height = image.size
        image_records.append({
            'id': image_id,
            'file_name': image_path.relative_to(image_root).as_posix(),
            'width': width,
            'height': height,
        })

        for obj in hbb_labels[stem]:
            if obj.class_name not in class_to_id:
                continue
            xmin, ymin, xmax, ymax = hbb_from_points(obj.points)
            xmin = max(0.0, min(xmin, float(width)))
            ymin = max(0.0, min(ymin, float(height)))
            xmax = max(xmin, min(xmax, float(width)))
            ymax = max(ymin, min(ymax, float(height)))
            box_width = xmax - xmin
            box_height = ymax - ymin
            if box_width <= 0.0 or box_height <= 0.0:
                raise ValueError(
                    f'{split}: non-positive HBB for {stem} ({obj.class_name}).')
            annotation_records.append({
                'id': len(annotation_records),
                'image_id': image_id,
                'category_id': class_to_id[obj.class_name],
                'bbox': [xmin, ymin, box_width, box_height],
                'area': box_width * box_height,
                # DOTA evaluation ignores difficulty=2 objects. Keeping it
                # as COCO crowd/ignore preserves this behaviour for both loss
                # construction and COCOeval.
                'iscrowd': int(obj.difficulty == 2),
                'segmentation': [],
            })

    return {
        'info': {'description': description},
        'licenses': [],
        'images': image_records,
        'annotations': annotation_records,
        'categories': [
            {'id': index, 'name': class_name, 'supercategory': 'object'}
            for index, class_name in enumerate(classes)
        ],
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding='utf-8')


def write_class_texts(root: Path) -> None:
    for name, classes in (
            ('classes_base_11cls.json', DOTA_BASE_CLASSES),
            ('classes_novel_4cls.json', DOTA_NOVEL_CLASSES),
            ('classes_full_15cls.json', DOTA_FULL_CLASSES)):
        write_json(root / name, [[class_name] for class_name in classes])


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Create DOTA-v1.0 HBB COCO JSONs without changing images.')
    parser.add_argument('--root', type=Path,
                        default=Path('data/DOTA-v1.0'))
    args = parser.parse_args()
    root = args.root

    train_raw = collect_labels(root / 'train/labelTxt-v1.0/labelTxt')
    train_hbb = collect_labels(
        root / 'train/labelTxt-v1.0/Train_Task2_gt/trainset_reclabelTxt')
    train_image_root = root / 'train/images'
    train_images = collect_images(train_image_root)
    ensure_matching_stems(train_raw, train_hbb, train_images, 'train')

    val_raw = collect_labels(root / 'val/labelTxt-v1.0/labelTxt')
    val_hbb = collect_labels(
        root / 'val/labelTxt-v1.0/Val_Task2_gt/valset_reclabelTxt')
    val_image_root = root / 'val/images'
    val_images = collect_images(val_image_root)
    ensure_matching_stems(val_raw, val_hbb, val_images, 'val')

    unknown_classes = {
        obj.class_name for labels in (train_raw, train_hbb, val_raw, val_hbb)
        for objects in labels.values() for obj in objects
        if obj.class_name not in DOTA_FULL_CLASSES
    }
    if unknown_classes:
        raise ValueError(
            f'This is not a compatible DOTA-v1.0 label set; unexpected '
            f'classes: {sorted(unknown_classes)}.')

    novel_train_stems = {
        stem for stem, objects in train_raw.items()
        if any(obj.class_name in DOTA_NOVEL_CLASSES for obj in objects)
    }
    retained_train_stems = set(train_raw) - novel_train_stems
    mismatched_train_stems = find_severely_mismatched_images(
        train_images, train_hbb, retained_train_stems)
    retained_train_stems -= mismatched_train_stems
    mismatched_val_stems = find_severely_mismatched_images(
        val_images, val_hbb, val_raw)
    if mismatched_val_stems:
        raise ValueError(
            'Validation image/label mismatches cannot be silently excluded: '
            f'{sorted(mismatched_val_stems)}.')
    output_root = root / 'annotations'
    train_payload = build_coco(
        split='train', images=train_images, image_root=train_image_root,
        hbb_labels=train_hbb,
        retained_stems=retained_train_stems, classes=DOTA_BASE_CLASSES,
        description=('DOTA-v1.0 Task2 HBB train annotations. Images containing '
                     'any Table-III Novel-4 object are excluded.'))
    val_full_payload = build_coco(
        split='val', images=val_images, image_root=val_image_root,
        hbb_labels=val_hbb,
        retained_stems=set(val_raw), classes=DOTA_FULL_CLASSES,
        description='DOTA-v1.0 Task2 HBB validation annotations, all 15 classes.')
    val_novel_payload = build_coco(
        split='val', images=val_images, image_root=val_image_root,
        hbb_labels=val_hbb,
        retained_stems=set(val_raw), classes=DOTA_NOVEL_CLASSES,
        description='DOTA-v1.0 Task2 HBB validation annotations, Novel-4 only.')

    outputs = {
        output_root / 'train_base_11cls_image_disjoint_hbb.json': train_payload,
        output_root / 'val_full_15cls_hbb.json': val_full_payload,
        output_root / 'val_novel_4cls_hbb.json': val_novel_payload,
    }
    for path, payload in outputs.items():
        write_json(path, payload)
    write_class_texts(root)

    base_counts = Counter(
        ann['category_id'] for ann in train_payload['annotations']
        if not ann['iscrowd'])
    print(
        'DOTA-v1.0 HBB JSON generation complete\n'
        f'  train source images: {len(train_raw)}\n'
        f'  train images removed for Novel-4: {len(novel_train_stems)}\n'
        f'  train images excluded for image/label mismatch: '
        f'{len(mismatched_train_stems)}'
        f' ({", ".join(sorted(mismatched_train_stems)) or "none"})\n'
        f'  train Base-11 images retained: {len(train_payload["images"])}\n'
        f'  train Base-11 annotations: {len(train_payload["annotations"])}\n'
        f'  val full images/annotations: {len(val_full_payload["images"])}'
        f'/{len(val_full_payload["annotations"])}\n'
        f'  val Novel-4 annotations: {len(val_novel_payload["annotations"])}\n'
        f'  non-ignore Base-11 instance counts: '
        f'{dict(sorted(base_counts.items()))}')


if __name__ == '__main__':
    main()

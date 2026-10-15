"""Build the 800x800 DOTA-v1.0 derivatives used by Table III.

The crop geometry follows the DescReg supplementary material inherited by
RT-OVAD:

* the number of windows on an axis is ``ceil(length / 800)``;
* first and last windows touch the two image boundaries;
* intermediate windows are evenly distributed;
* dimensions shorter than 800 are zero-padded on the right/bottom.

The paper writes ``length // 800 + 1``.  For dimensions exactly divisible by
800, its reported totals (18430 train and 6259 val) prove that the actual
implementation uses the mathematically equivalent ceiling rule without an
extra redundant window.

HBB handling is reproduced from the public ITC-OVAD processed annotations:
an object is assigned to every overlapping crop that contains its center,
the source HBB is expanded by one pixel on each side, then clipped to the
800x800 crop.  This prevents partial objects whose centers lie outside a crop
from becoming training/evaluation targets.

The source DOTA directory is read-only.  All generated files are written to
``data/DOTA800`` by default.  Existing valid patch images are reused so an
interrupted run can be resumed safely.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from PIL import Image


Image.MAX_IMAGE_PIXELS = None

PATCH_SIZE = 800
EXPECTED_PATCH_COUNTS = {'train': 18430, 'val': 6259}
IMAGE_SUFFIXES = frozenset({'.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff'})

DOTA_BASE_CLASSES = (
    'plane', 'ship', 'storage-tank', 'baseball-diamond',
    'basketball-court', 'ground-track-field', 'harbor', 'bridge',
    'large-vehicle', 'small-vehicle', 'roundabout')
DOTA_NOVEL_CLASSES = (
    'tennis-court', 'helicopter', 'soccer-ball-field', 'swimming-pool')
DOTA_FULL_CLASSES = DOTA_BASE_CLASSES + DOTA_NOVEL_CLASSES


@dataclass(frozen=True)
class DotaObject:
    class_name: str
    difficulty: int
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @property
    def center(self) -> tuple[float, float]:
        return ((self.xmin + self.xmax) / 2.0,
                (self.ymin + self.ymax) / 2.0)


@dataclass(frozen=True)
class SourceJob:
    split: str
    image_path: str
    label_path: str
    output_image_dir: str
    write_images: bool
    overwrite: bool
    png_compress_level: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Crop DOTA-v1.0 to the Table-III 800x800 protocol and '
                    'write HBB COCO annotations.')
    parser.add_argument(
        '--source-root', type=Path, default=Path('data/DOTA-v1.0'))
    parser.add_argument(
        '--output-root', type=Path, default=Path('data/DOTA800'))
    parser.add_argument(
        '--workers', type=int, default=min(4, os.cpu_count() or 1),
        help='Parallel source-image workers. Four limits peak memory while '
             'still accelerating PNG encoding.')
    parser.add_argument(
        '--png-compress-level', type=int, default=3, choices=range(10),
        metavar='0-9')
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Build and validate metadata/counts without writing images/JSON.')
    parser.add_argument(
        '--overwrite', action='store_true',
        help='Regenerate existing patch PNGs. By default valid 800x800 files '
             'are reused and invalid files cause a hard failure.')
    parser.add_argument(
        '--validate-only', action='store_true',
        help='Deep-check an existing output directory without cropping.')
    return parser.parse_args()


def axis_starts(length: int, patch_size: int = PATCH_SIZE) -> tuple[int, ...]:
    """Return evenly distributed integer starts including both boundaries."""
    if length <= 0:
        raise ValueError(f'Image dimension must be positive, got {length}.')
    count = max(1, math.ceil(length / patch_size))
    if count == 1:
        return (0, )
    last_start = length - patch_size
    # Conventional half-up rounding is deterministic across Python versions.
    starts = tuple(
        math.floor(index * last_start / (count - 1) + 0.5)
        for index in range(count))
    if starts[0] != 0 or starts[-1] != last_start:
        raise AssertionError('Crop starts must touch both image boundaries.')
    if len(set(starts)) != len(starts):
        raise AssertionError(f'Duplicate crop starts for dimension {length}.')
    return starts


def parse_task2_hbb(path: Path) -> list[DotaObject]:
    objects: list[DotaObject] = []
    for line_number, raw_line in enumerate(
            path.read_text(encoding='utf-8').splitlines(), start=1):
        tokens = raw_line.split()
        if len(tokens) < 10:
            continue
        try:
            points = tuple(float(value) for value in tokens[:8])
            difficulty = int(float(tokens[9]))
        except ValueError as exc:
            raise ValueError(
                f'Malformed DOTA annotation {path}:{line_number}: '
                f'{raw_line!r}') from exc
        class_name = tokens[8]
        if class_name not in DOTA_FULL_CLASSES:
            raise ValueError(
                f'Unexpected DOTA-v1.0 class {class_name!r} in '
                f'{path}:{line_number}.')
        xs, ys = points[0::2], points[1::2]
        xmin, ymin, xmax, ymax = min(xs), min(ys), max(xs), max(ys)
        if xmax <= xmin or ymax <= ymin:
            raise ValueError(
                f'Non-positive source HBB in {path}:{line_number}.')
        objects.append(DotaObject(
            class_name=class_name,
            difficulty=difficulty,
            xmin=xmin,
            ymin=ymin,
            xmax=xmax,
            ymax=ymax))
    return objects


def collect_images(image_root: Path) -> dict[str, Path]:
    paths = sorted(
        path for path in image_root.rglob('*')
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        grouped[path.stem].append(path)
    duplicates = {stem: values for stem, values in grouped.items()
                  if len(values) != 1}
    if duplicates:
        raise ValueError(
            f'Duplicate image stems below {image_root}: '
            f'{list(duplicates)[:10]}.')
    return {stem: values[0] for stem, values in grouped.items()}


def locate_split(source_root: Path, split: str) -> tuple[dict[str, Path], Path]:
    image_root = source_root / split / 'images'
    task2_dir = source_root / split / 'labelTxt-v1.0'
    if split == 'train':
        task2_dir /= 'Train_Task2_gt/trainset_reclabelTxt'
    else:
        task2_dir /= 'Val_Task2_gt/valset_reclabelTxt'
    images = collect_images(image_root)
    labels = {path.stem: path for path in sorted(task2_dir.glob('*.txt'))}
    if set(images) != set(labels):
        raise ValueError(
            f'{split}: image/Task2 label stems differ; '
            f'only_images={sorted(set(images) - set(labels))[:10]}, '
            f'only_labels={sorted(set(labels) - set(images))[:10]}.')
    expected_sources = 1411 if split == 'train' else 458
    if len(images) != expected_sources:
        raise ValueError(
            f'{split}: expected {expected_sources} source images, found '
            f'{len(images)}.')
    return images, task2_dir


def crop_annotation(obj: DotaObject, crop_x: int, crop_y: int,
                    source_width: int, source_height: int) -> dict[str, Any] | None:
    center_x, center_y = obj.center
    if not (crop_x <= center_x < min(crop_x + PATCH_SIZE, source_width) and
            crop_y <= center_y < min(crop_y + PATCH_SIZE, source_height)):
        return None

    # Public ITC-OVAD processed labels use a one-pixel HBB expansion before
    # clipping. This also converts DOTA's inclusive integer corners cleanly.
    abs_xmin = max(0, math.floor(obj.xmin) - 1)
    abs_ymin = max(0, math.floor(obj.ymin) - 1)
    abs_xmax = min(source_width, math.ceil(obj.xmax) + 1)
    abs_ymax = min(source_height, math.ceil(obj.ymax) + 1)
    xmin = max(0, abs_xmin - crop_x)
    ymin = max(0, abs_ymin - crop_y)
    xmax = min(PATCH_SIZE, abs_xmax - crop_x)
    ymax = min(PATCH_SIZE, abs_ymax - crop_y)
    width, height = xmax - xmin, ymax - ymin
    if width <= 0 or height <= 0:
        return None
    return {
        'class_name': obj.class_name,
        'bbox': [int(xmin), int(ymin), int(width), int(height)],
        'area': int(width * height),
        'iscrowd': int(obj.difficulty == 2),
        'source_difficulty': obj.difficulty,
    }


def save_patch(source: Image.Image, output_path: Path,
               crop_x: int, crop_y: int, overwrite: bool,
               png_compress_level: int) -> str:
    if output_path.is_file() and not overwrite:
        try:
            with Image.open(output_path) as existing:
                if existing.size != (PATCH_SIZE, PATCH_SIZE):
                    raise ValueError(
                        f'Existing patch has wrong size {existing.size}: '
                        f'{output_path}')
        except OSError as exc:
            raise ValueError(
                f'Existing patch cannot be opened: {output_path}') from exc
        return 'reused'

    right = min(crop_x + PATCH_SIZE, source.width)
    bottom = min(crop_y + PATCH_SIZE, source.height)
    cropped = source.crop((crop_x, crop_y, right, bottom)).convert('RGB')
    if cropped.size == (PATCH_SIZE, PATCH_SIZE):
        output = cropped
    else:
        output = Image.new('RGB', (PATCH_SIZE, PATCH_SIZE), (0, 0, 0))
        output.paste(cropped, (0, 0))
    temporary = output_path.with_suffix(output_path.suffix + '.part')
    output.save(
        temporary, format='PNG', compress_level=png_compress_level,
        optimize=False)
    temporary.replace(output_path)
    return 'written'


def process_source(job: SourceJob) -> dict[str, Any]:
    image_path = Path(job.image_path)
    label_path = Path(job.label_path)
    output_image_dir = Path(job.output_image_dir)
    objects = parse_task2_hbb(label_path)

    with Image.open(image_path) as source:
        source_width, source_height = source.size
        x_starts = axis_starts(source_width)
        y_starts = axis_starts(source_height)
        if job.write_images:
            source.load()

        outside_objects = [
            obj for obj in objects
            if (obj.center[0] < 0 or obj.center[0] >= source_width or
                obj.center[1] < 0 or obj.center[1] >= source_height)
        ]
        partial_source_boxes = [
            obj for obj in objects
            if (obj.xmin < 0 or obj.ymin < 0 or
                obj.xmax > source_width or obj.ymax > source_height)
            and obj not in outside_objects
        ]

        patches = []
        write_counts = Counter()
        for crop_y in y_starts:
            for crop_x in x_starts:
                filename = f'{image_path.stem}__{crop_x}__{crop_y}.png'
                patch_annotations = []
                for obj in objects:
                    annotation = crop_annotation(
                        obj, crop_x, crop_y, source_width, source_height)
                    if annotation is not None:
                        patch_annotations.append(annotation)
                if job.write_images:
                    write_counts[save_patch(
                        source, output_image_dir / filename, crop_x, crop_y,
                        job.overwrite, job.png_compress_level)] += 1
                patches.append({
                    'file_name': filename,
                    'width': PATCH_SIZE,
                    'height': PATCH_SIZE,
                    'source_image': image_path.name,
                    'source_width': source_width,
                    'source_height': source_height,
                    'crop_x': crop_x,
                    'crop_y': crop_y,
                    'annotations': patch_annotations,
                })

    return {
        'source_stem': image_path.stem,
        'source_objects': len(objects),
        'outside_source_objects': len(outside_objects),
        'partial_source_boxes': len(partial_source_boxes),
        'patches': patches,
        'patches_written': write_counts['written'],
        'patches_reused': write_counts['reused'],
    }


def categories(classes: Sequence[str]) -> list[dict[str, Any]]:
    return [
        {'id': index, 'name': name, 'supercategory': 'object'}
        for index, name in enumerate(classes)
    ]


def annotation_payload(description: str,
                       images: Sequence[dict[str, Any]],
                       annotations: Sequence[dict[str, Any]],
                       classes: Sequence[str]) -> dict[str, Any]:
    clean_images = []
    for image in images:
        clean_images.append({key: value for key, value in image.items()
                             if key != 'annotations'})
    clean_annotations = []
    for annotation_id, annotation in enumerate(annotations):
        copied = {key: value for key, value in annotation.items()
                  if key not in ('class_name', 'source_difficulty')}
        copied['id'] = annotation_id
        clean_annotations.append(copied)
    return {
        'info': {
            'description': description,
            'crop_size': PATCH_SIZE,
            'crop_rule': 'ceil-count, boundary-aligned, evenly-distributed',
            'object_rule': 'center-in-crop, HBB expanded 1px then clipped',
        },
        'licenses': [],
        'images': clean_images,
        'annotations': clean_annotations,
        'categories': categories(classes),
    }


def assemble_split(split: str, source_results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    full_class_to_id = {name: index
                        for index, name in enumerate(DOTA_FULL_CLASSES)}
    images = []
    annotations = []
    source_has_novel: dict[str, bool] = {}

    for result in source_results:
        source_has_novel[result['source_stem']] = any(
            annotation['class_name'] in DOTA_NOVEL_CLASSES
            for patch in result['patches']
            for annotation in patch['annotations'])
        for patch in result['patches']:
            image_id = len(images)
            image_record = dict(patch)
            patch_annotations = image_record.pop('annotations')
            image_record['id'] = image_id
            images.append(image_record)
            for annotation in patch_annotations:
                copied = dict(annotation)
                copied['image_id'] = image_id
                copied['category_id'] = full_class_to_id[copied['class_name']]
                annotations.append(copied)

    expected = EXPECTED_PATCH_COUNTS[split]
    if len(images) != expected:
        raise ValueError(
            f'{split}: crop rule produced {len(images)} patches, expected '
            f'{expected}. Source data/version is incompatible with the paper.')
    return {
        'images': images,
        'annotations': annotations,
        'source_has_novel': source_has_novel,
        'source_results': source_results,
    }


def subset_payload(assembled: dict[str, Any], classes: Sequence[str],
                   description: str, image_policy: str) -> dict[str, Any]:
    target_id = {name: index for index, name in enumerate(classes)}
    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    full_name = {index: name for index, name in enumerate(DOTA_FULL_CLASSES)}
    all_classes_by_image: dict[int, set[str]] = defaultdict(set)
    for annotation in assembled['annotations']:
        class_name = full_name[annotation['category_id']]
        all_classes_by_image[annotation['image_id']].add(class_name)
        if class_name in target_id:
            copied = dict(annotation)
            copied['category_id'] = target_id[class_name]
            copied['class_name'] = class_name
            annotations_by_image[annotation['image_id']].append(copied)

    selected_images = []
    selected_annotations = []
    for image in assembled['images']:
        image_id = image['id']
        target_annotations = annotations_by_image.get(image_id, [])
        source_stem = Path(image['source_image']).stem
        has_novel_patch = bool(
            all_classes_by_image[image_id] & set(DOTA_NOVEL_CLASSES))
        if image_policy == 'all':
            keep = True
        elif image_policy == 'nonempty_target':
            keep = bool(target_annotations)
        elif image_policy == 'patch_disjoint_nonempty_base':
            keep = bool(target_annotations) and not has_novel_patch
        elif image_policy == 'source_disjoint_nonempty_base':
            keep = (bool(target_annotations) and
                    not assembled['source_has_novel'][source_stem])
        else:
            raise ValueError(f'Unknown image policy: {image_policy}')
        if keep:
            selected_images.append(image)
            selected_annotations.extend(target_annotations)

    return annotation_payload(
        description, selected_images, selected_annotations, classes)


def atomic_write_json(path: Path, payload: Any, pretty: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.part')
    with temporary.open('w', encoding='utf-8') as file:
        if pretty:
            json.dump(payload, file, ensure_ascii=False, indent=2)
        else:
            json.dump(payload, file, ensure_ascii=False,
                      separators=(',', ':'))
    temporary.replace(path)


def run_split(source_root: Path, output_root: Path, split: str,
              workers: int, write_images: bool, overwrite: bool,
              png_compress_level: int) -> dict[str, Any]:
    images, label_dir = locate_split(source_root, split)
    output_image_dir = output_root / split / 'images'
    if write_images:
        output_image_dir.mkdir(parents=True, exist_ok=True)
    jobs = [
        SourceJob(
            split=split,
            image_path=str(images[stem]),
            label_path=str(label_dir / f'{stem}.txt'),
            output_image_dir=str(output_image_dir),
            write_images=write_images,
            overwrite=overwrite,
            png_compress_level=png_compress_level)
        for stem in sorted(images)
    ]

    results = []
    if workers == 1:
        iterator: Iterable[dict[str, Any]] = map(process_source, jobs)
        executor = None
    else:
        executor = concurrent.futures.ProcessPoolExecutor(max_workers=workers)
        iterator = executor.map(process_source, jobs, chunksize=1)
    try:
        for index, result in enumerate(iterator, start=1):
            results.append(result)
            if index == 1 or index % 25 == 0 or index == len(jobs):
                patch_count = sum(len(item['patches']) for item in results)
                written = sum(item['patches_written'] for item in results)
                reused = sum(item['patches_reused'] for item in results)
                print(
                    f'{split}: source {index}/{len(jobs)}, '
                    f'patches={patch_count}, written={written}, '
                    f'reused={reused}', flush=True)
    finally:
        if executor is not None:
            # ``cancel_futures`` was added in Python 3.9. Some YOLO-World
            # environments still use Python 3.8, where passing it raises a
            # TypeError after all crop jobs have already finished.
            if sys.version_info >= (3, 9):
                executor.shutdown(wait=True, cancel_futures=True)
            else:
                executor.shutdown(wait=True)
    return assemble_split(split, results)


def write_split_jsons(output_root: Path, split: str,
                      assembled: dict[str, Any]) -> dict[str, dict[str, int]]:
    annotation_root = output_root / 'annotations'
    payloads: dict[str, dict[str, Any]] = {}
    payloads[f'{split}_full_15cls_hbb_all_patches.json'] = subset_payload(
        assembled, DOTA_FULL_CLASSES,
        f'DOTA-v1.0 {split}: all paper-protocol 800x800 patches, Full-15 HBB.',
        'all')
    payloads[f'{split}_full_15cls_hbb.json'] = subset_payload(
        assembled, DOTA_FULL_CLASSES,
        f'DOTA-v1.0 {split}: non-empty paper-protocol 800x800 patches, '
        'Full-15 HBB.', 'nonempty_target')

    if split == 'train':
        payloads['train_base_11cls_hbb.json'] = subset_payload(
            assembled, DOTA_BASE_CLASSES,
            'DOTA-v1.0 train Base-11: keep every crop with Base labels; '
            'Novel labels are excluded.', 'nonempty_target')
        payloads['train_base_11cls_patch_disjoint_hbb.json'] = subset_payload(
            assembled, DOTA_BASE_CLASSES,
            'DOTA-v1.0 train Base-11: exclude any 800x800 crop containing a '
            'Novel-4 object.', 'patch_disjoint_nonempty_base')
        payloads['train_base_11cls_source_image_disjoint_hbb.json'] = (
            subset_payload(
                assembled, DOTA_BASE_CLASSES,
                'DOTA-v1.0 train Base-11: exclude every crop from any source '
                'image containing a Novel-4 object.',
                'source_disjoint_nonempty_base'))
    else:
        payloads['val_novel_4cls_hbb.json'] = subset_payload(
            assembled, DOTA_NOVEL_CLASSES,
            'DOTA-v1.0 val ZSD Novel-4: only crops containing at least one '
            'Novel target; Base labels are excluded.', 'nonempty_target')

    summaries = {}
    for filename, payload in payloads.items():
        atomic_write_json(annotation_root / filename, payload)
        summaries[filename] = {
            'images': len(payload['images']),
            'annotations': len(payload['annotations']),
            'categories': len(payload['categories']),
        }
    return summaries


def validate_payloads(output_root: Path,
                      summaries: dict[str, dict[str, dict[str, int]]]) -> None:
    for split, expected in EXPECTED_PATCH_COUNTS.items():
        image_dir = output_root / split / 'images'
        image_count = sum(
            1 for path in image_dir.glob('*.png') if path.is_file())
        if image_count != expected:
            raise ValueError(
                f'{split}: expected {expected} PNG patches, found '
                f'{image_count} under {image_dir}.')

    required = {
        'train': (
            'train_full_15cls_hbb_all_patches.json',
            'train_full_15cls_hbb.json',
            'train_base_11cls_hbb.json',
            'train_base_11cls_patch_disjoint_hbb.json',
            'train_base_11cls_source_image_disjoint_hbb.json'),
        'val': (
            'val_full_15cls_hbb_all_patches.json',
            'val_full_15cls_hbb.json',
            'val_novel_4cls_hbb.json'),
    }
    for split, filenames in required.items():
        if set(summaries[split]) != set(filenames):
            raise AssertionError(f'{split}: missing expected JSON outputs.')
        all_count = summaries[split][filenames[0]]['images']
        if all_count != EXPECTED_PATCH_COUNTS[split]:
            raise AssertionError(
                f'{filenames[0]} must retain every generated crop.')


def deep_validate_output(output_root: Path) -> dict[str, Any]:
    """Decode every PNG and validate every COCO reference/bounding box."""
    annotation_root = output_root / 'annotations'
    expected_jsons = {
        'train': (
            'train_full_15cls_hbb_all_patches.json',
            'train_full_15cls_hbb.json',
            'train_base_11cls_hbb.json',
            'train_base_11cls_patch_disjoint_hbb.json',
            'train_base_11cls_source_image_disjoint_hbb.json'),
        'val': (
            'val_full_15cls_hbb_all_patches.json',
            'val_full_15cls_hbb.json',
            'val_novel_4cls_hbb.json'),
    }
    report: dict[str, Any] = {'images': {}, 'jsons': {}}
    png_names_by_split: dict[str, set[str]] = {}

    part_files = list(output_root.rglob('*.part'))
    if part_files:
        raise ValueError(
            f'Found {len(part_files)} unfinished .part files, for example '
            f'{part_files[0]}.')

    for split, expected_count in EXPECTED_PATCH_COUNTS.items():
        image_dir = output_root / split / 'images'
        paths = sorted(image_dir.glob('*.png'))
        if len(paths) != expected_count:
            raise ValueError(
                f'{split}: expected {expected_count} PNGs, got {len(paths)}.')
        names = set()
        total_bytes = 0
        for index, path in enumerate(paths, start=1):
            if path.name in names:
                raise ValueError(f'{split}: duplicate patch name {path.name}.')
            names.add(path.name)
            total_bytes += path.stat().st_size
            try:
                with Image.open(path) as image:
                    if image.size != (PATCH_SIZE, PATCH_SIZE):
                        raise ValueError(
                            f'{path}: expected 800x800, got {image.size}.')
                    image.verify()
            except OSError as exc:
                raise ValueError(f'Corrupt PNG: {path}') from exc
            if index % 5000 == 0 or index == len(paths):
                print(
                    f'validate images {split}: {index}/{len(paths)}',
                    flush=True)
        png_names_by_split[split] = names
        report['images'][split] = {
            'count': len(paths),
            'bytes': total_bytes,
            'all_decodable_800x800': True,
        }

    loaded: dict[str, dict[str, Any]] = {}
    for split, filenames in expected_jsons.items():
        for filename in filenames:
            path = annotation_root / filename
            try:
                data = json.loads(path.read_text(encoding='utf-8'))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f'Cannot read valid COCO JSON: {path}') from exc
            loaded[filename] = data
            for key in ('images', 'annotations', 'categories'):
                if not isinstance(data.get(key), list):
                    raise TypeError(f'{path}: {key} must be a list.')

            image_ids = [image.get('id') for image in data['images']]
            annotation_ids = [ann.get('id') for ann in data['annotations']]
            category_ids = [cat.get('id') for cat in data['categories']]
            if len(set(image_ids)) != len(image_ids):
                raise ValueError(f'{filename}: duplicate image ids.')
            if len(set(annotation_ids)) != len(annotation_ids):
                raise ValueError(f'{filename}: duplicate annotation ids.')
            if len(set(category_ids)) != len(category_ids):
                raise ValueError(f'{filename}: duplicate category ids.')
            if annotation_ids != list(range(len(annotation_ids))):
                raise ValueError(f'{filename}: annotation ids are not dense.')

            image_by_id = {image['id']: image for image in data['images']}
            valid_categories = set(category_ids)
            image_names = set()
            annotated_image_ids = set()
            for image in data['images']:
                if image.get('width') != PATCH_SIZE or image.get('height') != PATCH_SIZE:
                    raise ValueError(f'{filename}: non-800 image record.')
                file_name = image.get('file_name')
                if file_name not in png_names_by_split[split]:
                    raise ValueError(
                        f'{filename}: missing patch file {file_name!r}.')
                image_names.add(file_name)
            if len(image_names) != len(data['images']):
                raise ValueError(f'{filename}: duplicate image file names.')

            for ann in data['annotations']:
                if ann.get('image_id') not in image_by_id:
                    raise ValueError(f'{filename}: annotation has missing image.')
                if ann.get('category_id') not in valid_categories:
                    raise ValueError(f'{filename}: annotation has missing category.')
                bbox = ann.get('bbox')
                if (not isinstance(bbox, list) or len(bbox) != 4 or
                        not all(isinstance(value, (int, float)) and
                                math.isfinite(float(value)) for value in bbox)):
                    raise ValueError(f'{filename}: malformed bbox {bbox!r}.')
                x, y, width, height = map(float, bbox)
                if (x < 0 or y < 0 or width <= 0 or height <= 0 or
                        x + width > PATCH_SIZE or y + height > PATCH_SIZE):
                    raise ValueError(f'{filename}: invalid bbox {bbox!r}.')
                if abs(float(ann.get('area', -1)) - width * height) > 1e-6:
                    raise ValueError(f'{filename}: incorrect bbox area.')
                annotated_image_ids.add(ann['image_id'])

            if '_all_patches' in filename:
                if image_names != png_names_by_split[split]:
                    raise ValueError(
                        f'{filename}: all-patches JSON does not cover every PNG.')
            elif annotated_image_ids != set(image_ids):
                raise ValueError(
                    f'{filename}: nonempty subset contains an empty image.')
            report['jsons'][filename] = {
                'images': len(data['images']),
                'annotations': len(data['annotations']),
                'categories': len(data['categories']),
                'valid': True,
            }

    def annotation_signatures(data: dict[str, Any],
                              accepted: set[str]) -> set[tuple[Any, ...]]:
        image_name = {image['id']: image['file_name']
                      for image in data['images']}
        category_name = {cat['id']: cat['name']
                         for cat in data['categories']}
        return {
            (image_name[ann['image_id']], category_name[ann['category_id']],
             *tuple(ann['bbox']), ann.get('iscrowd', 0))
            for ann in data['annotations']
            if category_name[ann['category_id']] in accepted
        }

    train_full = loaded['train_full_15cls_hbb.json']
    train_base = loaded['train_base_11cls_hbb.json']
    if annotation_signatures(train_full, set(DOTA_BASE_CLASSES)) != (
            annotation_signatures(train_base, set(DOTA_BASE_CLASSES))):
        raise ValueError('Base-11 train JSON is not the exact Base subset.')
    val_full = loaded['val_full_15cls_hbb.json']
    val_novel = loaded['val_novel_4cls_hbb.json']
    if annotation_signatures(val_full, set(DOTA_NOVEL_CLASSES)) != (
            annotation_signatures(val_novel, set(DOTA_NOVEL_CLASSES))):
        raise ValueError('Novel-4 val JSON is not the exact Novel subset.')

    full_category = {cat['id']: cat['name'] for cat in train_full['categories']}
    train_image_name = {image['id']: image['file_name']
                        for image in train_full['images']}
    novel_files = {
        train_image_name[ann['image_id']]
        for ann in train_full['annotations']
        if full_category[ann['category_id']] in DOTA_NOVEL_CLASSES
    }
    patch_disjoint_files = {
        image['file_name']
        for image in loaded['train_base_11cls_patch_disjoint_hbb.json']['images']}
    if patch_disjoint_files & novel_files:
        raise ValueError('Patch-disjoint train JSON still contains Novel crops.')

    source_has_novel = {
        file_name.split('__', 1)[0] for file_name in novel_files}
    source_disjoint_files = {
        image['file_name'] for image in
        loaded['train_base_11cls_source_image_disjoint_hbb.json']['images']}
    if any(name.split('__', 1)[0] in source_has_novel
           for name in source_disjoint_files):
        raise ValueError(
            'Source-image-disjoint JSON still contains a Novel source image.')

    report['cross_file_checks'] = {
        'train_base_is_exact_full_subset': True,
        'val_novel_is_exact_full_subset': True,
        'patch_disjoint_has_no_novel_patch': True,
        'source_image_disjoint_has_no_novel_source': True,
        'unfinished_part_files': 0,
    }
    atomic_write_json(
        output_root / 'validation_report.json', report, pretty=True)
    return report


def main() -> None:
    args = parse_args()
    if args.workers <= 0:
        raise ValueError('--workers must be positive.')
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    if source_root == output_root:
        raise ValueError('Source and output roots must be different.')
    if args.validate_only:
        if not output_root.is_dir():
            raise FileNotFoundError(f'DOTA800 output root not found: {output_root}')
        report = deep_validate_output(output_root)
        print('\nDeep validation passed.\n' + json.dumps(
            report, ensure_ascii=False, indent=2), flush=True)
        return
    if not source_root.is_dir():
        raise FileNotFoundError(f'DOTA source root not found: {source_root}')

    print(
        f'Source: {source_root}\nOutput: {output_root}\n'
        f'Write images: {not args.dry_run}\nWorkers: {args.workers}',
        flush=True)
    assembled_by_split = {}
    for split in ('train', 'val'):
        assembled_by_split[split] = run_split(
            source_root, output_root, split, args.workers,
            not args.dry_run, args.overwrite, args.png_compress_level)

    manifest = {
        'source_root': str(source_root),
        'output_root': str(output_root),
        'patch_size': PATCH_SIZE,
        'crop_rule': 'ceil-count, boundary-aligned, evenly-distributed',
        'object_rule': 'center-in-crop, HBB expanded 1px then clipped',
        'expected_patch_counts': EXPECTED_PATCH_COUNTS,
        'splits': {},
    }
    for split, assembled in assembled_by_split.items():
        source_results = assembled['source_results']
        manifest['splits'][split] = {
            'source_images': len(source_results),
            'patches': len(assembled['images']),
            'full_annotations_with_overlap': len(assembled['annotations']),
            'source_objects': sum(item['source_objects']
                                  for item in source_results),
            'outside_source_objects_dropped': sum(
                item['outside_source_objects'] for item in source_results),
            'partial_source_boxes_clipped': sum(
                item['partial_source_boxes'] for item in source_results),
            'source_anomalies': [
                {
                    'source_stem': item['source_stem'],
                    'outside_source_objects': item['outside_source_objects'],
                    'partial_source_boxes': item['partial_source_boxes'],
                }
                for item in source_results
                if (item['outside_source_objects'] or
                    item['partial_source_boxes'])
            ],
        }

    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
        return

    output_root.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for split, assembled in assembled_by_split.items():
        summaries[split] = write_split_jsons(output_root, split, assembled)
        manifest['splits'][split]['json_outputs'] = summaries[split]

    for filename, classes in (
            ('classes_base_11cls.json', DOTA_BASE_CLASSES),
            ('classes_novel_4cls.json', DOTA_NOVEL_CLASSES),
            ('classes_full_15cls.json', DOTA_FULL_CLASSES)):
        atomic_write_json(
            output_root / filename, [[name] for name in classes], pretty=True)
    atomic_write_json(output_root / 'crop_manifest.json', manifest, pretty=True)
    validate_payloads(output_root, summaries)
    deep_validate_output(output_root)

    print('\nDOTA800 generation complete.', flush=True)
    print(json.dumps({
        split: manifest['splits'][split] for split in ('train', 'val')
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()

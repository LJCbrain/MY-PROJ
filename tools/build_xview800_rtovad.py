"""Build the RT-OVAD xView 800x800 training derivative.

RT-OVAD reports that the 846 annotated xView training images are split into
665 training and 181 evaluation source images, producing 12,825 and 3,550
800x800 patches.  The authors do not spell out the source-image identifiers
in the paper.  Their public ``xView_GZSD.json`` recovers exactly the first 181
numeric TIFF identifiers; the remaining 665 images are therefore used here.

Both sides of the split are cropped and saved: 12,825 training patches and
3,550 evaluation/test patches.  The output directory calls the latter ``val``
to match the public ITC-OVAD repository's ``xView/val`` convention.  COCO HBB
files are written for Full-60/Base-48 training and Full-60/Novel-12 evaluation.
The patch-disjoint Base-48 file removes every training crop containing at
least one Novel-12 object, matching this workspace's DOTA800 preparation.

The script is resumable: valid existing PNG patches are reused.  It supports
Python 3.8 and deliberately avoids ``cancel_futures``.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from PIL import Image


Image.MAX_IMAGE_PIXELS = None

PATCH_SIZE = 800
EXPECTED_SOURCE_COUNTS = {'all': 846, 'train': 665, 'val': 181}
EXPECTED_PATCH_COUNTS = {'train': 12825, 'val': 3550}
EXPECTED_SPLIT_HASHES = {
    'val': 'de253a6177800bd59a56388b3b269f830483fe73a62fc8ea034cce0a557f1688',
    'train': '3c47481a26b55a010b171a6f984956bae521e73ebe6ee2d87acb47452df6c024',
}
IMAGE_SUFFIXES = frozenset({'.tif', '.tiff'})

# Canonical category ordering in the authors' public xView_GZSD annotation:
# Base-48 first, followed by Novel-12.
XVIEW_BASE_CLASSES = (
    'building', 'small-car', 'truck',
    'truck-tractor-with-flatbed-trailer', 'construction-site', 'trailer',
    'utility-truck', 'cargo-truck', 'vehicle-lot', 'hut-or-tent',
    'damaged-building', 'container-crane', 'cargo-car', 'storage-tank',
    'facility', 'dump-truck', 'front-loader-or-bulldozer',
    'ground-grader', 'shed', 'passenger-vehicle', 'tower-crane', 'sailboat',
    'fishing-vessel', 'shipping-container', 'tower', 'cargo-plane',
    'aircraft-hangar', 'haul-truck', 'yacht', 'oil-tanker', 'ferry',
    'tugboat', 'truck-tractor', 'passenger-car', 'helipad', 'crane-truck',
    'engineering-vehicle', 'small-aircraft', 'pylon', 'container-ship',
    'locomotive', 'truck-tractor-with-liquid-tank', 'flat-car', 'tank-car',
    'straddle-carrier', 'cement-mixer', 'fixed-wing-aircraft',
    'railway-vehicle')

XVIEW_NOVEL_CLASSES = (
    'helicopter', 'bus', 'pickup-truck',
    'truck-tractor-with-box-trailer', 'maritime-vessel', 'motorboat',
    'barge', 'reach-stacker', 'mobile-crane', 'scraper-or-tractor',
    'excavator', 'shipping-container-lot')

XVIEW_FULL_CLASSES = XVIEW_BASE_CLASSES + XVIEW_NOVEL_CLASSES

# Official xView 60-class label IDs mapped to RT-OVAD's canonical names.
TYPE_ID_TO_CLASS = {
    11: 'fixed-wing-aircraft',
    12: 'small-aircraft',
    13: 'cargo-plane',
    15: 'helicopter',
    17: 'passenger-vehicle',
    18: 'small-car',
    19: 'bus',
    20: 'pickup-truck',
    21: 'utility-truck',
    23: 'truck',
    24: 'cargo-truck',
    25: 'truck-tractor-with-box-trailer',
    26: 'truck-tractor',
    27: 'trailer',
    28: 'truck-tractor-with-flatbed-trailer',
    29: 'truck-tractor-with-liquid-tank',
    32: 'crane-truck',
    33: 'railway-vehicle',
    34: 'passenger-car',
    35: 'cargo-car',
    36: 'flat-car',
    37: 'tank-car',
    38: 'locomotive',
    40: 'maritime-vessel',
    41: 'motorboat',
    42: 'sailboat',
    44: 'tugboat',
    45: 'barge',
    47: 'fishing-vessel',
    49: 'ferry',
    50: 'yacht',
    51: 'container-ship',
    52: 'oil-tanker',
    53: 'engineering-vehicle',
    54: 'tower-crane',
    55: 'container-crane',
    56: 'reach-stacker',
    57: 'straddle-carrier',
    59: 'mobile-crane',
    60: 'dump-truck',
    61: 'haul-truck',
    62: 'scraper-or-tractor',
    63: 'front-loader-or-bulldozer',
    64: 'excavator',
    65: 'cement-mixer',
    66: 'ground-grader',
    71: 'hut-or-tent',
    72: 'shed',
    73: 'building',
    74: 'aircraft-hangar',
    76: 'damaged-building',
    77: 'facility',
    79: 'construction-site',
    83: 'vehicle-lot',
    84: 'helipad',
    86: 'storage-tank',
    89: 'shipping-container-lot',
    91: 'shipping-container',
    93: 'pylon',
    94: 'tower',
}


@dataclass(frozen=True)
class XViewObject:
    class_name: str
    type_id: int
    feature_id: int
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.xmin + self.xmax) / 2.0,
                (self.ymin + self.ymax) / 2.0)


@dataclass(frozen=True)
class SourceJob:
    image_path: str
    objects: Tuple[XViewObject, ...]
    output_image_dir: str
    write_images: bool
    overwrite: bool
    png_compress_level: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Build the RT-OVAD xView800 training set and COCO HBBs.')
    parser.add_argument(
        '--source-root', type=Path, default=Path('data/xView'))
    parser.add_argument(
        '--output-root', type=Path, default=Path('data/xView800'))
    parser.add_argument(
        '--official-val-json', type=Path,
        default=Path('tmp/xView_GZSD_official.json'),
        help='Optional public RT-OVAD xView_GZSD.json used to verify the 181 '
             'source identifiers. Missing files are allowed because the '
             'verified split hashes are embedded.')
    parser.add_argument(
        '--workers', type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument(
        '--png-compress-level', type=int, default=3, choices=range(10),
        metavar='0-9')
    parser.add_argument(
        '--overwrite', action='store_true',
        help='Rewrite existing PNG patches instead of reusing valid files.')
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Build/check annotations and counts without writing files.')
    parser.add_argument(
        '--validate-only', action='store_true',
        help='Deep-check an already generated output directory.')
    return parser.parse_args()


def numeric_stem(path: Path) -> int:
    try:
        return int(path.stem)
    except ValueError as exc:
        raise ValueError(
            'xView source TIFF names must have numeric stems; got {}'.format(
                path.name)) from exc


def split_hash(stems: Sequence[str]) -> str:
    # The trailing newline makes the representation explicit and shell-friendly.
    return hashlib.sha256(
        ('\n'.join(stems) + '\n').encode('utf-8')).hexdigest()


def axis_starts(length: int, patch_size: int = PATCH_SIZE) -> Tuple[int, ...]:
    """Return the crop starts reproduced from RT-OVAD's public filenames."""
    if length <= 0:
        raise ValueError('Image dimension must be positive: {}'.format(length))
    if length <= patch_size:
        return (0, )
    # Equivalent to L//800+1 for non-divisible dimensions, but avoids a
    # redundant window when L is exactly divisible by 800.  The paper's
    # reported 12,825 training total requires this ceiling behaviour.
    count = int(math.ceil(float(length) / patch_size))
    step = int(math.ceil(float(length) / count))
    starts = [index * step for index in range(count - 1)]
    starts.append(length - patch_size)
    if len(starts) != count or starts[0] != 0:
        raise AssertionError('Invalid xView crop starts.')
    if any(start < 0 or start + patch_size > length for start in starts):
        raise AssertionError(
            'Out-of-bounds crop for dimension {}: {}'.format(length, starts))
    if len(set(starts)) != len(starts):
        raise AssertionError(
            'Duplicate crop start for dimension {}: {}'.format(length, starts))
    return tuple(starts)


def collect_and_split_images(source_root: Path) -> Dict[str, Any]:
    image_root = source_root / 'train_images'
    if not image_root.is_dir():
        raise FileNotFoundError('Missing xView train_images: {}'.format(image_root))
    paths = sorted(
        (path for path in image_root.iterdir()
         if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
        key=numeric_stem)
    stems = [path.stem for path in paths]
    if len(paths) != EXPECTED_SOURCE_COUNTS['all']:
        raise ValueError(
            'Expected {} annotated source TIFFs, found {}.'.format(
                EXPECTED_SOURCE_COUNTS['all'], len(paths)))
    if len(set(stems)) != len(stems):
        raise ValueError('Duplicate xView numeric TIFF stems found.')
    val_paths = paths[:EXPECTED_SOURCE_COUNTS['val']]
    train_paths = paths[EXPECTED_SOURCE_COUNTS['val']:]
    split_paths = {'val': val_paths, 'train': train_paths}
    for split, selected in split_paths.items():
        selected_stems = [path.stem for path in selected]
        actual_hash = split_hash(selected_stems)
        if actual_hash != EXPECTED_SPLIT_HASHES[split]:
            raise ValueError(
                '{} source split hash differs from the verified RT-OVAD '
                'split: {} != {}.'.format(
                    split, actual_hash, EXPECTED_SPLIT_HASHES[split]))
    return {
        'all': paths,
        'train': train_paths,
        'val': val_paths,
    }


def iter_geojson_features(path: Path, chunk_size: int = 4 * 1024 * 1024
                          ) -> Iterator[Dict[str, Any]]:
    """Stream a FeatureCollection without materializing the 372 MB JSON."""
    decoder = json.JSONDecoder()
    with path.open('r', encoding='utf-8') as file:
        buffer = ''
        eof = False
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                eof = True
            buffer += chunk
            marker = buffer.find('"features"')
            if marker >= 0:
                array_start = buffer.find('[', marker)
                if array_start >= 0:
                    buffer = buffer[array_start + 1:]
                    break
            if eof:
                raise ValueError('GeoJSON has no features array: {}'.format(path))
            # Keep enough overlap to locate a marker crossing chunk boundaries.
            if len(buffer) > 128:
                buffer = buffer[-128:]

        position = 0
        while True:
            while True:
                while (position < len(buffer) and
                       buffer[position] in ' \t\r\n,'):
                    position += 1
                if position < len(buffer):
                    break
                if eof:
                    raise ValueError('Unexpected EOF in features array.')
                buffer = ''
                position = 0
                chunk = file.read(chunk_size)
                if not chunk:
                    eof = True
                buffer += chunk

            if buffer[position] == ']':
                return
            try:
                feature, end = decoder.raw_decode(buffer, position)
            except json.JSONDecodeError as exc:
                if eof:
                    raise ValueError(
                        'Malformed/truncated GeoJSON near byte buffer '
                        'position {}.'.format(position)) from exc
                buffer = buffer[position:]
                position = 0
                chunk = file.read(chunk_size)
                if not chunk:
                    eof = True
                buffer += chunk
                continue
            if not isinstance(feature, dict):
                raise TypeError('GeoJSON feature must be an object.')
            yield feature
            position = end
            if position > chunk_size:
                buffer = buffer[position:]
                position = 0


def load_objects(source_root: Path, train_stems: Sequence[str],
                 val_stems: Sequence[str]
                 ) -> Tuple[Dict[str, Dict[str, Tuple[XViewObject, ...]]],
                            Dict[str, Any]]:
    label_path = source_root / 'train_labels' / 'xView_train.geojson'
    if not label_path.is_file():
        raise FileNotFoundError('Missing xView GeoJSON: {}'.format(label_path))
    train_set, val_set = set(train_stems), set(val_stems)
    known_stems = train_set | val_set
    objects: Dict[str, Dict[str, List[XViewObject]]] = {
        'train': defaultdict(list),
        'val': defaultdict(list),
    }
    stats = Counter()
    unknown_type_ids = Counter()

    for feature in iter_geojson_features(label_path):
        stats['features_total'] += 1
        properties = feature.get('properties') or {}
        image_name = str(properties.get('image_id', ''))
        image_stem = Path(image_name).stem
        if image_stem not in known_stems:
            stats['missing_source_image'] += 1
            continue
        split = 'train' if image_stem in train_set else 'val'
        stats['features_{}'.format(split)] += 1
        try:
            type_id = int(properties['type_id'])
        except (KeyError, TypeError, ValueError):
            stats['malformed_type_id'] += 1
            continue
        class_name = TYPE_ID_TO_CLASS.get(type_id)
        if class_name is None:
            unknown_type_ids[type_id] += 1
            stats['unsupported_type_id'] += 1
            continue
        bounds = str(properties.get('bounds_imcoords', '')).split(',')
        if len(bounds) != 4:
            stats['malformed_bounds'] += 1
            continue
        try:
            xmin, ymin, xmax, ymax = (float(value) for value in bounds)
        except ValueError:
            stats['malformed_bounds'] += 1
            continue
        if xmax <= xmin or ymax <= ymin:
            stats['nonpositive_source_bbox'] += 1
            continue
        stats['supported_positive_{}'.format(split)] += 1
        try:
            feature_id = int(properties.get('feature_id', -1))
        except (TypeError, ValueError):
            feature_id = -1
        objects[split][image_stem].append(XViewObject(
            class_name=class_name,
            type_id=type_id,
            feature_id=feature_id,
            xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax))

        if stats['features_total'] % 100000 == 0:
            print('labels: parsed {}/601937 features'.format(
                stats['features_total']), flush=True)

    frozen = {
        'train': {
            stem: tuple(objects['train'].get(stem, ()))
            for stem in train_stems
        },
        'val': {
            stem: tuple(objects['val'].get(stem, ()))
            for stem in val_stems
        },
    }
    report = dict(stats)
    report['unknown_type_ids'] = {
        str(key): value for key, value in sorted(unknown_type_ids.items())}
    report['train_sources_with_objects'] = sum(
        bool(value) for value in frozen['train'].values())
    report['val_sources_with_objects'] = sum(
        bool(value) for value in frozen['val'].values())
    return frozen, report


def verify_official_val_split(path: Path, val_stems: Sequence[str]) -> Dict[str, Any]:
    report = {
        'path': str(path),
        'available': path.is_file(),
        'verified': False,
    }
    if not path.is_file():
        report['note'] = (
            'Public JSON unavailable locally; embedded verified SHA-256 '
            'split hashes were enforced.')
        return report
    data = json.loads(path.read_text(encoding='utf-8'))
    official_stems = set()
    for image in data.get('images', []):
        name = Path(str(image.get('file_name', ''))).stem
        source_stem = name.split('__', 1)[0]
        if source_stem:
            official_stems.add(source_stem)
    expected = set(val_stems)
    if official_stems != expected:
        raise ValueError(
            'Official xView_GZSD source IDs differ: missing={}, extra={}'.format(
                sorted(expected - official_stems, key=int)[:10],
                sorted(official_stems - expected, key=int)[:10]))
    report.update({
        'verified': True,
        'official_nonempty_crop_records': len(data.get('images', [])),
        'official_unique_source_ids': len(official_stems),
    })
    return report


def apply_official_val_annotations(
        path: Path, assembled: Dict[str, Any]) -> Dict[str, Any]:
    """Replace derived val HBBs with the authors' released evaluation HBBs.

    The public file contains only non-empty patches.  Images are matched by
    their ``source__x__y`` stem, so its TIFF names map losslessly to the PNGs
    generated here.  The all-patches COCO file still retains all 3,550 image
    records, including the 1,508 background crops.
    """
    if not path.is_file():
        return {
            'applied': False,
            'reason': 'official xView_GZSD.json is unavailable locally',
            'derived_annotations': len(assembled['annotations']),
        }
    data = json.loads(path.read_text(encoding='utf-8'))
    category_names = tuple(cat['name'] for cat in data.get('categories', []))
    if category_names != XVIEW_FULL_CLASSES:
        raise ValueError('Official xView_GZSD category order is unexpected.')
    generated_by_stem = {
        Path(image['file_name']).stem: image for image in assembled['images']}
    official_image_by_id = {image['id']: image for image in data['images']}
    official_stems = {
        Path(image['file_name']).stem for image in data['images']}
    missing = official_stems - set(generated_by_stem)
    if missing:
        raise ValueError(
            'Official evaluation JSON refers to missing generated patches: '
            '{}'.format(sorted(missing)[:10]))

    official_annotations = []
    for annotation in data['annotations']:
        official_image = official_image_by_id[annotation['image_id']]
        stem = Path(official_image['file_name']).stem
        generated_image = generated_by_stem[stem]
        category_id = int(annotation['category_id'])
        bbox = [int(value) for value in annotation['bbox']]
        official_annotations.append({
            'image_id': generated_image['id'],
            'category_id': category_id,
            'class_name': XVIEW_FULL_CLASSES[category_id],
            'bbox': bbox,
            'area': int(bbox[2] * bbox[3]),
            'iscrowd': int(annotation.get('iscrowd', 0)),
        })
    derived_count = len(assembled['annotations'])
    assembled['annotations'] = official_annotations
    return {
        'applied': True,
        'official_nonempty_images': len(data['images']),
        'official_annotations': len(official_annotations),
        'locally_derived_annotations_before_replacement': derived_count,
        'local_minus_official_count': derived_count - len(official_annotations),
        'filename_mapping': 'official .tif stem -> generated .png stem',
    }


def crop_annotation(obj: XViewObject, crop_x: int, crop_y: int,
                    source_width: int, source_height: int
                    ) -> Optional[Dict[str, Any]]:
    crop_right = min(crop_x + PATCH_SIZE, source_width)
    crop_bottom = min(crop_y + PATCH_SIZE, source_height)
    center_x, center_y = obj.center
    # Public xView labels behave like a strict point-in-window test.
    if not (crop_x < center_x < crop_right and
            crop_y < center_y < crop_bottom):
        return None

    raw_width = obj.xmax - obj.xmin
    raw_height = obj.ymax - obj.ymin
    visible_width = max(
        0.0, min(obj.xmax, crop_right) - max(obj.xmin, crop_x))
    visible_height = max(
        0.0, min(obj.ymax, crop_bottom) - max(obj.ymin, crop_y))
    retained_ratio = ((visible_width * visible_height) /
                      (raw_width * raw_height))
    if retained_ratio <= 0.5:
        return None

    # The released RT-OVAD xView HBBs expand every source side by one pixel,
    # then clip to the crop and source image.
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
        'iscrowd': 0,
        'source_type_id': obj.type_id,
        'source_feature_id': obj.feature_id,
    }


def save_patch(source: Image.Image, output_path: Path, crop_x: int,
               crop_y: int, overwrite: bool, png_compress_level: int) -> str:
    if output_path.is_file() and not overwrite:
        try:
            with Image.open(output_path) as existing:
                if existing.size == (PATCH_SIZE, PATCH_SIZE):
                    existing.verify()
                    return 'reused'
        except (OSError, ValueError):
            pass

    right = min(crop_x + PATCH_SIZE, source.width)
    bottom = min(crop_y + PATCH_SIZE, source.height)
    cropped = source.crop((crop_x, crop_y, right, bottom)).convert('RGB')
    if cropped.size == (PATCH_SIZE, PATCH_SIZE):
        output = cropped
    else:
        output = Image.new('RGB', (PATCH_SIZE, PATCH_SIZE), (0, 0, 0))
        output.paste(cropped, (0, 0))
    temporary = output_path.with_suffix(output_path.suffix + '.part')
    output.save(temporary, format='PNG', compress_level=png_compress_level,
                optimize=False)
    temporary.replace(output_path)
    return 'written'


def process_source(job: SourceJob) -> Dict[str, Any]:
    image_path = Path(job.image_path)
    output_image_dir = Path(job.output_image_dir)
    with Image.open(image_path) as source:
        source_width, source_height = source.size
        x_starts = axis_starts(source_width)
        y_starts = axis_starts(source_height)
        if job.write_images:
            source.load()
        patches = []
        write_counts = Counter()
        for crop_y in y_starts:
            for crop_x in x_starts:
                filename = '{}__{}__{}.png'.format(
                    image_path.stem, crop_x, crop_y)
                patch_annotations = []
                for obj in job.objects:
                    annotation = crop_annotation(
                        obj, crop_x, crop_y, source_width, source_height)
                    if annotation is not None:
                        patch_annotations.append(annotation)
                if job.write_images:
                    status = save_patch(
                        source, output_image_dir / filename, crop_x, crop_y,
                        job.overwrite, job.png_compress_level)
                    write_counts[status] += 1
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
        'source_objects': len(job.objects),
        'patches': patches,
        'patches_written': write_counts['written'],
        'patches_reused': write_counts['reused'],
    }


def categories(classes: Sequence[str]) -> List[Dict[str, Any]]:
    return [
        {'id': index, 'name': name, 'supercategory': 'object'}
        for index, name in enumerate(classes)
    ]


def annotation_payload(description: str, images: Sequence[Dict[str, Any]],
                       annotations: Sequence[Dict[str, Any]],
                       classes: Sequence[str]) -> Dict[str, Any]:
    clean_images = [dict(image) for image in images]
    clean_annotations = []
    for annotation_id, annotation in enumerate(annotations):
        copied = {
            key: value for key, value in annotation.items()
            if key not in ('class_name', 'source_type_id', 'source_feature_id')
        }
        copied['id'] = annotation_id
        clean_annotations.append(copied)
    return {
        'info': {
            'description': description,
            'crop_size': PATCH_SIZE,
            'crop_rule': (
                'count=ceil(L/800); step=ceil(L/count); final window '
                'boundary-aligned'),
            'object_rule': (
                'strict center-in-crop; retained raw area > 0.5; '
                'HBB expanded 1px then clipped'),
            'source_split': (
                'numeric source TIFF order: first 181 evaluation, '
                'remaining 665 training; verified against public '
                'RT-OVAD xView_GZSD.json'),
        },
        'licenses': [],
        'images': clean_images,
        'annotations': clean_annotations,
        'categories': categories(classes),
    }


def assemble_results(split: str,
                     results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    full_class_to_id = {
        name: index for index, name in enumerate(XVIEW_FULL_CLASSES)}
    images = []
    annotations = []
    for result in results:
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
    if len(images) != EXPECTED_PATCH_COUNTS[split]:
        raise ValueError(
            'Crop geometry produced {} {} patches; expected {}.'.format(
                len(images), split, EXPECTED_PATCH_COUNTS[split]))
    return {'images': images, 'annotations': annotations}


def subset_payload(assembled: Dict[str, Any], classes: Sequence[str],
                   description: str, image_policy: str) -> Dict[str, Any]:
    target_id = {name: index for index, name in enumerate(classes)}
    full_name = {
        index: name for index, name in enumerate(XVIEW_FULL_CLASSES)}
    target_by_image = defaultdict(list)
    classes_by_image = defaultdict(set)
    for annotation in assembled['annotations']:
        class_name = full_name[annotation['category_id']]
        classes_by_image[annotation['image_id']].add(class_name)
        if class_name in target_id:
            copied = dict(annotation)
            copied['category_id'] = target_id[class_name]
            copied['class_name'] = class_name
            target_by_image[annotation['image_id']].append(copied)

    selected_images = []
    selected_annotations = []
    novel_set = set(XVIEW_NOVEL_CLASSES)
    for image in assembled['images']:
        image_id = image['id']
        target_annotations = target_by_image.get(image_id, [])
        has_novel = bool(classes_by_image[image_id] & novel_set)
        if image_policy == 'all':
            keep = True
        elif image_policy == 'nonempty_target':
            keep = bool(target_annotations)
        elif image_policy == 'patch_disjoint_nonempty_base':
            keep = bool(target_annotations) and not has_novel
        else:
            raise ValueError('Unknown image policy: {}'.format(image_policy))
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


def class_text(classes: Sequence[str]) -> List[List[str]]:
    return [[name] for name in classes]


def run_split_crops(split: str, paths: Sequence[Path],
                    objects: Dict[str, Tuple[XViewObject, ...]],
                    output_root: Path, workers: int, write_images: bool,
                    overwrite: bool,
                    png_compress_level: int) -> Dict[str, Any]:
    output_image_dir = output_root / split / 'images'
    if write_images:
        output_image_dir.mkdir(parents=True, exist_ok=True)
    jobs = [
        SourceJob(
            image_path=str(path),
            objects=objects[path.stem],
            output_image_dir=str(output_image_dir),
            write_images=write_images,
            overwrite=overwrite,
            png_compress_level=png_compress_level)
        for path in paths
    ]
    results = []
    if workers == 1:
        iterator = map(process_source, jobs)  # type: Iterable[Dict[str, Any]]
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
                    '{}: source {}/{}, patches={}, written={}, reused={}'.format(
                        split, index, len(jobs), patch_count, written, reused),
                    flush=True)
    finally:
        if executor is not None:
            # Compatible with the Python 3.8 environments used on the servers.
            executor.shutdown(wait=True)
    return assemble_results(split, results)


def write_outputs(output_root: Path,
                  assembled_by_split: Dict[str, Dict[str, Any]]
                  ) -> Dict[str, Dict[str, int]]:
    annotation_root = output_root / 'annotations'
    train_assembled = assembled_by_split['train']
    val_assembled = assembled_by_split['val']
    payloads = {
        'train_full_60cls_hbb_all_patches.json': subset_payload(
            train_assembled, XVIEW_FULL_CLASSES,
            'xView RT-OVAD train: all 12,825 800x800 patches, Full-60 HBB.',
            'all'),
        'train_full_60cls_hbb.json': subset_payload(
            train_assembled, XVIEW_FULL_CLASSES,
            'xView RT-OVAD train: non-empty 800x800 patches, Full-60 HBB.',
            'nonempty_target'),
        'train_base_48cls_hbb.json': subset_payload(
            train_assembled, XVIEW_BASE_CLASSES,
            'xView RT-OVAD train Base-48: keep crops with Base targets; '
            'Novel labels are omitted.',
            'nonempty_target'),
        'train_base_48cls_patch_disjoint_hbb.json': subset_payload(
            train_assembled, XVIEW_BASE_CLASSES,
            'xView RT-OVAD train Base-48: exclude every 800x800 crop '
            'containing any Novel-12 target.',
            'patch_disjoint_nonempty_base'),
        'val_full_60cls_hbb_all_patches.json': subset_payload(
            val_assembled, XVIEW_FULL_CLASSES,
            'xView RT-OVAD paper test/evaluation split: all 3,550 800x800 '
            'patches, Full-60 HBB.',
            'all'),
        'val_full_60cls_hbb.json': subset_payload(
            val_assembled, XVIEW_FULL_CLASSES,
            'xView RT-OVAD paper test/evaluation split: non-empty 800x800 '
            'patches, Full-60 HBB.',
            'nonempty_target'),
        'val_novel_12cls_hbb.json': subset_payload(
            val_assembled, XVIEW_NOVEL_CLASSES,
            'xView RT-OVAD paper ZSD test/evaluation split: non-empty '
            'Novel-12 targets only.',
            'nonempty_target'),
    }
    summaries = {}
    for filename, payload in payloads.items():
        atomic_write_json(annotation_root / filename, payload)
        summaries[filename] = {
            'images': len(payload['images']),
            'annotations': len(payload['annotations']),
            'categories': len(payload['categories']),
        }
    atomic_write_json(
        output_root / 'classes_base_48cls.json',
        class_text(XVIEW_BASE_CLASSES), pretty=True)
    atomic_write_json(
        output_root / 'classes_novel_12cls.json',
        class_text(XVIEW_NOVEL_CLASSES), pretty=True)
    atomic_write_json(
        output_root / 'classes_full_60cls.json',
        class_text(XVIEW_FULL_CLASSES), pretty=True)
    return summaries


def count_expected_patches(paths: Sequence[Path]) -> Tuple[int, Dict[str, Any]]:
    total = 0
    dimensions = Counter()
    for path in paths:
        with Image.open(path) as image:
            width, height = image.size
        count = len(axis_starts(width)) * len(axis_starts(height))
        total += count
        dimensions['source_images'] += 1
        dimensions['patches'] += count
    return total, dict(dimensions)


def deep_validate_output(output_root: Path) -> Dict[str, Any]:
    if list(output_root.rglob('*.part')):
        raise ValueError('Unfinished .part files remain below output root.')
    loaded = {}
    report = {'images': {}, 'jsons': {}}
    names_by_split = {}
    jsons_by_split = {
        'train': (
            'train_full_60cls_hbb_all_patches.json',
            'train_full_60cls_hbb.json',
            'train_base_48cls_hbb.json',
            'train_base_48cls_patch_disjoint_hbb.json'),
        'val': (
            'val_full_60cls_hbb_all_patches.json',
            'val_full_60cls_hbb.json',
            'val_novel_12cls_hbb.json'),
    }
    expected_classes = {
        'train_full_60cls_hbb_all_patches.json': XVIEW_FULL_CLASSES,
        'train_full_60cls_hbb.json': XVIEW_FULL_CLASSES,
        'train_base_48cls_hbb.json': XVIEW_BASE_CLASSES,
        'train_base_48cls_patch_disjoint_hbb.json': XVIEW_BASE_CLASSES,
        'val_full_60cls_hbb_all_patches.json': XVIEW_FULL_CLASSES,
        'val_full_60cls_hbb.json': XVIEW_FULL_CLASSES,
        'val_novel_12cls_hbb.json': XVIEW_NOVEL_CLASSES,
    }

    for split, expected_count in EXPECTED_PATCH_COUNTS.items():
        image_dir = output_root / split / 'images'
        paths = sorted(image_dir.glob('*.png'))
        if len(paths) != expected_count:
            raise ValueError(
                '{}: expected {} PNG patches, found {}.'.format(
                    split, expected_count, len(paths)))
        names = set()
        total_bytes = 0
        for index, path in enumerate(paths, start=1):
            if path.name in names:
                raise ValueError('Duplicate patch name: {}'.format(path.name))
            names.add(path.name)
            total_bytes += path.stat().st_size
            try:
                with Image.open(path) as image:
                    if image.size != (PATCH_SIZE, PATCH_SIZE):
                        raise ValueError(
                            '{} has size {}.'.format(path, image.size))
                    image.verify()
            except OSError as exc:
                raise ValueError('Corrupt PNG: {}'.format(path)) from exc
            if index % 2500 == 0 or index == len(paths):
                print('validate {} images: {}/{}'.format(
                    split, index, len(paths)), flush=True)
        names_by_split[split] = names
        report['images'][split] = {
            'count': len(paths),
            'bytes': total_bytes,
            'all_decodable_800x800': True,
        }

        for filename in jsons_by_split[split]:
            path = output_root / 'annotations' / filename
            data = json.loads(path.read_text(encoding='utf-8'))
            loaded[filename] = data
            category_names = tuple(
                cat['name'] for cat in data['categories'])
            if category_names != expected_classes[filename]:
                raise ValueError(
                    '{} has wrong category order.'.format(filename))
            image_by_id = {image['id']: image for image in data['images']}
            if len(image_by_id) != len(data['images']):
                raise ValueError('{} has duplicate image IDs.'.format(filename))
            category_ids = {cat['id'] for cat in data['categories']}
            annotated_ids = set()
            file_names = set()
            for image in data['images']:
                if (image['width'] != PATCH_SIZE or
                        image['height'] != PATCH_SIZE):
                    raise ValueError(
                        '{} has a non-800 image record.'.format(filename))
                if image['file_name'] not in names:
                    raise ValueError(
                        '{} refers to missing {}.'.format(
                            filename, image['file_name']))
                file_names.add(image['file_name'])
            if len(file_names) != len(data['images']):
                raise ValueError('{} has duplicate filenames.'.format(filename))
            annotation_ids = []
            for ann in data['annotations']:
                annotation_ids.append(ann['id'])
                if ann['image_id'] not in image_by_id:
                    raise ValueError(
                        '{} has orphan annotation.'.format(filename))
                if ann['category_id'] not in category_ids:
                    raise ValueError(
                        '{} has invalid category.'.format(filename))
                x, y, width, height = ann['bbox']
                if (x < 0 or y < 0 or width <= 0 or height <= 0 or
                        x + width > PATCH_SIZE or y + height > PATCH_SIZE):
                    raise ValueError(
                        '{} has invalid bbox {}.'.format(filename, ann['bbox']))
                if ann['area'] != width * height:
                    raise ValueError(
                        '{} has incorrect area.'.format(filename))
                annotated_ids.add(ann['image_id'])
            if annotation_ids != list(range(len(annotation_ids))):
                raise ValueError(
                    '{} annotation IDs are not dense.'.format(filename))
            if '_all_patches' in filename:
                if file_names != names:
                    raise ValueError(
                        '{} does not cover every {} PNG.'.format(
                            filename, split))
            elif annotated_ids != set(image_by_id):
                raise ValueError('{} contains empty images.'.format(filename))
            report['jsons'][filename] = {
                'images': len(data['images']),
                'annotations': len(data['annotations']),
                'categories': len(data['categories']),
                'valid': True,
            }

    full = loaded['train_full_60cls_hbb.json']
    full_image_name = {image['id']: image['file_name']
                       for image in full['images']}
    full_category_name = {cat['id']: cat['name'] for cat in full['categories']}
    novel_files = {
        full_image_name[ann['image_id']] for ann in full['annotations']
        if full_category_name[ann['category_id']] in XVIEW_NOVEL_CLASSES}
    disjoint_files = {
        image['file_name'] for image in
        loaded['train_base_48cls_patch_disjoint_hbb.json']['images']}
    if novel_files & disjoint_files:
        raise ValueError('Patch-disjoint Base-48 JSON retains Novel crops.')

    def base_signatures(data: Dict[str, Any]) -> set:
        image_names = {image['id']: image['file_name']
                       for image in data['images']}
        category_names = {cat['id']: cat['name'] for cat in data['categories']}
        return {
            (image_names[ann['image_id']], category_names[ann['category_id']],
             tuple(ann['bbox'])) for ann in data['annotations']
            if category_names[ann['category_id']] in XVIEW_BASE_CLASSES}
    if base_signatures(full) != base_signatures(
            loaded['train_base_48cls_hbb.json']):
        raise ValueError('Base-48 JSON is not the exact Full-60 Base subset.')
    val_full = loaded['val_full_60cls_hbb.json']
    val_novel = loaded['val_novel_12cls_hbb.json']

    def novel_signatures(data: Dict[str, Any]) -> set:
        image_names = {image['id']: image['file_name']
                       for image in data['images']}
        category_names = {cat['id']: cat['name'] for cat in data['categories']}
        return {
            (image_names[ann['image_id']], category_names[ann['category_id']],
             tuple(ann['bbox'])) for ann in data['annotations']
            if category_names[ann['category_id']] in XVIEW_NOVEL_CLASSES}
    if novel_signatures(val_full) != novel_signatures(val_novel):
        raise ValueError('Novel-12 val JSON is not the exact Full-60 subset.')
    report['cross_file_checks'] = {
        'base_is_exact_full_subset': True,
        'patch_disjoint_has_no_novel_crop': True,
        'val_novel_is_exact_full_subset': True,
        'unfinished_part_files': 0,
    }
    atomic_write_json(output_root / 'validation_report.json', report,
                      pretty=True)
    return report


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    official_val_json = args.official_val_json.resolve()
    if args.workers < 1:
        raise ValueError('--workers must be at least 1.')
    if args.validate_only:
        report = deep_validate_output(output_root)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    print('Source: {}'.format(source_root), flush=True)
    print('Output: {}'.format(output_root), flush=True)
    print('Write images: {}'.format(not args.dry_run), flush=True)
    print('Workers: {}'.format(args.workers), flush=True)

    splits = collect_and_split_images(source_root)
    train_stems = [path.stem for path in splits['train']]
    val_stems = [path.stem for path in splits['val']]
    val_patch_count, _ = count_expected_patches(splits['val'])
    train_patch_count, _ = count_expected_patches(splits['train'])
    if val_patch_count != EXPECTED_PATCH_COUNTS['val']:
        raise ValueError(
            'Validation split produces {} crops, expected {}.'.format(
                val_patch_count, EXPECTED_PATCH_COUNTS['val']))
    if train_patch_count != EXPECTED_PATCH_COUNTS['train']:
        raise ValueError(
            'Training split produces {} crops, expected {}.'.format(
                train_patch_count, EXPECTED_PATCH_COUNTS['train']))
    official_report = verify_official_val_split(official_val_json, val_stems)
    print(
        'Split verified: val={} sources/{} patches; train={} sources/{} patches'.format(
            len(val_stems), val_patch_count, len(train_stems), train_patch_count),
        flush=True)

    objects_by_split, label_report = load_objects(
        source_root, train_stems, val_stems)
    print('Labels parsed: {}'.format(json.dumps(label_report,
                                                ensure_ascii=False)),
          flush=True)

    assembled_by_split = {}
    for split in ('train', 'val'):
        assembled_by_split[split] = run_split_crops(
            split, splits[split], objects_by_split[split], output_root,
            args.workers, write_images=not args.dry_run,
            overwrite=args.overwrite,
            png_compress_level=args.png_compress_level)
    official_annotation_report = apply_official_val_annotations(
        official_val_json, assembled_by_split['val'])
    print('Official val annotations: {}'.format(json.dumps(
        official_annotation_report, ensure_ascii=False)), flush=True)
    summaries = {}
    if not args.dry_run:
        summaries = write_outputs(output_root, assembled_by_split)
        split_manifest = {
            'protocol': 'RT-OVAD xView zero-shot split',
            'source_root': str(source_root),
            'sorting': 'numeric TIFF filename stem ascending',
            'all_source_count': len(splits['all']),
            'val': {
                'source_count': len(val_stems),
                'paper_role': 'test/evaluation split; named val for repository compatibility',
                'patch_count': val_patch_count,
                'source_ids': val_stems,
                'sha256_newline_delimited_ids': split_hash(val_stems),
                'images_generated': True,
            },
            'train': {
                'source_count': len(train_stems),
                'patch_count': train_patch_count,
                'source_ids': train_stems,
                'sha256_newline_delimited_ids': split_hash(train_stems),
                'images_generated': True,
            },
            'official_val_annotation_verification': official_report,
            'official_val_annotations': official_annotation_report,
        }
        atomic_write_json(output_root / 'split_manifest.json', split_manifest,
                          pretty=True)
        crop_manifest = {
            'patch_size': PATCH_SIZE,
            'crop_geometry': (
                'for L>800: count=ceil(L/800), step=ceil(L/count), starts are '
                'i*step plus final L-800; for L<=800: start 0 and pad'),
            'object_assignment': (
                'strict center-in-window and raw retained-area ratio > 0.5'),
            'hbb_conversion': 'expand source HBB by 1px, then clip',
            'label_parse': label_report,
            'official_val_annotations': official_annotation_report,
            'outputs': summaries,
        }
        atomic_write_json(output_root / 'crop_manifest.json', crop_manifest,
                          pretty=True)
        report = deep_validate_output(output_root)
        print('Deep validation passed.', flush=True)
        print(json.dumps(report['jsons'], ensure_ascii=False, indent=2),
              flush=True)
    else:
        print(
            'Dry-run passed: train={}/{} and val={}/{} '
            'patches/accepted annotations.'.format(
                len(assembled_by_split['train']['images']),
                len(assembled_by_split['train']['annotations']),
                len(assembled_by_split['val']['images']),
                len(assembled_by_split['val']['annotations'])),
            flush=True)


if __name__ == '__main__':
    main()

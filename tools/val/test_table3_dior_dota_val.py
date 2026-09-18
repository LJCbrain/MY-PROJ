"""Audit and evaluate RT-OVAD Table III (upper, non-asterisk) on val.

The script performs four independent evaluations with one checkpoint:

1. DIOR GZSD on the complete 20-class validation set.
2. DIOR ZSD with only Novel-4 annotations/prompts.
3. DOTA-v1.0 GZSD on the complete 15-class validation set.
4. DOTA-v1.0 ZSD on Novel-containing validation patches with only Novel-4
   annotations/prompts.

It reports every number in the upper part of Table III: Base/Novel/HM AP50,
Base/Novel/HM RE@100 at IoU=0.5, Novel-only AP50 and Novel-only RE@100.
Before inference it checks COCO structure, class order, image paths, bounding
boxes, the Novel-only annotation subsets, and the paper's dataset-size/crop
requirements.  Protocol differences are printed prominently and can be made
fatal with ``--require-paper-exact``.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


# Running this file directly normally adds only ``tools/val`` to sys.path. Put
# the repository root first so Config.fromfile can import custom modules.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DIOR_BASE_CLASSES = (
    'airplane', 'baseballfield', 'bridge', 'chimney', 'dam',
    'Expressway-Service-area', 'Expressway-toll-station', 'golffield',
    'harbor', 'overpass', 'ship', 'stadium', 'storagetank', 'tenniscourt',
    'trainstation', 'vehicle')
DIOR_NOVEL_CLASSES = (
    'airport', 'basketballcourt', 'groundtrackfield', 'windmill')
DIOR_FULL_CLASSES = (
    'airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge',
    'chimney', 'dam', 'Expressway-Service-area',
    'Expressway-toll-station', 'golffield', 'groundtrackfield', 'harbor',
    'overpass', 'ship', 'stadium', 'storagetank', 'tenniscourt',
    'trainstation', 'vehicle', 'windmill')

DOTA_BASE_CLASSES = (
    'plane', 'ship', 'storage-tank', 'baseball-diamond',
    'basketball-court', 'ground-track-field', 'harbor', 'bridge',
    'large-vehicle', 'small-vehicle', 'roundabout')
DOTA_NOVEL_CLASSES = (
    'tennis-court', 'helicopter', 'soccer-ball-field', 'swimming-pool')
DOTA_FULL_CLASSES = DOTA_BASE_CLASSES + DOTA_NOVEL_CLASSES


@dataclass(frozen=True)
class JsonSpec:
    name: str
    annotation: str
    image_prefix: str
    classes: tuple[str, ...]


JSON_SPECS = (
    JsonSpec('DIOR train Base-16',
             'data/DIOR/Annotations2/'
             'train_base_16cls_image_disjoint_yoloworld.json',
             'data/DIOR/JPEGImages', DIOR_BASE_CLASSES),
    JsonSpec('DIOR val GZSD-20', 'data/DIOR/Annotations2/val.json',
             'data/DIOR/JPEGImages', DIOR_FULL_CLASSES),
    JsonSpec('DIOR val ZSD-4',
             'data/DIOR/Annotations2/val_novel_4cls.json',
             'data/DIOR/JPEGImages', DIOR_NOVEL_CLASSES),
    JsonSpec('DOTA-v1.0 train All patches-15',
             'data/DOTA800/annotations/'
             'train_full_15cls_hbb_all_patches.json',
             'data/DOTA800/train/images', DOTA_FULL_CLASSES),
    JsonSpec('DOTA-v1.0 train Base-11 patch-disjoint',
             'data/DOTA800/annotations/'
             'train_base_11cls_patch_disjoint_hbb.json',
             'data/DOTA800/train/images', DOTA_BASE_CLASSES),
    JsonSpec('DOTA-v1.0 val GZSD-15',
             'data/DOTA800/annotations/val_full_15cls_hbb.json',
             'data/DOTA800/val/images', DOTA_FULL_CLASSES),
    JsonSpec('DOTA-v1.0 val All patches-15',
             'data/DOTA800/annotations/val_full_15cls_hbb_all_patches.json',
             'data/DOTA800/val/images', DOTA_FULL_CLASSES),
    JsonSpec('DOTA-v1.0 val ZSD-4',
             'data/DOTA800/annotations/val_novel_4cls_hbb.json',
             'data/DOTA800/val/images', DOTA_NOVEL_CLASSES),
)


@dataclass(frozen=True)
class EvalSpec:
    dataset: str
    mode: str
    dataset_key: str
    evaluator_key: str


EVAL_SPECS = (
    EvalSpec('DIOR', 'GZSD', 'dior_gzsd_val_dataset',
             'dior_gzsd_val_evaluator'),
    EvalSpec('DIOR', 'ZSD', 'dior_zsd_val_dataset',
             'dior_zsd_val_evaluator'),
    EvalSpec('DOTA-v1.0', 'GZSD', 'dota_gzsd_val_dataset',
             'dota_gzsd_val_evaluator'),
    EvalSpec('DOTA-v1.0', 'ZSD', 'dota_zsd_val_dataset',
             'dota_zsd_val_evaluator'),
)


@dataclass
class AuditMessage:
    level: str
    code: str
    message: str


@dataclass
class JsonSummary:
    name: str
    images: int
    annotations: int
    categories: int
    boxes_invalid: int
    image_files_missing: int
    image_size_mismatches: int
    images_800x800: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Audit and test RT-OVAD Table III upper protocol on '
                    'DIOR and DOTA-v1.0 validation data.')
    parser.add_argument(
        '--config', default='configs/pretrain/DIOR_DOTAv2.5.py',
        help='Config that defines all four *_val_dataset/evaluator objects.')
    parser.add_argument(
        '--checkpoint', default="work_dirs/DIOR_DOTAv2.5_ALL/best_coco_bbox_mAP_50_HM_epoch_20.pth",
        help='Checkpoint to test. Required unless --audit-only is used.')
    parser.add_argument(
        '--work-dir', default='work_dirs/DIOR_DOTAv1/table3_upper_val',
        help='Directory for four inference runs and result tables.')
    parser.add_argument(
        '--audit-only', action='store_true',
        help='Only audit datasets; do not import MMEngine or run inference.')
    parser.add_argument(
        '--require-paper-exact', action='store_true',
        help='Treat paper-protocol differences (not only broken data) as '
             'fatal.')
    parser.add_argument(
        '--skip-image-size-check', action='store_true',
        help='Skip opening every image to verify dimensions. File existence '
             'and JSON-declared dimensions are still checked.')
    parser.add_argument(
        '--format-only', action='store_true',
        help='Regenerate aligned tables from an existing metrics JSON.')
    parser.add_argument(
        '--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none')
    parser.add_argument('--local_rank', type=int, default=0)
    return parser.parse_args()


def resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def load_coco(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except OSError as exc:
        raise FileNotFoundError(f'Cannot read annotation JSON: {path}') from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f'Invalid JSON: {path}: {exc}') from exc
    if not isinstance(data, dict):
        raise TypeError(f'COCO root must be an object: {path}')
    for key in ('images', 'annotations', 'categories'):
        if not isinstance(data.get(key), list):
            raise TypeError(f'{path}: {key!r} must be a list.')
    return data


def duplicates(values: Iterable[Any]) -> set[Any]:
    seen: set[Any] = set()
    repeated: set[Any] = set()
    for value in values:
        if value in seen:
            repeated.add(value)
        seen.add(value)
    return repeated


def _finite_number(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(float(value)))


def audit_json(spec: JsonSpec, check_image_sizes: bool,
               messages: list[AuditMessage]) -> tuple[dict[str, Any], JsonSummary]:
    annotation_path = resolve_project_path(spec.annotation)
    image_root = resolve_project_path(spec.image_prefix)
    data = load_coco(annotation_path)

    categories = data['categories']
    category_names = tuple(category.get('name') for category in categories)
    category_ids = [category.get('id') for category in categories]
    if category_names != spec.classes:
        messages.append(AuditMessage(
            'ERROR', 'CATEGORY_ORDER',
            f'{spec.name}: categories/order differ. Expected {spec.classes}; '
            f'got {category_names}.'))
    if duplicates(category_ids):
        messages.append(AuditMessage(
            'ERROR', 'DUPLICATE_CATEGORY_ID',
            f'{spec.name}: duplicate category ids {duplicates(category_ids)}.'))

    image_ids = [image.get('id') for image in data['images']]
    annotation_ids = [ann.get('id') for ann in data['annotations']]
    if duplicates(image_ids):
        messages.append(AuditMessage(
            'ERROR', 'DUPLICATE_IMAGE_ID',
            f'{spec.name}: duplicate image ids detected.'))
    if duplicates(annotation_ids):
        messages.append(AuditMessage(
            'ERROR', 'DUPLICATE_ANNOTATION_ID',
            f'{spec.name}: duplicate annotation ids detected.'))

    image_by_id = {image.get('id'): image for image in data['images']}
    valid_category_ids = set(category_ids)
    invalid_boxes = 0
    invalid_box_examples: list[str] = []
    broken_refs = 0
    for ann in data['annotations']:
        if (ann.get('image_id') not in image_by_id or
                ann.get('category_id') not in valid_category_ids):
            broken_refs += 1
            continue
        bbox = ann.get('bbox')
        image = image_by_id[ann['image_id']]
        width, height = image.get('width'), image.get('height')
        if (not isinstance(bbox, list) or len(bbox) != 4 or
                not all(_finite_number(value) for value in bbox) or
                not _finite_number(width) or not _finite_number(height)):
            invalid_boxes += 1
            if len(invalid_box_examples) < 4:
                invalid_box_examples.append(
                    f'ann={ann.get("id")}, image={image.get("file_name")}, '
                    f'bbox={bbox!r}')
            continue
        x, y, box_width, box_height = map(float, bbox)
        if (x < 0 or y < 0 or box_width <= 0 or box_height <= 0 or
                x + box_width > float(width) + 1e-6 or
                y + box_height > float(height) + 1e-6):
            invalid_boxes += 1
            if len(invalid_box_examples) < 4:
                invalid_box_examples.append(
                    f'ann={ann.get("id")}, image={image.get("file_name")}, '
                    f'bbox={bbox!r}')
    if broken_refs:
        messages.append(AuditMessage(
            'ERROR', 'BROKEN_ANNOTATION_REFERENCE',
            f'{spec.name}: {broken_refs} annotations reference a missing '
            'image/category.'))
    if invalid_boxes:
        # DIOR's official HBB XML includes four known malformed/out-of-range
        # boxes. Do not silently alter validation ground truth.
        messages.append(AuditMessage(
            'WARNING', 'SOURCE_BBOX_ANOMALY',
            f'{spec.name}: {invalid_boxes} source-derived non-positive or '
            f'out-of-image boxes ({"; ".join(invalid_box_examples)}). These '
            'are reported rather than silently changing the benchmark GT.'))

    files_missing = 0
    size_mismatches = 0
    images_800 = 0
    pil_image = None
    if check_image_sizes:
        try:
            from PIL import Image as pil_image
        except ImportError as exc:
            raise ImportError(
                'Pillow is required for the default image-size audit. '
                'Install Pillow or pass --skip-image-size-check.') from exc

    for image in data['images']:
        file_name = image.get('file_name')
        width, height = image.get('width'), image.get('height')
        if width == 800 and height == 800:
            images_800 += 1
        if not isinstance(file_name, str) or not file_name:
            files_missing += 1
            continue
        image_path = image_root / Path(file_name)
        if not image_path.is_file():
            files_missing += 1
            continue
        if check_image_sizes and pil_image is not None:
            try:
                with pil_image.open(image_path) as opened:
                    actual_size = opened.size
            except OSError:
                size_mismatches += 1
            else:
                if actual_size != (width, height):
                    size_mismatches += 1
    if files_missing:
        messages.append(AuditMessage(
            'ERROR', 'MISSING_IMAGES',
            f'{spec.name}: {files_missing} image files are missing.'))
    if size_mismatches:
        messages.append(AuditMessage(
            'ERROR', 'IMAGE_SIZE_MISMATCH',
            f'{spec.name}: {size_mismatches} actual image dimensions differ '
            'from JSON.'))

    return data, JsonSummary(
        name=spec.name,
        images=len(data['images']),
        annotations=len(data['annotations']),
        categories=len(categories),
        boxes_invalid=invalid_boxes,
        image_files_missing=files_missing,
        image_size_mismatches=size_mismatches,
        images_800x800=images_800)


def annotation_signature(annotation: Mapping[str, Any]) -> tuple[Any, ...]:
    bbox = annotation.get('bbox', [])
    return (annotation.get('image_id'), annotation.get('category_id'),
            *(round(float(value), 6) for value in bbox),
            annotation.get('iscrowd', 0))


def audit_novel_subset(full: dict[str, Any], novel: dict[str, Any],
                       novel_names: Sequence[str], dataset: str,
                       messages: list[AuditMessage]) -> None:
    full_images = {image['id']: image.get('file_name')
                   for image in full['images']}
    novel_images = {image['id']: image.get('file_name')
                    for image in novel['images']}
    full_name_by_id = {category['id']: category['name']
                       for category in full['categories']}
    novel_target_ids = {
        ann['image_id'] for ann in full['annotations']
        if full_name_by_id.get(ann.get('category_id')) in novel_names}
    allowed_image_sets = (set(full_images), novel_target_ids)
    if (set(novel_images) not in allowed_image_sets or
            any(full_images.get(image_id) != file_name
                for image_id, file_name in novel_images.items())):
        messages.append(AuditMessage(
            'ERROR', 'ZSD_IMAGE_SET',
            f'{dataset}: ZSD images must be either the complete GZSD image '
            'set or exactly the subset containing Novel-4 ground truth.'))

    novel_id_by_name = {category['name']: category['id']
                        for category in novel['categories']}
    expected = []
    for ann in full['annotations']:
        name = full_name_by_id.get(ann.get('category_id'))
        if name in novel_names:
            converted = dict(ann)
            converted['category_id'] = novel_id_by_name[name]
            expected.append(annotation_signature(converted))
    actual = [annotation_signature(ann) for ann in novel['annotations']]
    if sorted(expected) != sorted(actual):
        messages.append(AuditMessage(
            'ERROR', 'ZSD_ANNOTATION_SUBSET',
            f'{dataset}: Novel-only JSON is not the exact Novel-4 subset of '
            'the full validation annotation.'))


def audit_training_filter(dior_source: dict[str, Any],
                          dior_train: dict[str, Any],
                          messages: list[AuditMessage]) -> None:
    name_by_id = {category['id']: category['name']
                  for category in dior_source['categories']}
    novel_image_ids = {
        ann['image_id'] for ann in dior_source['annotations']
        if name_by_id.get(ann['category_id']) in DIOR_NOVEL_CLASSES}
    retained_ids = {image['id'] for image in dior_train['images']}
    overlap = novel_image_ids & retained_ids
    expected_count = len(dior_source['images']) - len(novel_image_ids)
    if overlap or len(retained_ids) != expected_count:
        messages.append(AuditMessage(
            'ERROR', 'DIOR_NOVEL_IMAGE_FILTER',
            f'DIOR Base-16 training JSON is not strictly image-disjoint: '
            f'{len(overlap)} retained images contain Novel-4 annotations.'))
    messages.append(AuditMessage(
        'PROTOCOL', 'DIOR_STRICT_IMAGE_DISJOINT',
        f'DIOR training keeps {len(retained_ids)}/{len(dior_source["images"])} '
        f'images after removing all {len(novel_image_ids)} images containing '
        'Novel-4. RT-OVAD only states that novel categories are excluded; it '
        'does not document this stricter whole-image removal.'))


def audit_dota_training_filter(dota_full: dict[str, Any],
                               dota_train: dict[str, Any],
                               messages: list[AuditMessage]) -> None:
    full_name_by_id = {category['id']: category['name']
                       for category in dota_full['categories']}
    base_ids = {category_id for category_id, name in full_name_by_id.items()
                if name in DOTA_BASE_CLASSES}
    novel_ids = {category_id for category_id, name in full_name_by_id.items()
                 if name in DOTA_NOVEL_CLASSES}
    image_ids_with_base = {
        ann['image_id'] for ann in dota_full['annotations']
        if ann.get('category_id') in base_ids}
    image_ids_with_novel = {
        ann['image_id'] for ann in dota_full['annotations']
        if ann.get('category_id') in novel_ids}
    expected_ids = image_ids_with_base - image_ids_with_novel
    retained_ids = {image['id'] for image in dota_train['images']}
    if retained_ids != expected_ids:
        messages.append(AuditMessage(
            'ERROR', 'DOTA_NOVEL_IMAGE_FILTER',
            'DOTA Base-11 training JSON is not the exact set of non-empty '
            'crop patches that contain Base-11 ground truth and no Novel-4 '
            f'ground truth: expected {len(expected_ids)}, got '
            f'{len(retained_ids)}.'))

    expected_annotations = [
        annotation_signature(ann) for ann in dota_full['annotations']
        if ann.get('image_id') in expected_ids and
        ann.get('category_id') in base_ids]
    actual_annotations = [
        annotation_signature(ann) for ann in dota_train['annotations']]
    if sorted(expected_annotations) != sorted(actual_annotations):
        messages.append(AuditMessage(
            'ERROR', 'DOTA_BASE_ANNOTATION_SUBSET',
            'DOTA Base-11 patch-disjoint annotations are not the exact Base '
            'subset of the full training annotations.'))

    messages.append(AuditMessage(
        'INFO', 'DOTA_PATCH_DISJOINT',
        f'DOTA training keeps {len(retained_ids)} non-empty Base-11 crop '
        f'patches and excludes {len(image_ids_with_novel)} non-empty patches '
        'containing Novel-4 ground truth.'))


def audit_paper_protocol(data: dict[str, dict[str, Any]],
                         summaries: Sequence[JsonSummary],
                         messages: list[AuditMessage]) -> None:
    by_name = {summary.name: summary for summary in summaries}
    dior_val = by_name['DIOR val GZSD-20']
    dota_train_all = by_name['DOTA-v1.0 train All patches-15']
    dota_val_eval = by_name['DOTA-v1.0 val GZSD-15']
    dota_val_all = by_name['DOTA-v1.0 val All patches-15']
    dota_zsd_eval = by_name['DOTA-v1.0 val ZSD-4']

    if dior_val.images != 5863:
        messages.append(AuditMessage(
            'PROTOCOL', 'DIOR_VAL_COUNT',
            f'DIOR paper validation set has 5863 images; current JSON has '
            f'{dior_val.images}.'))

    # Table III upper uses the DescReg DOTA-v1.0 preprocessing: raw 1411/458
    # train/val images are cropped/padded into 18430/6259 800x800 patches.
    if (dota_val_all.images != 6259 or
            dota_val_all.images_800x800 != dota_val_all.images):
        messages.append(AuditMessage(
            'PROTOCOL', 'DOTA_VAL_NOT_PAPER_CROPPED',
            'Table III requires 6259 DOTA-v1.0 validation patches of '
            f'800x800; current all-patches JSON has {dota_val_all.images} '
            f'images, of which {dota_val_all.images_800x800} are 800x800.'))
    if dota_val_eval.images != dota_val_all.images:
        messages.append(AuditMessage(
            'PROTOCOL', 'DOTA_GZSD_NONEMPTY_SUBSET',
            f'The selected GZSD evaluation uses {dota_val_eval.images} '
            f'non-empty patches instead of all {dota_val_all.images} '
            'generated validation patches. This is the requested local '
            'non-empty-patch protocol.'))
    if dota_zsd_eval.images != dota_val_all.images:
        messages.append(AuditMessage(
            'PROTOCOL', 'DOTA_ZSD_NOVEL_TARGET_SUBSET',
            f'The selected ZSD evaluation uses {dota_zsd_eval.images} '
            'Novel-containing patches instead of all '
            f'{dota_val_all.images} generated validation patches.'))
    if (dota_train_all.images != 18430 or
            dota_train_all.images_800x800 != dota_train_all.images):
        messages.append(AuditMessage(
            'PROTOCOL', 'DOTA_TRAIN_NOT_PAPER_CROPPED',
            'Table III preprocessing requires 18430 DOTA-v1.0 training '
            f'patches of 800x800; current all-patches JSON has '
            f'{dota_train_all.images} images, of which '
            f'{dota_train_all.images_800x800} are 800x800.'))

    dior_source_path = resolve_project_path(
        'data/DIOR/Annotations2/train.json')
    if dior_source_path.is_file():
        audit_training_filter(load_coco(dior_source_path),
                              data['DIOR train Base-16'], messages)
    audit_dota_training_filter(
        data['DOTA-v1.0 train All patches-15'],
        data['DOTA-v1.0 train Base-11 patch-disjoint'], messages)


def aligned_table(headers: Sequence[str], rows: Sequence[Sequence[str]],
                  numeric_columns: set[int] | None = None) -> str:
    numeric_columns = numeric_columns or set()
    normalized = [tuple(str(cell) for cell in headers)] + [
        tuple(str(cell) for cell in row) for row in rows]
    if any(len(row) != len(headers) for row in normalized):
        raise ValueError('Every table row must have the same number of cells.')
    widths = [max(len(row[index]) for row in normalized)
              for index in range(len(headers))]

    def render(row: Sequence[str]) -> str:
        cells = []
        for index, value in enumerate(row):
            cells.append(value.rjust(widths[index])
                         if index in numeric_columns
                         else value.ljust(widths[index]))
        return '| ' + ' | '.join(cells) + ' |'

    separator = '|-' + '-|-'.join('-' * width for width in widths) + '-|'
    return '\n'.join((render(headers), separator,
                      *(render(row) for row in rows)))


def print_audit(summaries: Sequence[JsonSummary],
                messages: Sequence[AuditMessage]) -> None:
    headers = ('Dataset JSON', 'Images', 'Annotations', 'Classes',
               'Bad boxes', 'Missing files', 'Size mismatch', '800x800')
    rows = [
        (summary.name, str(summary.images), str(summary.annotations),
         str(summary.categories), str(summary.boxes_invalid),
         str(summary.image_files_missing),
         str(summary.image_size_mismatches), str(summary.images_800x800))
        for summary in summaries]
    print('\nDATASET AUDIT')
    print(aligned_table(headers, rows, set(range(1, len(headers)))))
    if messages:
        print('\nAUDIT FINDINGS')
        for item in messages:
            print(f'[{item.level}] {item.code}: {item.message}')
    else:
        print('\nAUDIT FINDINGS\n[OK] No structural or protocol issue found.')


def write_audit_outputs(work_dir: Path,
                        summaries: Sequence[JsonSummary],
                        messages: Sequence[AuditMessage]) -> tuple[Path, Path]:
    headers = ('Dataset JSON', 'Images', 'Annotations', 'Classes',
               'Bad boxes', 'Missing files', 'Size mismatch', '800x800')
    rows = [
        (summary.name, str(summary.images), str(summary.annotations),
         str(summary.categories), str(summary.boxes_invalid),
         str(summary.image_files_missing),
         str(summary.image_size_mismatches), str(summary.images_800x800))
        for summary in summaries]
    lines = [aligned_table(headers, rows, set(range(1, len(headers)))), '']
    lines.extend(
        f'[{item.level}] {item.code}: {item.message}' for item in messages)
    text_path = work_dir / 'table3_dataset_audit.txt'
    text_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    json_path = work_dir / 'table3_dataset_audit.json'
    json_path.write_text(json.dumps({
        'summaries': [asdict(summary) for summary in summaries],
        'findings': [asdict(item) for item in messages],
        'paper_exact': not any(item.level == 'PROTOCOL' for item in messages),
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    return text_path, json_path


def run_audit(skip_image_size_check: bool) -> tuple[
        list[JsonSummary], list[AuditMessage]]:
    messages: list[AuditMessage] = []
    summaries: list[JsonSummary] = []
    datasets: dict[str, dict[str, Any]] = {}
    for spec in JSON_SPECS:
        data, summary = audit_json(
            spec, not skip_image_size_check, messages)
        datasets[spec.name] = data
        summaries.append(summary)

    audit_novel_subset(
        datasets['DIOR val GZSD-20'], datasets['DIOR val ZSD-4'],
        DIOR_NOVEL_CLASSES, 'DIOR', messages)
    audit_novel_subset(
        datasets['DOTA-v1.0 val GZSD-15'],
        datasets['DOTA-v1.0 val ZSD-4'], DOTA_NOVEL_CLASSES,
        'DOTA-v1.0', messages)
    audit_paper_protocol(datasets, summaries, messages)
    print_audit(summaries, messages)
    return summaries, messages


def metric_value(metrics: Mapping[str, Any], suffix: str) -> float:
    matches = [float(value) for key, value in metrics.items()
               if str(key).split('/')[-1] == suffix]
    if len(matches) != 1:
        raise KeyError(
            f'Expected exactly one metric ending in {suffix!r}; got '
            f'{len(matches)}. Available keys: {sorted(metrics)}')
    return matches[0]


def percentage(metrics: Mapping[str, Any], suffix: str) -> float:
    return round(metric_value(metrics, suffix) * 100.0, 1)


def build_runner(config: Any) -> Any:
    from mmengine.runner import Runner
    from mmyolo.registry import RUNNERS
    if 'runner_type' in config:
        return RUNNERS.build(config)
    return Runner.from_cfg(config)


def run_four_evaluations(args: argparse.Namespace) -> dict[str, Any]:
    from mmengine.config import Config
    from mmyolo.utils import is_metainfo_lower

    if args.checkpoint is None:
        raise ValueError('--checkpoint is required unless --audit-only is used.')
    checkpoint = resolve_project_path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint}')
    config_path = resolve_project_path(args.config)
    if not config_path.is_file():
        raise FileNotFoundError(f'Config not found: {config_path}')

    base_config = Config.fromfile(str(config_path))
    base_config.launcher = args.launcher
    is_metainfo_lower(base_config)
    for spec in EVAL_SPECS:
        if spec.dataset_key not in base_config:
            raise KeyError(f'{config_path} does not define {spec.dataset_key}.')
        if spec.evaluator_key not in base_config:
            raise KeyError(f'{config_path} does not define {spec.evaluator_key}.')

    work_dir = resolve_project_path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    raw: dict[str, dict[str, float]] = {}
    for spec in EVAL_SPECS:
        key = f'{spec.dataset}_{spec.mode}'
        run_config = copy.deepcopy(base_config)
        run_config.load_from = str(checkpoint)
        run_config.work_dir = str(work_dir / key.replace('.', '_'))
        # Keep DataLoader's outer sampler/batch/worker settings. Replacing the
        # whole dict with a dataset caused the earlier ``unexpected type``
        # DataLoader failure.
        run_config.test_dataloader = copy.deepcopy(
            base_config.test_dataloader)
        run_config.test_dataloader['dataset'] = copy.deepcopy(
            base_config[spec.dataset_key])
        run_config.test_evaluator = copy.deepcopy(
            base_config[spec.evaluator_key])
        # Enforce the paper metric even when an older uploaded config still
        # says 300. Detector max_per_img may remain 300; the evaluator keeps
        # the highest-scoring 100 predictions for RE@100.
        run_config.test_evaluator['recall_max_dets'] = 100

        print(f'\n===== {spec.dataset} {spec.mode} validation =====')
        runner = build_runner(run_config)
        metrics = runner.test()
        raw[key] = {str(name): float(value)
                    for name, value in metrics.items()}
        del runner
        try:
            import torch
            torch.cuda.empty_cache()
        except ImportError:
            pass

    return {
        'DIOR': result_row(raw['DIOR_GZSD'], raw['DIOR_ZSD']),
        'DOTA-v1.0': result_row(
            raw['DOTA-v1.0_GZSD'], raw['DOTA-v1.0_ZSD']),
        'raw_metrics': raw,
        'checkpoint': str(checkpoint),
        'config': str(config_path),
    }


def result_row(gzsd: Mapping[str, Any],
               zsd: Mapping[str, Any]) -> dict[str, float]:
    return {
        'GZSD_AP50_Base': percentage(gzsd, 'bbox_mAP_50_base'),
        'GZSD_AP50_Novel': percentage(gzsd, 'bbox_mAP_50_novel'),
        'GZSD_AP50_HM': percentage(gzsd, 'bbox_mAP_50_HM'),
        'GZSD_RE100_Base': percentage(gzsd, 'bbox_recall_50_base'),
        'GZSD_RE100_Novel': percentage(gzsd, 'bbox_recall_50_novel'),
        'GZSD_RE100_HM': percentage(gzsd, 'bbox_recall_50_HM'),
        'ZSD_AP50_Novel': percentage(zsd, 'bbox_mAP_50_novel'),
        'ZSD_RE100_Novel': percentage(zsd, 'bbox_recall_50_novel'),
    }


RESULT_COLUMNS = (
    'GZSD_AP50_Base', 'GZSD_AP50_Novel', 'GZSD_AP50_HM',
    'GZSD_RE100_Base', 'GZSD_RE100_Novel', 'GZSD_RE100_HM',
    'ZSD_AP50_Novel', 'ZSD_RE100_Novel')


def result_table(payload: Mapping[str, Any]) -> str:
    headers = (
        'Dataset', 'GZSD AP50 Base', 'GZSD AP50 Novel', 'GZSD AP50 HM',
        'GZSD RE@100 Base', 'GZSD RE@100 Novel', 'GZSD RE@100 HM',
        'ZSD AP50 Novel', 'ZSD RE@100 Novel')
    rows = []
    for dataset in ('DIOR', 'DOTA-v1.0'):
        values = payload[dataset]
        rows.append((dataset, *(f'{float(values[key]):.1f}'
                                for key in RESULT_COLUMNS)))
    return aligned_table(headers, rows, set(range(1, len(headers))))


def write_outputs(work_dir: Path, payload: dict[str, Any],
                  summaries: Sequence[JsonSummary],
                  messages: Sequence[AuditMessage]) -> tuple[Path, Path, Path]:
    table = result_table(payload) + '\n'
    table_path = work_dir / 'table3_upper_val_metrics.txt'
    table_path.write_text(table, encoding='utf-8')

    tsv_path = work_dir / 'table3_upper_val_metrics.tsv'
    tsv_headers = ('Dataset', *RESULT_COLUMNS)
    tsv_rows = ['\t'.join(tsv_headers)]
    for dataset in ('DIOR', 'DOTA-v1.0'):
        tsv_rows.append('\t'.join((
            dataset, *(f'{float(payload[dataset][key]):.1f}'
                       for key in RESULT_COLUMNS))))
    tsv_path.write_text('\n'.join(tsv_rows) + '\n', encoding='utf-8')

    json_path = work_dir / 'table3_upper_val_metrics.json'
    serializable = dict(payload)
    serializable['audit'] = {
        'summaries': [asdict(summary) for summary in summaries],
        'findings': [asdict(item) for item in messages],
        'paper_exact': not any(item.level == 'PROTOCOL' for item in messages),
        'recall_definition': 'RE@100 at IoU=0.5',
    }
    json_path.write_text(
        json.dumps(serializable, ensure_ascii=False, indent=2),
        encoding='utf-8')
    return table_path, tsv_path, json_path


def format_existing(work_dir: Path) -> None:
    json_path = work_dir / 'table3_upper_val_metrics.json'
    try:
        payload = json.loads(json_path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f'Cannot read existing results: {json_path}') from exc
    table = result_table(payload) + '\n'
    table_path = work_dir / 'table3_upper_val_metrics.txt'
    table_path.write_text(table, encoding='utf-8')
    print('\nTABLE III UPPER (VAL, %)')
    print(table)
    print(f'Reformatted: {table_path}')


def main() -> None:
    args = parse_args()
    work_dir = resolve_project_path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    if args.format_only:
        format_existing(work_dir)
        return

    summaries, messages = run_audit(args.skip_image_size_check)
    audit_text, audit_json = write_audit_outputs(
        work_dir, summaries, messages)
    print(f'\nAudit report files:\n  {audit_text}\n  {audit_json}')
    structural_errors = [item for item in messages if item.level == 'ERROR']
    protocol_errors = [item for item in messages if item.level == 'PROTOCOL']
    if structural_errors:
        raise RuntimeError(
            f'Dataset audit failed with {len(structural_errors)} structural '
            'error(s). Fix them before testing.')
    if args.require_paper_exact and protocol_errors:
        raise RuntimeError(
            f'Paper-exact audit failed with {len(protocol_errors)} protocol '
            'difference(s).')
    if args.audit_only:
        return

    payload = run_four_evaluations(args)
    table_path, tsv_path, json_path = write_outputs(
        work_dir, payload, summaries, messages)
    print('\nTABLE III UPPER (VAL, %)')
    print(result_table(payload))
    if protocol_errors:
        print('\nWARNING: these values are valid for the current local '
              'validation JSONs, but are not paper-exact until every '
              '[PROTOCOL] finding above is resolved.')
    print('\nWrote results to:\n'
          f'  {table_path}\n  {tsv_path}\n  {json_path}')


if __name__ == '__main__':
    main()

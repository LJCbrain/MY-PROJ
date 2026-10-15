"""Evaluate a YOLO-World checkpoint on the LAE-FOD DIOR and DOTAv2 sets.

The script uses the category order stored in each COCO annotation file as
the test prompt order.  This is important for zero-shot evaluation: the
DIOR and DOTAv2 validation sets have different local label spaces.  Each
run uses ``mmdet.CocoMetric`` and consequently prints the standard MMEngine
metrics, for example ``coco/bbox_mAP``, ``coco/bbox_mAP_50`` and
``coco/bbox_mAP_75``.

Example:
    python tools/val/test_lae_fod_dior_dotav2.py \
        --config configs/pretrain/LAEv2.py \
        --checkpoint work_dirs/LAEv2/best_coco_bbox_mAP_epoch_10.pth
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable



# Direct execution adds only ``tools/val`` to sys.path.  The project root is
# needed so config custom imports can find ``yolo_world``.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass(frozen=True)
class DatasetSpec:
    """Static locations for one LAE-FOD validation benchmark."""

    name: str
    annotation: str
    image_prefix: str


DATASETS = (
    DatasetSpec(
        name='DIOR',
        annotation='LAE-FOD/DIOR/DIOR_val.json',
        image_prefix='LAE-FOD/DIOR/trainval_images/JPEGImages-trainval'),
    DatasetSpec(
        name='DOTAv2',
        annotation='LAE-FOD/DOTAv2/DOTAv2_val.json',
        image_prefix='LAE-FOD/DOTAv2/images/images'),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Test one checkpoint on LAE-FOD DIOR and DOTAv2.')
    parser.add_argument(
        '--config', default='configs/pretrain/LAEv2.py',
        help='Model config that matches --checkpoint.')
    parser.add_argument(
        '--checkpoint', default="/home/wxj185133/YOLO-World/work_dirs/LAEv2/epoch_10.pth",
        help='Checkpoint to evaluate (required unless --check-only is used).')
    parser.add_argument(
        '--data-root', default='data/LAE-1M',
        help='Directory containing the LAE-FOD directory.')
    parser.add_argument(
        '--dior-ann', default=DATASETS[0].annotation,
        help='DIOR COCO annotation path, relative to --data-root by default.')
    parser.add_argument(
        '--dotav2-ann', default=DATASETS[1].annotation,
        help='DOTAv2 COCO annotation path, relative to --data-root by default.')
    parser.add_argument(
        '--dior-img-prefix', default=DATASETS[0].image_prefix,
        help='DIOR image directory, relative to --data-root by default.')
    parser.add_argument(
        '--dotav2-img-prefix', default=DATASETS[1].image_prefix,
        help='DOTAv2 image directory, relative to --data-root by default.')
    parser.add_argument(
        '--work-dir', default='work_dirs/lae_fod_dior_dotav2_val',
        help='Directory for per-dataset logs, generated prompt files, and metrics.')
    parser.add_argument(
        '--check-only', action='store_true',
        help='Validate annotations, categories, and image paths without inference.')
    parser.add_argument(
        '--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none', help='Job launcher.')
    parser.add_argument('--local_rank', type=int, default=0)
    return parser.parse_args()


def project_path(path: str | Path) -> Path:
    """Interpret relative command-line paths from the repository root."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def data_path(data_root: Path, path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else data_root / candidate


def load_annotation(annotation_path: Path) -> dict[str, Any]:
    try:
        with annotation_path.open('r', encoding='utf-8') as file:
            annotation = json.load(file)
    except OSError as exc:
        raise FileNotFoundError(
            f'Cannot read annotation file: {annotation_path}') from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f'Invalid JSON in {annotation_path}: {exc}') from exc

    if not isinstance(annotation, dict):
        raise TypeError(f'COCO annotation must be an object: {annotation_path}')
    for key in ('images', 'annotations', 'categories'):
        if not isinstance(annotation.get(key), list):
            raise TypeError(
                f'{annotation_path}: {key!r} must be a list in a COCO file.')
    return annotation


def validate_categories(annotation: dict[str, Any], annotation_path: Path) -> tuple[str, ...]:
    """Return the source category order after checking the COCO label space."""
    category_ids: set[int] = set()
    names: set[str] = set()
    classes: list[str] = []
    for index, category in enumerate(annotation['categories']):
        if not isinstance(category, dict):
            raise TypeError(f'{annotation_path}: category {index} is not an object.')
        category_id = category.get('id')
        name = category.get('name')
        if not isinstance(category_id, int) or not isinstance(name, str) or not name:
            raise TypeError(
                f'{annotation_path}: category {index} needs an integer id and '
                'a non-empty name.')
        if category_id in category_ids or name in names:
            raise ValueError(
                f'{annotation_path}: duplicate category id or name at index {index}.')
        category_ids.add(category_id)
        names.add(name)
        classes.append(name)

    if not classes:
        raise ValueError(f'{annotation_path}: no categories found.')

    image_ids = {
        image.get('id') for image in annotation['images']
        if isinstance(image, dict) and isinstance(image.get('id'), int)
    }
    if len(image_ids) != len(annotation['images']):
        raise ValueError(f'{annotation_path}: every image needs a unique integer id.')
    for index, item in enumerate(annotation['annotations']):
        if not isinstance(item, dict):
            raise TypeError(f'{annotation_path}: annotation {index} is not an object.')
        if item.get('image_id') not in image_ids:
            raise ValueError(
                f'{annotation_path}: annotation {index} references missing '
                f'image_id={item.get("image_id")!r}.')
        if item.get('category_id') not in category_ids:
            raise ValueError(
                f'{annotation_path}: annotation {index} references invalid '
                f'category_id={item.get("category_id")!r}.')
    return tuple(classes)


def validate_image_paths(images: Iterable[dict[str, Any]], image_root: Path,
                         dataset_name: str) -> None:
    """Fail early with a useful error if an annotation/image root mismatches."""
    missing: list[str] = []
    for image in images:
        file_name = image.get('file_name') if isinstance(image, dict) else None
        if not isinstance(file_name, str) or not file_name:
            missing.append(f'<invalid file_name: {file_name!r}>')
        elif not (image_root / file_name).is_file():
            missing.append(file_name)
        if len(missing) == 10:
            break
    if missing:
        examples = ', '.join(repr(value) for value in missing)
        raise FileNotFoundError(
            f'{dataset_name}: images were not found below {image_root}. '
            f'Examples: {examples}')


def write_class_prompts(work_dir: Path, dataset_name: str,
                        classes: tuple[str, ...]) -> Path:
    """Create the ``LoadText`` prompt table paired to this local label space."""
    prompt_dir = work_dir / 'generated_class_texts'
    prompt_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = prompt_dir / f'{dataset_name.lower()}_classes.json'
    prompt_path.write_text(
        json.dumps([[class_name] for class_name in classes],
                   ensure_ascii=False, indent=2),
        encoding='utf-8')
    return prompt_path


def build_dataset(data_root: Path, annotation_path: Path, image_prefix: str,
                  classes: tuple[str, ...], prompt_path: Path,
                  pipeline: list[dict[str, Any]]) -> dict[str, Any]:
    """Build one local-vocabulary multimodal COCO test dataset."""
    try:
        ann_file = str(annotation_path.relative_to(data_root))
    except ValueError:
        ann_file = str(annotation_path)

    return dict(
        type='MultiModalDataset',
        dataset=dict(
            type='LAEYOLOv5CocoDataset',
            data_root=str(data_root),
            ann_file=ann_file,
            data_prefix=dict(img=image_prefix),
            metainfo=dict(classes=classes),
            class_text_path=str(prompt_path),
            strict_category_validation=True,
            test_mode=True,
            batch_shapes_cfg=None),
        class_text_path=str(prompt_path),
        pipeline=copy.deepcopy(pipeline))


def make_runner(config: Any):
    from mmengine.runner import Runner
    from mmyolo.registry import RUNNERS

    return RUNNERS.build(config) if 'runner_type' in config else Runner.from_cfg(config)


def serializable_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    return {str(key): float(value) for key, value in metrics.items()}


def main() -> None:
    args = parse_args()
    os.chdir(PROJECT_ROOT)
    os.environ.setdefault('LOCAL_RANK', str(args.local_rank))

    data_root = project_path(args.data_root).resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f'LAE data root not found: {data_root}')
    work_dir = project_path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    requested = (
        (DATASETS[0], args.dior_ann, args.dior_img_prefix),
        (DATASETS[1], args.dotav2_ann, args.dotav2_img_prefix),
    )
    prepared: list[tuple[DatasetSpec, Path, str, tuple[str, ...]]] = []
    for spec, annotation_arg, image_prefix in requested:
        annotation_path = data_path(data_root, annotation_arg).resolve()
        image_root = data_path(data_root, image_prefix).resolve()
        annotation = load_annotation(annotation_path)
        classes = validate_categories(annotation, annotation_path)
        validate_image_paths(annotation['images'], image_root, spec.name)
        prepared.append((spec, annotation_path, image_prefix, classes))
        print(f'{spec.name}: {len(annotation["images"])} images, '
              f'{len(annotation["annotations"])} annotations, '
              f'{len(classes)} classes')

    if args.check_only:
        print('Dataset checks passed; inference was skipped (--check-only).')
        return

    if args.checkpoint is None:
        raise ValueError('--checkpoint is required unless --check-only is used.')
    checkpoint_path = project_path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')

    from mmengine.config import Config
    from mmyolo.utils import is_metainfo_lower

    config_path = project_path(args.config)
    if not config_path.is_file():
        raise FileNotFoundError(f'Config not found: {config_path}')
    base_config = Config.fromfile(str(config_path))
    base_config.launcher = args.launcher
    is_metainfo_lower(base_config)
    if 'test_dataloader' not in base_config:
        raise KeyError(f'{config_path} does not define test_dataloader.')
    if 'test_pipeline' not in base_config:
        raise KeyError(f'{config_path} does not define test_pipeline.')

    all_metrics: dict[str, dict[str, float]] = {}
    for spec, annotation_path, image_prefix, classes in prepared:
        prompt_path = write_class_prompts(work_dir, spec.name, classes)
        run_config = copy.deepcopy(base_config)
        run_config.load_from = str(checkpoint_path.resolve())
        run_config.work_dir = str((work_dir / spec.name).resolve())
        run_config.test_dataloader = copy.deepcopy(base_config.test_dataloader)
        run_config.test_dataloader['dataset'] = build_dataset(
            data_root, annotation_path, image_prefix, classes, prompt_path,
            copy.deepcopy(base_config.test_pipeline))
        run_config.test_evaluator = dict(
            type='mmdet.CocoMetric', ann_file=str(annotation_path), metric='bbox')

        print(f'\n===== Testing {spec.name} =====')
        runner = make_runner(run_config)
        metrics = runner.test()
        all_metrics[spec.name] = serializable_metrics(metrics)
        del runner
        try:
            import torch
            torch.cuda.empty_cache()
        except ImportError:
            pass

    output_path = work_dir / 'lae_fod_dior_dotav2_metrics.json'
    output_path.write_text(
        json.dumps(
            dict(checkpoint=str(checkpoint_path), config=str(config_path),
                 metrics=all_metrics),
            ensure_ascii=False, indent=2),
        encoding='utf-8')
    print(f'\nSaved complete COCO metrics to: {output_path}')


if __name__ == '__main__':
    main()

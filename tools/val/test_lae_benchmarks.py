"""Evaluate one YOLO-World baseline checkpoint on the three LAE benchmarks.

The script reads the classes directly from each supplied COCO annotation JSON,
then writes matching prompt JSON files under ``--work-dir``.  It therefore
does not rely on manually maintained category-text files and evaluates the
exact label vocabulary in every benchmark.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

# Running ``python tools/val/test_lae_benchmarks.py`` makes Python add only
# ``tools/val`` to sys.path.  Insert the repository root so Config.fromfile()
# can import ``yolo_world`` without requiring a manually exported PYTHONPATH.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mmengine.config import Config
from mmengine.runner import Runner
from mmyolo.registry import RUNNERS
from mmyolo.utils import is_metainfo_lower


@dataclass(frozen=True)
class BenchmarkSpec:
    """One LAE benchmark and the paper metric to report for it."""

    name: str
    annotation: str
    metric_suffix: str
    image_prefix: Optional[str]
    default_image_prefixes: tuple[str, ...]


BENCHMARKS = (
    BenchmarkSpec(
        name='DIOR',
        annotation='LAE-FOD/DIOR/DIOR_val.json',
        metric_suffix='bbox_mAP_50',
        image_prefix=None,
        default_image_prefixes=(
            'LAE-FOD/DIOR/trainval_images/JPEGImages-trainval/',
            'LAE-FOD/DIOR/images/images/',
            'LAE-FOD/DIOR/images/',
            'LAE-FOD/DIOR/')),
    BenchmarkSpec(
        name='DOTAv2.0',
        annotation='LAE-FOD/DOTAv2/DOTAv2_val.json',
        metric_suffix='bbox_mAP',
        image_prefix=None,
        default_image_prefixes=(
            'LAE-FOD/DOTAv2/images/images/',
            'LAE-FOD/DOTAv2/images/',
            'LAE-FOD/DOTAv2/')),
    BenchmarkSpec(
        name='LAE-80C',
        annotation='LAE-80C/LAE-80C-benchmark.json',
        metric_suffix='bbox_mAP',
        image_prefix=None,
        default_image_prefixes=(
            'LAE-80C/images/images/',
            'LAE-80C/images/',
            'LAE-80C/')),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Evaluate a YOLO-World baseline on DIOR, DOTAv2.0, and '
                    'LAE-80C and write a paper-style result table.')
    parser.add_argument(
        '--config', default='configs/pretrain/DIOR_DOTAbaseline.py',
        help='Model config matching the checkpoint.')
    parser.add_argument(
        '--checkpoint',
        default=(
            'weights/yolo_world_v2_l_clip_large_o365v1_goldg_pretrain_800ft-9df82e55.pth'),
        help='Checkpoint to evaluate.')
    parser.add_argument(
        '--data-root', default='data/LAE-1M',
        help='Directory containing LAE-FOD/ and LAE-80C/.')
    parser.add_argument('--dior-ann', default=BENCHMARKS[0].annotation)
    parser.add_argument('--dotav2-ann', default=BENCHMARKS[1].annotation)
    parser.add_argument('--lae80c-ann', default=BENCHMARKS[2].annotation)
    parser.add_argument(
        '--dior-img-prefix', default=None,
        help='Override DIOR image prefix relative to --data-root.')
    parser.add_argument(
        '--dotav2-img-prefix', default=None,
        help='Override DOTAv2.0 image prefix relative to --data-root.')
    parser.add_argument(
        '--lae80c-img-prefix', default=None,
        help='Override LAE-80C image prefix relative to --data-root.')
    parser.add_argument(
        '--method-name', default='YOLO-World v2-L baseline',
        help='Method label in the output table.')
    parser.add_argument(
        '--pretraining-data', default='O365, GoldG',
        help='Pre-training-data label in the output table.')
    parser.add_argument(
        '--work-dir', default='work_dirs/lae_baseline_benchmarks')
    parser.add_argument(
        '--format-only', action='store_true',
        help='Regenerate the aligned Markdown/TSV table from an existing '
             'lae_baseline_benchmark_metrics.json without rerunning inference.')
    parser.add_argument(
        '--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none')
    parser.add_argument('--local_rank', type=int, default=0)
    return parser.parse_args()


def absolute_path(root: Path, path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def read_annotation(annotation_path: Path) -> dict[str, Any]:
    try:
        with annotation_path.open('r', encoding='utf-8') as file:
            annotation = json.load(file)
    except OSError as exc:
        raise FileNotFoundError(
            f'Cannot read benchmark annotation: {annotation_path}') from exc
    if not isinstance(annotation, dict):
        raise TypeError(f'COCO annotation must be a JSON object: {annotation_path}')
    if not isinstance(annotation.get('images'), list):
        raise TypeError(f'Missing list-valued images field: {annotation_path}')
    if not isinstance(annotation.get('categories'), list):
        raise TypeError(f'Missing list-valued categories field: {annotation_path}')
    return annotation


def category_names(annotation: dict[str, Any], annotation_path: Path) -> tuple[str, ...]:
    names = []
    ids = set()
    for index, category in enumerate(annotation['categories']):
        if not isinstance(category, dict):
            raise TypeError(f'Invalid category {index} in {annotation_path}.')
        category_id = category.get('id')
        name = category.get('name')
        if not isinstance(category_id, int) or not isinstance(name, str) or not name:
            raise TypeError(
                f'Category {index} in {annotation_path} needs an integer id '
                'and non-empty string name.')
        if category_id in ids or name in names:
            raise ValueError(f'Duplicate category id/name in {annotation_path}.')
        ids.add(category_id)
        names.append(name)
    if not names:
        raise ValueError(f'No categories found in {annotation_path}.')
    return tuple(names)


def file_exists_for_prefix(data_root: Path, prefix: str,
                           image_records: Iterable[dict[str, Any]]) -> bool:
    """Test a representative sample before committing to an image prefix."""
    records = list(image_records)
    if not records:
        return False
    for record in records[:min(32, len(records))]:
        file_name = record.get('file_name')
        if not isinstance(file_name, str) or not file_name:
            return False
        if not (data_root / prefix / file_name).is_file():
            return False
    return True


def choose_image_prefix(data_root: Path, annotation: dict[str, Any],
                        override: Optional[str],
                        candidates: tuple[str, ...],
                        benchmark_name: str) -> str:
    choices = (override,) if override is not None else candidates
    for prefix in choices:
        if prefix is not None and file_exists_for_prefix(
                data_root, prefix, annotation['images']):
            return prefix

    sample = annotation['images'][0].get('file_name', '<missing file_name>')
    attempted = ', '.join(str(data_root / prefix) for prefix in choices)
    raise FileNotFoundError(
        f'{benchmark_name}: cannot find images such as {sample!r}. Tried: '
        f'{attempted}. Pass the correct --*-img-prefix explicitly.')


def write_prompt_file(work_dir: Path, benchmark_name: str,
                      classes: tuple[str, ...]) -> Path:
    prompt_dir = work_dir / 'generated_class_texts'
    prompt_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = prompt_dir / f'{benchmark_name.lower().replace(".", "_")}_classes.json'
    prompt_path.write_text(
        json.dumps([[name] for name in classes], ensure_ascii=False, indent=2),
        encoding='utf-8')
    return prompt_path


def test_pipeline(config: Config) -> list[dict[str, Any]]:
    if 'test_pipeline' not in config:
        raise KeyError(f'{config.filename} must define test_pipeline.')
    return copy.deepcopy(config.test_pipeline)


def benchmark_dataset(data_root: Path, annotation_path: Path,
                      image_prefix: str, classes: tuple[str, ...],
                      prompt_path: Path, pipeline: list[dict[str, Any]]) -> dict:
    """Build a zero-shot multimodal COCO dataset for one LAE benchmark."""
    try:
        ann_file = str(annotation_path.relative_to(data_root))
    except ValueError:
        # An absolute external annotation path is valid too; MMEngine accepts
        # it without prefixing ``data_root``.
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


def build_runner(config: Config):
    if 'runner_type' in config:
        return RUNNERS.build(config)
    return Runner.from_cfg(config)


def metric_value(metrics: dict[str, Any], suffix: str) -> float:
    values = [value for key, value in metrics.items()
              if str(key).split('/')[-1] == suffix]
    if len(values) != 1:
        raise KeyError(
            f'Expected exactly one metric ending in {suffix!r}; got '
            f'{values}. Available metrics: {sorted(metrics)}')
    return float(values[0])


def percent(value: float) -> float:
    return round(value * 100.0, 1)


def render_markdown_table(method_name: str, pretraining_data: str,
                          results: dict[str, float]) -> str:
    headers = (
        'Method', 'Pre-Training Data', 'DIOR (AP50)', 'DOTAv2.0 (mAP)',
        'LAE-80C (mAP)')
    values = (
        method_name, pretraining_data, f'{results["DIOR"]:.1f}',
        f'{results["DOTAv2.0"]:.1f}', f'{results["LAE-80C"]:.1f}')
    widths = tuple(max(len(header), len(value))
                   for header, value in zip(headers, values))

    def row(cells: tuple[str, ...]) -> str:
        return '| ' + ' | '.join(
            value.ljust(width) for value, width in zip(cells, widths)) + ' |'

    # Keep the Markdown separator so the file renders as a table in editors,
    # while padding all cells so ``cat`` also displays aligned columns.
    separator = '| ' + ' | '.join('-' * width for width in widths) + ' |'
    return '\n'.join((row(headers), separator, row(values), ''))


def write_result_tables(work_dir: Path, method_name: str,
                        pretraining_data: str,
                        results: dict[str, float]) -> tuple[Path, Path, str]:
    """Write the human-readable and spreadsheet result tables."""
    markdown_table = render_markdown_table(method_name, pretraining_data, results)
    markdown_path = work_dir / 'lae_baseline_benchmark_metrics.md'
    markdown_path.write_text(markdown_table, encoding='utf-8')
    tsv_path = work_dir / 'lae_baseline_benchmark_metrics.tsv'
    tsv_path.write_text(
        'Method\tPre-Training Data\tDIOR (AP50)\tDOTAv2.0 (mAP)\t'
        'LAE-80C (mAP)\n'
        f'{method_name}\t{pretraining_data}\t'
        f'{results["DIOR"]:.1f}\t{results["DOTAv2.0"]:.1f}\t'
        f'{results["LAE-80C"]:.1f}\n',
        encoding='utf-8')
    return markdown_path, tsv_path, markdown_table


def reformat_existing_results(work_dir: Path) -> None:
    """Regenerate only the presentation files after a formatting update."""
    json_path = work_dir / 'lae_baseline_benchmark_metrics.json'
    try:
        payload = json.loads(json_path.read_text(encoding='utf-8'))
        results = payload['reported_metrics_percent']
        method_name = payload['method']
        pretraining_data = payload['pretraining_data']
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f'Cannot format existing results from {json_path}. Run the full '
            'benchmark once first.') from exc
    required = {'DIOR', 'DOTAv2.0', 'LAE-80C'}
    if set(results) != required:
        raise ValueError(
            f'Unexpected result keys in {json_path}: {sorted(results)}')
    normalized = {key: float(value) for key, value in results.items()}
    markdown_path, tsv_path, markdown_table = write_result_tables(
        work_dir, str(method_name), str(pretraining_data), normalized)
    print(f'\n{markdown_table}')
    print(f'Reformatted results:\n  {markdown_path}\n  {tsv_path}')


def main() -> None:
    args = parse_args()
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    if args.format_only:
        reformat_existing_results(work_dir)
        return

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')
    data_root = Path(args.data_root).resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f'LAE data root not found: {data_root}')

    base_config = Config.fromfile(args.config)
    base_config.launcher = args.launcher
    is_metainfo_lower(base_config)

    annotations = (args.dior_ann, args.dotav2_ann, args.lae80c_ann)
    overrides = (args.dior_img_prefix, args.dotav2_img_prefix,
                 args.lae80c_img_prefix)
    raw_metrics: dict[str, dict[str, float]] = {}
    result_values: dict[str, float] = {}

    for template, annotation, override in zip(BENCHMARKS, annotations, overrides):
        annotation_path = absolute_path(data_root, annotation)
        annotation_data = read_annotation(annotation_path)
        classes = category_names(annotation_data, annotation_path)
        image_prefix = choose_image_prefix(
            data_root, annotation_data, override,
            template.default_image_prefixes, template.name)
        prompt_path = write_prompt_file(work_dir, template.name, classes)
        dataset = benchmark_dataset(
            data_root, annotation_path, image_prefix, classes, prompt_path,
            test_pipeline(base_config))

        run_config = copy.deepcopy(base_config)
        run_config.load_from = str(checkpoint_path)
        run_config.work_dir = str(work_dir / template.name)
        # Preserve the inherited loader's sampler, batch size, and workers;
        # only replace the benchmark dataset.
        run_config.test_dataloader = copy.deepcopy(base_config.test_dataloader)
        run_config.test_dataloader['dataset'] = dataset
        run_config.test_evaluator = dict(
            type='mmdet.CocoMetric', ann_file=str(annotation_path), metric='bbox')

        print(f'\n===== {template.name}: {len(classes)} classes, '
              f'images under {image_prefix} =====')
        runner = build_runner(run_config)
        metrics = runner.test()
        raw_metrics[template.name] = {
            str(key): float(value) for key, value in metrics.items()}
        result_values[template.name] = percent(
            metric_value(metrics, template.metric_suffix))
        del runner
        try:
            import torch
            torch.cuda.empty_cache()
        except ImportError:
            pass

    output = {
        'method': args.method_name,
        'pretraining_data': args.pretraining_data,
        'reported_metrics_percent': result_values,
        'raw_metrics': raw_metrics,
        'checkpoint': str(checkpoint_path),
        'data_root': str(data_root),
    }
    json_path = work_dir / 'lae_baseline_benchmark_metrics.json'
    json_path.write_text(json.dumps(output, ensure_ascii=False, indent=2),
                         encoding='utf-8')
    markdown_path, tsv_path, markdown_table = write_result_tables(
        work_dir, args.method_name, args.pretraining_data, result_values)
    print(f'\n{markdown_table}')
    print('Wrote results to:\n'
          f'  {markdown_path}\n  {tsv_path}\n  {json_path}')


if __name__ == '__main__':
    main()

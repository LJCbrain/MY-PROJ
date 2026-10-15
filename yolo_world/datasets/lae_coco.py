"""Strict COCO dataset adapter for the LAE-1M annotations.

LAE-1M provides a per-source class-text table.  The normal COCO dataset
loader maps COCO category ids to those local zero-based labels, but xView has
one inconsistent annotation file: its categories are numbered 0..59 while
its annotations are numbered 1..60.  This adapter validates the complete
annotation/text contract before construction and applies the explicitly
configured annotation-id offset only while parsing, leaving source data
untouched.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

from mmyolo.datasets.yolov5_coco import YOLOv5CocoDataset
from mmyolo.registry import DATASETS


def _as_text(entry: Any, source: Path, index: int) -> str:
    """Extract a single class prompt from an LAE class-text entry."""
    if isinstance(entry, str):
        text = entry
    elif (isinstance(entry, list) and len(entry) == 1
          and isinstance(entry[0], str)):
        text = entry[0]
    else:
        raise TypeError(
            f'Invalid class-text entry at index {index} in {source}. '
            'Expected a string or a one-item string list.')
    if not text:
        raise ValueError(f'Empty class text at index {index} in {source}.')
    return text


@DATASETS.register_module()
class LAEYOLOv5CocoDataset(YOLOv5CocoDataset):
    """YOLOv5 COCO dataset with strict LAE category/text validation.

    Args:
        class_text_path: JSON class-text table paired with ``ann_file``.
        annotation_category_id_offset: Offset applied to every raw annotation
            category id before the standard COCO parser maps it to the local
            zero-based label.  xView must use ``-1``; all other LAE files use
            the default ``0``.
        strict_category_validation: Validate every category and annotation at
            startup.  It should remain enabled for LAE training.
    """

    def __init__(self,
                 *args,
                 class_text_path: str,
                 annotation_category_id_offset: int = 0,
                 strict_category_validation: bool = True,
                 **kwargs) -> None:
        self.class_text_path = class_text_path
        self.annotation_category_id_offset = int(
            annotation_category_id_offset)
        self.strict_category_validation = bool(strict_category_validation)
        self._lae_num_classes = None
        super().__init__(*args, **kwargs)

    @staticmethod
    def _read_json(path: str) -> Dict[str, Any]:
        source = Path(path)
        try:
            with source.open('r', encoding='utf-8') as file:
                payload = json.load(file)
        except OSError as exc:
            raise FileNotFoundError(
                f'Cannot read LAE annotation/text file: {source}') from exc
        if not isinstance(payload, dict):
            raise TypeError(f'LAE annotation file must be a JSON object: {source}')
        return payload

    def _resolve_path(self, path: str) -> Path:
        candidate = Path(path)
        if candidate.is_absolute() or candidate.exists():
            return candidate
        return Path(self.data_root) / candidate

    def _load_class_texts(self) -> List[str]:
        text_path = self._resolve_path(self.class_text_path)
        try:
            with text_path.open('r', encoding='utf-8') as file:
                raw_texts = json.load(file)
        except OSError as exc:
            raise FileNotFoundError(
                f'Cannot read LAE class-text file: {text_path}') from exc
        if not isinstance(raw_texts, list):
            raise TypeError(
                f'LAE class-text file must be a JSON list: {text_path}')
        return [_as_text(entry, text_path, index)
                for index, entry in enumerate(raw_texts)]

    def _validate_annotation_contract(self) -> None:
        ann_path = self._resolve_path(self.ann_file)
        annotation = self._read_json(str(ann_path))
        class_texts = self._load_class_texts()
        categories = annotation.get('categories')
        annotations = annotation.get('annotations')
        images = annotation.get('images')
        if not isinstance(categories, list) or not isinstance(annotations, list):
            raise TypeError(
                f'LAE annotations require list-valued categories and '
                f'annotations: {ann_path}')
        if not isinstance(images, list):
            raise TypeError(
                f'LAE annotations require a list-valued images field: {ann_path}')

        category_ids = []
        category_names = []
        for index, category in enumerate(categories):
            if not isinstance(category, dict):
                raise TypeError(
                    f'Invalid category at index {index} in {ann_path}.')
            category_id = category.get('id')
            category_name = category.get('name')
            if not isinstance(category_id, int) or not isinstance(category_name,
                                                                  str):
                raise TypeError(
                    f'Category {index} in {ann_path} needs integer id and '
                    'string name.')
            category_ids.append(category_id)
            category_names.append(category_name)

        if len(category_ids) != len(set(category_ids)):
            raise ValueError(f'Duplicate category ids in {ann_path}.')
        if len(category_names) != len(class_texts):
            raise ValueError(
                f'Category/text count mismatch for {ann_path}: '
                f'{len(category_names)} categories vs {len(class_texts)} texts.')
        if category_names != class_texts:
            mismatch = next(index for index, (category_name, text) in enumerate(
                zip(category_names, class_texts)) if category_name != text)
            raise ValueError(
                f'Category/text order mismatch for {ann_path} at index '
                f'{mismatch}: category={category_names[mismatch]!r}, '
                f'text={class_texts[mismatch]!r}.')

        configured_classes: Sequence[str] = self.metainfo.get('classes', ())
        if tuple(class_texts) != tuple(configured_classes):
            raise ValueError(
                f'metainfo.classes does not exactly match {self.class_text_path} '
                f'for {ann_path}.')

        image_ids = set()
        for image in images:
            if not isinstance(image, dict) or not isinstance(image.get('id'), int):
                raise TypeError(f'Every image in {ann_path} must have an integer id.')
            image_ids.add(image['id'])

        valid_category_ids = set(category_ids)
        invalid_records = []
        observed_category_ids = set()
        for annotation_index, item in enumerate(annotations):
            if not isinstance(item, dict):
                raise TypeError(
                    f'Invalid annotation at index {annotation_index} in {ann_path}.')
            raw_category_id = item.get('category_id')
            image_id = item.get('image_id')
            if not isinstance(raw_category_id, int):
                raise TypeError(
                    f'Annotation {annotation_index} in {ann_path} has a '
                    'non-integer category_id.')
            if image_id not in image_ids:
                raise ValueError(
                    f'Annotation {annotation_index} in {ann_path} references '
                    f'missing image_id={image_id}.')
            mapped_category_id = (raw_category_id +
                                  self.annotation_category_id_offset)
            observed_category_ids.add(mapped_category_id)
            if mapped_category_id not in valid_category_ids:
                invalid_records.append((annotation_index, image_id,
                                        raw_category_id, mapped_category_id))
                if len(invalid_records) == 5:
                    break
        if invalid_records:
            raise ValueError(
                f'Invalid LAE category mapping for {ann_path}; offset='
                f'{self.annotation_category_id_offset}, examples='
                f'{invalid_records}.')
        missing_category_ids = valid_category_ids - observed_category_ids
        if missing_category_ids:
            raise ValueError(
                f'Categories without annotations in {ann_path}: '
                f'{sorted(missing_category_ids)[:10]}.')

        self._lae_num_classes = len(class_texts)

    def load_data_list(self) -> List[dict]:
        if self.strict_category_validation:
            self._validate_annotation_contract()
        return super().load_data_list()

    def parse_data_info(self, raw_data_info: dict):
        """Apply the validated raw-id offset before standard COCO parsing."""
        if self.annotation_category_id_offset == 0:
            parsed = super().parse_data_info(raw_data_info)
        else:
            shifted_annotations = []
            for annotation in raw_data_info['raw_ann_info']:
                shifted = dict(annotation)
                shifted['category_id'] = (
                    int(shifted['category_id']) +
                    self.annotation_category_id_offset)
                shifted_annotations.append(shifted)
            parsed_input = dict(raw_data_info)
            parsed_input['raw_ann_info'] = shifted_annotations
            parsed = super().parse_data_info(parsed_input)

        if self._lae_num_classes is not None:
            for instance in parsed['instances']:
                label = int(instance['bbox_label'])
                if label < 0 or label >= self._lae_num_classes:
                    raise IndexError(
                        'LAE parser emitted an out-of-range local label: '
                        f'label={label}, class_count={self._lae_num_classes}, '
                        f'ann_file={self.ann_file}.')
        return parsed

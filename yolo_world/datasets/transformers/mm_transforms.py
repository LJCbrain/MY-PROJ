# Copyright (c) Tencent Inc. All rights reserved.
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import random
from typing import Tuple

import numpy as np
from mmyolo.registry import TRANSFORMS
from mmcv.transforms import BaseTransform


@TRANSFORMS.register_module()
class RandomLoadText:

    def __init__(self,
                 text_path: str = None,
                 prompt_format: str = '{}',
                 num_neg_samples: Tuple[int, int] = (80, 80),
                 max_num_samples: int = 80,
                 padding_to_max: bool = False,
                 padding_value: str = '') -> None:
        self.prompt_format = prompt_format
        self.num_neg_samples = num_neg_samples
        self.max_num_samples = max_num_samples
        self.padding_to_max = padding_to_max
        self.padding_value = padding_value
        if text_path is not None:
            with open(text_path, 'r') as f:
                self.class_texts = json.load(f)

    def __call__(self, results: dict) -> dict:
        assert 'texts' in results or hasattr(self, 'class_texts'), (
            'No texts found in results.')
        class_texts = results.get(
            'texts',
            getattr(self, 'class_texts', None))

        num_classes = len(class_texts)
        if 'gt_labels' in results:
            gt_label_tag = 'gt_labels'
        elif 'gt_bboxes_labels' in results:
            gt_label_tag = 'gt_bboxes_labels'
        else:
            raise ValueError('No valid labels found in results.')
        positive_labels = set(results[gt_label_tag])
        if len(positive_labels) > self.max_num_samples:
            positive_labels = set(random.sample(list(positive_labels),
                                                k=self.max_num_samples))
        num_neg_samples = min(
            min(num_classes, self.max_num_samples) - len(positive_labels),
            random.randint(*self.num_neg_samples))
        candidate_neg_labels = []
        for idx in range(num_classes):
            if idx not in positive_labels:
                candidate_neg_labels.append(idx)
        negative_labels = random.sample(
            candidate_neg_labels, k=num_neg_samples)

        sampled_labels = list(positive_labels) + list(negative_labels)
        random.shuffle(sampled_labels)

        label2ids = {label: i for i, label in enumerate(sampled_labels)}

        gt_valid_mask = np.zeros(len(results['gt_bboxes']), dtype=bool)
        for idx, label in enumerate(results[gt_label_tag]):
            if label in label2ids:
                gt_valid_mask[idx] = True
                results[gt_label_tag][idx] = label2ids[label]
        results['gt_bboxes'] = results['gt_bboxes'][gt_valid_mask]
        results[gt_label_tag] = results[gt_label_tag][gt_valid_mask]

        if 'instances' in results:
            retaged_instances = []
            for idx, inst in enumerate(results['instances']):
                label = inst['bbox_label']
                if label in label2ids:
                    inst['bbox_label'] = label2ids[label]
                    retaged_instances.append(inst)
            results['instances'] = retaged_instances

        texts = []
        for label in sampled_labels:
            cls_caps = class_texts[label]
            assert len(cls_caps) > 0
            cap_id = random.randrange(len(cls_caps))
            sel_cls_cap = self.prompt_format.format(cls_caps[cap_id])
            texts.append(sel_cls_cap)

        if self.padding_to_max:
            num_valid_labels = len(positive_labels) + len(negative_labels)
            num_padding = self.max_num_samples - num_valid_labels
            if num_padding > 0:
                texts += [self.padding_value] * num_padding

        results['texts'] = texts

        return results


@TRANSFORMS.register_module()
class LoadText:

    def __init__(self,
                 text_path: str = None,
                 prompt_format: str = '{}',
                 multi_prompt_flag: str = '/') -> None:
        self.prompt_format = prompt_format
        self.multi_prompt_flag = multi_prompt_flag
        if text_path is not None:
            with open(text_path, 'r') as f:
                self.class_texts = json.load(f)

    def __call__(self, results: dict) -> dict:
        assert 'texts' in results or hasattr(self, 'class_texts'), (
            'No texts found in results.')
        class_texts = results.get(
            'texts',
            getattr(self, 'class_texts', None))

        texts = []
        for idx, cls_caps in enumerate(class_texts):
            assert len(cls_caps) > 0
            sel_cls_cap = cls_caps[0]
            sel_cls_cap = self.prompt_format.format(sel_cls_cap)
            texts.append(sel_cls_cap)

        results['texts'] = texts

        return results



@TRANSFORMS.register_module()
class MapDatasetToGlobal(BaseTransform):
    r"""阶段 1：将各数据集原始类名映射到 global canonical（不改 GT label\)。

    期望输入（至少满足其一）：
      \- results[dataset_key]\: str（例如 "DIOR" 或 "DOTAv2"）
      \- 或在初始化时固定 dataset_name

    期望 results['texts'] 结构：
      \- List[List[str]]，通常每项是 [class\_name]

    映射文件格式（每个数据集一份）：
      \- Dict[str, str]，key=原始类名（建议原样写或小写均可），value=global canonical

    参数：
      \- maps\_root\: 映射文件根目录（包含各数据集子目录）
      \- map\_filename\: 每个数据集的映射文件名
      \- dataset\_key\: 从 results 取数据集名的 key
      \- dataset\_name\: 固定的数据集名（可选）
      \- lowercase/strip\: 用于匹配 key 的归一化
      \- strict\: 若 True，遇到未映射类名直接报错；否则回退原名
    """

    def __init__(
        self,
        maps_root: str,
        map_filename: str = "dataset_class_to_global.json",
        dataset_key: str = "dataset_name",
        dataset_name: Optional[str] = None,
        lowercase: bool = True,
        strip: bool = True,
        strict: bool = False,
    ) -> None:
        self.maps_root = Path(maps_root)
        self.map_filename = map_filename
        self.dataset_key = dataset_key
        self.dataset_name = dataset_name
        self.lowercase = lowercase
        self.strip = strip
        self.strict = strict

        self._cache: Dict[str, Dict[str, str]] = {}

    def _norm(self, s: Any) -> Any:
        if not isinstance(s, str):
            return s
        if self.strip:
            s = s.strip()
        if self.lowercase:
            s = s.lower()
        return s

    def _load_map(self, dataset_name: str) -> Dict[str, str]:
        if dataset_name in self._cache:
            return self._cache[dataset_name]

        p = (self.maps_root / dataset_name / self.map_filename).resolve()
        if not p.exists():
            raise FileNotFoundError(f"找不到映射文件: {p}")

        with p.open("r", encoding="utf-8") as f:
            raw = json.load(f)

        if not isinstance(raw, dict):
            raise ValueError(f"映射文件必须是 dict: {p}")

        m = {
            self._norm(k): self._norm(v)
            for k, v in raw.items()
            if isinstance(k, str) and isinstance(v, str)
        }
        self._cache[dataset_name] = m
        return m

    def transform(self, results: Dict[str, Any]) -> Dict[str, Any]:
        texts: Optional[List[List[str]]] = results.get("texts", None)
        if texts is None:
            return results

        dataset_name = self.dataset_name or results.get(self.dataset_key, None)
        if not isinstance(dataset_name, str) or not dataset_name.strip():
            raise KeyError(
                f"无法获取 dataset 名称，请在 results['{self.dataset_key}'] 提供，或初始化时传入 dataset_name"
            )

        m = self._load_map(dataset_name)

        new_texts: List[List[str]] = []
        mapped_names: List[str] = []

        for item in texts:
            if not item:
                new_texts.append(item)
                mapped_names.append("")
                continue

            raw_name = item[0]
            key = self._norm(raw_name)
            if key in m:
                g = m[key]
            else:
                if self.strict:
                    raise KeyError(f"未映射类名: dataset={dataset_name}, name={raw_name}")
                g = raw_name

            new_texts.append([g])
            mapped_names.append(g)

        results["texts"] = new_texts
        results["global_texts"] = mapped_names
        results["dataset_name_used_for_map"] = dataset_name
        return results
#
# @TRANSFORMS.register_module()
# class MapGlobalToFinal(BaseTransform):
#     r"""阶段 2：将 global canonical 合并到 final canonical。
#
#     输入：
#       \- results['texts']\: List[List[str]]
#
#     映射文件格式：
#       \- Dict[str, str]，key=global canonical，value=final canonical
#
#     参数：
#       \- map\_path\: global\-\>final 映射文件路径
#       \- lowercase/strip\: 用于匹配 key 的归一化
#       \- strict\: 若 True，未命中的 global 直接报错；否则回退原名
#     """
#
#     def __init__(
#             self,
#             map_path: str,
#             lowercase: bool = True,
#             strip: bool = True,
#             strict: bool = False,
#     ) -> None:
#         self.map_path = Path(map_path)
#         self.lowercase = lowercase
#         self.strip = strip
#         self.strict = strict
#
#         if not self.map_path.exists():
#             raise FileNotFoundError(f"找不到: {self.map_path}")
#
#         with self.map_path.open("r", encoding="utf-8") as f:
#             raw = json.load(f)
#
#         if not isinstance(raw, dict):
#             raise ValueError(f"映射文件必须是 dict: {self.map_path}")
#
#         self.map = {
#             self._norm(k): self._norm(v)
#             for k, v in raw.items()
#             if isinstance(k, str) and isinstance(v, str)
#         }
#
#     def _norm(self, s: Any) -> Any:
#         if not isinstance(s, str):
#             return s
#         if self.strip:
#             s = s.strip()
#         if self.lowercase:
#             s = s.lower()
#         return s
#
#     def transform(self, results: Dict[str, Any]) -> Dict[str, Any]:
#         texts: Optional[List[List[str]]] = results.get("texts", None)
#         if texts is None:
#             return results
#
#         new_texts: List[List[str]] = []
#         final_names: List[str] = []
#
#         for item in texts:
#             if not item:
#                 new_texts.append(item)
#                 final_names.append("")
#                 continue
#
#             raw = item[0]
#             key = self._norm(raw)
#             if key in self.map:
#                 final = self.map[key]
#             else:
#                 if self.strict:
#                     raise KeyError(f"未命中 global_to_final: {raw}")
#                 final = raw
#
#             new_texts.append([final])
#             final_names.append(final)
#
#         results["texts"] = new_texts
#         results["final_texts"] = final_names
#         return results
#
# @TRANSFORMS.register_module()
# class SetDatasetName(BaseTransform):
#     """Inject dataset_name into results for downstream text transforms."""
#
#     def __init__(self, dataset_name: str, dataset_key: str = "dataset_name", overwrite: bool = True):
#         self.dataset_name = dataset_name
#         self.dataset_key = dataset_key
#         self.overwrite = overwrite
#
#     def transform(self, results: dict) -> dict:
#         if self.overwrite or (self.dataset_key not in results):
#             results[self.dataset_key] = self.dataset_name
#         return results
#
#     def __repr__(self) -> str:
#         return f"{self.__class__.__name__}(dataset_name={self.dataset_name}, dataset_key={self.dataset_key}, overwrite={self.overwrite})"
#
# @TRANSFORMS.register_module()
# class LimitTrainClasses(BaseTransform):
#     """Limit texts/classes and gt_labels for training.
#
#     Expected keys in results:
#       - `texts`: list[str] or None
#       - `gt_bboxes`: (N, 4) array-like (optional)
#       - `gt_labels`: (N,) array-like (optional)
#     """
#
#     def __init__(
#         self,
#         num_train_classes: int,
#         padding_to_max: bool = True,
#         padding_value: str = "",
#         drop_invalid_labels: bool = True,
#     ) -> None:
#         self.num_train_classes = int(num_train_classes)
#         self.padding_to_max = bool(padding_to_max)
#         self.padding_value = padding_value
#         self.drop_invalid_labels = bool(drop_invalid_labels)
#
#     def transform(self, results: Dict[str, Any]) -> Optional[Dict[str, Any]]:
#         k = self.num_train_classes
#
#         # 1) texts: truncate/pad to k
#         texts = results.get("texts", None)
#         if texts is not None:
#             texts = list(texts)
#             if len(texts) >= k:
#                 texts = texts[:k]
#             elif self.padding_to_max:
#                 texts = texts + [self.padding_value] * (k - len(texts))
#             results["texts"] = texts
#
#         # 2) gt_labels: keep only [0, k-1]
#         if "gt_labels" in results and results["gt_labels"] is not None:
#             labels = results["gt_labels"]
#             labels = np.asarray(labels, dtype=np.int64)
#
#             valid = (labels >= 0) & (labels < k)
#             if not valid.all():
#                 if self.drop_invalid_labels:
#                     # filter bboxes/masks along with labels if they exist
#                     results["gt_labels"] = labels[valid]
#                     if "gt_bboxes" in results and results["gt_bboxes"] is not None:
#                         bboxes = np.asarray(results["gt_bboxes"])
#                         results["gt_bboxes"] = bboxes[valid]
#                     if "gt_bboxes_labels" in results and results["gt_bboxes_labels"] is not None:
#                         results["gt_bboxes_labels"] = np.asarray(results["gt_bboxes_labels"])[valid]
#                 else:
#                     # clamp into range (not recommended, but prevents index error)
#                     results["gt_labels"] = np.clip(labels, 0, k - 1)
#
#         return results
#
#     def __repr__(self) -> str:
#         return (
#             f"{self.__class__.__name__}(num_train_classes={self.num_train_classes}, "
#             f"padding_to_max={self.padding_to_max}, drop_invalid_labels={self.drop_invalid_labels})"
#         )

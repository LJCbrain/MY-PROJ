# Copyright (c) Tencent Inc. All rights reserved.
import copy
import json
import logging
from typing import Callable, List, Union

from mmengine.logging import print_log
from mmengine.dataset.base_dataset import (
        BaseDataset, Compose, force_full_init)
from mmyolo.registry import DATASETS


@DATASETS.register_module()
class MultiModalDataset:
    """Multi-modal dataset."""

    def __init__(self,
                 dataset: Union[BaseDataset, dict],
                 class_text_path: str = None,
                 test_mode: bool = True,
                 pipeline: List[Union[dict, Callable]] = [],
                 lazy_init: bool = False) -> None:
        self.dataset: BaseDataset
        if isinstance(dataset, dict):
            self.dataset = DATASETS.build(dataset)
        elif isinstance(dataset, BaseDataset):
            self.dataset = dataset
        else:
            raise TypeError(
                'dataset must be a dict or a BaseDataset, '
                f'but got {dataset}')

        self.class_text_path = class_text_path  # 解决文本类别问题：保留文本类别文件路径用于报错定位
        if class_text_path is not None:  # 若提供了类别文本路径
            self.class_texts = json.load(open(class_text_path, 'r'))  # 读取 JSON 得到类别文本
            ori_classes = self.dataset.metainfo.get('classes', None)
            if ori_classes is not None and len(ori_classes) != len(self.class_texts):
                raise ValueError(
                    'Class/text mismatch in MultiModalDataset: '
                    f'{len(ori_classes)} dataset classes but '
                    f'{len(self.class_texts)} text entries in '
                    f'{class_text_path}.')  # 解决文本类别问题：确保数据集类别数和文本类别数一致
            # ori_classes = self.dataset.metainfo['classes']
            # assert len(ori_classes) == len(self.class_texts), \
            #     ('The number of classes in the dataset and the class text'
            #      'file must be the same.')
        else:
            self.class_texts = None

        self.test_mode = test_mode
        self._metainfo = self.dataset.metainfo
        self.pipeline = Compose(pipeline)

        self._fully_initialized = False
        if not lazy_init:
            self.full_init()

    @property
    def metainfo(self) -> dict:  # 返回元信息字典
        return copy.deepcopy(self._metainfo)  # 返回深拷贝，避免外部修改内部状态

    def _validate_class_texts(self) -> None:
        if self.class_texts is None:
            return
        ori_classes = self.dataset.metainfo.get('classes', None)
        if ori_classes is not None and len(ori_classes) != len(self.class_texts):
            raise ValueError(
                'Class/text mismatch in MultiModalDataset: '
                f'{len(ori_classes)} dataset classes but '
                f'{len(self.class_texts)} text entries in '
                f'{self.class_text_path}.')  # 解决文本类别问题：确保 full_init 后数据集类别数和文本类别数一致

    def full_init(self) -> None:  # 完整初始化数据集
        """``full_init`` dataset."""  # 方法文档：执行 full_init
        if self._fully_initialized:
            return

        self.dataset.full_init()  # 调用底层数据集初始化
        self._validate_class_texts()  # 解决文本类别问题：底层数据集初始化后再次校验类别/text 对齐
        self._ori_len = len(self.dataset)  # 记录底层数据集原始长度
        self._fully_initialized = True

    @force_full_init
    def get_data_info(self, idx: int) -> dict:
        """Get annotation by index."""
        data_info = self.dataset.get_data_info(idx)  # 从底层数据集获取 data_info
        if self.class_texts is not None:  # 若存在类别文本
            data_info.update({'texts': self.class_texts})  # 注入 texts 字段给后续流程使用
            data_info.update({'class_text_path': self.class_text_path})  # 解决文本类别问题：传递文本类别文件路径用于越界报错
        return data_info  # 返回 data_info

    def __getitem__(self, idx):
        if not self._fully_initialized:
            print_log(
                'Please call `full_init` method manually to '
                'accelerate the speed.',
                logger='current',
                level=logging.WARNING)
            self.full_init()

        data_info = self.get_data_info(idx)

        if hasattr(self.dataset, 'test_mode') and not self.dataset.test_mode:
            data_info['dataset'] = self
        elif not self.test_mode:
            data_info['dataset'] = self
        return self.pipeline(data_info)

    @force_full_init
    def __len__(self) -> int:
        return self._ori_len


@DATASETS.register_module()
class MultiModalMixedDataset(MultiModalDataset):
    """Multi-modal Mixed dataset.
    mix "detection dataset" and "caption dataset"
    Args:
        dataset_type (str): dataset type, 'detection' or 'caption'
    """
    def __init__(self,
                 dataset: Union[BaseDataset, dict],
                 class_text_path: str = None,
                 dataset_type: str = 'detection',
                 test_mode: bool = True,
                 pipeline: List[Union[dict, Callable]] = [],
                 lazy_init: bool = False) -> None:
        self.dataset_type = dataset_type
        super().__init__(dataset,
                         class_text_path,
                         test_mode,
                         pipeline,
                         lazy_init)

    @force_full_init
    def get_data_info(self, idx: int) -> dict:
        """Get annotation by index."""
        data_info = self.dataset.get_data_info(idx)  # 从底层数据集获取 data_info
        if self.class_texts is not None:  # 若存在类别文本
            data_info.update({'texts': self.class_texts})  # 注入 texts 字段
            data_info.update({'class_text_path': self.class_text_path})  # 解决文本类别问题：传递文本类别文件路径用于越界报错
        data_info['is_detection'] = 1 \
            if self.dataset_type == 'detection' else 0
        return data_info

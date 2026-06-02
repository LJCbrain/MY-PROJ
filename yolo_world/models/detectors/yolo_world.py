# Copyright (c) Tencent Inc. All rights reserved.  # 版权声明

from typing import List, Tuple, Union  # �?typing 导入类型注解工具
import torch  # 导入 PyTorch 主包
import torch.nn as nn  # 导入 PyTorch 神经网络模块并简写为 nn
import torch.nn.functional as F
from torch import Tensor  # 直接导入 Tensor 类型
from mmdet.structures import OptSampleList, SampleList  # 导入 mmdet 中样本列表类型（可�?必选）
from mmyolo.models.detectors import YOLODetector  # �?mmyolo 导入基础 YOLO 检测器
from mmyolo.registry import MODELS  # 导入模型注册器，用于注册自定义检测器q


@MODELS.register_module()  # 将该类注册到 MODELS 注册表中，方便通过配置构建
class YOLOWorldDetector(YOLODetector):  # 定义 YOLOWorldDetector 类，继承 YOLODetector
    """Implementation of YOLOW Series"""  # YOLOW 系列的实现说�?
    def __init__(self,  # 构造函数，初始化检测器
                 *args,  # 位置参数，透传给父�?
                 mm_neck: bool = False,  # 是否使用多模�?neck 的标�?
                 num_train_classes=80,  # 训练阶段类别�?
                 num_test_classes=80,  # 测试阶段类别�?
                 loss_fgbg_bce_weight: float = 0.0,
                 loss_fgbg_dice_weight: float = 0.0,
                 loss_fgbg_consistency_weight: float = 0.0,
                 **kwargs) -> None:  # 关键字参数，透传给父�?/-> None 表示这个函数不会返回任何有意义的�?
        self.mm_neck = mm_neck  # 保存是否使用多模�?neck 的配�?
        self.num_train_classes = num_train_classes  # 保存训练类别�?
        self.num_test_classes = num_test_classes  # 保存测试类别�?
        self.loss_fgbg_bce_weight = loss_fgbg_bce_weight
        self.loss_fgbg_dice_weight = loss_fgbg_dice_weight
        self.loss_fgbg_consistency_weight = loss_fgbg_consistency_weight
        super().__init__(*args, **kwargs)  # 调用父类构造函数完成基础初始�?

    def loss(self, batch_inputs: Tensor,  # 定义 loss 计算函数，batch_inputs 为图像张�?
             batch_data_samples: SampleList) -> Union[dict, list]:  # batch_data_samples 为标注信息列表，返回字典或列表形式的损失
                                                                    #SampleList = List[DetDataSample]
        """Calculate losses from a batch of inputs and data samples."""  # 文档字符串：根据一批数据计算损�?
        self.bbox_head.num_classes = self.num_train_classes  # 设置检测头的类别数为训练类别数
        img_feats, txt_feats, txt_masks = self.extract_feat(  # 调用特征提取函数，获取图像和文本特征以及文本 mask
            batch_inputs, batch_data_samples)  # 传入图像和样本信�?
        losses = self.bbox_head.loss(img_feats, txt_feats, txt_masks,  # 调用检测头�?loss 函数计算损失
                                     batch_data_samples)  # 传入样本列表作为标签
        losses.update(self._loss_fgbg(batch_inputs, batch_data_samples))
        return losses  # 返回损失结果

    def _loss_fgbg(self, batch_inputs: Tensor,
                   batch_data_samples: SampleList) -> dict:
        """Compute class-agnostic foreground auxiliary losses for training."""
        if not self.with_neck or not hasattr(self.neck, '_fgbg_parts'):
            return {}
        fgbg_parts = self.neck._fgbg_parts
        if not fgbg_parts or 'masks' not in fgbg_parts:
            return {}

        pred_masks = fgbg_parts['masks']
        if len(pred_masks) == 0:
            return {}

        input_shape = batch_inputs.shape[-2:]
        target_masks = [
            self._build_bbox_pseudo_masks(batch_data_samples, mask.shape[-2:],
                                          input_shape, batch_inputs.shape[0],
                                          mask.device, mask.dtype)
            for mask in pred_masks
        ]

        losses = {}
        if self.loss_fgbg_bce_weight > 0:
            bce_loss = sum(
                F.binary_cross_entropy(mask, target)
                for mask, target in zip(pred_masks, target_masks))
            losses['loss_fgbg_bce'] = (
                bce_loss / len(pred_masks) * self.loss_fgbg_bce_weight)

        if self.loss_fgbg_dice_weight > 0:
            dice_loss = sum(
                self._dice_loss(mask, target)
                for mask, target in zip(pred_masks, target_masks))
            losses['loss_fgbg_dice'] = (
                dice_loss / len(pred_masks) * self.loss_fgbg_dice_weight)

        if self.loss_fgbg_consistency_weight > 0 and len(pred_masks) > 1:
            consistency = 0.0
            for src, dst in zip(pred_masks[:-1], pred_masks[1:]):
                src = F.interpolate(
                    src,
                    size=dst.shape[-2:],
                    mode='bilinear',
                    align_corners=False)
                consistency = consistency + F.l1_loss(src, dst.detach())
            losses['loss_fgbg_consistency'] = (
                consistency / (len(pred_masks) - 1) *
                self.loss_fgbg_consistency_weight)

        return losses

    def _build_bbox_pseudo_masks(self, batch_data_samples: SampleList,
                                 feat_shape: Tuple[int, int],
                                 input_shape: Tuple[int, int],
                                 batch_size: int,
                                 device: torch.device,
                                 dtype: torch.dtype) -> Tensor:
        """Rasterize GT boxes into feature-level class-agnostic masks."""
        feat_h, feat_w = feat_shape
        input_h, input_w = input_shape
        masks = torch.zeros(
            (batch_size, 1, feat_h, feat_w),
            device=device,
            dtype=dtype)

        scale_x = feat_w / max(float(input_w), 1.0)
        scale_y = feat_h / max(float(input_h), 1.0)

        if isinstance(batch_data_samples, dict):
            bboxes_labels = batch_data_samples.get('bboxes_labels', None)
            if bboxes_labels is None or bboxes_labels.numel() == 0:
                return masks
            bboxes_labels = bboxes_labels.to(device=device, dtype=dtype)
            batch_indices = bboxes_labels[:, 0].long()
            boxes = bboxes_labels[:, 2:6].clone()
            boxes[:, 0::2] = boxes[:, 0::2] * scale_x
            boxes[:, 1::2] = boxes[:, 1::2] * scale_y
            x1 = boxes[:, 0].floor().clamp(0, feat_w - 1).long()
            y1 = boxes[:, 1].floor().clamp(0, feat_h - 1).long()
            x2 = boxes[:, 2].ceil().clamp(1, feat_w).long()
            y2 = boxes[:, 3].ceil().clamp(1, feat_h).long()
            for batch_idx, left, top, right, bottom in zip(
                    batch_indices, x1, y1, x2, y2):
                if 0 <= batch_idx < batch_size and right > left and bottom > top:
                    masks[batch_idx, 0, top:bottom, left:right] = 1.0
            return masks

        for batch_idx, data_sample in enumerate(batch_data_samples):
            if not hasattr(data_sample, 'gt_instances'):
                continue
            bboxes = data_sample.gt_instances.bboxes
            if len(bboxes) == 0:
                continue
            boxes = bboxes.tensor if hasattr(bboxes, 'tensor') else bboxes
            boxes = boxes.to(device=device, dtype=dtype).clone()
            boxes[:, 0::2] = boxes[:, 0::2] * scale_x
            boxes[:, 1::2] = boxes[:, 1::2] * scale_y
            x1 = boxes[:, 0].floor().clamp(0, feat_w - 1).long()
            y1 = boxes[:, 1].floor().clamp(0, feat_h - 1).long()
            x2 = boxes[:, 2].ceil().clamp(1, feat_w).long()
            y2 = boxes[:, 3].ceil().clamp(1, feat_h).long()
            for left, top, right, bottom in zip(x1, y1, x2, y2):
                if right > left and bottom > top:
                    masks[batch_idx, 0, top:bottom, left:right] = 1.0
        return masks

    @staticmethod
    def _dice_loss(pred_mask: Tensor, target_mask: Tensor) -> Tensor:
        pred = pred_mask.flatten(1)
        target = target_mask.flatten(1)
        intersection = (pred * target).sum(dim=1)
        denominator = pred.sum(dim=1) + target.sum(dim=1)
        return (1 - (2 * intersection + 1.0) / (denominator + 1.0)).mean()
    def predict(self,  # 定义预测函数
                batch_inputs: Tensor,  # 输入图像张量
                batch_data_samples: SampleList,  # 输入样本元信息（�?img_meta 等）
                rescale: bool = True) -> SampleList:  # 是否在预测后进行缩放恢复，返回预测后�?SampleList
        """Predict results from a batch of inputs and data samples with post-
        processing.
        """  # 文档字符串：根据一批数据进行前向推理并做后处理

        img_feats, txt_feats, txt_masks = self.extract_feat(  # 提取图像和文本特�?
            batch_inputs, batch_data_samples)  # 传入图像和样本元信息

        # self.bbox_head.num_classes = self.num_test_classes  # 原始逻辑：将类别数设为预定义测试类别数（目前注释掉）
        self.bbox_head.num_classes = txt_feats[0].shape[0]  # 根据当前文本特征的类别维度动态设�?num_classes
        results_list = self.bbox_head.predict(img_feats,  # 使用检测头进行预测
                                              txt_feats,  # 文本特征作为分类参�?
                                              txt_masks,  # 文本 mask，标记有效文�?
                                              batch_data_samples,  # 样本元信�?
                                              rescale=rescale)  # 是否对输出框进行尺度还原

        batch_data_samples = self.add_pred_to_datasample(  # 将预测结果写回到 batch_data_samples �?
            batch_data_samples, results_list)  # 传入原始样本信息和预测列�?
        return batch_data_samples  # 返回包含预测结果的样本列�?

    def reparameterize(self, texts: List[List[str]]) -> None:  # 重新参数化函数，将文本编码为特征并缓�?
        # encode text embeddings into the detector  # 注释：将文本嵌入编码并缓存到检测器�?
        self.texts = texts  # 缓存文本字符串列�?
        self.text_feats, _ = self.backbone.forward_text(texts)  # 通过 backbone 的文本分支编码文本，得到文本特征并缓�?

    def _forward(  # 定义不带后处理的前向函数（用于导�?调试�?
            self,
            batch_inputs: Tensor,  # 输入图像
            batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:  # 可选的样本信息，返回前向结果张量列�?
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.
        """  # 文档字符串：仅做网络前向，不带后处理
        img_feats, txt_feats, txt_masks = self.extract_feat(  # 先提取特�?
            batch_inputs, batch_data_samples)  # 传入图像和样本信�?
        results = self.bbox_head.forward(img_feats, txt_feats, txt_masks)  # 调用检测头前向，输出中间结�?

        return results  # 返回前向输出

    def extract_feat(  # 定义特征提取函数
            self, batch_inputs: Tensor,  # 输入图像张量
            batch_data_samples: SampleList) -> Tuple[Tuple[Tensor], Tensor]:  # 输入样本信息，返�?(img_feats, txt_feats, txt_masks)
        """Extract features."""  # 文档字符串：提取图像和文本特�?
        #
        # txt_masks = None#修改部分
        txt_feats = None  # 初始化文本特征为 None
        if batch_data_samples is None:  # 如果没有传入样本信息
            texts = self.texts  # 使用缓存的文本列�?
            txt_feats = self.text_feats  # 使用缓存的文本特�?
        elif isinstance(batch_data_samples,  # 如果样本信息是字典并且包�?'texts' 字段
                        dict) and 'texts' in batch_data_samples:
            texts = batch_data_samples['texts']  # 从字典中取出文本列表
        elif isinstance(batch_data_samples, list) and hasattr(  # 如果样本信息是列表并且元素包�?texts 属�?
                batch_data_samples[0], 'texts'):
            texts = [data_sample.texts for data_sample in batch_data_samples]  # 从每个样本中收集 texts 列表
        elif hasattr(self, 'text_feats'):  # 否则如果检测器已经缓存了文本特�?
            texts = self.texts  # 使用缓存文本
            txt_feats = self.text_feats  # 使用缓存文本特征
        else:  # 以上情况都不满足
            raise TypeError('batch_data_samples should be dict or list.')  # 抛出类型错误，提示输入不合法
        if txt_feats is not None:  # 如果已经有文本特征（缓存�?
            # forward image only  # 注释：只需要前向图像分�?
            img_feats = self.backbone.forward_image(batch_inputs)  # 使用 backbone 的图像分支前向，得到图像特征
        else:  # 否则需要同时前向图像和文本
            img_feats, (txt_feats,  # �?backbone 返回图像特征�?(文本特征, 文本 mask)
                        txt_masks) = self.backbone(batch_inputs, texts)  # 将图像和文本一起送入 backbone
        if self.with_neck:  # 如果模型配置�?neck
            if self.mm_neck:  # 如果使用多模�?neck
                img_feats = self.neck(img_feats, txt_feats)  # neck 同时接收图像和文本特�?
            else:  # 普�?neck
                img_feats = self.neck(img_feats)  # neck 只接收图像特�?
        return img_feats, txt_feats, txt_masks  # 返回图像特征、文本特征和文本 mask


@MODELS.register_module()  # �?SimpleYOLOWorldDetector 注册�?MODELS
class SimpleYOLOWorldDetector(YOLODetector):  # 简化版 YOLO World 检测器，继�?YOLODetector
    """Implementation of YOLO World Series"""  # 文档字符串：YOLO World 的一个实�?
    def __init__(self,  # 构造函�?
                 *args,  # 透传的位置参�?
                 mm_neck: bool = False,  # 是否使用多模�?neck
                 num_train_classes=80,  # 训练类别�?
                 num_test_classes=80,  # 测试类别�?
                 prompt_dim=512,  # 文本/提示 embedding 维度
                 num_prompts=80,  # 提示 token（类别）数量
                 embedding_path='',  # 预存�?embedding 路径（可选）
                 reparameterized=False,  # 是否已经将文本特征重参数化到检测头�?
                 freeze_prompt=False,  # 是否冻结 prompt 参数
                 use_mlp_adapter=False,  # 是否使用 MLP adapter �?prompt 做映�?
                 **kwargs) -> None:  # 其他关键字参�?
        self.mm_neck = mm_neck  # 保存是否使用多模�?neck 的配�?
        self.num_training_classes = num_train_classes  # 训练阶段类别数（注意属性名与上一个类略有区别�?
        self.num_test_classes = num_test_classes  # 测试阶段类别�?
        self.prompt_dim = prompt_dim  # 保存 prompt 维度
        self.num_prompts = num_prompts  # 保存 prompt 数量
        self.reparameterized = reparameterized  # 标记是否重参数化
        self.freeze_prompt = freeze_prompt  # 是否冻结 prompt 梯度
        self.use_mlp_adapter = use_mlp_adapter  # 是否使用 MLP adapter
        super().__init__(*args, **kwargs)  # 调用父类构造函�?

        if not self.reparameterized:  # 如果还没有重参数化，需要显式管�?embeddings
            if len(embedding_path) > 0:  # 如果提供了预训练 embedding 路径
                import numpy as np  # 延迟导入 numpy
                self.embeddings = torch.nn.Parameter(  # �?.npy 文件加载 embedding，并注册为可学习参数
                    torch.from_numpy(np.load(embedding_path)).float())  # 读取 npy 文件并转�?float32 Tensor
            else:  # 否则随机初始�?embeddings
                # random init  # 注释：随机初始化 embeddings
                embeddings = nn.functional.normalize(torch.randn(  # 生成正态分布随机向量并�?L2 归一�?
                    (num_prompts, prompt_dim)),  # 张量形状�?(类别�? 维度)
                                                     dim=-1)  # 在最后一维上归一�?
                self.embeddings = nn.Parameter(embeddings)  # 将随机生成的 embeddings 注册为可学习参数

            if self.freeze_prompt:  # 如果配置冻结 prompt
                self.embeddings.requires_grad = False  # 关闭 embeddings 的梯�?
            else:  # 否则允许优化
                self.embeddings.requires_grad = True  # 打开 embeddings 的梯�?

            if use_mlp_adapter:  # 如果需要使�?MLP adapter
                self.adapter = nn.Sequential(  # 定义一个两层的 MLP 适配�?
                    nn.Linear(prompt_dim, prompt_dim * 2), nn.ReLU(True),  # 线性映射到 2 倍维度并�?ReLU
                    nn.Linear(prompt_dim * 2, prompt_dim))  # 再映射回原始 prompt 维度
            else:  # 不使�?adapter
                self.adapter = None  # 适配器置�?None

    def  loss(self, batch_inputs: Tensor,  # 定义损失计算函数
             batch_data_samples: SampleList) -> Union[dict, list]:  # 输入样本列表，返回损�?
        """Calculate losses from a batch of inputs and data samples."""  # 文档字符串：计算一批样本的损失
        self.bbox_head.num_classes = self.num_training_classes  # 设置训练阶段的类别数
        img_feats, txt_feats = self.extract_feat(batch_inputs,  # 提取图像和（可能的）文本特征
                                                 batch_data_samples)  # 传入图像和样本列�?
        if self.reparameterized:  # 如果已经重参数化
            losses = self.bbox_head.loss(img_feats, batch_data_samples)  # 检测头只需要图像特征和标签
        else:  # 未重参数化，需要传入文本特�?
            losses = self.bbox_head.loss(img_feats, txt_feats,  # 检测头使用图像+文本特征一起计算损�?
                                         batch_data_samples)  # 传入标签数据
        return losses  # 返回损失

    def predict(self,  # 定义预测函数
                batch_inputs: Tensor,  # 输入图像
                batch_data_samples: SampleList,  # 样本元信�?
                rescale: bool = True) -> SampleList:  # 是否做尺度还原，返回预测后的样本列表
        """Predict results from a batch of inputs and data samples with post-
        processing.
        """  # 文档字符串：带后处理的预�?

        img_feats, txt_feats = self.extract_feat(batch_inputs,  # 提取特征
                                                 batch_data_samples)  # 传入图像和样本信�?


        self.bbox_head.num_classes = self.num_test_classes  # 设置检测头的类别数为测试类别数
        if self.reparameterized:  # 如果已经重参数化
            results_list = self.bbox_head.predict(img_feats,  # 只用图像特征做预�?
                                                  batch_data_samples,  # 样本元信�?
                                                  rescale=rescale)  # 是否还原尺度
        else:  # 未重参数�?
            results_list = self.bbox_head.predict(img_feats,  # 同时使用图像和文本特�?
                                                  txt_feats,  # 文本特征用作类别原型
                                                  batch_data_samples,  # 样本元信�?
                                                  rescale=rescale)  # 是否还原尺度

        batch_data_samples = self.add_pred_to_datasample(  # 将预测结果写回数据样�?
            batch_data_samples, results_list)  # 传入原始样本和预测结�?
        return batch_data_samples  # 返回包含预测的样本列�?

    def _forward(  # 定义不带后处理的前向函数
            self,
            batch_inputs: Tensor,  # 输入图像
            batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:  # 可选的样本信息，返回结果张量列�?
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.
        """  # 文档字符串：只做网络前向，不做后处理
        img_feats, txt_feats = self.extract_feat(batch_inputs,  # 提取特征
                                                 batch_data_samples)  # 传入图像和样本信�?
        if self.reparameterized:  # 如果已重参数�?
            results = self.bbox_head.forward(img_feats)  # 检测头只需要图像特征前�?
        else:  # 否则仍需文本特征
            results = self.bbox_head.forward(img_feats, txt_feats)  # 检测头使用图像+文本特征前向
        return results  # 返回前向输出

    def extract_feat(  # 定义特征提取函数
            self, batch_inputs: Tensor,  # 输入图像
            batch_data_samples: SampleList) -> Tuple[Tuple[Tensor], Tensor]:  # 返回 (img_feats, txt_feats)
        """Extract features."""  # 文档字符串：提取特征
        # only image features  # 注释：只�?backbone 中提取图像特�?
        img_feats, _ = self.backbone(batch_inputs, None)  # 调用 backbone，第二个参数�?None 表示不处理文�?

        if not self.reparameterized:  # 如果还未重参数化
            # use embeddings  # 注释：使用内部维护的 embeddings 作为文本特征
            txt_feats = self.embeddings[None]  # �?batch 维度上扩一维，形状变成 (1, num_prompts, prompt_dim)
            if self.adapter is not None:  # 如果配置�?MLP adapter
                txt_feats = self.adapter(txt_feats) + txt_feats  # 通过 adapter 得到增量特征并与原始特征相加（残差连接）
                txt_feats = nn.functional.normalize(txt_feats, dim=-1, p=2)  # 对最后一维做 L2 归一�?
            txt_feats = txt_feats.repeat(img_feats[0].shape[0], 1, 1)  # �?batch 维度上复制到与图�?batch 大小一�?
        else:  # 已重参数�?
            txt_feats = None  # 不再显式使用文本特征
        if self.with_neck:  # 如果定义�?neck
            if self.mm_neck:  # 使用多模�?neck
                img_feats = self.neck(img_feats, txt_feats)  # neck 同时接收图像和文本特�?
            else:  # 普�?neck
                img_feats = self.neck(img_feats)  # neck 只接收图像特�?
        return img_feats, txt_feats  # 返回图像特征和文本特�?




# Copyright (c) Tencent Inc. All rights reserved.  # 版权声明

from typing import List, Tuple, Union  # 从 typing 导入类型注解工具
import torch  # 导入 PyTorch 主包
import torch.nn as nn  # 导入 PyTorch 神经网络模块并简写为 nn
from torch import Tensor  # 直接导入 Tensor 类型
from mmdet.structures import OptSampleList, SampleList  # 导入 mmdet 中样本列表类型（可选/必选）
from mmyolo.models.detectors import YOLODetector  # 从 mmyolo 导入基础 YOLO 检测器
from mmyolo.registry import MODELS  # 导入模型注册器，用于注册自定义检测器q


@MODELS.register_module()  # 将该类注册到 MODELS 注册表中，方便通过配置构建
class YOLOWorldDetector(YOLODetector):  # 定义 YOLOWorldDetector 类，继承 YOLODetector
    """Implementation of YOLOW Series"""  # YOLOW 系列的实现说明
    def __init__(self,  # 构造函数，初始化检测器
                 *args,  # 位置参数，透传给父类
                 mm_neck: bool = False,  # 是否使用多模态 neck 的标志
                 num_train_classes=80,  # 训练阶段类别数
                 num_test_classes=80,  # 测试阶段类别数
                 **kwargs) -> None:  # 关键字参数，透传给父类//-> None 表示这个函数不会返回任何有意义的值
        self.mm_neck = mm_neck  # 保存是否使用多模态 neck 的配置
        self.num_train_classes = num_train_classes  # 保存训练类别数
        self.num_test_classes = num_test_classes  # 保存测试类别数
        super().__init__(*args, **kwargs)  # 调用父类构造函数完成基础初始化

    def loss(self, batch_inputs: Tensor,  # 定义 loss 计算函数，batch_inputs 为图像张量
             batch_data_samples: SampleList) -> Union[dict, list]:  # batch_data_samples 为标注信息列表，返回字典或列表形式的损失
                                                                    #SampleList = List[DetDataSample]
        """Calculate losses from a batch of inputs and data samples."""  # 文档字符串：根据一批数据计算损失
        self.bbox_head.num_classes = self.num_train_classes  # 设置检测头的类别数为训练类别数
        img_feats, txt_feats, txt_masks = self.extract_feat(  # 调用特征提取函数，获取图像和文本特征以及文本 mask
            batch_inputs, batch_data_samples)  # 传入图像和样本信息
        losses = self.bbox_head.loss(img_feats, txt_feats, txt_masks,  # 调用检测头的 loss 函数计算损失
                                     batch_data_samples)  # 传入样本列表作为标签
        return losses  # 返回损失结果

    def predict(self,  # 定义预测函数
                batch_inputs: Tensor,  # 输入图像张量
                batch_data_samples: SampleList,  # 输入样本元信息（如 img_meta 等）
                rescale: bool = True) -> SampleList:  # 是否在预测后进行缩放恢复，返回预测后的 SampleList
        """Predict results from a batch of inputs and data samples with post-
        processing.
        """  # 文档字符串：根据一批数据进行前向推理并做后处理

        img_feats, txt_feats, txt_masks = self.extract_feat(  # 提取图像和文本特征 
            batch_inputs, batch_data_samples)  # 传入图像和样本元信息

        # self.bbox_head.num_classes = self.num_test_classes  # 原始逻辑：将类别数设为预定义测试类别数（目前注释掉）
        self.bbox_head.num_classes = txt_feats[0].shape[0]  # 根据当前文本特征的类别维度动态设置 num_classes
        results_list = self.bbox_head.predict(img_feats,  # 使用检测头进行预测
                                              txt_feats,  # 文本特征作为分类参考
                                              txt_masks,  # 文本 mask，标记有效文本
                                              batch_data_samples,  # 样本元信息
                                              rescale=rescale)  # 是否对输出框进行尺度还原

        batch_data_samples = self.add_pred_to_datasample(  # 将预测结果写回到 batch_data_samples 中
            batch_data_samples, results_list)  # 传入原始样本信息和预测列表
        return batch_data_samples  # 返回包含预测结果的样本列表

    def reparameterize(self, texts: List[List[str]]) -> None:  # 重新参数化函数，将文本编码为特征并缓存
        # encode text embeddings into the detector  # 注释：将文本嵌入编码并缓存到检测器中
        self.texts = texts  # 缓存文本字符串列表
        self.text_feats, _ = self.backbone.forward_text(texts)  # 通过 backbone 的文本分支编码文本，得到文本特征并缓存

    def _forward(  # 定义不带后处理的前向函数（用于导出/调试）
            self,
            batch_inputs: Tensor,  # 输入图像
            batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:  # 可选的样本信息，返回前向结果张量列表
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.
        """  # 文档字符串：仅做网络前向，不带后处理
        img_feats, txt_feats, txt_masks = self.extract_feat(  # 先提取特征
            batch_inputs, batch_data_samples)  # 传入图像和样本信息
        results = self.bbox_head.forward(img_feats, txt_feats, txt_masks)  # 调用检测头前向，输出中间结果

        return results  # 返回前向输出

    def extract_feat(  # 定义特征提取函数
            self, batch_inputs: Tensor,  # 输入图像张量
            batch_data_samples: SampleList) -> Tuple[Tuple[Tensor], Tensor]:  # 输入样本信息，返回 (img_feats, txt_feats, txt_masks)
        """Extract features."""  # 文档字符串：提取图像和文本特征
        #
        # txt_masks = None#修改部分
        txt_feats = None  # 初始化文本特征为 None
        if batch_data_samples is None:  # 如果没有传入样本信息
            texts = self.texts  # 使用缓存的文本列表
            txt_feats = self.text_feats  # 使用缓存的文本特征
        elif isinstance(batch_data_samples,  # 如果样本信息是字典并且包含 'texts' 字段
                        dict) and 'texts' in batch_data_samples:
            texts = batch_data_samples['texts']  # 从字典中取出文本列表
        elif isinstance(batch_data_samples, list) and hasattr(  # 如果样本信息是列表并且元素包含 texts 属性
                batch_data_samples[0], 'texts'):
            texts = [data_sample.texts for data_sample in batch_data_samples]  # 从每个样本中收集 texts 列表
        elif hasattr(self, 'text_feats'):  # 否则如果检测器已经缓存了文本特征
            texts = self.texts  # 使用缓存文本
            txt_feats = self.text_feats  # 使用缓存文本特征
        else:  # 以上情况都不满足
            raise TypeError('batch_data_samples should be dict or list.')  # 抛出类型错误，提示输入不合法
        if txt_feats is not None:  # 如果已经有文本特征（缓存）
            # forward image only  # 注释：只需要前向图像分支
            img_feats = self.backbone.forward_image(batch_inputs)  # 使用 backbone 的图像分支前向，得到图像特征
        else:  # 否则需要同时前向图像和文本
            img_feats, (txt_feats,  # 从 backbone 返回图像特征和 (文本特征, 文本 mask)
                        txt_masks) = self.backbone(batch_inputs, texts)  # 将图像和文本一起送入 backbone
        if self.with_neck:  # 如果模型配置了 neck
            if self.mm_neck:  # 如果使用多模态 neck
                img_feats = self.neck(img_feats, txt_feats)  # neck 同时接收图像和文本特征
            else:  # 普通 neck
                img_feats = self.neck(img_feats)  # neck 只接收图像特征
        return img_feats, txt_feats, txt_masks  # 返回图像特征、文本特征和文本 mask


@MODELS.register_module()  # 将 SimpleYOLOWorldDetector 注册进 MODELS
class SimpleYOLOWorldDetector(YOLODetector):  # 简化版 YOLO World 检测器，继承 YOLODetector
    """Implementation of YOLO World Series"""  # 文档字符串：YOLO World 的一个实现
    def __init__(self,  # 构造函数
                 *args,  # 透传的位置参数
                 mm_neck: bool = False,  # 是否使用多模态 neck
                 num_train_classes=80,  # 训练类别数
                 num_test_classes=80,  # 测试类别数
                 prompt_dim=512,  # 文本/提示 embedding 维度
                 num_prompts=80,  # 提示 token（类别）数量
                 embedding_path='',  # 预存的 embedding 路径（可选）
                 reparameterized=False,  # 是否已经将文本特征重参数化到检测头中
                 freeze_prompt=False,  # 是否冻结 prompt 参数
                 use_mlp_adapter=False,  # 是否使用 MLP adapter 对 prompt 做映射
                 **kwargs) -> None:  # 其他关键字参数
        self.mm_neck = mm_neck  # 保存是否使用多模态 neck 的配置
        self.num_training_classes = num_train_classes  # 训练阶段类别数（注意属性名与上一个类略有区别）
        self.num_test_classes = num_test_classes  # 测试阶段类别数
        self.prompt_dim = prompt_dim  # 保存 prompt 维度
        self.num_prompts = num_prompts  # 保存 prompt 数量
        self.reparameterized = reparameterized  # 标记是否重参数化
        self.freeze_prompt = freeze_prompt  # 是否冻结 prompt 梯度
        self.use_mlp_adapter = use_mlp_adapter  # 是否使用 MLP adapter
        super().__init__(*args, **kwargs)  # 调用父类构造函数

        if not self.reparameterized:  # 如果还没有重参数化，需要显式管理 embeddings
            if len(embedding_path) > 0:  # 如果提供了预训练 embedding 路径
                import numpy as np  # 延迟导入 numpy
                self.embeddings = torch.nn.Parameter(  # 从 .npy 文件加载 embedding，并注册为可学习参数
                    torch.from_numpy(np.load(embedding_path)).float())  # 读取 npy 文件并转为 float32 Tensor
            else:  # 否则随机初始化 embeddings
                # random init  # 注释：随机初始化 embeddings
                embeddings = nn.functional.normalize(torch.randn(  # 生成正态分布随机向量并做 L2 归一化
                    (num_prompts, prompt_dim)),  # 张量形状为 (类别数, 维度)
                                                     dim=-1)  # 在最后一维上归一化
                self.embeddings = nn.Parameter(embeddings)  # 将随机生成的 embeddings 注册为可学习参数

            if self.freeze_prompt:  # 如果配置冻结 prompt
                self.embeddings.requires_grad = False  # 关闭 embeddings 的梯度
            else:  # 否则允许优化
                self.embeddings.requires_grad = True  # 打开 embeddings 的梯度

            if use_mlp_adapter:  # 如果需要使用 MLP adapter
                self.adapter = nn.Sequential(  # 定义一个两层的 MLP 适配器
                    nn.Linear(prompt_dim, prompt_dim * 2), nn.ReLU(True),  # 线性映射到 2 倍维度并接 ReLU
                    nn.Linear(prompt_dim * 2, prompt_dim))  # 再映射回原始 prompt 维度
            else:  # 不使用 adapter
                self.adapter = None  # 适配器置为 None

    def  loss(self, batch_inputs: Tensor,  # 定义损失计算函数
             batch_data_samples: SampleList) -> Union[dict, list]:  # 输入样本列表，返回损失
        """Calculate losses from a batch of inputs and data samples."""  # 文档字符串：计算一批样本的损失
        self.bbox_head.num_classes = self.num_training_classes  # 设置训练阶段的类别数
        img_feats, txt_feats = self.extract_feat(batch_inputs,  # 提取图像和（可能的）文本特征
                                                 batch_data_samples)  # 传入图像和样本列表
        if self.reparameterized:  # 如果已经重参数化
            losses = self.bbox_head.loss(img_feats, batch_data_samples)  # 检测头只需要图像特征和标签
        else:  # 未重参数化，需要传入文本特征
            losses = self.bbox_head.loss(img_feats, txt_feats,  # 检测头使用图像+文本特征一起计算损失
                                         batch_data_samples)  # 传入标签数据
        return losses  # 返回损失

    def predict(self,  # 定义预测函数
                batch_inputs: Tensor,  # 输入图像
                batch_data_samples: SampleList,  # 样本元信息
                rescale: bool = True) -> SampleList:  # 是否做尺度还原，返回预测后的样本列表
        """Predict results from a batch of inputs and data samples with post-
        processing.
        """  # 文档字符串：带后处理的预测

        img_feats, txt_feats = self.extract_feat(batch_inputs,  # 提取特征
                                                 batch_data_samples)  # 传入图像和样本信息


        self.bbox_head.num_classes = self.num_test_classes  # 设置检测头的类别数为测试类别数
        if self.reparameterized:  # 如果已经重参数化
            results_list = self.bbox_head.predict(img_feats,  # 只用图像特征做预测
                                                  batch_data_samples,  # 样本元信息
                                                  rescale=rescale)  # 是否还原尺度
        else:  # 未重参数化
            results_list = self.bbox_head.predict(img_feats,  # 同时使用图像和文本特征
                                                  txt_feats,  # 文本特征用作类别原型
                                                  batch_data_samples,  # 样本元信息
                                                  rescale=rescale)  # 是否还原尺度

        batch_data_samples = self.add_pred_to_datasample(  # 将预测结果写回数据样本
            batch_data_samples, results_list)  # 传入原始样本和预测结果
        return batch_data_samples  # 返回包含预测的样本列表

    def _forward(  # 定义不带后处理的前向函数
            self,
            batch_inputs: Tensor,  # 输入图像
            batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:  # 可选的样本信息，返回结果张量列表
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.
        """  # 文档字符串：只做网络前向，不做后处理
        img_feats, txt_feats = self.extract_feat(batch_inputs,  # 提取特征
                                                 batch_data_samples)  # 传入图像和样本信息
        if self.reparameterized:  # 如果已重参数化
            results = self.bbox_head.forward(img_feats)  # 检测头只需要图像特征前向
        else:  # 否则仍需文本特征
            results = self.bbox_head.forward(img_feats, txt_feats)  # 检测头使用图像+文本特征前向
        return results  # 返回前向输出

    def extract_feat(  # 定义特征提取函数
            self, batch_inputs: Tensor,  # 输入图像
            batch_data_samples: SampleList) -> Tuple[Tuple[Tensor], Tensor]:  # 返回 (img_feats, txt_feats)
        """Extract features."""  # 文档字符串：提取特征
        # only image features  # 注释：只从 backbone 中提取图像特征
        img_feats, _ = self.backbone(batch_inputs, None)  # 调用 backbone，第二个参数传 None 表示不处理文本

        if not self.reparameterized:  # 如果还未重参数化
            # use embeddings  # 注释：使用内部维护的 embeddings 作为文本特征
            txt_feats = self.embeddings[None]  # 在 batch 维度上扩一维，形状变成 (1, num_prompts, prompt_dim)
            if self.adapter is not None:  # 如果配置了 MLP adapter
                txt_feats = self.adapter(txt_feats) + txt_feats  # 通过 adapter 得到增量特征并与原始特征相加（残差连接）
                txt_feats = nn.functional.normalize(txt_feats, dim=-1, p=2)  # 对最后一维做 L2 归一化
            txt_feats = txt_feats.repeat(img_feats[0].shape[0], 1, 1)  # 在 batch 维度上复制到与图像 batch 大小一致
        else:  # 已重参数化
            txt_feats = None  # 不再显式使用文本特征
        if self.with_neck:  # 如果定义了 neck
            if self.mm_neck:  # 使用多模态 neck
                img_feats = self.neck(img_feats, txt_feats)  # neck 同时接收图像和文本特征
            else:  # 普通 neck
                img_feats = self.neck(img_feats)  # neck 只接收图像特征
        return img_feats, txt_feats  # 返回图像特征和文本特征




# Copyright (c) Tencent Inc. All rights reserved.
import math
import copy
from typing import List, Optional, Tuple, Union, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmcv.cnn import ConvModule
from mmengine.config import ConfigDict
from mmengine.model import BaseModule
from torch import Tensor
from torch.nn.modules.batchnorm import _BatchNorm

from mmengine.dist import get_dist_info
from mmengine.structures import InstanceData
from mmdet.structures import SampleList
from mmdet.utils import OptConfigType, InstanceList, OptInstanceList
from mmdet.models.utils import (multi_apply, unpack_gt_instances,
                                filter_scores_and_topk)
from mmyolo.registry import MODELS
from mmyolo.models.dense_heads import YOLOv8HeadModule, YOLOv8Head
from mmyolo.models.utils import gt_instances_preprocess
from mmcv.cnn.bricks import build_norm_layer


@MODELS.register_module()
class ContrastiveHead(BaseModule):
    """Contrastive Head for YOLO-World
    compute the region-text scores according to the
    similarity between image and text features
    Args:
        embed_dims (int): embed dim of text and image features
    """
    def __init__(self,
                 embed_dims: int,
                 init_cfg: OptConfigType = None,
                 use_einsum: bool = True) -> None:

        super().__init__(init_cfg=init_cfg)

        self.bias = nn.Parameter(torch.zeros([]))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.use_einsum = use_einsum

    def forward(self, x: Tensor, w: Tensor) -> Tensor:
        """Forward function of contrastive learning."""
        x = F.normalize(x, dim=1, p=2)
        w = F.normalize(w, dim=-1, p=2)

        if self.use_einsum:
            x = torch.einsum('bchw,bkc->bkhw', x, w)
        else:
            batch, channel, height, width = x.shape
            _, k, _ = w.shape
            x = x.permute(0, 2, 3, 1)  # bchw->bhwc
            x = x.reshape(batch, -1, channel)  # bhwc->b(hw)c
            w = w.permute(0, 2, 1)  # bkc->bck
            x = torch.matmul(x, w)
            x = x.reshape(batch, height, width, k)
            x = x.permute(0, 3, 1, 2)

        x = x * self.logit_scale.exp() + self.bias
        return x


@MODELS.register_module()
class BNContrastiveHead(BaseModule):
    """ Batch Norm Contrastive Head for YOLO-World
    using batch norm instead of l2-normalization
    Args:
        embed_dims (int): embed dim of text and image features
        norm_cfg (dict): normalization params
    """
    def __init__(self,
                 embed_dims: int,
                 norm_cfg: ConfigDict,
                 init_cfg: OptConfigType = None,
                 use_einsum: bool = True) -> None:

        super().__init__(init_cfg=init_cfg)
        self.norm = build_norm_layer(norm_cfg, embed_dims)[1]
        self.bias = nn.Parameter(torch.zeros([]))
        # use -1.0 is more stable
        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))
        self.use_einsum = use_einsum

    def forward(self, x: Tensor, w: Tensor) -> Tensor:
        """Forward function of contrastive learning."""
        x = self.norm(x)
        w = F.normalize(w, dim=-1, p=2)

        if self.use_einsum:
            x = torch.einsum('bchw,bkc->bkhw', x, w)
        else:
            batch, channel, height, width = x.shape
            _, k, _ = w.shape
            x = x.permute(0, 2, 3, 1)  # bchw->bhwc
            x = x.reshape(batch, -1, channel)  # bhwc->b(hw)c
            w = w.permute(0, 2, 1)  # bkc->bck
            x = torch.matmul(x, w)
            x = x.reshape(batch, height, width, k)
            x = x.permute(0, 3, 1, 2)

        x = x * self.logit_scale.exp() + self.bias
        return x


@MODELS.register_module()
class RepBNContrastiveHead(BaseModule):
    """ Batch Norm Contrastive Head for YOLO-World
    using batch norm instead of l2-normalization
    Args:
        embed_dims (int): embed dim of text and image features
        norm_cfg (dict): normalization params
    """
    def __init__(self,
                 embed_dims: int,
                 num_guide_embeds: int,
                 norm_cfg: ConfigDict,
                 init_cfg: OptConfigType = None) -> None:

        super().__init__(init_cfg=init_cfg)
        self.norm = build_norm_layer(norm_cfg, embed_dims)[1]
        self.conv = nn.Conv2d(embed_dims, num_guide_embeds, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        """Forward function of contrastive learning."""
        x = self.norm(x)
        x = self.conv(x)
        return x


@MODELS.register_module()
class YOLOWorldHeadModule(YOLOv8HeadModule):
    """Head Module for YOLO-World

    Args:
        embed_dims (int): embed dim for text feautures and image features
        use_bn_head (bool): use batch normalization head
    """
    def __init__(self,
                 *args,
                 embed_dims: int,
                 use_bn_head: bool = False,
                 use_einsum: bool = True,
                 freeze_all: bool = False,
                 **kwargs) -> None:
        self.embed_dims = embed_dims
        self.use_bn_head = use_bn_head
        self.use_einsum = use_einsum
        self.freeze_all = freeze_all
        super().__init__(*args, **kwargs)

    def init_weights(self, prior_prob=0.01):
        """Initialize the weight and bias of PPYOLOE head."""
        super().init_weights()
        for cls_pred, cls_contrast, stride in zip(self.cls_preds,
                                                  self.cls_contrasts,
                                                  self.featmap_strides):
            cls_pred[-1].bias.data[:] = 0.0  # reset bias
            if hasattr(cls_contrast, 'bias'):
                nn.init.constant_(
                    cls_contrast.bias.data,
                    math.log(5 / self.num_classes / (640 / stride)**2))

    def _init_layers(self) -> None:
        """initialize conv layers in YOLOv8 head."""
        # Init decouple head
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.cls_contrasts = nn.ModuleList()

        reg_out_channels = max(
            (16, self.in_channels[0] // 4, self.reg_max * 4))
        cls_out_channels = max(self.in_channels[0], self.num_classes)

        for i in range(self.num_levels):#（8 16 32）
            self.reg_preds.append(
                nn.Sequential(
                    ConvModule(in_channels=self.in_channels[i],
                               out_channels=reg_out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               norm_cfg=self.norm_cfg,
                               act_cfg=self.act_cfg),
                    ConvModule(in_channels=reg_out_channels,
                               out_channels=reg_out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               norm_cfg=self.norm_cfg,
                               act_cfg=self.act_cfg),
                    nn.Conv2d(in_channels=reg_out_channels,
                              out_channels=4 * self.reg_max,
                              kernel_size=1)))
            self.cls_preds.append(
                nn.Sequential(
                    ConvModule(in_channels=self.in_channels[i],
                               out_channels=cls_out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               norm_cfg=self.norm_cfg,
                               act_cfg=self.act_cfg),
                    ConvModule(in_channels=cls_out_channels,
                               out_channels=cls_out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               norm_cfg=self.norm_cfg,
                               act_cfg=self.act_cfg),
                    nn.Conv2d(in_channels=cls_out_channels,
                              out_channels=self.embed_dims,
                              kernel_size=1)))
            if self.use_bn_head:
                self.cls_contrasts.append(
                    BNContrastiveHead(self.embed_dims,
                                      self.norm_cfg,
                                      use_einsum=self.use_einsum))
            else:
                self.cls_contrasts.append(
                    ContrastiveHead(self.embed_dims,
                                    use_einsum=self.use_einsum))

        proj = torch.arange(self.reg_max, dtype=torch.float)
        self.register_buffer('proj', proj, persistent=False)

        if self.freeze_all:
            self._freeze_all()

    def _freeze_all(self):
        """Freeze the model."""
        for m in self.modules():
            if isinstance(m, _BatchNorm):
                m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_all:
            self._freeze_all()

    def forward(self, img_feats: Tuple[Tensor], txt_feats: Tensor,
                txt_masks: Tensor) -> Tuple[List]:
        """Forward features from the upstream network."""
        assert len(img_feats) == self.num_levels
        txt_feats = [txt_feats for _ in range(self.num_levels)]
        txt_masks = [txt_masks for _ in range(self.num_levels)]
        return multi_apply(self.forward_single, img_feats, txt_feats,
                           txt_masks, self.cls_preds, self.reg_preds,
                           self.cls_contrasts)

    def forward_single(self, img_feat: Tensor, txt_feat: Tensor,
                       txt_masks: Tensor, cls_pred: nn.ModuleList,
                       reg_pred: nn.ModuleList,
                       cls_contrast: nn.ModuleList) -> Tuple:
        """Forward feature of a single scale level."""
        b, _, h, w = img_feat.shape
        cls_embed = cls_pred(img_feat)
        cls_logit = cls_contrast(cls_embed, txt_feat)

        if txt_masks is not None:

            # # 临时增加这一行，强制将 mask 设为 1
            # txt_masks = torch.ones_like(txt_masks)
            txt_masks = txt_masks.view(b, -1, 1, 1).expand(-1, -1, h, w)
            if self.training:
                cls_logit = cls_logit * txt_masks
                cls_logit[txt_masks == 0] = -10e6
            else:
                cls_logit[txt_masks == 0] = -10e6

        bbox_dist_preds = reg_pred(img_feat)
        if self.reg_max > 1:
            bbox_dist_preds = bbox_dist_preds.reshape(
                [-1, 4, self.reg_max, h * w]).permute(0, 3, 1, 2)

            # TODO: The get_flops script cannot handle the situation of
            #  matmul, and needs to be fixed later
            # bbox_preds = bbox_dist_preds.softmax(3).matmul(self.proj)
            bbox_preds = bbox_dist_preds.softmax(3).matmul(
                self.proj.view([-1, 1])).squeeze(-1)
            bbox_preds = bbox_preds.transpose(1, 2).reshape(b, -1, h, w)
        else:
            bbox_preds = bbox_dist_preds
        if self.training:
            return cls_logit, bbox_preds, bbox_dist_preds
        else:
            return cls_logit, bbox_preds


@MODELS.register_module()
class RepYOLOWorldHeadModule(YOLOWorldHeadModule):
    def __init__(self,
                 *args,
                 embed_dims: int,
                 num_guide: int,
                 freeze_all: bool = False,
                 **kwargs) -> None:
        super().__init__(*args,
                         embed_dims=embed_dims,
                         use_bn_head=True,
                         use_einsum=False,
                         freeze_all=freeze_all,
                         **kwargs)

        # using rep head
        cls_contrasts = []
        for _ in range(self.num_levels):
            cls_contrasts.append(
                RepBNContrastiveHead(embed_dims=embed_dims,
                                     num_guide_embeds=num_guide,
                                     norm_cfg=self.norm_cfg))
        self.cls_contrasts = nn.ModuleList(cls_contrasts)

    def forward_single(self, img_feat: Tensor, cls_pred: nn.ModuleList,
                       reg_pred: nn.ModuleList,
                       cls_contrast: nn.ModuleList) -> Tuple:
        """Forward features from the upstream network."""
        b, _, h, w = img_feat.shape
        cls_embed = cls_pred(img_feat)
        cls_logit = cls_contrast(cls_embed)
        bbox_dist_preds = reg_pred(img_feat)
        if self.reg_max > 1:
            bbox_dist_preds = bbox_dist_preds.reshape(
                [-1, 4, self.reg_max, h * w]).permute(0, 3, 1, 2)

            # TODO: The get_flops script cannot handle the situation of
            #  matmul, and needs to be fixed later
            # bbox_preds = bbox_dist_preds.softmax(3).matmul(self.proj)
            bbox_preds = bbox_dist_preds.softmax(3).matmul(
                self.proj.view([-1, 1])).squeeze(-1)
            bbox_preds = bbox_preds.transpose(1, 2).reshape(b, -1, h, w)
        else:
            bbox_preds = bbox_dist_preds
        if self.training:
            return cls_logit, bbox_preds, bbox_dist_preds
        else:
            return cls_logit, bbox_preds

    def forward(self, img_feats: Tuple[Tensor]) -> Tuple[List]:
        assert len(img_feats) == self.num_levels
        return multi_apply(self.forward_single, img_feats, self.cls_preds,
                           self.reg_preds, self.cls_contrasts)


@MODELS.register_module()
class YOLOWorldHead(YOLOv8Head):
    """YOLO-World Head
    """
    def __init__(self, world_size=-1, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.world_size = world_size

    """YOLO World v8 head."""

    def loss(self, img_feats: Tuple[Tensor], txt_feats: Tensor,  # 定义损失函数并接收图像与文本特征
             txt_masks: Tensor, batch_data_samples: Union[list, dict]) -> dict:  # 接收文本掩码与批量数据样本
        """Perform forward propagation and loss calculation of the detection head on
         the features of the upstream network."""  # 描述函数用途，执行上游网络特征的前向传播并计算检测头的损失。
        outs = self(img_feats, txt_feats, txt_masks)  # 执行前向传播获取头部输出
        # Fast version  # 使用加速版损失计算流程
        # print(batch_data_samples['bboxes_labels'])
        loss_inputs = outs + (txt_masks,
                              batch_data_samples['bboxes_labels'],
                              batch_data_samples['img_metas'])  # 拼接标注框与图像元信息作为损失输入
        losses = self.loss_by_feat(*loss_inputs)  # 基于特征输出计算具体损失
        return losses  # 返回损失字典

    def loss_and_predict(
        self,
        img_feats: Tuple[Tensor],
        txt_feats: Tensor,
        txt_masks: Tensor,
        batch_data_samples: SampleList,
        proposal_cfg: Optional[ConfigDict] = None
    ) -> Tuple[dict, InstanceList]:
        """Perform forward propagation of the head, then calculate loss and
        predictions from the features and data samples.
        """
        outputs = unpack_gt_instances(batch_data_samples)
        (batch_gt_instances, batch_gt_instances_ignore,
         batch_img_metas) = outputs

        outs = self(img_feats, txt_feats, txt_masks)

        loss_inputs = outs + (txt_masks, batch_gt_instances, batch_img_metas,
                              batch_gt_instances_ignore)
        losses = self.loss_by_feat(*loss_inputs)

        predictions = self.predict_by_feat(*outs,
                                           batch_img_metas=batch_img_metas,
                                           cfg=proposal_cfg)
        return losses, predictions

    def forward(self, img_feats: Tuple[Tensor], txt_feats: Tensor,
                txt_masks: Tensor) -> Tuple[List]:
        """Forward features from the upstream network."""
        return self.head_module(img_feats, txt_feats, txt_masks)

    def predict(self,
                img_feats: Tuple[Tensor],
                txt_feats: Tensor,
                txt_masks: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = False) -> InstanceList:
        """Perform forward propagation of the detection head and predict
        detection results on the features of the upstream network.
        执行检测头的前向传播，并在上游网络特征上预测检测结果。
        """
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        outs = self(img_feats, txt_feats, txt_masks)
        predictions = self.predict_by_feat(*outs,
                                           batch_img_metas=batch_img_metas,
                                           rescale=rescale)
        return predictions

    def aug_test(self,
                 aug_batch_feats,
                 aug_batch_img_metas,
                 rescale=False,
                 with_ori_nms=False,
                 **kwargs):
        """Test function with test time augmentation."""
        raise NotImplementedError('aug_test is not implemented yet.')

    def loss_by_feat(
            self,
            cls_scores: Sequence[Tensor],
            bbox_preds: Sequence[Tensor],
            bbox_dist_preds: Sequence[Tensor],
            batch_text_masks: Tensor,
            batch_gt_instances: Sequence[InstanceData],
            batch_img_metas: Sequence[dict],
            batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        """Calculate the loss based on the features extracted by the detection
        head.

        Args:
            cls_scores (Sequence[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_priors * num_classes.
            bbox_preds (Sequence[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_priors * 4.
            bbox_dist_preds (Sequence[Tensor]): Box distribution logits for
                each scale level with shape (bs, reg_max + 1, H*W, 4).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
        Returns:
            dict[str, Tensor]: A dictionary of losses.
        """
        num_imgs = len(batch_img_metas)

        current_featmap_sizes = [
            cls_score.shape[2:] for cls_score in cls_scores
        ]
        # If the shape does not equal, generate new one
        if current_featmap_sizes != self.featmap_sizes_train:
            self.featmap_sizes_train = current_featmap_sizes

            mlvl_priors_with_stride = self.prior_generator.grid_priors(
                self.featmap_sizes_train,
                dtype=cls_scores[0].dtype,
                device=cls_scores[0].device,
                with_stride=True)

            self.num_level_priors = [len(n) for n in mlvl_priors_with_stride]
            self.flatten_priors_train = torch.cat(mlvl_priors_with_stride,
                                                  dim=0)
            self.stride_tensor = self.flatten_priors_train[..., [2]]

        # gt info
        gt_info = gt_instances_preprocess(batch_gt_instances, num_imgs)
        gt_labels = gt_info[:, :, :1]
        gt_bboxes = gt_info[:, :, 1:]  # xyxy
        pad_bbox_flag = (gt_bboxes.sum(-1, keepdim=True) > 0).float()

        # pred info
        flatten_cls_preds = [
            cls_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                 self.num_classes)
            for cls_pred in cls_scores
        ]
        flatten_pred_bboxes = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]
        # (bs, n, 4 * reg_max)
        flatten_pred_dists = [
            bbox_pred_org.reshape(num_imgs, -1, self.head_module.reg_max * 4)
            for bbox_pred_org in bbox_dist_preds
        ]

        flatten_dist_preds = torch.cat(flatten_pred_dists, dim=1)
        flatten_cls_preds = torch.cat(flatten_cls_preds, dim=1)
        flatten_pred_bboxes = torch.cat(flatten_pred_bboxes, dim=1)
        flatten_pred_bboxes = self.bbox_coder.decode(
            self.flatten_priors_train[..., :2], flatten_pred_bboxes,
            self.stride_tensor[..., 0])

        assigned_result = self.assigner(
            (flatten_pred_bboxes.detach()).type(gt_bboxes.dtype),
            flatten_cls_preds.detach().sigmoid(), self.flatten_priors_train,
            gt_labels, gt_bboxes, pad_bbox_flag)

        assigned_bboxes = assigned_result['assigned_bboxes']
        assigned_scores = assigned_result['assigned_scores']
        fg_mask_pre_prior = assigned_result['fg_mask_pre_prior']

        assigned_scores_sum = assigned_scores.sum().clamp(min=1)

        if batch_text_masks is not None:
            cls_weight = batch_text_masks.view(num_imgs, 1, -1).expand(
                -1, flatten_cls_preds.shape[1], -1).to(flatten_cls_preds)

            loss_cls = self.loss_cls(flatten_cls_preds, assigned_scores)
            _loss_cls = (loss_cls * cls_weight).sum(dim=-1)
            loss_cls = _loss_cls.sum()
        else:
            loss_cls = self.loss_cls(flatten_cls_preds, assigned_scores).sum()
        loss_cls /= assigned_scores_sum

        # rescale bbox
        assigned_bboxes /= self.stride_tensor
        flatten_pred_bboxes /= self.stride_tensor

        # select positive samples mask
        num_pos = fg_mask_pre_prior.sum()
        if num_pos > 0:
            # when num_pos > 0, assigned_scores_sum will >0, so the loss_bbox
            # will not report an error
            # iou loss
            prior_bbox_mask = fg_mask_pre_prior.unsqueeze(-1).repeat([1, 1, 4])
            pred_bboxes_pos = torch.masked_select(
                flatten_pred_bboxes, prior_bbox_mask).reshape([-1, 4])
            assigned_bboxes_pos = torch.masked_select(
                assigned_bboxes, prior_bbox_mask).reshape([-1, 4])
            bbox_weight = torch.masked_select(assigned_scores.sum(-1),
                                              fg_mask_pre_prior).unsqueeze(-1)
            loss_bbox = self.loss_bbox(
                pred_bboxes_pos, assigned_bboxes_pos,
                weight=bbox_weight) / assigned_scores_sum

            # dfl loss
            pred_dist_pos = flatten_dist_preds[fg_mask_pre_prior]
            assigned_ltrb = self.bbox_coder.encode(
                self.flatten_priors_train[..., :2] / self.stride_tensor,
                assigned_bboxes,
                max_dis=self.head_module.reg_max - 1,
                eps=0.01)
            assigned_ltrb_pos = torch.masked_select(
                assigned_ltrb, prior_bbox_mask).reshape([-1, 4])
            loss_dfl = self.loss_dfl(pred_dist_pos.reshape(
                -1, self.head_module.reg_max),
                                     assigned_ltrb_pos.reshape(-1),
                                     weight=bbox_weight.expand(-1,
                                                               4).reshape(-1),
                                     avg_factor=assigned_scores_sum)
        else:
            loss_bbox = flatten_pred_bboxes.sum() * 0
            loss_dfl = flatten_pred_bboxes.sum() * 0
        if self.world_size == -1:
            _, world_size = get_dist_info()
        else:
            world_size = self.world_size
            world_size = self.world_size

        return dict(loss_cls=loss_cls * num_imgs * world_size,
                    loss_bbox=loss_bbox * num_imgs * world_size,
                    loss_dfl=loss_dfl * num_imgs * world_size)

    def predict_by_feat(self,  # 将头部输出特征转换为检测结果
                        cls_scores: List[Tensor],  # 分类分支特征列表
                        bbox_preds: List[Tensor],  # 回归分支特征列表
                        objectnesses: Optional[List[Tensor]] = None,  # 目标存在性分支特征
                        batch_img_metas: Optional[List[dict]] = None,  # 批次图像元信息
                        cfg: Optional[ConfigDict] = None,  # 推理配置
                        rescale: bool = True,  # 是否映射回原图尺度
                        with_nms: bool = True) -> List[InstanceData]:  # 是否执行NMS

        """Transform a batch of output features extracted by the head into bbox results."""  # 函数文档说明



        # #检测是否是由于文本特征过弱导致的分类分数过低问题
        # for cls_logit in cls_scores:
        #     # 在此处添加打印检查
        #     if cls_logit.max().item() < 0.001:  # 假设 0.001 为你的 score_thr
        #         print(f"DEBUG: Max abstract cls_logit {cls_logit.max().item()} is lower than threshold.")
        #


        assert len(cls_scores) == len(bbox_preds)  # 分类与回归预测数量应一致
        if objectnesses is None:  # 判断是否包含objectness分支
            with_objectnesses = False  # 标记缺少objectness
        else:
            with_objectnesses = True  # 标记存在objectness
            assert len(cls_scores) == len(objectnesses)  # 三分支层数需一致

        cfg = self.test_cfg if cfg is None else cfg  # 若未传入cfg则使用默认测试配置
        cfg = copy.deepcopy(cfg)  # 拷贝配置以免修改原配置

        multi_label = cfg.multi_label  # 读取multi_label参数
        multi_label &= self.num_classes > 1  # 在单类情况下禁用multi_label
        cfg.multi_label = multi_label  # 回写multi_label结果

        num_imgs = len(batch_img_metas)  # 批次图像数量
        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]  # 各层特征图尺寸

        if featmap_sizes != self.featmap_sizes:  # 尺寸变化时需重新生成先验
            self.mlvl_priors = self.prior_generator.grid_priors(  # 动态生成网格先验
                featmap_sizes,
                dtype=cls_scores[0].dtype,
                device=cls_scores[0].device)
            self.featmap_sizes = featmap_sizes  # 更新缓存的特征图尺寸
        flatten_priors = torch.cat(self.mlvl_priors)  # 拼接各尺度先验

        mlvl_strides = [  # 为每层生成步长张量
            flatten_priors.new_full(
                (featmap_size.numel() * self.num_base_priors, ), stride)  # 使用对应步长填充
            for featmap_size, stride in zip(featmap_sizes, self.featmap_strides)  # 遍历每层尺寸与步长
        ]
        flatten_stride = torch.cat(mlvl_strides)  # 展平特征步长

        flatten_cls_scores = [  # 保存展平后的分类特征
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                  self.num_classes)  # 调整维度并拉平
            for cls_score in cls_scores  # 遍历各尺度分类特征
        ]
        flatten_bbox_preds = [  # 保存展平后的回归特征
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)  # 调整维度并拉平
            for bbox_pred in bbox_preds  # 遍历各尺度回归特征
        ]

        flatten_cls_scores = torch.cat(flatten_cls_scores, dim=1).sigmoid()  # 拼接并映射分类分数到0-1
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)  # 拼接所有回归预测
        flatten_decoded_bboxes = self.bbox_coder.decode(  # 将归一化回归量解码为实际框
            flatten_priors[None], flatten_bbox_preds, flatten_stride)

        if with_objectnesses:  # 若存在objectness分支则需同样展平
            flatten_objectness = [
                objectness.permute(0, 2, 3, 1).reshape(num_imgs, -1)  # 展平objectness特征
                for objectness in objectnesses  # 遍历各尺度objectness
            ]
            flatten_objectness = torch.cat(flatten_objectness, dim=1).sigmoid()  # 拼接并归一化objectness得分
        else:
            flatten_objectness = [None for _ in range(num_imgs)]  # 不存在objectness时填充None

        results_list = []  # 初始化结果列表
        for (bboxes, scores, objectness,
             img_meta) in zip(flatten_decoded_bboxes, flatten_cls_scores,
                              flatten_objectness, batch_img_metas):  # 遍历每张图像的预测
            ori_shape = img_meta['ori_shape']  # 原图尺寸
            scale_factor = img_meta['scale_factor']  # 尺度缩放因子
            if 'pad_param' in img_meta:  # 判断是否存在padding参数
                pad_param = img_meta['pad_param']  # 读取padding信息
            else:
                pad_param = None  # 无padding信息时置空

            score_thr = cfg.get('score_thr', -1)  # 读取分数阈值
            if objectness is not None and score_thr > 0 and not cfg.get(  # 按需求使用objectness筛选
                    'yolox_style', False):
                conf_inds = objectness > score_thr  # objectness筛选掩码
                bboxes = bboxes[conf_inds, :]  # 过滤框
                scores = scores[conf_inds, :]  # 过滤分类分数
                objectness = objectness[conf_inds]  # 过滤objectness

            if objectness is not None:  # 若objectness有效则融合置信度
                scores *= objectness[:, None]  # 分类分数乘以objectness

            if scores.shape[0] == 0:  # 若没有候选框
                empty_results = InstanceData()  # 创建空结果
                empty_results.bboxes = bboxes  # 占位框
                empty_results.scores = scores[:, 0]  # 占位分数
                empty_results.labels = scores[:, 0].int()  # 占位标签
                results_list.append(empty_results)  # 保存空结果
                continue  # 处理下一张图像

            nms_pre = cfg.get('nms_pre', 100000)  # NMS前保留数量
            if cfg.multi_label is False:  # 单标签模式
                scores, labels = scores.max(1, keepdim=True)  # 取最大类分数及标签
                scores, _, keep_idxs, results = filter_scores_and_topk(  # 依据阈值筛选候选
                    scores,
                    score_thr,
                    nms_pre,
                    results=dict(labels=labels[:, 0]))
                labels = results['labels']  # 提取筛选后的标签
            else:
                scores, labels, keep_idxs, _ = filter_scores_and_topk(  # 多标签模式直接筛选
                    scores, score_thr, nms_pre)

            results = InstanceData(scores=scores,  # 根据筛选结果构建实例数据
                                   labels=labels,
                                   bboxes=bboxes[keep_idxs])

            if rescale:  # 需要映射回原图时处理
                if pad_param is not None:  # 先去除padding偏移
                    results.bboxes -= results.bboxes.new_tensor([
                        pad_param[2], pad_param[0], pad_param[2], pad_param[0]
                    ])
                results.bboxes /= results.bboxes.new_tensor(  # 按缩放因子还原
                    scale_factor).repeat((1, 2))

            if cfg.get('yolox_style', False):  # yolox模式不做max_per_img裁剪
                cfg.max_per_img = len(results)  # 使用全部候选

            results = self._bbox_post_process(results=results,  # 执行后处理（含NMS与裁剪）
                                              cfg=cfg,
                                              rescale=False,
                                              with_nms=with_nms,
                                              img_meta=img_meta)
            results.bboxes[:, 0::2].clamp_(0, ori_shape[1])  # 约束x坐标到图像宽度
            results.bboxes[:, 1::2].clamp_(0, ori_shape[0])  # 约束y坐标到图像高度

            results_list.append(results)  # 保存当前图像结果
        return results_list  # 返回全部图像的检测结果
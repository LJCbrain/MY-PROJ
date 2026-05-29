_base_ = ('../../third_party/mmyolo/configs/yolov8/'
          'yolov8_l_syncbn_fast_8xb16-500e_coco.py')
custom_imports = dict(imports=['yolo_world'],
                      allow_failed_imports=False)

# hyper-parameters
prompt_budget = 80  # 解决文本类别问题：训练阶段每张图固定 prompt 数，不代表全训练集类别总数
num_classes = 80
num_training_classes = prompt_budget
max_epochs = 50  # Maximum training epochs
close_mosaic_epochs = 2
save_epoch_intervals = 2
text_channels = 768
neck_embed_channels = [128, 256, _base_.last_stage_out_channels // 2]
neck_num_heads = [4, 8, _base_.last_stage_out_channels // 2 // 32]
base_lr = 2e-3
weight_decay = 0.0125
train_batch_size_per_gpu = 8
# text_model_name = '../pretrained_models/clip-vit-large-patch14-336'
text_model_name = r'pretrained_models/clip-vit-large-patch14-336'
img_scale = (800, 800)
load_from = 'weights/yolo_world_v2_l_clip_large_o365v1_goldg_pretrain_800ft-9df82e55.pth'

# model settings
model = dict(
    type='YOLOWorldDetector',  # 新增代码：使用当前真实存在的 detector，前景门控放到 neck 中
    mm_neck=True,
    num_train_classes=num_training_classes,
    num_test_classes=num_classes,
    data_preprocessor=dict(type='YOLOWDetDataPreprocessor'),
    backbone=dict(
        _delete_=True,
        type='MultiModalYOLOBackbone',
        image_model={{_base_.model.backbone}},
        text_model=dict(
            type='HuggingCLIPLanguageBackbone',
            model_name=text_model_name,
            frozen_modules=['all'],
            add_mask=True,  # 解决文本类别问题：启用 padding prompt 的文本 mask
            pad_value='')),  # 解决文本类别问题：统一用空字符串作为 padding prompt
    neck=dict(type='YOLOWorldPAFPN',
              guide_channels=text_channels,
              embed_channels=neck_embed_channels,
              num_heads=neck_num_heads,
              block_cfg=dict(type='MaxSigmoidCSPLayerWithTwoConv'),
              fgbg_cfg=dict(  # 新增代码：在 neck 内加入类别无关前景门控
                  type='ClassAgnosticForegroundGate',  # 新增代码：只输出 [B,1,H,W] 前景 mask
                  in_channels=[256, 512, _base_.last_stage_out_channels],  # 新增代码：匹配 P3/P4/P5 输出通道
                  hidden_ratio=0.25,  # 新增代码：mask 分支隐藏通道比例
                  min_hidden=16,  # 新增代码：mask 分支最小隐藏通道数
                  init_gamma=0.0,  # 新增代码：初始等价于原始 neck
              )),
    bbox_head=dict(type='YOLOWorldHead',
                   head_module=dict(type='YOLOWorldHeadModule',
                                    use_bn_head=True,
                                    embed_dims=text_channels,
                                    num_classes=num_training_classes)),
    train_cfg=dict(assigner=dict(num_classes=num_training_classes)))

# dataset settings
text_transform = [
    
    dict(type='RandomLoadText',
         num_neg_samples=(num_training_classes, num_training_classes),
         max_num_samples=num_training_classes,
         padding_to_max=True,
         padding_value=''),
    #
    dict(type='mmdet.PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                    'flip_direction', 'texts'))
]
train_pipeline = [
    *_base_.pre_transform,
    dict(type='MultiModalMosaic',
         img_scale=img_scale,
         pad_val=114.0,
         pre_transform=_base_.pre_transform),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(1 - _base_.affine_scale, 1 + _base_.affine_scale),
        max_aspect_ratio=_base_.max_aspect_ratio,
        border=(-img_scale[0] // 2, -img_scale[1] // 2),
        border_val=(114, 114, 114)),
    *_base_.last_transform[:-1],
    *text_transform,
]

train_pipeline_stage2 = [
    *_base_.pre_transform,
    dict(type='YOLOv5KeepRatioResize', scale=img_scale),
    dict(
        type='LetterResize',
        scale=img_scale,
        allow_scale_up=True,
        pad_val=dict(img=114.0)),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(1 - _base_.affine_scale, 1 + _base_.affine_scale),
        max_aspect_ratio=_base_.max_aspect_ratio,
        border_val=(114, 114, 114)),
    *_base_.last_transform[:-1],
    *text_transform
]


# 1. 基础路径设置
data_root = 'data/LAE-1M'  # 你的数据集根目录

sub_datasets_info = [
    # ('LAE-COD/AID/images/images', 'LAE-COD/AID/AID.json', 'data/LAE-1M/LAE-COD/AID/AID_categories.json'),
    # ('LAE-COD/EMS/images/images', 'LAE-COD/EMS/EMS.json', 'data/LAE-1M/LAE-COD/EMS/EMS_categories.json'),
    # ('LAE-COD/NWPU-RESISC45/images/images',   'LAE-COD/NWPU-RESISC45/NWPURESISC45.json',   'data/LAE-1M/LAE-COD/NWPU-RESISC45/NWPU-RESISC45_categories.json'),
    # ('LAE-COD/SLM/images/images', 'LAE-COD/SLM/SLM.json', 'data/LAE-1M/LAE-COD/SLM/SLM_categories.json'),
    ('LAE-FOD/DIOR/trainval_images/JPEGImages-trainval', 'LAE-FOD/DIOR/processed_LAE-1M_DIOR_train.json', 'data/LAE-1M/LAE-FOD/DIOR/DIOR_categories.json'),
    ('LAE-FOD/DOTAv2/images/images', 'LAE-FOD/DOTAv2/processed_LAE-1M_DOTAv2_train.json', 'data/LAE-1M/LAE-FOD/DOTAv2/DOTAv2_categories.json'),
    # ('LAE-FOD/FAIR1M/images/images', 'LAE-FOD/FAIR1M/processed_LAE-1M_FAIR1M_train.json', 'data/LAE-1M/LAE-FOD/FAIR1M/FAIR1M_categories.json'),
    # ('LAE-FOD/HRSC2016/images/images', 'LAE-FOD/HRSC2016/processed_LAE-1M_HRSC2016_train.json', 'data/LAE-1M/LAE-FOD/HRSC2016/HRSC2016_categories.json'),
    # ('LAE-FOD/NWPU VHR-10/images/images', 'LAE-FOD/NWPU VHR-10/processed_LAE-1M_NWPU-VHR-10_train.json', 'data/LAE-1M/LAE-FOD/NWPU VHR-10/NWPU VHR-10_categories.json'),
    # ('LAE-FOD/Power-Plant/Images/Images', 'LAE-FOD/Power-Plant/processed_LAE-1M_Power-Plant_train.json', 'data/LAE-1M/LAE-FOD/Power-Plant/Power-Plant_categories.json'),
    # ('LAE-FOD/RSOD/images/images', 'LAE-FOD/RSOD/processed_LAE-1M_RSOD_train.json', 'data/LAE-1M/LAE-FOD/RSOD/RSOD_categories.json'),
    # ('LAE-FOD/xview/train_images_1024_05/train_images_1024_05', 'LAE-FOD/xview/processed_LAE-1M_Xview_train_1024_05.json', 'data/LAE-1M/LAE-FOD/xview/xview_categories.json'),
]


# 3. 构建数据集配置列表
train_dataset_list = []
for img_path, ann_path, class_text_path in sub_datasets_info:

    train_dataset_list.append(
        dict(
            type='MultiModalDataset',
            dataset=dict(
                type='YOLOv5CocoDataset',
                data_root=data_root,
                ann_file=ann_path,
                data_prefix=dict(img=img_path),
                filter_cfg=dict(filter_empty_gt=False, min_size=32),
            ),
            class_text_path=class_text_path,  # 每个子集使用自己的类别文本
            pipeline=train_pipeline,
            # NOTE: MultiModalDataset 不支持 `pipeline_stage2` 这个参数。
            # 第二阶段 pipeline 通过 PipelineSwitchHook 切换 `train_dataloader.dataset.pipeline` 来实现。
        )
    )

# 4. 配置 DataLoader 使用 ConcatDataset
train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    collate_fn=dict(type='yolow_collate'),
    dataset=dict(
        _delete_=True,        # 删除继承的旧配置
        type='ConcatDataset', # 关键：拼接器
        datasets=train_dataset_list, # 传入上面生成的列表
        ignore_keys=['classes', 'palette'] # 忽略可能冲突的元信息
    )
)

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='YOLOv5KeepRatioResize', scale=img_scale),
    dict(
        type='LetterResize',
        scale=img_scale,
        allow_scale_up=False,
        pad_val=dict(img=114)),
    dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
    dict(type='LoadText'),
    dict(type='mmdet.PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor', 'pad_param', 'texts'))
]


lae_80c_test_dataset = dict(
    _delete_=True,
    type='MultiModalDataset',
    dataset=dict(type='YOLOv5CocoDataset',
                 data_root=data_root,
                 test_mode=True,
                 ann_file='LAE-80C/LAE-80C-benchmark.json',
                 data_prefix=dict(img='LAE-80C/images/images/'),
                 batch_shapes_cfg=None),
    class_text_path=r'data/LAE-1M/LAE-80C/LAE-80C-benchmark_categories.json',
    pipeline=test_pipeline)
val_dataloader = dict(dataset=lae_80c_test_dataset)
test_dataloader = val_dataloader

# val_evaluator = dict(type='mmdet.LVISMetric',
#                      ann_file='data/coco/lvis/instances_val2017.json.json',
#                      metric='bbox')

val_evaluator = dict(type='mmdet.CocoMetric',
                     ann_file=r'data/LAE-1M/LAE-80C/LAE-80C-benchmark.json',  # 评测用标注文件（与 val 一致）
                     metric='bbox')  # 评测 bbox 检测指标
test_evaluator = val_evaluator

# training settings
default_hooks = dict(param_scheduler=dict(max_epochs=max_epochs),
                     checkpoint=dict(interval=save_epoch_intervals,
                                     rule='greater'))
custom_hooks = [
    dict(type='EMAHook',
         ema_type='ExpMomentumEMA',
         momentum=0.0001,
         update_buffers=True,
         strict_load=False,
         priority=49),
    dict(type='mmdet.PipelineSwitchHook',
         switch_epoch=max_epochs - close_mosaic_epochs,
         switch_pipeline=train_pipeline_stage2)
]
train_cfg = dict(max_epochs=max_epochs,
                 val_interval=5,
                 dynamic_intervals=[((max_epochs - close_mosaic_epochs),
                                     _base_.val_interval_stage2)])
visualizer = dict(
    type='mmdet.DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend')  # 增加 TensorBoard 后端
    ],
    name='visualizer'
)
optim_wrapper = dict(optimizer=dict(
    _delete_=True,
    type='AdamW',
    lr=base_lr,
    weight_decay=weight_decay,
    batch_size_per_gpu=train_batch_size_per_gpu),
                     paramwise_cfg=dict(bias_decay_mult=0.0,
                                        norm_decay_mult=0.0,
                                        custom_keys={
                                            'backbone.text_model':
                                            dict(lr_mult=0.01),
                                            'logit_scale':
                                            dict(weight_decay=0.0)
                                        }),
                     constructor='YOLOWv5OptimizerConstructor')


# conda create -n yoloworld python=3.9
# pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 -i https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu113
# pip install openmim
# mim install mmcv==2.0.0
# pip install -r requirements/basic_requirements.txt
# pip show torch
# pip show mmcv
# pip uninstall -y numpy
# pip install "numpy<2.0.0"
# tensorboard --logdir ./work_dirs


# screen -S my_session：创建一个新的 screen 会话。
# Ctrl + A 然后按 D：分离当前 screen 会话，任务在后台继续运行。
# ctrl+a esc
# screen -r my_session：重新连接到一个指定的 screen 会话。
# screen -ls：查看所有活动的 screen 会话

# watch -n 1 nvidia-smi 查看剩余显存和 GPU 使用情况，确认训练是否正常进行以及资源占用情况。

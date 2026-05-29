
_base_ = ('../../third_party/mmyolo/configs/yolov8/'  # 继承基础 YOLOv8 配置（相对路径，分行拼接字符串）
          'yolov8_s_syncbn_fast_8xb16-500e_coco.py')  # 基础配置文件名（YOLOv8-s，SyncBN，COCO 训练设定）
custom_imports = dict(imports=['yolo_world'],  # mmengine/mmdet 配置：自定义导入模块列表
                      allow_failed_imports=False)  # 若导入失败则报错（不允许静默失败）


# hyper-parameters  # 注释：超参数区
num_classes = 80# 开放词表/测试时类别总数（OVOD 评测类别数）
num_training_classes = 38  # 训练阶段参与分类/对齐的类别数（通常 COCO 80 类）
max_epochs = 50  # Maximum training epochs  # 最大训练轮数（epoch）
close_mosaic_epochs = 2  # 训练末期关闭 mosaic 的 epoch 数（用于收敛/稳定）
#这种技术会随机取出 4 张图片，通过随机缩放、裁剪、排列，拼接成一张大图作为输入，为了增加鲁棒性，最后两轮关闭为了让BN收敛平稳
save_epoch_intervals = 2  # checkpoint 保存间隔（每隔多少个 epoch 保存一次）
text_channels = 512  # 文本分支/跨模态对齐的嵌入通道维度
neck_embed_channels = [128, 256, _base_.last_stage_out_channels // 2]  # neck 各层 embed 通道配置（含来自 base 的通道）
neck_num_heads = [4, 8, _base_.last_stage_out_channels // 2 // 32]  # neck 多头注意力 head 数配置（最后一层按通道推导）
base_lr = 2e-4  # 基础学习率
weight_decay = 0.025   # 权重衰减系数（这里相当于 0.025）
train_batch_size_per_gpu = 8  # 单 GPU batch size（用于 dataloader/optimizer 构造）
# load_from = 'weights/yolo_world_v2_s_obj365v1_goldg_pretrain-55b943ea.pth'
# img_scale = (800, 800)

# model settings  # 注释：模型结构配置区
model = dict(  # 顶层 model 配置字典
    type='YOLOWorldDetector',  # 模型类型：YOLO-World 检测器
    mm_neck=True,  # 是否启用多模态 neck（与文本引导相关）
    num_train_classes=num_training_classes,  # 训练类别数（用于 head/assigner 等）
    num_test_classes=num_classes,  # 测试/开放词表类别数（推理时文本类别数）
    data_preprocessor=dict(type='YOLOWDetDataPreprocessor'),  # 数据预处理器类型（YOLO-World 检测预处理）
    backbone=dict(  # backbone 配置
        _delete_=True,  # 删除并完全替换 base 配置中的 backbone
        type='MultiModalYOLOBackbone',  # 多模态 YOLO backbone（图像+文本）
        image_model={{_base_.model.backbone}},  # 复用基础配置中的图像 backbone 配置（使用��置插值语法）
        text_model=dict(  # 文本 backbone 配置
            type='HuggingCLIPLanguageBackbone',  # 文本编码器：CLIP 语言侧（HuggingFace）
            model_name=r'pretrained_models/clip-vit-base-patch32',  # 预训练文本模型名称
            frozen_modules=['all'])),  # 冻结文本模型全部模块（避免训练更新）
    neck=dict(type='YOLOWorldPAFPN',  # neck 类型：YOLOWorld 的 PAFPN
              guide_channels=text_channels,  # 文本引导通道数���与 text embedding 对齐）
              embed_channels=neck_embed_channels,  # neck 内部各尺度 embed 通道
              num_heads=neck_num_heads,  # neck 内部各尺度 attention head 数
              block_cfg=dict(type='MaxSigmoidCSPLayerWithTwoConv')),  # neck block 结构配置（CSPLayer 变体）
    bbox_head=dict(type='YOLOWorldHead',  # 检测 head 类型
                   head_module=dict(type='YOLOWorldHeadModule',  # head 内部模块类型
                                    use_bn_head=True,  # head 中是否使用 BN
                                    embed_dims=text_channels,  # head 使用的 embedding 维度（与文本通道一致）
                                    num_classes=num_training_classes)),  # head 训练类别数（用于分类分支）
    train_cfg=dict(assigner=dict(num_classes=num_training_classes)),
    test_cfg = dict(score_thr=0.001, # 在这里调低或调高置信度阈值
                    nms=dict(type='nms', iou_threshold=0.7) ) # 以及非极大值抑制的 NMS 阈值
    )  # 训练配置：分配器使用的类别数



# dataset settings  # 注释：数据集与 pipeline 配置区
text_transform = [  # 文本相关的数据增强/打包流程
    dict(type='RandomLoadText',  # 随机加载/采样文本（含负样本）
         num_neg_samples=(10, 37),  # 负样本采样数量范围（这里固定为全量候选）
         max_num_samples=num_training_classes,  # 每次最多采样的文本条目数（与训练类别数一致）
         padding_to_max=True,  # 是否 padding 到 max_num_samples
         padding_value=''),  # padding 时用空字符串填充
    dict(type='mmdet.PackDetInputs',  # mmdet 打包输入（统一数据结构）
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',  # 需要保留到 metainfo 的键
                    'flip_direction', 'texts'))  # 额外保留 texts（文本提示/类别名）
]  # 文本 transform 列表结束
train_pipeline = [  # 训练数据 pipeline（阶段 1）
    *_base_.pre_transform,  # 复用 base 的预处理步骤
    dict(type='MultiModalMosaic',  # 多模态 mosaic（图像拼接增强）
         img_scale=_base_.img_scale,  # mosaic 的目标尺度（来自 base）
         pad_val=114.0,  # padding 像素值（YOLO 常用 114）
         pre_transform=_base_.pre_transform),  # mosaic 前的预处理步骤
    dict(  # 仿射增强配置块
        type='YOLOv5RandomAffine',  # YOLOv5 风格随���仿射
        max_rotate_degree=0.0,  # 最大旋转角（这里关闭旋转）
        max_shear_degree=0.0,  # 最大错切角（这里关闭错切）
        scaling_ratio_range=(1 - _base_.affine_scale, 1 + _base_.affine_scale),  # 缩放范围（围绕 1 的区间）
        max_aspect_ratio=_base_.max_aspect_ratio,  # 最大宽高比变化（来自 base）
        border=(-_base_.img_scale[0] // 2, -_base_.img_scale[1] // 2),  # 边界偏��（配合 mosaic）
        border_val=(114, 114, 114)),  # 边界填充值（RGB）
    *_base_.last_transform[:-1],  # 复用 base 的最后 transform（去掉最后一步以插入文本相关处理）
    *text_transform,  # 追加文本 transform（加载/打包 texts）
]  # 训练 pipeline（阶段 1）结束
train_pipeline_stage2 = [*_base_.train_pipeline_stage2[:-1], *text_transform]
# 阶段 2 pipeline：复用 base 并追加文本 transform


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
            pipeline=train_pipeline
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

test_pipeline = [  # 测试/验证 pipeline
    *_base_.test_pipeline[:-1],  # 复用 base 的 test pipeline（去掉最后一步以插入文本相关处理）
    dict(type='LoadText'),  # 加载 texts（用于开放词表推理/评测）
    dict(type='mmdet.PackDetInputs',  # 打包成 mmdet 统一输入
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',  # 需要保留的元信息
                    'scale_factor', 'pad_param', 'texts'))  # 追加 texts 元信息
]  # 测试 pipeline 结束
# 1) 验证集：来自 LAE-1M 的独立划分（示例路径）
# lae_1m_val_dataset = dict(
#     _delete_=True,
#     type='MultiModalDataset',
#     dataset=dict(
#         type='YOLOv5CocoDataset',
#         data_root=data_root,
#         test_mode=True,
#         ann_file='LAE-FOD/DDTEST/DDTEST.json',  # 你新建的 5% 验证集标注
#         data_prefix=dict(img='LAE-FOD/DDTEST/images/'),
#         batch_shapes_cfg=None),
#     class_text_path=r'data/LAE-1M/LAE-FOD/DDTEST/DDTEST_categories.json',
#     pipeline=test_pipeline
# )
# 2) 测试集：保持 LAE-80C 不变
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
#
# # 4) evaluator 解耦
# val_evaluator = dict(
#     type='mmdet.CocoMetric',
#     ann_file=r'data/LAE-1M/LAE-FOD/DDTEST/DDTEST.json',
#     metric='bbox'
# )
val_evaluator = dict(type='mmdet.CocoMetric',
                     ann_file=r'data/LAE-1M/LAE-80C/LAE-80C-benchmark.json',  # 评测用标注文件（与 val 一致）
                     metric='bbox')  # 评测 bbox 检测指标

test_evaluator = val_evaluator
# training settings  # 注释：训练过程与优化器配置区
default_hooks = dict(param_scheduler=dict(max_epochs=max_epochs),  # 默认 hook：学习率/参数调度器最大 epoch
                     checkpoint=dict(interval=save_epoch_intervals,  # 默认 hook：checkpoint 保存间隔
                                     rule='greater'))  # 保存规则：按更优指标保留（greater 更好）
custom_hooks = [  # 自定义 hooks 列表
    dict(type='EMAHook',  # EMA（指数滑动平均）hook
         ema_type='ExpMomentumEMA',  # EMA 类型：指数动量
         momentum=0.0001,  # EMA 动量系数
         update_buffers=True,  # 是否同步/更新 buffer（如 BN 统计）
         strict_load=False,  # 加载 EMA 权重时是否严格匹配 key
         priority=49),  # hook 优先级（影响调用顺序）
    dict(type='mmdet.PipelineSwitchHook',  # pipeline 切换 hook（训练后期关闭强增强）
         switch_epoch=max_epochs - close_mosaic_epochs,  # 切换 epoch（训��末 close_mosaic_epochs 前）
         switch_pipeline=train_pipeline_stage2)  # 切换到阶段 2 pipeline
]  # 自定义 hooks 结束
train_cfg = dict(max_epochs=max_epochs,  # 训练最大 epoch
                 val_interval=5,  # 验证间隔（每 10 个 epoch 验证一次）
                 dynamic_intervals=[((max_epochs - close_mosaic_epochs),  # 动态验证间隔切换点
                                     _base_.val_interval_stage2)])  # 切到 stage2 的验证间隔（来自 base）
visualizer = dict(
    type='mmdet.DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend')  # 增加 TensorBoard 后端
    ],
    name='visualizer'
)
optim_wrapper = dict(optimizer=dict(  # 优化器包装器（mmengine）
    _delete_=True,  # 删除并替换 base 的 optimizer
    type='AdamW',  # 优化器类型：AdamW
    lr=base_lr,  # 学习率
    weight_decay=weight_decay,  # 权重衰减
    batch_size_per_gpu=train_batch_size_per_gpu),  # 传入 batch_size（某些构造器用于 lr 缩放）
                     paramwise_cfg=dict(bias_decay_mult=0.0,  # bias 不施加 weight decay
                                        norm_decay_mult=0.0,  # norm 层（BN/LN）不施加 weight decay
                                        custom_keys={  # 针对特定参数子树的自定义超参
                                            'backbone.text_model':  # 文本 backbone 参数
                                            dict(lr_mult=0.01),  # 文本分支学习率倍率（更小）gzai
                                            'logit_scale':  # CLIP 的 logit_scale 参数
                                            dict(weight_decay=0.0)  # 对该参数禁用 weight decay
                                        }),  # paramwise_cfg 结束
                     constructor='YOLOWv5OptimizerConstructor')  # 使用自定义优化器构造器（按 YOLO 规则分组）






# conda create -n yoloworld python=3.8.10
# conda activate yoloworld
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
# screen -r my_session：重新连接到一个指定的 screen 会话。
# screen -ls：查看所有活动的 screen 会话

# watch -n 1 nvidia-smi 查看剩余显存和 GPU 使用情况，确认训练是否正常进行以及资源占用情况。
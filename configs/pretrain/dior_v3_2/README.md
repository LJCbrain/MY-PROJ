# DIORv3.2：以2.17为基线的隔离描述关系实验

3.2新增完整映射的Semantic Adapter及文本描述关系损失，不实现RNR，
不改O分支，不替换图像骨干。Adapter不是O/OVC的Visual Adapter。
配置直接继承 `configs/pretrain/DIORv2.17best.py`，不是继承3.0或3.1。
2026-10-28 协议修正：2.17及继承配置的训练锚点均限制为基类；旧实验结果不代表修正后的协议。
新文本子类仅由3.2的 `custom_imports` 注册；不导入公共 `__init__.py`。
描述源文件和说明放在专属 `configs/pretrain/dior_v3_2/` 目录，避免被
仓库现有的 `assets/`、`docs/` 忽略规则排除。生成的 `.pt` 矩阵是可重建缓存，
按现有权重文件规则被Git忽略；默认首次训练前向自动编码基类描述，也可提前生成缓存。

## 方法与边界

16个基类航空外观描述 → 冻结CLIP → 原始16×16余弦关系矩阵。
排除自身的行softmax只在计算损失时应用，初始温度0.03。
每类选择最多3个最相似邻居与3个最不相似邻居，使用两组的笛卡尔积
构造确定性三元组；不是原论文的随机采样复现，不额外消耗训练RNG。
自适应间隔为两个邻居的教师关系概率之差；全批次及有效三元组取均值。
学生是提示后CLIP文本经共享Semantic Adapter得到的 `[B,16,768]` 基类原型。
实际检测头使用同一个Adapter的输出，而不是另建仅用于辅助损失的投影器。

检测文本链路为：类别名称 → PromptBank + 冻结CLIP → 归一化文本向量e
→ `w = normalize(W e + b)` → 与检测头的视觉向量匹配。
Adapter为单层768→768全连接，共590,592个可训练参数；Xavier随机初始化权重、
零偏置。没有残差、门控、原向量直通或恒等初始化，允许完整重塑语义空间。
随机映射可能降低初始预训练对齐，这是本次完整映射实验需要实测的代价。
输入类别数不写死：Base16训练和关系学习、全部20类推理共享W和b。
未提示的冻结CLIP向量仍用于neck引导和PromptBank路由，不经过Adapter。

总损失为2.17原损失加 `loss_text_description`。单点语义锚定保持0.05，
只约束Adapter之前的提示后CLIP向量，不把Adapter输出拉回原冻结语义。
旧类别名称拓扑KL关闭；描述损失初始权重0.01，独立1000次前向warmup。
这些超参数是待验证的初始值，不代表最优值或保证Novel提升。
CLIP冻结；Base分类损失直接训练新Adapter，也沿原检测链路训练提示词和视觉侧。
描述关系损失训练新Adapter及被选中的提示词；沿用锚定的硬路由，因此不通过
此项关系损失更新路由器、图像骨干或O/OVC Visual Adapter。
新Adapter显式设置 `lr_mult=1.0`，实际初始LR=0.001，避免继承整个文本骨干的
0.01倍学习率；其他优化器设置保持2.17。训练仍是一阶段联合学习，没有新增冻结阶段。

检测分类、语义锚定、文本拓扑和描述关系损失均只使用Base16。
Novel4名称和描述只可用于验证/测试，不能进入训练编码器或教师关系矩阵。
冻结CLIP不能消除通过新类文本训练提示词或Adapter的泄漏。
不得根据目标测试图片、预测或指标反复编辑描述或调参。

## 描述版本

当前训练源为 `aerial_descriptions_base16.json`，只保留v2中的16个基类描述，
版本为 `v2_appearance_review_base16`。旧的 `aerial_descriptions.json`、
`aerial_descriptions_v1.json` 和20类矩阵仅作历史存档，不得接入当前训练。
加载器要求类别集合严格一致，不会静默裁剪旧20类教师。
候选描述尚不代表已经证明提升检测效果。

## 生成矩阵与运行

从仓库根目录执行，使用已有YOLO-World训练环境。编码器只读本地权重，
不下载模型、不读取任何图像或数据集标注。描述太长会报错，不静默截断。
默认 `matrix_path=None`，无需预先生成矩阵；首次训练前向只编码16个基类描述。

```powershell
python tools/build_dior_description_matrix.py --device cpu
python tools/train.py --config configs/pretrain/DIORv3.2.py --work-dir work_dirs/DIORv3.2_semantic_full
```

如需离线缓存，生成后显式指定新的Base16路径：

```powershell
python tools/train.py --config configs/pretrain/DIORv3.2.py --work-dir work_dirs/DIORv3.2_base16 --cfg-options model.backbone.text_model.description_regularization_cfg.matrix_path=assets/dior_v3_2_description_matrix_base16.pt
```

矩阵保存原始余弦、类别名、描述和源文件SHA256。加载时按名字重排两轴，
类别缺失、额外类别、重复、矩阵不对称或校验和变化都会报错。
必须使用新实验目录和原始预训练初始化，不能从已见过Novel文本的实验检查点resume。
已有训练权重的泄漏无法通过删除文本或缓存逆转，旧结果需要重新训练和评估。
推理不读取描述或矩阵、不计算关系损失，但保留Adapter完整映射。
教师缓存不进入EMA/state_dict，warmup整数计数器进入检查点。

## 历史随机种子来源（不代表修正后的实验结果）

3.2显式固定 `randomness=dict(seed=179871018)`，与2.17取得HM约14.3%的
实际运行一致，不只根据当前baseline配置推测。核对日志为
`work_dirs/DIORv2.17/20260917_151954.log`：第37行运行时seed为179871018，
第5226行HM为0.14296774，第5229行记录最优检查点在iter=25968。
原3.2已经使用这个数值；本次补充来源注释和与baseline随机性配置的一致性测试，
不改2.17，不额外开启deterministic或改变cuDNN策略。
正式运行不要用命令行覆盖成另一个seed。相同seed不保证3.2复现2.17的HM：
新增关系损失、Adapter初始化和完整映射会改变优化轨迹及RNG消费，
设备、软件版本及非确定性算子也可能影响结果。

## 最小消融与验收

```powershell
# Adapter-only：保留完整映射，关闭老师。
python tools/train.py --config configs/pretrain/DIORv3.2.py --work-dir work_dirs/DIORv3.2_adapter_only --cfg-options model.backbone.text_model.description_regularization_cfg.loss_weight=0
# 原提示词关系版本：关闭新Adapter，保留老师。
python tools/train.py --config configs/pretrain/DIORv3.2.py --work-dir work_dirs/DIORv3.2_prompt_relation --cfg-options model.backbone.text_model.semantic_adapter_cfg.enabled=False
# 原2.17计算路径：同时关闭Adapter和老师。
python tools/train.py --config configs/pretrain/DIORv3.2.py --work-dir work_dirs/DIORv3.2_baseline_path --cfg-options model.backbone.text_model.semantic_adapter_cfg.enabled=False model.backbone.text_model.description_regularization_cfg.loss_weight=0
python -m unittest discover -s tests -p test_description_regularization.py -v
```

只有同时关闭Adapter且关系权重为0，才直接回到2.17原分类和锚定路径；
此时不读取教师文件、不产生新增损失，
原检测输出、锚定损失、提示词梯度和RNG路径应一致。
检查3.2日志中的 `loss_text_description`，并比较同协议、同种子的
Novel AP50、Base AP50、HM及固定区域Novel→Base错分。
不要将3.2从2.17最佳训练检查点继续微调后，直接与2.17原始训练比较；
默认保持2.17相同预训练初始化 `load_from`，不改变 `resume`。

论文依据：[DescReg方法](https://arxiv.org/html/2402.18233v1#S3.SS2)。
本实现是其关系正则思想对现有提示词模型的适配，不是完整论文复现。

完整模型的有界检查（不测AP、不启动正式训练）：

```powershell
python tools/validate_ovc.py --config configs/pretrain/DIORv3.2.py --device cpu --batch-size 2 --image-size 128 --steps 2 --output-dir work_dirs/DIORv3.2_semantic_full_smoke
```

检查报告中的 `semantic_adapter_parameter_changed`、
`semantic_adapter_grad_norms`、`semantic_adapter_learning_rates` 和
`inference_semantic_adapter_shapes`，分别验证真实参数更新、梯度、LR与20类推理。
旧提示词关系版本的检查结果不能充当新Adapter的验证结果。

以下为协议修正前的历史验收，不能作为Base16版本的验收：

2026-10-10验收：21项描述/Adapter测试和3项原拓扑测试通过；完整预训练
检测器在两张真实Base训练图片上以CPU、128×128、两步更新通过。
新Adapter梯度非零、权重实际变化，20类推理输出为 `[2,20,768]`。
报告位于 `work_dirs/DIORv3.2_semantic_full_smoke/report.json`。
这个检查没有Runner学习率调度、不测AP；随机完整映射下分类损失两步为
13.20和711.91，虽然没有非有限值，但波动明显，不能宣称训练稳定或效果提升。
是否改善Novel→Base错分必须通过正式同协议训练与消融判断。

同一输入的CUDA AMP六步检查中，损失均有限，但动态loss scaler跳过5步、
完成1次真实更新；新Adapter权重发生变化。整项GPU检查未通过：最终推理
在MMCV CUDA NMS处报 `no kernel image is available for execution on the device`。
本机RTX 5060 Laptop（sm_120）与当前Torch 2.0.1+cu118组合也发出架构兼容告警。
不经过检测器、仅对两个框独立调用CUDA NMS也复现相同错误；CPU NMS正常，
该问题不依赖新Adapter。
GPU报告为 `work_dirs/DIORv3.2_semantic_full_amp_smoke/report.json`；其中
梯度hook记录的是AMP缩放后、裁剪前的梯度，溢出跳步中的NaN不应当作成功更新。
不要把这份GPU失败报告写成全链路通过；未修改环境依赖或添加NMS回退。

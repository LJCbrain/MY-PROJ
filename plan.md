# YOLO-World 遥感开放词汇检测改进计划

## 核心思想

这次改动的主线是把任务拆开：

- `neck` 只负责判断“哪里像目标”。
- `head` 继续负责判断“目标是什么”和“目标框在哪里”。

因此，前景模块必须是类别无关的，只输出单通道前景响应：

```python
mask: [B, 1, H, W]
```

不能在 `neck` 中输出类别 logits，也不能替代 `YOLOWorldHead` 的图文分类。

## 整体路线

计划分三步完成：

1. 多尺度类别无关前景门控。
2. 基于 bbox pseudo mask 的前景监督。
3. 多尺度前景一致性约束。

后续如果前三步稳定，再考虑加入文本解耦的 objectness prior。

## 第一阶段：类别无关前景门控

状态：已完成。

目标是在 `YOLOWorldPAFPN` 输出特征后加入一个轻量门控模块，让 P3/P4/P5 每个尺度都学习一个类别无关前景 mask。

当前机制：

```python
mask = sigmoid(mask_conv(feat))
fg = feat * mask
bg = feat * (1 - mask)
out = feat + gamma * fg
```

设计要点：

- 每个尺度单独预测一个 `[B,1,H,W]` mask。
- 输出仍然是 `tuple[Tensor]`，不改变 `YOLOWorldHead` 接口。
- 第一阶段不新增 loss，让模块先通过检测主损失间接学习。
- `gamma` 初始很小或为 0，避免一开始破坏预训练特征。

第一阶段主要验证：

- 模型能否正常 build。
- 单次训练迭代能否跑通。
- 加入 gate 后训练是否稳定。
- 与 baseline 做初步 ablation。

## 第二阶段：bbox pseudo mask 前景监督

状态：下一步。

目标是用检测框生成弱监督前景 mask，指导 `ClassAgnosticForegroundGate` 学到更明确的“目标区域”，避免它只关注遥感图像中的纹理、道路、阴影等背景模式。

基本思路：

```python
for each feature_level:
    pseudo_mask = boxes_to_mask(gt_boxes, feature_size)
    pred_mask = neck._fgbg_parts["masks"][level]
    loss = BCE(pred_mask, pseudo_mask) + Dice(pred_mask, pseudo_mask)
```

可以先加入三个辅助项：

```python
loss_fgbg_bce
loss_fgbg_dice
loss_fgbg_area
```

推荐初始权重：

```python
fgbg_w_bce = 0.05
fgbg_w_dice = 0.05
fgbg_w_area = 0.005
```

注意事项：

- pseudo mask 只监督前景 gate，不传给 `YOLOWorldHead`。
- 不改变 head 的 `forward / loss / predict`。
- 不做类别区分，所有 GT box 区域都视作 objectness 前景。
- 暂时不加入 entropy、margin、bg energy 这类不稳定约束。

## 第三阶段：多尺度前景一致性

状态：第二阶段稳定后再做。

目标是让 P3/P4/P5 的前景判断在尺度间保持一致，同时保留各尺度自己的分工。

遥感检测中，小目标依赖高分辨率 P3，大目标和上下文依赖 P4/P5。因此前景 mask 既要有细节，也要有跨尺度稳定性。

基本伪代码：

```python
mask_p3_to_p4 = downsample(mask_p3, size=mask_p4.shape[-2:])
mask_p4_to_p5 = downsample(mask_p4, size=mask_p5.shape[-2:])

loss_consistency = (
    L1(mask_p3_to_p4, mask_p4.detach()) +
    L1(mask_p4_to_p5, mask_p5.detach())
)
```

推荐初始权重：

```python
fgbg_w_consistency = 0.01
```

注意事项：

- 这个 loss 权重要小，不能强行让所有尺度 mask 完全一样。
- 优先把高分辨率 mask 下采样到低分辨率。
- 可以先对低分辨率目标 mask 使用 `detach()`，减少训练震荡。

## 后续可选：文本解耦 objectness prior

如果前三步有效，再考虑让文本特征参与“哪里像目标”的判断。

但文本只能被压缩成类别无关 objectness prior，不能在 neck 中保留类别维度。

允许：

```python
sim = image_feat @ text_feat.T
objectness = max_over_text(sim)
mask = sigmoid(local_logit + alpha * objectness)
```

禁止：

```python
neck_cls_logits = sim  # [B, K, H, W]
```

也就是说，文本可以辅助判断“这里像不像任意目标”，但最终“是什么类别”仍然必须交给 `YOLOWorldHead`。

## 实验顺序

推荐按下面顺序做 ablation：

1. 原始 YOLOWorld baseline。
2. baseline + 第一阶段 gate。
3. baseline + gate + bbox pseudo mask loss。
4. baseline + gate + bbox pseudo mask loss + 多尺度一致性。
5. 如果前面有效，再尝试文本解耦 objectness prior。

当前下一步：

实现第二阶段，也就是给 `ClassAgnosticForegroundGate` 加 bbox pseudo mask 弱监督。

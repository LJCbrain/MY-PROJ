# DIOR 3.2：20类航空描述的第二版审校

本文件的模型检查记录对应添加Semantic Adapter之前的提示词关系版本。
后续完整映射实验不改变本版描述或教师矩阵；当前链路及独立验收见README。

## 目标与证据边界

本版是**可冻结用于关系正则实验的候选老师**，不是已经证明有效的视觉关系真值。
由Codex依据公开论文的方法原则和通用物体知识改写；没有调用论文作者的生成模型，
没有逐字复制作者的描述列表，也没有进行航空专家人工认证。
起草和选择描述未读取DIOR图片、标注、预测、测试分数，也没有预设需要达到的相似度排序。
先完成20类规则审校，再编码和诊断；不根据诊断结果反复改词追求某组邻居。
编码后的相似度仍是冻结CLIP的文本先验，不是经过视觉样本验证的相似度。

Novel类别名称和通用描述仅作为已知目标词表的文本先验；训练检测标签仍为Base16。
需要披露这种ZSD设定。不得用目标Novel测试集选择描述或超参数；如需开发选择，
应另行预先定义Base内部的伪Novel划分，不改动本次实验协议。

## 借鉴了哪些论文，没有复现哪些部分

1. [DescReg，Zang等，AAAI 2024](https://arxiv.org/html/2402.18233v1#S3.SS2)：
   借鉴航空外观描述和类间关系约束，突出形状、部件与空间布局。
   本版并未复现其全部训练结构；编码器和3.2的三元组实现保持原适配方案。
2. [DCLIP，Menon与Vondrick，ICLR 2023](https://arxiv.org/abs/2210.07183)：
   借鉴可识别视觉属性的写法，把功能说明改成可见特征。
   没有添加其逐属性分类打分或描述集合。
3. [CuPL，Pratt等，ICCV 2023](https://arxiv.org/abs/2209.03320)：
   借鉴面向视觉外观生成文本的思路。其多描述集成不在本次范围内；
   当前仍是一类一条描述，没有把多段文本拼成长句，也没有增添集成变量。
4. [Does VLM Classification Benefit from LLM Description Semantics?，Ma等，AAAI 2025](https://arxiv.org/html/2412.11917v2)：
   借鉴类别名称与描述内容分离的控制思路。老师正文不重复完整类别名称，
   分类器名称及原语义锚定不变。这是本项目的设计选择，不能从该分类论文
   直接推断关系正则或航空检测也一定受益。

## 本次统一写作规则

- 视角：光学航空俯视/高角度观察；不规定地理朝向、绝对米数或固定颜色。
- 范围：飞机、船、车辆、储罐等写单体；机场、港口、车站等写设施内部布局。
- 内容次序：整体轮廓 → 可见部件 → 内部布局 → 条件性的细节/阴影。
- 核心信息用肯定描述；不写“像某个其他类”，也不靠否定另一个类制造距离。
- 类别可共享真实外观词汇，不强行把相似类改得毫不相干。
- 细小场地线条、顶部开口、阴影等依赖分辨率/角度的特征明确写为可选。
- 不把飞机与机场、船与港口、烟囱与厂房等共现关系写成单体的必要属性。
- 各类采用相近粒度和篇幅；CLIP实际token数检查包含特殊token，不静默截断。
- 保留原20个类别ID与顺序；设施级场景属性不可完全删除，但要与附带目标共现区分。

“不重复类别名称”不是删除所有有语义的部件词。跑道、船体、转子等词本身仍包含
类别信息，这是有效外观描述所需的知识，并不意味着老师已与类别语义完全解耦。

## 逐类改写依据

| 类别ID | 新版保留的主信息 | 相比v1的重点处理 |
| --- | --- | --- |
| airplane | 纵向机身、双侧主翼、较小尾翼、双侧对称 | 删除停机坪/跑道共现，突出单体部件相对位置 |
| airport | 超长跑道带、较窄连接道、宽阔机坪、分支连接 | 删除停靠飞机作为识别依据，强调完整设施布局 |
| baseballfield | 菱形内场、扇形外场、张开的直边与弧形外边界 | 明确内外场组合，避免把整体误写成普通椭圆场 |
| basketballcourt | 矩形场地、对称标线、中圈、端部弧线与矩形区域 | 不依赖篮筐小目标；细线明确以可分辨为条件 |
| bridge | 连续细长跨越结构、平行边缘、两端接续 | 核心写跨越结构；支撑/阴影只作可选高度线索 |
| chimney | 小圆/多边形顶部、细长竖向轴体、窄长阴影 | 删除工业厂房共现；开口和阴影不是强制条件 |
| dam | 横跨水道的宽厚直/曲屏障、两侧水面布局差异 | 用可见水道和结构关系代替用途/发电定义 |
| Expressway-Service-area | 道路旁停车区、分散建筑、内部通路与接入弯道 | 区分旁侧展开布局和横跨多车道的条带布局 |
| Expressway-toll-station | 横向棚/亭列、多平行窄车道、先展开再收拢 | 强化棚与车道的相对方向；不依赖收费用途 |
| golffield | 蜿蜒草地带、较小圆润草地区、散布浅色沙地 | 删除固定绿色；用分散地表纹理和布局描述 |
| groundtrackfield | 窄椭圆跑道环、多平行道、直边接半圆端、中央场 | 更完整写内部布局；与体育场可共享真实外观 |
| harbor | 水陆复合边界、多码头/岸边、重复泊位、内陆场地 | 删除停泊船作为必要信息，强调港区自身结构 |
| overpass | 多条道路分层交叉、连续上层结构、各自引道 | 弯曲匝道/错位阴影可选，不规定必须有环形匝道 |
| ship | 单个细长船体、渐窄艏部、较宽中段、甲板结构 | 删除码头共现；水域为对象介质，尾迹仅可选 |
| stadium | 宽厚看台/屋顶围合、中央开敞场、嵌套布局 | 加强建筑围合；内部跑道为可选的真实共享属性 |
| storagetank | 圆盘屋顶、圆柱轮廓、曲边、可选同心纹理/阴影 | 删除工业用途与厂区依赖，单罐形状优先 |
| tenniscourt | 内嵌矩形、发球区格子、平行边线、横向中线 | 强化直线网格；不以相邻多场地替代单场地属性 |
| trainstation | 多平行轨道、狭长站台/棚、站台与轨道相对位置 | 强调设施内部线性结构，不要求停靠列车 |
| vehicle | 紧凑矩形顶面、短前后段、圆角与局部阴影 | 删除停车场/道路队列共现，避免场景替代单体 |
| windmill | 细长塔架、小机舱、同一轮毂伸出的窄长转子叶片 | 不规定恰有三片或固定投影；视角变化与阴影可选 |

## 版本、接入与对照

原描述逐字节备份为 `aerial_descriptions_v1.json`，SHA256为
`13522ef87bda8d760eb80c3655d55cd7c710b21e6c7db90b7c61cf2e19f50bfc`。
原矩阵 `assets/dior_v3_2_description_matrix.pt` 不覆盖。
新版源仍为 `aerial_descriptions.json`，新版缓存为
`assets/dior_v3_2_description_matrix_v2.pt`，3.2配置与生成工具默认值同步切换。

只变老师文本/缓存路径及其说明；2.17、检测器结构、O分支、RNR、编码器、
关系损失实现、权重0.01、温度0.03、topk=3、warmup=1000均不改。
新版本必须在独立目录从与2.17相同的预训练初始化开始；不要续接旧老师的训练。
在对比v1/v2时，类别名称有无及多条外观改写是一个组合变化，不能将AP差异
单独归因于“去类别名”。需要做单因素归因时应另设匹配对照，不能事后改本版文本。

代码验收检查类别对齐、长度、矩阵有限/对称/单位对角、源哈希及损失/梯度路径。
关系审计检查全部类的近邻和远邻，不仅看四个Novel类。可见的不合理候选关系
应如实记录，不能把文字或矩阵改成想要的数字后再宣称老师正确。
最终效果需要同协议、同种子消融比较Base AP50、Novel AP50、HM，以及固定区域
Novel→Base错分；损失有限或邻居看起来合理都不能替代检测指标。

## 冻结后的CLIP关系审计

本次使用与2.17相同的本地冻结 `clip-vit-large-patch14-336`，CPU、float32，
输入为20条正文，不拼接类别ID、不读取图像。老师源文件SHA256：
`0282a7a2a69db4071132509ef6d2b41a1124bf46a130f10c12d163559ba9ce58`。
下面是这一固定版本的诊断结果，不是按这些数值再改写的目标。

- 20个名称与baseline锚定顺序一致；实际长度44–56个token，含特殊token，均小于77。
- 原始矩阵20×20，非有限值0；最大不对称误差2.39e-7，单位对角最大误差4.18e-7。
- 非对角余弦范围0.423851–0.820604，均值0.632273；v1均值0.638413。
- 相比v1非对角元素平均绝对变化0.085957；均值降低本身不代表关系更正确。
- 老师行softmax排除自身，温度保持0.03；余弦不是视觉真值，也不是检测置信度。

| 检查的类对 | v1余弦 | v2余弦 | 可作出的有限判断 |
| --- | ---: | ---: | --- |
| airport / airplane | 0.810258 | 0.657699 | 设施与单体的文本关系减弱；不能据此断言更易识别机场 |
| basketballcourt / tenniscourt | 0.831795 | 0.820604 | 保留矩形运动场共享结构；不代表两个类应被合并 |
| groundtrackfield / stadium | 0.561811 | 0.704985 | 共享场地布局更突出，但仍未进入跑道类top3 |
| windmill / airplane | 0.745466 | 0.670868 | 关系减弱，叶片/机翼共享形状仍可能影响文本编码 |
| harbor / ship | 0.811248 | 0.654310 | 港区结构与单船轮廓的描述更分离；不代表分类收益已证实 |
| golffield / harbor | 0.726859 | 0.746914 | 可疑近邻，应重点观察；广域布局或地表词汇可能造成混淆 |
| vehicle / storagetank | 0.562551 | 0.775707 | 可疑近邻，说明去类别名不能自动得到可靠外观关系 |

最后两项是风险信号，不是已经查明的误差原因。仅凭文本编码不能判定所有类对的
视觉相似度：有的细长/圆形轮廓确实可能跨类共享，有的关系则可能来自编码器偏差。
此版不手工修改矩阵元素，不按主观指定顺序替换topk；也不换编码器以掩盖这次审计。

### 全部20类近邻/远邻

以下顺序均按原始余弦排序，括号内为余弦，排除自身。S-area和T-station分别是
`Expressway-Service-area`、`Expressway-toll-station`的显示缩写，不改变矩阵类别名。

| 类别 | top3近邻 | bottom3远邻 |
| --- | --- | --- |
| airplane | vehicle(.760), ship(.698), harbor(.694) | groundtrackfield(.528), basketballcourt(.579), chimney(.589) |
| airport | S-area(.708), trainstation(.696), overpass(.689) | storagetank(.445), chimney(.473), windmill(.533) |
| baseballfield | groundtrackfield(.780), stadium(.769), tenniscourt(.748) | windmill(.503), bridge(.557), chimney(.564) |
| basketballcourt | tenniscourt(.821), stadium(.744), baseballfield(.740) | windmill(.424), chimney(.499), airport(.548) |
| bridge | harbor(.768), ship(.758), trainstation(.753) | groundtrackfield(.527), windmill(.548), baseballfield(.557) |
| chimney | storagetank(.744), windmill(.675), vehicle(.663) | airport(.473), basketballcourt(.499), stadium(.502) |
| dam | harbor(.756), bridge(.748), overpass(.739) | windmill(.506), chimney(.552), groundtrackfield(.569) |
| S-area | T-station(.795), harbor(.777), overpass(.732) | windmill(.512), ship(.545), chimney(.556) |
| T-station | S-area(.795), overpass(.765), trainstation(.746) | windmill(.512), chimney(.574), ship(.577) |
| golffield | harbor(.747), S-area(.732), stadium(.728) | windmill(.488), ship(.556), chimney(.567) |
| groundtrackfield | baseballfield(.780), tenniscourt(.742), basketballcourt(.730) | windmill(.466), harbor(.519), bridge(.527) |
| harbor | S-area(.777), bridge(.768), dam(.756) | windmill(.499), groundtrackfield(.519), chimney(.520) |
| overpass | T-station(.765), dam(.739), S-area(.732) | storagetank(.559), stadium(.565), windmill(.577) |
| ship | bridge(.758), vehicle(.712), airplane(.698) | stadium(.523), airport(.541), S-area(.545) |
| stadium | baseballfield(.769), basketballcourt(.744), harbor(.741) | windmill(.433), chimney(.502), ship(.523) |
| storagetank | vehicle(.776), chimney(.744), stadium(.724) | airport(.445), tenniscourt(.557), overpass(.559) |
| tenniscourt | basketballcourt(.821), baseballfield(.748), groundtrackfield(.742) | windmill(.493), chimney(.522), storagetank(.557) |
| trainstation | bridge(.753), T-station(.746), overpass(.715) | chimney(.519), windmill(.548), basketballcourt(.566) |
| vehicle | storagetank(.776), airplane(.760), S-area(.728) | groundtrackfield(.539), airport(.541), dam(.588) |
| windmill | chimney(.675), airplane(.671), ship(.652) | basketballcourt(.424), stadium(.433), groundtrackfield(.466) |

top3是当前实现的相对正邻居，不是正样本标签；三元组约束仍会保留间隔，不会直接
把邻居类别映射为同一类。但可疑正邻居确实可能干扰原型几何，不能忽略。
因此本版满足文本规范与代码接入验收，**关系质量与检测收益仍待消融**。

### 接入验证结果

14项描述关系/版本测试与3项既有文本拓扑测试通过。另在完整3.2检测器上完成
CPU、batch=2、输入128×128、2步训练及推理检查，新老师成功加载，
`loss_text_description` 两步均为有限正值，主分支和OVC Adapter均有参数更新，
推理图像骨干调用次数保持1。报告为
`work_dirs/DIORv3.2_desc_v2_smoke/report.json`，状态 `passed`。
这只是小尺寸接入检查，不是完整训练、AP评测或GPU/AMP验证。
这些检查在文本冻结之后进行，其图像和结果没有参与描述起草或筛选。

2.17配置及本次保护的9个既有配置/模型文件修改前后SHA256一致；
原v1矩阵SHA256亦保持不变。没有启动正式训练，没有覆盖既有实验输出。

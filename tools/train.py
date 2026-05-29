import sys  # 导入 sys：用于访问解释器相关变量/函数（如 sys.path）
import os.path as osp  # 导入 os.path 并命名为 osp：用于路径拼接/规范化等
import os
import builtins
import json
# Windows 编码修复补丁
# 强制 Windows 下 open() 默认使用 utf-8，解决读取 JSON 数据集的 UnicodeDecodeError
if os.name == 'nt':
    _original_open = builtins.open
    def _utf8_open(file, mode='r', *args, **kwargs):
        # 如果是文本模式('r', 'w'等)且未指定 encoding，则强制使用 utf-8
        if 'b' not in mode and 'encoding' not in kwargs:
            kwargs['encoding'] = 'utf-8'
        return _original_open(file, mode, *args, **kwargs)
    builtins.open = _utf8_open

proj_root = osp.abspath(osp.join(osp.dirname(__file__), '..'))  # 计算项目根目录：当前文件所在目录的上一级的绝对路径
if proj_root not in sys.path:  # 若项目根目录不在模块搜索路径中
    sys.path.insert(0, proj_root)  # 将项目根目录插入到 sys.path 首位：确保可优先导入项目内模块

    # Copyright (c) OpenMMLab. All rights reserved.  # 版权声明（上游项目 OpenMMLab）
import argparse  # 导入 argparse：用于解析命令行参数
import logging  # 导入 logging：用于日志级别/日志输出相关


from mmengine.config import Config, DictAction  # 从 mmengine 导入 Config/DictAction：用于读取配置与解析 key=val 覆盖
from mmengine.logging import print_log  # 导入 print_log：用于带 logger 名称/级别的日志输出
from mmengine.runner import Runner  # 导入 Runner：mmengine 的训练/评测运行器

from mmyolo.registry import RUNNERS  # 导入 RUNNERS 注册表：用于构建自定义 runner
from mmyolo.utils import is_metainfo_lower  # 导入 is_metainfo_lower：校验 metainfo 字段是否全小写


def parse_args():  # 定义命令行参数解析函数
    parser = argparse.ArgumentParser(description='Train a detector')  # 创建参数解析器：描述为“训练检测器”
    parser.add_argument('--config', help='train config file path',  # 添加 --config 参数：配置文件路径
                        default=r'configs/pretrain/CKyolo_world_v2_l_clip_large_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_800ft_lvis_minival.py')  # 默认配置路径：你本地的预训练配置
    parser.add_argument('--work-dir', help='the dir to save logs and models')  # 添加 --work-dir 参数：日志与模型保存目录
    parser.add_argument(  # 开始定义 --amp 参数
        '--amp',  # 参数名：--amp
        action='store_true',  # 行为：出现则为 True
        default=False,  # 默认不启用 AMP
        help='enable automatic-mixed-precision training')  # 说明：启用自动混合精度训练
    parser.add_argument(  # 开始定义 --resume 参数
        '--resume',  # 参数名：--resume
        nargs='?',  # 可选地带一个值；若不给值则使用 const
        type=str,  # 值类型：字符串（checkpoint 路径或 'auto' 逻辑）
        const='auto',  # 仅写 --resume 时的默认值：auto
        help='If specify checkpoint path, resume from it, while if not '  # help 文本：若指定路径则从该 checkpoint 恢复
        'specify, try to auto resume from the latest checkpoint '  # help 文本续行：否则尝试自动从最新 checkpoint 恢复
        'in the work directory.')  # help 文本续行：work_dir 中寻找最新的
    parser.add_argument(  # 开始定义 --cfg-options 参数
        '--cfg-options',  # 参数名：--cfg-options
        nargs='+',  # 接收一个或多个 key=val
        action=DictAction,  # 使用 DictAction：将 key=val 解析为字典（支持嵌套/列表等）
        help='override some settings in the used config, the key-value pair '  # help：覆盖配置中的部分设置
        'in xxx=yyy format will be merged into config file. If the value to '  # help：xxx=yyy 会 merge 到 cfg
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '  # help：列表形式示例
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '  # help：允许嵌套 list/tuple 示例
        'Note that the quotation marks are necessary and that no white space '  # help：注意需要引号且不能有空格
        'is allowed.')  # help：结束
    parser.add_argument(  # 开始定义 --launcher 参数
        '--launcher',  # 参数名：--launcher
        choices=['none', 'pytorch', 'slurm', 'mpi'],  # 允许的 launcher 选项：单机/torchrun/slurm/mpi
        default='none',  # 默认 none：不走分布式 launcher
        help='job launcher')  # help：任务启动器
    parser.add_argument('--local_rank', type=int, default=0)  # 添加 --local_rank：分布式本地 rank（默认 0）
    args = parser.parse_args()  # 解析命令行参数
    if 'LOCAL_RANK' not in os.environ:  # 若环境变量里没有 LOCAL_RANK（某些启动方式不会自动设置）
        os.environ['LOCAL_RANK'] = str(args.local_rank)  # 将 CLI 的 local_rank 写入环境变量：供下游分布式初始化读取
    return args  # 返回解析后的参数对象


def main():  # 定义主入口函数
    args = parse_args()  # 解析命令行参数
    # load config  # 注释：加载配置
    cfg = Config.fromfile(args.config)  # 从文件读取配置：mmengine Config
    # replace the ${key} with the value of cfg.key  # 注释：（可选）替换配置中的 ${key} 占位符
    # cfg = replace_cfg_vals(cfg)  # 注释：原逻辑被注释掉：不执行 replace_cfg_vals
    cfg.launcher = args.launcher  # 将 launcher 写入配置：Runner 构建时会使用
    if args.cfg_options is not None:  # 若有通过命令行传入 cfg 覆盖项
        cfg.merge_from_dict(args.cfg_options)  # 将覆盖项合并进 cfg（优先级高于文件内默认）
    # work_dir is determined in this priority: CLI > segment in file > filename  # 注释：work_dir 优先级：CLI > 配置段 > 配置文件名
    if args.work_dir is not None:  # 若命令行显式指定了 work_dir
        # update configs according to CLI args if args.work_dir is not None  # 注释：根据 CLI 更新配置
        cfg.work_dir = args.work_dir  # 设置输出目录：保存日志与权重
    elif cfg.get('work_dir', None) is None:  # 否则，如果配置文件里也没设置 work_dir
        # use config filename as default work_dir if cfg.work_dir is None  # 注释：用配置文件名生成默认 work_dir
        if args.config.startswith('projects/'):  # 若配置在 projects/ 子路径下（约定的工程结构）
            config = args.config[len('projects/'):]  # 去掉 projects/ 前缀
            config = config.replace('/configs/', '/')  # 将 /configs/ 结构压平：生成更短的 work_dir 路径
            cfg.work_dir = osp.join('./work_dirs', osp.splitext(config)[0])  # work_dirs/<config-without-ext> 作为输出目录
        else:  # 否则：普通 configs 路径
            cfg.work_dir = osp.join('./work_dirs',  # 拼接 work_dirs 根目录
                                    osp.splitext(osp.basename(args.config))[0])  # 使用配置文件 basename 去扩展名作为子目录
    # enable automatic-mixed-precision training  # 注释：启用自动混合精度训练（AMP）
    if args.amp is True:  # 若 CLI 指定了 --amp
        optim_wrapper = cfg.optim_wrapper.type  # 读取配置中优化器包装器类型
        if optim_wrapper == 'AmpOptimWrapper':  # 若配置本身已经是 AmpOptimWrapper
            print_log(  # 打印日志提示：重复启用
                'AMP training is already enabled in your config.',  # 提示文本：AMP 已启用
                logger='current',  # logger 名称：current
                level=logging.WARNING)  # 日志级别：WARNING
        else:  # 否则：需要从普通 OptimWrapper 切到 AmpOptimWrapper
            assert optim_wrapper == 'OptimWrapper', (  # 断言：仅支持从 OptimWrapper 切换（否则配置不兼容）
                '`--amp` is only supported when the optimizer wrapper type is '  # 断言信息：--amp 仅支持 OptimWrapper
                f'`OptimWrapper` but got {optim_wrapper}.')  # 断言信息：展示当前 wrapper 类型
            cfg.optim_wrapper.type = 'AmpOptimWrapper'  # 将 wrapper 类型改为 AmpOptimWrapper
            cfg.optim_wrapper.loss_scale = 'dynamic'  # 设置动态 loss scale：减少溢出风险

    # resume is determined in this priority: resume from > auto_resume  # 注释：resume 优先级：显式 resume_from > auto_resume
    if args.resume == 'auto':  # 若 --resume 未给路径而是 auto（或仅写 --resume）
        cfg.resume = True  # 开启断点续训
        cfg.load_from = None  # 不指定权重路径：由框架自动寻找最新 checkpoint
    elif args.resume is not None:  # 若指定了具体 checkpoint 路径
        cfg.resume = True  # 开启断点续训
        cfg.load_from = args.resume  # 从指定 checkpoint 加载并恢复
    # Determine whether the custom metainfo fields are all lowercase  # 注释：检查自定义 metainfo 字段是否全小写（规范要求）
    is_metainfo_lower(cfg)  # 执行检查：必要时可能报错/告警（由实现决定）
    # build the runner from config  # 注释：根据配置构建 Runner
    if 'runner_type' not in cfg:  # 若配置中没有 runner_type：使用默认 Runner
        # build the default runner  # 注释：构建默认 runner
        runner = Runner.from_cfg(cfg)  # 使用 mmengine 默认方式从 cfg 构建 Runner
    else:  # 否则：配置指定了自定义 runner_type
        # build customized runner from the registry  # 注释：从注册表构建自定义 runner
        # if 'runner_type' is set in the cfg  # 注释：前提：cfg 中设置了 runner_type
        runner = RUNNERS.build(cfg)  # 使用 mmyolo 的 RUNNERS 注册表按 cfg 构建 Runner
    # start training  # 注释：启动训练流程
    runner.train()  # 调用 Runner.train：开始训练


if __name__ == '__main__':  # Python 脚本入口：仅当直接运行该文件时执行
    main()  # 调用 main：开始解析参数/构建 runner/训练

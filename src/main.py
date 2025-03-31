import numpy as np  # 用于数值计算
import os  # 操作文件和目录
import collections  # 提供容器数据类型
from os.path import dirname, abspath  # 用于获取文件绝对路径和目录
from copy import deepcopy  # 用于深拷贝对象
from sacred import Experiment, SETTINGS  # 导入 Sacred 实验相关类和全局设置
from sacred.observers import FileStorageObserver  # 用于将实验日志保存到文件
from sacred.utils import apply_backspaces_and_linefeeds  # 用于格式化捕获的输出信息
import sys  # 系统相关操作，比如获取命令行参数
import torch as th  # PyTorch，用于深度学习计算
from utils.logging import get_logger  # 自定义日志模块，获取全局 logger
import yaml  # 用于读取 YAML 配置文件

from run import run  # 导入运行流程的入口函数

# 设置 Sacred 的捕获模式，这里设为 "no" 则不捕获标准输出/错误流（可选 "fd"）
SETTINGS['CAPTURE_MODE'] = "no"  # "no" 或 "fd"
# 获取全局日志对象
logger = get_logger()

# 创建一个名为 "pymarl" 的 Sacred 实验
ex = Experiment("pymarl")
# 将自定义的 logger 赋值给 Sacred 实验
ex.logger = logger
# 设置输出过滤器，用于处理捕获的输出中的退格符等
ex.captured_out_filter = apply_backspaces_and_linefeeds

# 定义保存结果的路径，结果目录位于项目根目录下的 "results" 文件夹
results_path = os.path.join(dirname(dirname(abspath(__file__))), "results")


@ex.main
def my_main(_run, _config, _log):
    """
    Sacred 实验主函数入口
    参数:
      _run: Sacred 运行对象
      _config: 实验配置字典
      _log: 日志对象
    """
    # 对配置进行深拷贝，避免在后续修改时影响原始配置
    config = config_copy(_config)
    # 根据配置中的随机种子，设置 numpy 和 torch 的随机种子，确保实验可复现
    np.random.seed(config["seed"])
    th.manual_seed(config["seed"])
    # 同时，将随机种子传递到环境参数中
    config['env_args']['seed'] = config["seed"]

    # 运行整个训练/评估框架
    run(_run, config, _log)


def _get_config(params, arg_name, subfolder):
    """
    从命令行参数中获取指定配置文件的内容
    参数:
      params: 命令行参数列表
      arg_name: 需要查找的参数名称（例如 "--env-config" 或 "--config"）
      subfolder: 配置文件所在的子文件夹（例如 "envs" 或 "algs"）
    返回:
      解析后的配置字典
    """
    config_name = None
    # 遍历命令行参数，查找匹配指定参数名称的项
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            config_name = _v.split("=")[1]  # 获取参数值，即配置文件的名称
            del params[_i]  # 删除该参数，避免后续处理时重复解析
            break

    if config_name is not None:
        # 构造配置文件的完整路径，并读取 YAML 文件
        with open(os.path.join(os.path.dirname(__file__), "config", subfolder, "{}.yaml".format(config_name)), "r", encoding="utf-8") as f:
            try:
                # 使用 FullLoader 加载 YAML 文件
                config_dict = yaml.load(f, Loader=yaml.FullLoader)
            except yaml.YAMLError as exc:
                # 如果加载出错，则终止程序并报错
                assert False, "{}.yaml error: {}".format(config_name, exc)
        return config_dict


def recursive_dict_update(d, u):
    """
    递归地将字典 u 的键值对更新到字典 d 中
    参数:
      d: 原始字典（可为 None）
      u: 待更新的字典（可为 None）
    返回:
      更新后的字典
    """
    from collections.abc import Mapping
    if d is None:
        d = {}
    if u is None:
        u = {}
    # 遍历待更新字典中的所有键值对
    for k, v in u.items():
        if isinstance(v, Mapping):
            # 如果值为字典，则递归更新
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            # 否则直接覆盖
            d[k] = v
    return d


def config_copy(config):
    """
    深拷贝配置对象，支持字典和列表的递归拷贝
    参数:
      config: 配置对象（可能为字典、列表或其他类型）
    返回:
      拷贝后的配置对象
    """
    if isinstance(config, dict):
        return {k: config_copy(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [config_copy(v) for v in config]
    else:
        return deepcopy(config)


if __name__ == '__main__':
    # 复制命令行参数列表，用于后续处理
    params = deepcopy(sys.argv)

    # 从 default.yaml 获取默认配置，文件位于 config 目录下
    with open(os.path.join(os.path.dirname(__file__), "config", "default.yaml"), "r", encoding="utf-8") as f:
        try:
            # 使用 FullLoader 加载 YAML 文件
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)

    # 根据命令行参数加载环境和算法的基础配置
    env_config = _get_config(params, "--env-config", "envs")
    alg_config = _get_config(params, "--config", "algs")
    # 递归合并配置：先合并环境配置，再合并算法配置
    config_dict = recursive_dict_update(config_dict, env_config)
    config_dict = recursive_dict_update(config_dict, alg_config)

    # 将合并后的配置添加到 Sacred 实验中
    ex.add_config(config_dict)

    # 将实验日志保存到 results/sacred 目录
    logger.info("Saving to FileStorageObserver in results/sacred.")
    file_obs_path = os.path.join(results_path, "sacred")
    ex.observers.append(FileStorageObserver.create(file_obs_path))

    # 解析命令行参数并运行实验
    ex.run_commandline(params)

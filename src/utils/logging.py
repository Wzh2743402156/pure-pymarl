from collections import defaultdict
import logging
import numpy as np # type: ignore
import torch as th

class Logger:
    def __init__(self, console_logger):
        """
        初始化 Logger 类

        参数：
            console_logger: 控制台日志记录器，用于在终端打印日志信息

        作用：
            初始化 Logger 对象，设置是否启用 Tensorboard、Sacred 或 HDF 保存日志，
            并创建一个用于存储统计数据的 defaultdict
        """
        self.console_logger = console_logger

        # 标记是否启用 Tensorboard、Sacred 或 HDF 日志记录
        self.use_tb = False
        self.use_sacred = False
        self.use_hdf = False

        # stats 字典用于存储每个统计项的记录，默认值为列表
        self.stats = defaultdict(lambda: [])

    def setup_tb(self, directory_name):
        """
        配置 Tensorboard 日志记录

        参数：
            directory_name (str): Tensorboard 日志文件保存的目录

        作用：
            导入 tensorboard_logger 模块，配置日志保存目录，
            将 log_value 函数赋值给 self.tb_logger，并标记 use_tb 为 True
        """
        # 延迟导入 tensorboard_logger 模块，避免未安装时出错
        from tensorboard_logger import configure, log_value
        configure(directory_name)
        self.tb_logger = log_value
        self.use_tb = True

    def setup_sacred(self, sacred_run_dict):
        """
        配置 Sacred 日志记录

        参数：
            sacred_run_dict: Sacred 运行对象的字典（包含运行时的相关信息）

        作用：
            将 sacred_run_dict.info 保存到 self.sacred_info，并将 use_sacred 标记为 True，
            以便后续将统计数据同步到 Sacred 中。
        """
        self.sacred_info = sacred_run_dict.info
        self.use_sacred = True

    def log_stat(self, key, value, t, to_sacred=True):
        """
        记录统计数据

        参数：
            key (str): 统计数据的名称
            value: 统计数据的值（可能为 tensor、numpy 数组、list 或其他数值类型）
            t (int): 当前时间步或环境步数，用于标记统计数据的记录时刻
            to_sacred (bool): 是否将数据记录到 Sacred，默认为 True

        作用：
            将传入的 value 转换为 float 数值（如为 tensor、numpy 数组或 list 则取均值），
            然后记录到 self.stats 字典中，并在启用 Tensorboard 或 Sacred 时同步记录。
        """
        # 如果 value 为 torch.Tensor，则处理为标量或计算均值
        if isinstance(value, th.Tensor):
            if value.numel() == 1:
                value = value.item()
            else:
                value = float(th.mean(value).item())
        # 如果 value 为 numpy 数组，则处理为标量或计算均值
        elif isinstance(value, np.ndarray):
            if value.size == 1:
                value = float(value.item())
            else:
                value = float(np.mean(value))
        # 如果 value 为 list，则转换为 numpy 数组再计算均值
        elif isinstance(value, list):
            arr = np.array(value)
            if arr.size == 1:
                value = float(arr.item())
            else:
                value = float(np.mean(arr))
        else:
            try:
                value = float(value)
            except Exception:
                pass

        # 将 (t, value) 记录到对应的统计项中
        self.stats[key].append((t, value))

        # 如果启用了 Tensorboard，则调用 tb_logger 记录数据
        if self.use_tb:
            self.tb_logger(key, value, t)

        # 如果启用了 Sacred 并且允许记录，则将数据同步到 sacred_info 中
        if self.use_sacred and to_sacred:
            if key in self.sacred_info:
                self.sacred_info["{}_T".format(key)].append(t)
                self.sacred_info[key].append(value)
            else:
                self.sacred_info["{}_T".format(key)] = [t]
                self.sacred_info[key] = [value]

    def print_recent_stats(self):
        """
        打印最近的统计数据

        作用：
            根据 self.stats 中保存的统计数据，计算最近几个时间步（默认窗口大小为 5）的均值，
            并格式化输出到控制台。特殊情况下，如 "epsilon" 只取最近一个数据。
        """
        # 构造日志字符串，首先记录环境步数和回合数（假设 "episode" 键存在且其最后一项为 (t_env, episode)）
        log_str = "Recent Stats | t_env: {:>10} | Episode: {:>8}\n".format(*self.stats["episode"][-1])
        i = 0
        # 遍历所有统计项，跳过 "episode" 项
        for (k, v) in sorted(self.stats.items()):
            if k == "episode":
                continue
            i += 1
            # 设置统计窗口：默认取最近 5 个数据，对于 "epsilon" 只取最近 1 个
            window = 5 if k != "epsilon" else 1
            # 针对最近 window 个数据计算均值，若数据为 tensor，则先转换为 numpy 数组
            import torch
            item = "{:.4f}".format(np.mean([x[1].cpu().numpy() if isinstance(x[1], torch.Tensor) else x[1] for x in self.stats[k][-window:]]))
            log_str += "{:<25}{:>8}".format(k + ":", item)
            log_str += "\n" if i % 4 == 0 else "\t"
        self.console_logger.info(log_str)


def get_logger():
    """
    创建并返回一个自定义的全局日志记录器

    作用：
        - 清除默认 logger 中的所有 handler
        - 添加一个 StreamHandler，并设置日志格式与时间格式
        - 设置日志级别为 DEBUG
        - 返回配置好的 logger 对象

    返回：
        logger: 配置好的全局日志记录器对象
    """
    logger = logging.getLogger()
    # 清除默认 handler
    logger.handlers = []
    ch = logging.StreamHandler()
    # 设置日志输出格式：包括日志级别、时间、记录器名称和消息内容
    formatter = logging.Formatter('[%(levelname)s %(asctime)s] %(name)s %(message)s', '%H:%M:%S')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel('DEBUG')
    return logger

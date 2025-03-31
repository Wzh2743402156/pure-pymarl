import numpy as np # type: ignore

class DecayThenFlatSchedule():
    def __init__(self, start, finish, time_length, decay="exp"):
        """
        初始化衰减然后平稳调度器

        参数：
            start (float): 起始值（初始阶段的值）。
            finish (float): 最终平稳值（衰减结束后保持的值）。
            time_length (int 或 float): 衰减阶段的持续时间（步数或时间长度）。
            decay (str): 衰减方式，默认为 "exp" 表示指数衰减，也可选 "linear" 表示线性衰减。

        作用：
            根据提供的参数初始化调度器，计算线性衰减时的步长 delta，以及在指数衰减方式下的 scaling 参数，
            以便后续通过 eval 方法计算给定时间 T 时的调度值。
        """
        self.start = start
        self.finish = finish
        self.time_length = time_length
        # 计算线性衰减每步的变化量 delta，即 (start - finish) 除以衰减持续的时间长度
        self.delta = (self.start - self.finish) / self.time_length
        self.decay = decay

        if self.decay in ["exp"]:
            # 对于指数衰减，计算一个缩放因子 exp_scaling，用于控制衰减速率
            # 当 finish > 0 时，根据公式 scaling = -time_length / log(finish) 计算，否则默认设为 1
            self.exp_scaling = (-1) * self.time_length / np.log(self.finish) if self.finish > 0 else 1

    def eval(self, T):
        """
        根据当前时间步 T 评估调度值

        参数：
            T (int 或 float): 当前时间步数或时间点

        返回：
            float: 当前的调度值

        作用：
            根据设定的衰减方式（线性或指数）计算当前时间 T 时的调度值：
            - 如果采用线性衰减，则返回 max(finish, start - delta * T)；
            - 如果采用指数衰减，则计算 np.exp(- T / exp_scaling)，并确保结果介于 finish 与 start 之间。
        """
        if self.decay in ["linear"]:
            # 线性衰减：从 start 线性减少，每步减 delta，但不会低于 finish
            return max(self.finish, self.start - self.delta * T)
        elif self.decay in ["exp"]:
            # 指数衰减：计算指数衰减值，同时确保结果不会超过 start，也不会低于 finish
            return min(self.start, max(self.finish, np.exp(- T / self.exp_scaling)))
    pass

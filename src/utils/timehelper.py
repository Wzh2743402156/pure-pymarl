import time
import numpy as np # type: ignore

def print_time(start_time, T, t_max, episode, episode_rewards):
    """
    打印当前训练进度的时间信息

    参数：
        start_time (float): 训练开始的时间戳（由 time.time() 得到）
        T (int): 当前已进行的时间步数（或当前回合的步数）
        t_max (int): 训练的最大时间步数
        episode (int): 当前回合的编号
        episode_rewards (list): 保存各回合奖励的列表

    作用：
        1. 计算从训练开始到当前的已用时间（time_elapsed）。
        2. 根据当前步数与最大步数估计剩余时间（time_left）。
        3. 若剩余时间过长（超过 100 天），则将其限制在 100 天以内。
        4. 计算最近一段回合的平均奖励（如果 episode_rewards 长度足够）。
        5. 格式化并打印训练进度、步数、奖励、已用时间和预计剩余时间等信息，
           利用 ANSI 转义序列在终端刷新输出（end="\r" 实现覆盖显示）。
    """
    # 计算已经消耗的时间
    time_elapsed = time.time() - start_time
    # 避免 T 为 0，至少保证为 1
    T = max(1, T)
    # 根据当前时间步 T 计算预计剩余时间：已用时间 * (剩余步数 / 已完成步数)
    time_left_value = time_elapsed * (t_max - T) / T
    # 防止剩余时间超过 100 天（60*60*24*100 秒）
    time_left_value = min(time_left_value, 60 * 60 * 24 * 100)
    # 默认最后奖励设为 "N\A"
    last_reward = "N\A"
    # 如果记录的回合奖励数量超过 5，则计算最近 50 个回合奖励的平均值
    if len(episode_rewards) > 5:
        last_reward = "{:.2f}".format(np.mean(episode_rewards[-50:]))
    # 使用 ANSI 转义序列刷新终端输出，显示当前回合、当前步数、最大步数、平均奖励、已用时间和剩余时间
    print("\033[F\033[F\x1b[KEp: {:,}, T: {:,}/{:,}, Reward: {}, \n\x1b[KElapsed: {}, Left: {}\n".format(
          episode, T, t_max, last_reward, time_str(time_elapsed), time_str(time_left_value)), " " * 10, end="\r")


def time_left(start_time, t_start, t_current, t_max):
    """
    计算预计剩余时间

    参数：
        start_time (float): 训练开始的时间戳
        t_start (int): 计时起始步数（通常是某个标记步数）
        t_current (int): 当前步数
        t_max (int): 最大步数

    返回：
        str: 格式化后的剩余时间字符串

    作用：
        1. 如果当前步数已达到或超过最大步数，则返回 "-" 表示无剩余时间。
        2. 否则，计算从 t_start 到 t_current 的时间消耗，再估计剩余时间。
        3. 防止剩余时间超过 100 天，将其限制在 100 天以内。
        4. 返回一个经过格式化的时间字符串（如 "1 hours, 23 minutes, 45 seconds"）。
    """
    if t_current >= t_max:
        return "-"
    # 计算已经消耗的时间
    time_elapsed = time.time() - start_time
    t_current = max(1, t_current)  # 避免除数为 0
    # 计算剩余时间，比例为：已用时间 * (剩余步数 / (当前步数 - 起始步数))
    time_left_value = time_elapsed * (t_max - t_current) / (t_current - t_start)
    # 限制剩余时间不超过 100 天
    time_left_value = min(time_left_value, 60 * 60 * 24 * 100)
    return time_str(time_left_value)


def time_str(s):
    """
    将秒数转换为更易读的格式（天、小时、分钟和秒）

    参数：
        s (float): 时间长度（秒）

    返回：
        str: 格式化后的时间字符串

    作用：
        1. 使用 divmod 将秒数转换为天、小时、分钟和秒。
        2. 根据天、小时、分钟的非零情况拼接成字符串，最后总是显示秒数。
    """
    # 将 s 转换为天数和剩余秒数
    days, remainder = divmod(s, 60 * 60 * 24)
    # 将剩余秒数转换为小时数和剩余秒数
    hours, remainder = divmod(remainder, 60 * 60)
    # 将剩余秒数转换为分钟数和秒数
    minutes, seconds = divmod(remainder, 60)
    string = ""
    # 如果天数大于 0，则添加天的信息
    if days > 0:
        string += "{:d} days, ".format(int(days))
    # 如果小时数大于 0，则添加小时的信息
    if hours > 0:
        string += "{:d} hours, ".format(int(hours))
    # 如果分钟数大于 0，则添加分钟的信息
    if minutes > 0:
        string += "{:d} minutes, ".format(int(minutes))
    # 始终添加秒的信息
    string += "{:d} seconds".format(int(seconds))
    return string

from envs import REGISTRY as env_REGISTRY           # 从 env 模块的注册表中加载环境
from functools import partial                       # 用于构造部分函数
from components.episode_buffer import EpisodeBatch  # EpisodeBatch 用于存储和处理采样的回合数据
import numpy as np  # type: ignore                  # 用于数值计算

class EpisodeRunner:
    def __init__(self, args, logger):
        """
        初始化 EpisodeRunner
        
        参数：
            args: 配置参数对象，包含环境名称、环境参数、批次大小、测试回合数、日志间隔等信息
            logger: 日志记录器，用于记录训练和测试过程中的统计数据
        
        作用：
            根据配置初始化环境、设置回合长度、计数器以及日志记录变量。
        """
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run  # 批次大小（并行运行的环境数量），此处要求必须为 1
        assert self.batch_size == 1

        # 根据配置创建环境实例（通过 env_REGISTRY 动态加载对应环境类）
        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.episode_limit = self.env.episode_limit  # 当前环境的最大回合步数
        self.t = 0  # 当前回合的步数

        self.t_env = 0  # 全局环境时间步数计数

        # 用于记录训练和测试过程中每回合的总奖励
        self.train_returns = []
        self.test_returns = []
        # 用于记录训练和测试过程中的统计数据（例如各项指标的累积）
        self.train_stats = {}
        self.test_stats = {}

        # 初始化日志记录的时间步（用于控制训练日志的输出频率）
        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, mac):
        """
        设置 EpisodeRunner 的数据处理方式和多智能体控制器
        
        参数：
            scheme: 数据结构方案，描述 EpisodeBatch 的字段和数据形状
            groups: 数据中不同组（例如智能体组）的配置信息
            preprocess: 数据预处理方法（例如对动作进行 one-hot 编码）
            mac: 多智能体控制器，用于在每个时间步根据观测选择动作
        
        作用：
            构造一个新的 EpisodeBatch 部分函数，并保存传入的多智能体控制器（mac）。
        """
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac

    def get_env_info(self):
        """
        获取环境信息
        
        返回：
            dict: 包含环境基本信息，如全局状态维度、智能体观察维度、可用动作数、智能体数量、回合步数上限等
        """
        return self.env.get_env_info()

    def save_replay(self):
        """
        保存环境交互回放
        
        作用：
            调用环境的 save_replay 方法，保存当前交互过程的回放数据。
        """
        self.env.save_replay()

    def close_env(self):
        """
        关闭环境
        
        作用：
            调用环境的 close 方法，释放资源或关闭窗口。
        """
        self.env.close()

    def reset(self):
        """
        重置当前回合数据
        
        作用：
            创建一个新的 EpisodeBatch，并重置环境和回合步数计数。
        """
        self.batch = self.new_batch()  # 新建一个回合数据批次
        self.env.reset()               # 重置环境到初始状态
        self.t = 0                     # 重置当前回合步数

    def run(self, test_mode=False):
        """
        执行一个完整的回合
        
        参数：
            test_mode (bool): 是否为测试模式，测试模式下可能使用不同的策略（例如贪心策略）
        
        作用：
            1. 重置环境和回合数据。
            2. 在每个时间步采集状态、可用动作和观察数据，并传递给控制器选择动作。
            3. 根据所选动作与环境交互，收集奖励、终止标志和其他环境信息。
            4. 更新回合数据批次，直至回合结束。
            5. 记录回合总奖励、更新统计信息，并根据设置输出日志。
        
        返回：
            EpisodeBatch: 包含当前回合所有数据的批次对象。
        """
        self.reset()

        terminated = False  # 回合结束标志
        episode_return = 0  # 回合累计奖励
        # 初始化多智能体控制器的隐藏状态，batch_size 为 1
        self.mac.init_hidden(batch_size=self.batch_size)

        while not terminated:
            # 收集时间步 t 前的状态、可用动作和观测数据
            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()]
            }
            # 更新回合数据批次（在时间步 t 处添加 pre_transition_data）
            self.batch.update(pre_transition_data, ts=self.t)

            # 传入目前为止的回合数据，获取当前时间步各智能体选择的动作（批次大小为 1）
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)

            # 与环境交互，执行所选动作（注意 actions[0]，因为 batch_size 为 1）
            reward, terminated, env_info = self.env.step(actions[0])
            episode_return += reward  # 累加本回合奖励
            
            # 构造后续数据，包括选择的动作、奖励和终止标志
            post_transition_data = {
                "actions": actions,
                "reward": [(reward,)],  # 奖励包装成元组
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }
            # 更新回合数据批次（在时间步 t 处添加后续数据）
            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1  # 时间步加 1

        # 回合结束后，再次记录环境的最后状态、可用动作和观察
        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()]
        }
        self.batch.update(last_data, ts=self.t)

        # 在最后状态下再次选择动作（通常用于保存最终决策）
        actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        self.batch.update({"actions": actions}, ts=self.t)

        # 根据模式选择使用训练或测试的统计数据和奖励列表
        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        # 更新统计数据：将当前环境信息中的各项值累加到 cur_stats 中
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t  # 更新全局环境步数

        cur_returns.append(episode_return)  # 记录当前回合累计奖励

        # 根据测试模式或训练日志间隔判断是否需要记录日志
        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            # 如果智能体选择器具有 epsilon 属性，则记录当前 epsilon 值
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        # 返回当前回合收集的全部数据
        return self.batch

    def _log(self, returns, stats, prefix):
        """
        记录统计信息
        
        参数：
            returns (list): 回合累计奖励列表
            stats (dict): 统计数据字典（例如环境信息的累计）
            prefix (str): 日志前缀（例如 "test_" 用于测试日志）
        
        作用：
            计算并记录奖励的均值、标准差以及其他统计指标，然后清空相应数据。
        """
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()  # 清空奖励列表

        # 对统计数据中的每一项（除了 n_episodes）计算平均值后记录
        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean", v / stats["n_episodes"], self.t_env)
        stats.clear()  # 清空统计数据

from envs.multiagentenv import MultiAgentEnv
import numpy as np # type: ignore
import random
import logging
import sys

# 设置全局日志等级为 INFO，并输出到控制台
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s %(asctime)s] %(message)s',
    datefmt='%H:%M:%S',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)


class CustomEnv(MultiAgentEnv):
    def __init__(self, n_agents=3, state_dim=10, obs_dim=5, episode_limit=100, **kwargs):
        """
        初始化自定义环境实例
        
        参数：
            n_agents (int): 环境中智能体的数量。
            state_dim (int): 全局状态的维度（特征数量）。
            obs_dim (int): 每个智能体的局部观察维度（特征数量）。
            episode_limit (int): 每个回合的最大步数。
            **kwargs: 其他额外参数，例如随机种子 seed。
        
        作用：
            初始化环境中的基本参数、状态和步数计数，并调用 reset 方法生成初始状态。
        """
        logger.info("[ENV] CustomEnv Initialized")
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.episode_limit = episode_limit
        self.t = 0  # 当前步数
        self.seed_val = kwargs.get("seed", None)
        if self.seed_val is not None:
            self.seed(self.seed_val)
        self.reset()  # 初始化状态和观察
        logger.info("[ENV] CustomEnv Initialized")

    def reset(self):
        """
        重置环境
        
        作用：
            重置内部步数计数，并随机生成初始全局状态和各智能体的初始观察值。
        
        返回：
            tuple: (observations, state)
                - observations: 一个形状为 (n_agents, obs_dim) 的 numpy 数组，表示所有智能体的初始观察。
                - state: 一个形状为 (state_dim,) 的 numpy 数组，表示初始的全局状态。
        """
        self.t = 0
        self.state = np.random.rand(self.state_dim)
        # 生成形状为 (n_agents, obs_dim) 的随机观察，表示每个智能体的初始观察
        self.obs = np.random.rand(self.n_agents, self.obs_dim)
        return self.get_obs(), self.get_state()

    def step(self, actions):
        """
        执行环境一步交互
        
        参数：
            actions: 当前所有智能体选择的动作（本示例中未具体使用该参数）。
        
        作用：
            根据传入的动作更新环境状态，增加步数计数，并随机生成新的状态和观察，
            同时计算并返回奖励、是否结束以及附加信息。
        
        返回：
            tuple: (reward, done, info)
                - reward: 一个形状为 (1,) 的 numpy 数组，表示本步的全局奖励（随机生成）。
                - done: 布尔值，指示当前回合是否结束（当步数达到 episode_limit 时为 True）。
                - info: 字典，包含当前环境的状态和观察（例如 {"state": state, "obs": obs}）。
        """
        self.t += 1
        # 随机生成一个奖励值，这里可根据需要修改为更合理的奖励计算方式
        reward_value = np.random.rand()
        reward = np.array([reward_value])  # 保证奖励的形状为 (1,)
        done = self.t >= self.episode_limit  # 判断是否达到回合结束条件
        # 更新状态和观察，这里均随机生成
        self.state = np.random.rand(self.state_dim)
        self.obs = np.random.rand(self.n_agents, self.obs_dim)
        info = {"state": self.state, "obs": self.obs}
        return reward, done, info

    def get_obs(self):
        """
        获取所有智能体的观察
        
        作用：
            返回当前所有智能体的局部观察数据。
        
        返回：
            numpy 数组: 形状为 (n_agents, obs_dim) 的数组，表示每个智能体的观察。
        """
        return self.obs

    def get_obs_agent(self, agent_id):
        """
        获取指定智能体的观察
        
        参数：
            agent_id (int): 智能体的索引。
        
        作用：
            返回指定智能体的局部观察数据。
        
        返回：
            numpy 数组: 形状为 (obs_dim,) 的数组，表示该智能体的观察。
        """
        return self.obs[agent_id]

    def get_obs_size(self):
        """
        获取单个智能体观察的维度
        
        作用：
            返回每个智能体观察向量的长度（维度）。
        
        返回：
            int: 观察维度，即 obs_dim。
        """
        return self.obs_dim

    def get_state(self):
        """
        获取当前全局状态
        
        作用：
            返回当前环境的全局状态信息。
        
        返回：
            numpy 数组: 形状为 (state_dim,) 的数组，表示当前全局状态。
        """
        return self.state

    def get_state_size(self):
        """
        获取全局状态的维度
        
        作用：
            返回全局状态向量的长度（维度）。
        
        返回：
            int: 全局状态的维度，即 state_dim。
        """
        return self.state_dim

    def get_avail_actions(self):
        """
        获取每个智能体可用的动作
        
        作用：
            返回当前状态下，每个智能体可执行的动作列表。
            此处假设每个智能体都有 5 个离散动作可选，并用 1 表示该动作可用。
        
        返回：
            numpy 数组: 形状为 (n_agents, 5) 的数组，每个元素为 1，表示对应动作可用。
        """
        return np.ones((self.n_agents, 5), dtype=np.int32)

    def get_avail_agent_actions(self, agent_id):
        """
        获取指定智能体可用的动作
        
        参数：
            agent_id (int): 智能体的索引。
        
        作用：
            返回指定智能体在当前状态下可执行的动作列表。
        
        返回：
            numpy 数组: 一维数组，长度为 5，每个元素为 1，表示对应动作可用。
        """
        return np.ones(5, dtype=np.int32)

    def get_total_actions(self):
        """
        获取每个智能体的动作总数
        
        作用：
            返回环境中每个智能体可以选择的总动作数量。
        
        返回：
            int: 总动作数，这里固定为 5。
        """
        return 5

    def render(self):
        """
        渲染环境
        
        作用：
            打印当前全局状态以及每个智能体的观察，用于调试或简单的可视化。
        """
        print("State:", self.state)
        for i in range(self.n_agents):
            print(f"Agent {i} obs:", self.obs[i])

    def close(self):
        """
        关闭环境
        
        作用：
            进行环境关闭操作，释放相关资源。
            此处无额外资源需要释放，方法为空。
        """
        pass

    def seed(self, seed=None):
        """
        设置随机种子
        
        参数：
            seed (int): 随机种子值，用于保证实验的可复现性。
        
        作用：
            设置 numpy 和 random 模块的随机种子，并记录该种子值。
        """
        self.seed_val = seed
        np.random.seed(seed)
        random.seed(seed)

    def save_replay(self):
        """
        保存回放
        
        作用：
            用于保存环境交互过程的回放数据。
            当前未实现，用户可根据需要扩展此接口。
        """
        pass

    def get_env_info(self):
        """
        获取环境基本信息
        
        作用：
            返回一个字典，包含环境的基础信息，供 PyMARL 框架使用。
            主要信息包括全局状态维度、单个智能体观察维度、动作总数、智能体数量以及回合最大步数。
        
        返回：
            dict: 包含以下键值对：
                - "state_shape": 全局状态维度（state_dim）
                - "obs_shape": 单个智能体观察维度（obs_dim）
                - "n_actions": 每个智能体的动作总数
                - "n_agents": 环境中智能体的数量
                - "episode_limit": 每个回合的最大步数
        """
        return {
            "state_shape": self.get_state_size(),
            "obs_shape": self.get_obs_size(),
            "n_actions": self.get_total_actions(),
            "n_agents": self.n_agents,
            "episode_limit": self.episode_limit
        }

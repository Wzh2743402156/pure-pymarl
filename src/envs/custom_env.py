from envs.multiagentenv import MultiAgentEnv
import numpy as np
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
        logger.info("[ENV] CustomEnv Initialized")
        """
        初始化自定义环境
        参数：
            n_agents: 智能体数量
            state_dim: 全局状态的维度
            obs_dim: 每个智能体的观察维度
            episode_limit: 每个回合的最大步数
            **kwargs: 其他额外参数，如 seed
        """
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.episode_limit = episode_limit
        self.t = 0  # 当前步数
        self.seed_val = kwargs.get("seed", None)
        if self.seed_val is not None:
            self.seed(self.seed_val)
        self.reset()
        logger.info("[ENV] CustomEnv Initialized")

    def reset(self):
        """
        重置环境，返回初始观察和状态
        """
        self.t = 0
        self.state = np.random.rand(self.state_dim)
        # 生成 shape=(n_agents, obs_dim) 的随机观察
        self.obs = np.random.rand(self.n_agents, self.obs_dim)
        return self.get_obs(), self.get_state()

    def step(self, actions):
        self.t += 1
        # print("[STEP] ", self.t)
        # 生成全局奖励：例如，所有智能体奖励的均值（也可以使用求和）
        reward_value = np.random.rand()  
        reward = np.array([reward_value])  # 确保奖励形状为 (1,)
        done = self.t >= self.episode_limit
        self.state = np.random.rand(self.state_dim)
        self.obs = np.random.rand(self.n_agents, self.obs_dim)
        info = {"state": self.state, "obs": self.obs}
        return reward, done, info

    def get_obs(self):
        """
        返回所有智能体的观察，形状为 (n_agents, obs_dim)
        """
        return self.obs

    def get_obs_agent(self, agent_id):
        """
        返回指定智能体的观察
        """
        return self.obs[agent_id]

    def get_obs_size(self):
        """
        返回单个智能体观察的维度
        """
        return self.obs_dim

    def get_state(self):
        """
        返回全局状态
        """
        return self.state

    def get_state_size(self):
        """
        返回全局状态的维度
        """
        return self.state_dim

    def get_avail_actions(self):
        """
        返回每个智能体可执行的动作。
        此处假设每个智能体都有 5 个离散动作可选，返回 shape=(n_agents, 5) 的数组。
        """
        return np.ones((self.n_agents, 5), dtype=np.int32)

    def get_avail_agent_actions(self, agent_id):
        """
        返回指定智能体可执行的动作（一个长度为5的一维数组）
        """
        return np.ones(5, dtype=np.int32)

    def get_total_actions(self):
        """
        返回每个智能体的总动作数（假设为 5）
        """
        return 5

    def render(self):
        """
        简单的渲染方法，打印当前状态及各智能体观察
        """
        print("State:", self.state)
        for i in range(self.n_agents):
            print(f"Agent {i} obs:", self.obs[i])

    def close(self):
        """
        关闭环境（此处无需释放额外资源）
        """
        pass

    def seed(self, seed=None):
        """
        设置随机种子，保证实验可复现
        """
        self.seed_val = seed
        np.random.seed(seed)
        random.seed(seed)

    def save_replay(self):
        """
        保存回放（此处未实现）
        """
        pass

    def get_env_info(self):
        """
        返回一个字典，包含环境的基本信息，
        用于 PyMARL 注册环境时调用。
        """
        return {
            "state_shape": self.get_state_size(),
            "obs_shape": self.get_obs_size(),
            "n_actions": self.get_total_actions(),
            "n_agents": self.n_agents,
            "episode_limit": self.episode_limit
        }

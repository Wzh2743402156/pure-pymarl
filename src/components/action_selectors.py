import torch as th
from torch.distributions import Categorical
from .epsilon_schedules import DecayThenFlatSchedule

# 定义一个全局字典 REGISTRY，用于注册不同的动作选择器
REGISTRY = {}

# ----------------------- MultinomialActionSelector -----------------------

class MultinomialActionSelector():
    def __init__(self, args):
        """
        初始化 MultinomialActionSelector（多项式动作选择器）

        参数：
            args: 参数对象，包含 epsilon 开始值、结束值、退火时间以及测试时是否采用贪心策略等超参数

        作用：
            根据参数创建一个衰减调度器，用于动态调整 epsilon 值（探索率）。
            初始化 epsilon 值，并读取是否在测试模式下采用贪心策略（test_greedy）。
        """
        self.args = args
        # 使用线性衰减方式初始化调度器
        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time, decay="linear")
        # 在时间步 0 时获取初始 epsilon 值
        self.epsilon = self.schedule.eval(0)
        # 从参数中获取 test_greedy 属性，默认为 True（测试时采用贪心选择动作）
        self.test_greedy = getattr(args, "test_greedy", True)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        """
        根据智能体输出和可用动作选择最终动作

        参数：
            agent_inputs (Tensor): 智能体输出的策略（例如 Q 值或概率分布），形状为 (batch, n_agents, n_actions)
            avail_actions (Tensor): 可用动作掩码，形状与 agent_inputs 相同或最后一维为动作数，
                                      值为 0 表示该动作不可用
            t_env (int): 当前环境全局时间步数，用于更新 epsilon
            test_mode (bool): 是否处于测试模式

        过程：
            1. 克隆智能体输出到 masked_policies，并将不可用动作位置置为 0；
            2. 根据当前 t_env 更新 epsilon 值；
            3. 如果处于测试模式且 test_greedy 为 True，则直接选择每个智能体输出中最大值对应的动作；
            4. 否则，使用多项式分布根据 masked_policies 采样动作。

        返回：
            picked_actions (Tensor): 选定的动作，形状为 (batch, n_agents)
        """
        # 克隆策略输出，避免直接修改原始 tensor
        masked_policies = agent_inputs.clone()
        # 对于不可用的动作，将对应的概率或 Q 值置为 0
        masked_policies[avail_actions == 0.0] = 0.0

        # 根据当前环境时间步 t_env 更新 epsilon 值
        self.epsilon = self.schedule.eval(t_env)

        if test_mode and self.test_greedy:
            # 测试模式下且采用贪心策略：选择最大概率/值的动作
            picked_actions = masked_policies.max(dim=2)[1]
        else:
            # 非测试模式或不使用贪心策略：使用多项式分布采样动作
            picked_actions = Categorical(masked_policies).sample().long()

        return picked_actions

# 将 MultinomialActionSelector 注册到 REGISTRY 中，键名为 "multinomial"
REGISTRY["multinomial"] = MultinomialActionSelector

# ----------------------- EpsilonGreedyActionSelector -----------------------

class EpsilonGreedyActionSelector():
    def __init__(self, args):
        """
        初始化 EpsilonGreedyActionSelector（ε-贪心动作选择器）

        参数：
            args: 参数对象，包含 epsilon 的起始、结束值、退火时间等参数

        作用：
            根据参数创建一个线性衰减调度器，用于动态调整 epsilon 值（探索率）。
            初始化 epsilon 值。
        """
        self.args = args
        # 初始化线性衰减调度器
        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time, decay="linear")
        self.epsilon = self.schedule.eval(0)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        """
        根据智能体输出和可用动作选择动作，采用 ε-贪心策略

        参数：
            agent_inputs (Tensor): 智能体输出的 Q 值，形状为 (batch, n_agents, n_actions)
            avail_actions (Tensor): 可用动作掩码，形状与 agent_inputs 相同，
                                      不可用动作标记为 0.0
            t_env (int): 当前环境全局时间步数，用于更新 epsilon
            test_mode (bool): 是否处于测试模式

        过程：
            1. 根据当前 t_env 更新 epsilon 值；
            2. 如果处于测试模式，则将 epsilon 设为 0（完全贪心）；
            3. 将不可用动作对应的 Q 值设置为 -inf，确保不被选中；
            4. 生成与每个智能体动作空间相同形状的随机数，用于判断是否随机选取动作；
            5. 对于随机选择的情况，从可用动作中采样随机动作；
            6. 对于非随机选择的情况，选择 Q 值最大的动作；
            7. 根据随机选择的掩码组合得到最终的选定动作。

        返回：
            picked_actions (Tensor): 选定的动作，形状为 (batch, n_agents)
        """
        # 根据当前环境时间步更新 epsilon
        self.epsilon = self.schedule.eval(t_env)

        if test_mode:
            # 测试模式下完全采用贪心策略，不进行随机选择
            self.epsilon = 0.0

        # 克隆 Q 值 tensor，用于掩码处理
        masked_q_values = agent_inputs.clone()
        # 将不可用动作的 Q 值设为负无穷，确保不会被选中
        masked_q_values[avail_actions == 0.0] = -float("inf")

        # 生成与 (batch, n_agents) 形状相同的随机数 tensor，范围为 [0, 1)
        random_numbers = th.rand_like(agent_inputs[:, :, 0])
        # 生成随机选择掩码：若随机数小于 epsilon，则该位置为 1（随机选择），否则为 0
        pick_random = (random_numbers < self.epsilon).long()
        # 对可用动作（avail_actions）进行采样，得到随机动作
        random_actions = Categorical(avail_actions.float()).sample().long()

        # 对于每个位置：若 pick_random 为 1，则使用随机动作；否则使用贪心选择的最大 Q 值对应的动作
        picked_actions = pick_random * random_actions + (1 - pick_random) * masked_q_values.max(dim=2)[1]
        return picked_actions

# 将 EpsilonGreedyActionSelector 注册到 REGISTRY 中，键名为 "epsilon_greedy"
REGISTRY["epsilon_greedy"] = EpsilonGreedyActionSelector

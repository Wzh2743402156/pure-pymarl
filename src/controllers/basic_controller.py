from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th

# 这个多智能体控制器（MAC）在各个智能体之间共享参数
class BasicMAC:
    def __init__(self, scheme, groups, args):
        """
        初始化 BasicMAC 控制器

        参数：
            scheme (dict): 数据结构方案，描述回合数据中各字段的形状（例如观察、动作等）
            groups (dict): 数据中的分组信息（例如智能体分组）
            args: 参数对象，包含智能体数量、智能体类型、动作选择器类型、是否观察上一次动作、是否包含智能体ID等超参数

        作用：
            根据输入方案构造智能体，并初始化动作选择器以及隐藏状态。
        """
        self.n_agents = args.n_agents
        self.args = args
        # 根据方案计算每个智能体输入的维度
        input_shape = self._get_input_shape(scheme)
        # 构建智能体模型（共享参数）
        self._build_agents(input_shape)
        # 保存智能体输出类型（例如 "q" 或 "pi_logits"）
        self.agent_output_type = args.agent_output_type

        # 初始化动作选择器，根据 args.action_selector 从注册表中加载相应的类实例
        self.action_selector = action_REGISTRY[args.action_selector](args)

        # 初始化隐藏状态（后续在 forward 时会更新）
        self.hidden_states = None

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        """
        根据当前回合数据选择动作

        参数：
            ep_batch: 当前回合的批次数据，包含各个时间步的观察、可用动作等信息
            t_ep (int): 当前回合的时间步索引
            t_env (int): 当前环境的全局时间步数
            bs: 批次选择器，默认为全部批次（本控制器批次大小通常为1）
            test_mode (bool): 是否为测试模式

        作用：
            1. 从批次数据中获取当前时间步所有智能体的可用动作；
            2. 调用 forward 方法计算智能体输出（例如 Q 值或 logits）；
            3. 使用动作选择器从智能体输出和可用动作中选择最终动作。
        
        返回：
            chosen_actions: 选定的动作（通常是一个张量）
        """
        # 从批次中取出当前时间步的可用动作数据
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        # 计算当前时间步智能体的输出
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        # 调用动作选择器根据当前输出和可用动作选择动作，bs 用于选择部分样本
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False):
        """
        前向传播计算智能体输出

        参数：
            ep_batch: 当前回合数据批次
            t (int): 当前时间步索引
            test_mode (bool): 是否为测试模式

        作用：
            1. 构造智能体输入；
            2. 获取当前时间步的可用动作；
            3. 使用智能体模型（agent）计算输出以及更新隐藏状态；
            4. 如果输出为策略 logits，则对输出进行 softmax 归一化，同时处理不可用动作和 epsilon 探索。
        
        返回：
            agent_outs: 形状为 (batch_size, n_agents, n_actions) 的输出张量
        """
        # 构造智能体输入
        agent_inputs = self._build_inputs(ep_batch, t)
        # 获取当前时间步各智能体的可用动作信息
        avail_actions = ep_batch["avail_actions"][:, t]
        # 通过智能体模型进行前向传播，得到输出和更新后的隐藏状态
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)

        # 如果智能体输出类型为 "pi_logits"，则需要进行 softmax 归一化转换为概率分布
        if self.agent_output_type == "pi_logits":
            if getattr(self.args, "mask_before_softmax", True):
                # 对于不可用的动作，将对应的 logits 设为一个极小值，以减小其 softmax 后的概率
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10

            # 对输出进行 softmax 归一化
            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
            if not test_mode:
                # 应用 epsilon 探索机制
                epsilon_action_num = agent_outs.size(-1)
                if getattr(self.args, "mask_before_softmax", True):
                    # 计算可用动作的数量，作为均匀分布的基数
                    epsilon_action_num = reshaped_avail_actions.sum(dim=1, keepdim=True).float()

                # 根据 epsilon 值平滑处理输出分布
                agent_outs = ((1 - self.action_selector.epsilon) * agent_outs
                               + th.ones_like(agent_outs) * self.action_selector.epsilon / epsilon_action_num)

                if getattr(self.args, "mask_before_softmax", True):
                    # 将不可用动作的概率置为 0
                    agent_outs[reshaped_avail_actions == 0] = 0.0

        # 将输出重塑为 (batch_size, n_agents, n_actions)
        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def init_hidden(self, batch_size):
        """
        初始化智能体的隐藏状态

        参数：
            batch_size (int): 当前批次大小

        作用：
            调用智能体模型的 init_hidden 方法生成初始隐藏状态，并将其扩展为 (batch_size, n_agents, hidden_dim) 的形状。
        """
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)

    def parameters(self):
        """
        获取智能体模型的所有参数

        返回：
            智能体模型的参数迭代器
        """
        return self.agent.parameters()

    def load_state(self, other_mac):
        """
        从另一个多智能体控制器加载状态

        参数：
            other_mac: 其他控制器对象，从中获取智能体模型的状态字典

        作用：
            将其他控制器中智能体模型的状态复制到当前智能体模型中。
        """
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        """
        将智能体模型迁移到 GPU 上
        """
        self.agent.cuda()

    def save_models(self, path):
        """
        保存智能体模型参数

        参数：
            path (str): 模型保存的路径
        """
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        """
        从磁盘加载智能体模型参数

        参数：
            path (str): 模型参数加载的路径
        """
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape):
        """
        构建智能体模型

        参数：
            input_shape (int): 输入数据的维度（由 _get_input_shape 得到）

        作用：
            根据参数 args.agent 从智能体注册表中加载对应的智能体类，并实例化该智能体。
        """
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)

    def _build_inputs(self, batch, t):
        """
        构造智能体输入

        参数：
            batch: 当前回合的数据批次，包含各个时间步的观测、动作等数据
            t (int): 当前时间步索引

        作用：
            1. 从 batch 中提取当前时间步所有智能体的观测；
            2. 如果设置了 obs_last_action，则在 t=0 时添加全零向量，否则添加 t-1 时的 one-hot 编码动作；
            3. 如果设置了 obs_agent_id，则添加智能体的 one-hot 编码；
            4. 将所有输入拼接成一个扁平向量，形状为 (batch_size * n_agents, input_dim)。
        
        返回：
            inputs (tensor): 拼接后的输入张量，用于传入智能体模型
        """
        bs = batch.batch_size
        inputs = []
        # 添加当前时间步的观测，形状为 (bs, n_agents, obs_dim)
        inputs.append(batch["obs"][:, t])  # b1av
        # 如果启用了观察上一次动作
        if self.args.obs_last_action:
            if t == 0:
                # t=0 时，没有上一个动作，使用全零向量
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                # 否则使用上一个时间步的动作 one-hot 表示
                inputs.append(batch["actions_onehot"][:, t-1])
        # 如果启用了观察智能体 ID，则添加 one-hot 编码（对每个智能体）
        if self.args.obs_agent_id:
            # 创建形状为 (n_agents, n_agents) 的单位矩阵，然后扩展为 (bs, n_agents, n_agents)
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))
        # 将所有输入 reshape 后在最后一维进行拼接，结果形状为 (bs * n_agents, input_dim)
        inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)
        return inputs

    def _get_input_shape(self, scheme):
        """
        计算智能体输入的总维度

        参数：
            scheme (dict): 数据方案，包含 "obs" 和 "actions_onehot" 的维度信息

        作用：
            1. 从 scheme 中获取智能体观测的维度；
            2. 如果启用了 obs_last_action，则添加动作 one-hot 编码的维度；
            3. 如果启用了 obs_agent_id，则添加智能体 ID 的维度（即智能体数量）。
        
        返回：
            input_shape (int): 构造智能体输入时所需的总维度
        """
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape

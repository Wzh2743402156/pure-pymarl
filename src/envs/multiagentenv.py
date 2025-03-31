class MultiAgentEnv(object):
    def step(self, actions):
        """
        执行环境中的一步交互
        
        参数：
            actions: 所有智能体在当前时间步选择的动作（通常是一个列表或数组）
        
        返回：
            reward: 当前时间步获得的奖励（可能为单个值或数组，取决于实现）
            terminated: 布尔值，表示当前回合是否结束
            info: 附加信息字典（例如调试信息或环境状态信息）
        
        说明：
            此函数需要根据传入的动作更新环境状态，并计算奖励。该接口在具体环境中必须实现。
        """
        raise NotImplementedError

    def get_obs(self):
        """
        获取所有智能体当前的观察
        
        返回：
            list 或 numpy 数组：包含每个智能体的观察信息。每个元素通常为一个向量或数组。
        
        说明：
            用于返回环境中所有智能体的局部观察。具体格式由具体环境定义。
        """
        raise NotImplementedError

    def get_obs_agent(self, agent_id):
        """
        获取指定智能体的观察
        
        参数：
            agent_id (int): 智能体的索引或标识符
        
        返回：
            numpy 数组或列表：表示该智能体的局部观察信息。
        
        说明：
            用于返回单个智能体的观察数据，方便针对单一智能体的操作或调试。
        """
        raise NotImplementedError

    def get_obs_size(self):
        """
        获取单个智能体观察的维度
        
        返回：
            int: 观察向量的长度或维度
        
        说明：
            用于描述每个智能体观察的形状信息，在构建神经网络等模型时会用到。
        """
        raise NotImplementedError

    def get_state(self):
        """
        获取当前全局状态
        
        返回：
            numpy 数组或列表：表示整个环境的全局状态信息
        
        说明：
            与 get_obs 不同，全局状态通常包含所有智能体的信息，可能比单个智能体的观察更完整。
        """
        raise NotImplementedError

    def get_state_size(self):
        """
        获取全局状态的维度
        
        返回：
            int: 全局状态向量的长度或维度
        
        说明：
            用于描述环境全局状态的形状信息，方便后续建模和数据处理。
        """
        raise NotImplementedError

    def get_avail_actions(self):
        """
        获取所有智能体在当前状态下可用的动作
        
        返回：
            numpy 数组或列表：通常形状为 (n_agents, n_actions) 的数组，
            其中每个元素表示对应智能体在对应动作上是否可用（例如 1 表示可用）。
        
        说明：
            用于返回当前状态下每个智能体可执行的动作列表。对于离散动作空间，这个接口很常见。
        """
        raise NotImplementedError

    def get_avail_agent_actions(self, agent_id):
        """
        获取指定智能体在当前状态下可用的动作
        
        参数：
            agent_id (int): 智能体的索引或标识符
        
        返回：
            numpy 数组或列表：表示该智能体可执行的动作列表，通常是一维数组，每个元素表示动作是否可用。
        
        说明：
            与 get_avail_actions 类似，但只返回单个智能体的数据。
        """
        raise NotImplementedError

    def get_total_actions(self):
        """
        获取每个智能体可以选择的动作总数
        
        返回：
            int: 动作总数
        
        说明：
            返回一个固定的数值，表示在任何状态下，智能体可以采取的总动作数。
            注意：此接口通常只适用于离散的一维动作空间。
        """
        # TODO: 此实现仅适用于离散的一维动作空间
        raise NotImplementedError

    def reset(self):
        """
        重置环境
        
        返回：
            tuple: (observations, state)
                - observations: 所有智能体的初始观察（通常为一个列表或数组）
                - state: 环境的初始全局状态
        
        说明：
            重置环境到初始状态，通常用于开始一个新的回合。
        """
        raise NotImplementedError

    def render(self):
        """
        渲染环境
        
        说明：
            可选接口，用于在屏幕上显示环境的当前状态，
            便于调试或可视化环境的动态变化。
        """
        raise NotImplementedError

    def close(self):
        """
        关闭环境
        
        说明：
            释放环境占用的资源，例如关闭窗口或释放内存等。
        """
        raise NotImplementedError

    def seed(self):
        """
        设置随机种子
        
        说明：
            用于设置环境内部随机数生成器的种子，以确保实验可复现。
        """
        raise NotImplementedError

    def save_replay(self):
        """
        保存环境交互回放
        
        说明：
            用于将环境交互过程保存下来，便于后续回放或分析。
            当前接口未实现，具体实现可根据需求扩展。
        """
        raise NotImplementedError

    def get_env_info(self):
        """
        获取环境的基本信息
        
        返回：
            dict: 包含环境的关键信息，例如：
                - "state_shape": 全局状态的维度
                - "obs_shape": 单个智能体观察的维度
                - "n_actions": 每个智能体可选的动作总数
                - "n_agents": 环境中智能体的数量
                - "episode_limit": 每个回合的最大步数
        
        说明：
            该接口用于在注册环境到 PyMARL 框架时提供必要的环境信息。
            注意：此接口中使用了 self.n_agents 和 self.episode_limit，
            因此在具体实现中需要保证这两个属性已经被正确设置。
        """
        env_info = {
            "state_shape": self.get_state_size(),
            "obs_shape": self.get_obs_size(),
            "n_actions": self.get_total_actions(),
            "n_agents": self.n_agents,
            "episode_limit": self.episode_limit
        }
        return env_info

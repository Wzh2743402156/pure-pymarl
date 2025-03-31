import torch as th

def build_td_lambda_targets(rewards, terminated, mask, target_qs, n_agents, gamma, td_lambda):
    """
    计算 TD(lambda) 目标值，用于 1-step TD 目标的扩展，即 lambda-return

    参数：
        rewards: 奖励张量，形状为 (B, T-1, 1)
        terminated: 终止标志张量，形状为 (B, T-1, 1)
                    (在某些实现中也可能为 (B, T-1)；表示每个时间步是否终止)
        mask: 掩码张量，形状为 (B, T-1, 1)
              表示哪些时间步为有效数据（1 为有效，0 为填充数据）
        target_qs: 目标 Q 值张量，形状为 (B, T, A)
                   表示目标网络在每个时间步对每个动作预测的 Q 值
        n_agents: 智能体数量（本函数中未直接使用，但通常在扩展版本中可能需要）
        gamma: 折扣因子，用于计算未来奖励的折扣和
        td_lambda: TD(lambda) 参数，控制 n 步 TD 和 Monte Carlo 之间的权衡

    作用：
        1. 初始化返回的 lambda-return 张量 ret，其形状与 target_qs 相同 (B, T, A)；
        2. 设置最后一个时间步的 lambda-return，对于未终止的回合，其值等于目标 Q 值；
        3. 从倒数第二个时间步开始，采用递归公式向前更新每个时间步的 lambda-return：
           ret[:, t] = td_lambda * gamma * ret[:, t+1] +
                       mask[:, t] * (rewards[:, t] + (1 - td_lambda) * gamma * target_qs[:, t+1] * (1 - terminated[:, t]))
        4. 返回从 t=0 到 t=T-1 的 lambda-return（去掉最后一步），形状为 (B, T-1, A)

    注意：
        - 这里假设 target_qs 的形状为 (B, T, A)，而 rewards、terminated、mask 的形状至少为 (B, T-1, 1)。
        - 其中，mask 用于屏蔽填充数据，确保只计算有效数据对应的 TD-error。
    """
    # 初始化 ret 张量，与 target_qs 形状相同，并填充为 0
    ret = target_qs.new_zeros(*target_qs.shape)
    # 对于最后一个时间步，如果回合没有终止，则将其目标 Q 值作为 lambda-return
    # th.sum(terminated, dim=1) 对每个 batch 求和，如果和为 0，则表示该回合未终止
    ret[:, -1] = target_qs[:, -1] * (1 - th.sum(terminated, dim=1))
    
    # 从倒数第二个时间步开始，向前递归计算 lambda-return
    for t in range(ret.shape[1] - 2, -1, -1):
        ret[:, t] = td_lambda * gamma * ret[:, t + 1] + mask[:, t] * (
            rewards[:, t] + (1 - td_lambda) * gamma * target_qs[:, t + 1] * (1 - terminated[:, t])
        )
    
    # 返回从 t=0 到 t=T-1 的 lambda-return，去除最后一项
    return ret[:, 0:-1]

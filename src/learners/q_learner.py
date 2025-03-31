import copy
from components.episode_buffer import EpisodeBatch  # 用于存储和采样 episode 数据
from modules.mixers.qmix import QMixer  # QMixer 用于将各智能体的 Q 值融合成全局 Q 值
import torch as th
from torch.optim import RMSprop  # 优化器，这里采用 RMSProp

class QLearner:
    def __init__(self, mac, scheme, logger, args):
        """
        初始化 QLearner

        参数：
            mac: 多智能体控制器（Multi-Agent Controller），负责生成各智能体的 Q 值
            scheme: 数据结构方案（描述回放数据的格式），暂时未在此处使用
            logger: 日志记录器，用于记录训练过程中的各种统计信息
            args: 参数对象，包含学习率、双重 Q-learning 标志、混合器类型、目标更新间隔等超参数

        作用：
            构建学习器，包括设置控制器、混合器（如果使用）、优化器、以及目标网络的初始化。
        """
        self.args = args
        self.mac = mac
        self.logger = logger

        # 获取 mac 中所有需要优化的参数
        self.params = list(mac.parameters())

        # 记录上次目标网络更新时的 episode 数
        self.last_target_update_episode = 0

        # 初始化混合器
        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "qmix":
                # 如果使用 QMIX 混合器，传入 args 参数
                self.mixer = QMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            # 将混合器参数也加入到优化器参数列表中
            self.params += list(self.mixer.parameters())
            # 创建混合器的目标网络，通过深拷贝获得
            self.target_mixer = copy.deepcopy(self.mixer)

        # 初始化优化器，采用 RMSprop 优化器
        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # 生成目标网络，采用深拷贝的方式复制 mac
        # 注意：这样会复制整个 MAC，包括动作选择器，但一般不会在训练中使用该部分
        self.target_mac = copy.deepcopy(mac)

        # 用于记录上一次记录日志的时间步
        self.log_stats_t = -self.args.learner_log_interval - 1

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        """
        训练学习器

        参数：
            batch (EpisodeBatch): 从回放缓冲区采样到的一个批次数据
            t_env (int): 当前环境时间步数
            episode_num (int): 当前的 episode 数

        作用：
            1. 从批次数据中提取奖励、动作、终止标志、以及填充掩码；
            2. 使用当前控制器（mac）计算 Q 值；
            3. 根据动作选择获得对应的 Q 值；
            4. 通过目标网络计算目标 Q 值（支持双重 Q-learning）；
            5. 如果使用混合器，则融合各智能体的 Q 值；
            6. 计算 1 步 Q-learning 的目标，并计算 TD-error 和 L2 损失；
            7. 进行反向传播和梯度裁剪，然后更新参数；
            8. 根据设定的目标更新间隔，更新目标网络；
            9. 定期记录统计信息（如损失、梯度范数、TD-error 等）。
        """
        # 从批次数据中提取奖励、动作、终止标志以及填充掩码
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        # 在 mask 中将已经终止的步对应的位置置为 0（从第二个时间步开始）
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # -----------------------------
        # 计算当前网络输出的 Q 值
        # -----------------------------
        mac_out = []
        # 初始化当前控制器的隐藏状态，保证与批次大小一致
        self.mac.init_hidden(batch.batch_size)
        # 对整个序列（时间步）进行前向传播
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        # 将每个时间步的输出拼接成一个 tensor，维度： (batch, seq_len, n_agents, n_actions)
        mac_out = th.stack(mac_out, dim=1)

        # 根据智能体实际选择的动作，提取对应的 Q 值
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # squeeze 去掉最后一个维度

        # -----------------------------
        # 计算目标网络输出的 Q 值
        # -----------------------------
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)
        # 去掉第一个时间步的估计，因为目标采用 1 步 TD 误差
        target_mac_out = th.stack(target_mac_out[1:], dim=1)

        # 将不可用的动作 Q 值置为一个极低的数（屏蔽掉），保证不会被选中
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        # -----------------------------
        # 计算目标 Q 值（支持双重 Q-learning）
        # -----------------------------
        if self.args.double_q:
            # 使用当前网络选择动作（注意 detach 使得梯度不传递）
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            # 从第 1 个时间步开始，找到最大 Q 值对应的动作索引
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            # 根据选择的动作，从目标网络中提取对应的 Q 值
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            # 直接对目标网络输出取最大值（在动作维度上）
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # -----------------------------
        # 如果使用混合器，对各智能体 Q 值进行融合
        # -----------------------------
        if self.mixer is not None:
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])

        # 计算 1 步 Q-learning 目标：reward + gamma * (1 - terminated) * target_max_qvals
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals

        # 计算 TD-error
        td_error = (chosen_action_qvals - targets.detach())

        # 扩展 mask 的维度以匹配 td_error
        mask = mask.expand_as(td_error)

        # 对填充部分（无效数据）置零
        masked_td_error = td_error * mask

        # 计算均方误差损失，并归一化（只对有效数据求平均）
        loss = (masked_td_error ** 2).sum() / mask.sum()

        # -----------------------------
        # 反向传播与优化
        # -----------------------------
        self.optimiser.zero_grad()
        loss.backward()
        # 对梯度进行裁剪，防止梯度爆炸
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        # -----------------------------
        # 更新目标网络：每隔一定 episode 更新一次目标网络参数
        # -----------------------------
        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        # -----------------------------
        # 日志记录：定期记录训练统计信息
        # -----------------------------
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm.item(), t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item() / mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
            self.log_stats_t = t_env

    def _update_targets(self):
        """
        更新目标网络

        作用：
            将当前控制器（mac）的参数复制到目标控制器（target_mac），
            如果使用混合器，则同时更新目标混合器的参数。
        """
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        """
        将模型迁移到 GPU

        作用：
            将当前控制器、目标控制器以及混合器（如果存在）迁移到 CUDA 设备上。
        """
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        """
        保存模型参数

        参数：
            path (str): 模型保存的路径

        作用：
            保存当前控制器、混合器（如果存在）和优化器的状态到磁盘上，
            以便后续恢复训练或进行评估。
        """
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        """
        加载模型参数

        参数：
            path (str): 模型参数加载路径

        作用：
            从磁盘加载之前保存的控制器、混合器（如果存在）和优化器的状态，
            恢复模型到之前的训练状态。
        """
        self.mac.load_models(path)
        # 注意：目标网络不单独保存，这里直接加载到目标网络中
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))

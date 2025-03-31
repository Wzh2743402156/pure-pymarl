import torch.nn as nn
import torch.nn.functional as F

class RNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        """
        初始化 RNNAgent 模型

        参数：
            input_shape (int): 输入数据的维度，通常为智能体观测的维度
            args (object): 参数对象，包含模型超参数，例如 rnn_hidden_dim、n_actions 等

        作用：
            构建一个包含全连接层、GRUCell 和全连接层的 RNN 模型，用于处理时序观测并输出各动作的 Q 值。
        """
        super(RNNAgent, self).__init__()
        self.args = args

        # 第一个全连接层：将输入映射到隐藏层维度
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        # GRUCell：作为循环单元，接收前一隐藏状态和当前输入更新隐藏状态
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        # 第二个全连接层：将更新后的隐藏状态映射到动作空间，输出每个动作的 Q 值
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self):
        """
        初始化 RNN 的隐藏状态

        作用：
            根据模型的参数生成一个全零的隐藏状态，并确保其所在设备与模型一致。

        返回：
            tensor: 形状为 (1, rnn_hidden_dim) 的全零张量，作为初始隐藏状态。
        """
        # 使用 fc1 权重创建新张量，使其与模型位于同一设备，并初始化为 0
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        """
        模型的前向传播

        参数：
            inputs (tensor): 当前时刻的输入数据（例如智能体的观测），形状为 (batch_size, input_shape)
            hidden_state (tensor): 前一时刻的隐藏状态，形状应为 (batch_size, rnn_hidden_dim)

        作用：
            1. 将输入通过全连接层 fc1 进行线性变换，并通过 ReLU 激活函数进行非线性映射。
            2. 利用 GRUCell 根据当前输入和前一隐藏状态更新隐藏状态。
            3. 通过全连接层 fc2 将更新后的隐藏状态映射到动作空间，得到各动作的 Q 值。

        返回：
            q (tensor): 输出的 Q 值张量，形状为 (batch_size, n_actions)
            h (tensor): 更新后的隐藏状态，形状为 (batch_size, rnn_hidden_dim)
        """
        # 对输入数据进行线性变换和 ReLU 激活
        x = F.relu(self.fc1(inputs))
        # 重塑隐藏状态以确保形状为 (batch_size, rnn_hidden_dim)
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        # 使用 GRUCell 更新隐藏状态
        h = self.rnn(x, h_in)
        # 通过全连接层将隐藏状态映射到动作空间，得到 Q 值
        q = self.fc2(h)
        return q, h

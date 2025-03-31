# 定义一个全局字典 REGISTRY，用于存储所有智能体类的注册信息
REGISTRY = {}

# 从当前包的 rnn_agent 模块中导入 RNNAgent 类
from .rnn_agent import RNNAgent

# 将 RNNAgent 类注册到 REGISTRY 字典中，键名为 "rnn"
# 这样，在其他模块中可以通过 REGISTRY["rnn"] 动态加载 RNNAgent 类
REGISTRY["rnn"] = RNNAgent

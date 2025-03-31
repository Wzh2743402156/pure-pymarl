# 从当前包的 q_learner 模块中导入 QLearner 类
from .q_learner import QLearner

# 定义一个全局注册字典 REGISTRY，用于存储所有学习器类
REGISTRY = {}

# 将 QLearner 类注册到 REGISTRY 字典中，键名为 "q_learner"
# 这样在其他模块中可以通过 REGISTRY["q_learner"] 动态加载 QLearner 类
REGISTRY["q_learner"] = QLearner

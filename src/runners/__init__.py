# 定义一个全局字典 REGISTRY，用于存储所有 Runner 类的注册信息
REGISTRY = {}

# 从当前包的 episode_runner 模块中导入 EpisodeRunner 类
from .episode_runner import EpisodeRunner

# 将 EpisodeRunner 类注册到 REGISTRY 字典中，键名为 "episode"
# 这样在其他模块中可以通过 REGISTRY["episode"] 动态加载并创建 EpisodeRunner 实例
REGISTRY["episode"] = EpisodeRunner

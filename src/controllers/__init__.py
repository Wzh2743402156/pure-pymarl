# 定义一个全局字典 REGISTRY，用于存储所有基本控制器（MAC）的注册信息
REGISTRY = {}

# 从当前包的 basic_controller 模块中导入 BasicMAC 类
from .basic_controller import BasicMAC

# 将 BasicMAC 类注册到 REGISTRY 字典中，键名为 "basic_mac"
# 这样在其他模块中可以通过 REGISTRY["basic_mac"] 动态加载并创建 BasicMAC 的实例
REGISTRY["basic_mac"] = BasicMAC

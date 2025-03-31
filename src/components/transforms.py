import torch as th

# 定义一个 Transform 基类，用于数据转换
class Transform:
    def transform(self, tensor):
        """
        对输入的 tensor 进行转换

        参数：
            tensor (Tensor): 输入张量

        返回：
            Tensor: 转换后的张量

        说明：
            此方法需要在子类中实现，提供具体的转换逻辑。
        """
        raise NotImplementedError

    def infer_output_info(self, vshape_in, dtype_in):
        """
        推断转换后数据的形状和数据类型

        参数：
            vshape_in (tuple): 输入张量的形状信息（不包括 batch 维度）
            dtype_in (torch.dtype): 输入张量的数据类型

        返回：
            tuple: (vshape_out, dtype_out)
                - vshape_out: 转换后张量的形状
                - dtype_out: 转换后张量的数据类型

        说明：
            此方法需要在子类中实现，用于描述转换后的数据属性。
        """
        raise NotImplementedError

# 定义 OneHot 类，继承自 Transform，用于将离散变量转换为 one-hot 编码
class OneHot(Transform):
    def __init__(self, out_dim):
        """
        初始化 OneHot 转换器

        参数：
            out_dim (int): 输出 one-hot 向量的维度，即类别总数
        """
        self.out_dim = out_dim

    def transform(self, tensor):
        """
        将输入张量转换为 one-hot 编码

        参数：
            tensor (Tensor): 输入张量，其最后一个维度应包含类别索引

        返回：
            Tensor: 转换后的 one-hot 编码张量，数据类型为 float

        过程：
            1. 创建一个新的张量 y_onehot，其形状与输入 tensor 除最后一维外加上 out_dim，初始全为 0；
            2. 使用 scatter_ 方法，在最后一个维度上按照 tensor 中的类别索引将对应位置置为 1；
            3. 最后将结果转换为 float 类型返回。
        """
        # 创建一个全零张量，形状为 (*tensor.shape[:-1], self.out_dim)
        y_onehot = tensor.new(*tensor.shape[:-1], self.out_dim).zero_()
        # 使用 scatter_ 方法将 tensor.long() 中的索引对应位置赋值为 1
        y_onehot.scatter_(-1, tensor.long(), 1)
        return y_onehot.float()

    def infer_output_info(self, vshape_in, dtype_in):
        """
        推断 one-hot 转换后数据的形状和数据类型

        参数：
            vshape_in (tuple): 输入张量的形状信息
            dtype_in (torch.dtype): 输入张量的数据类型

        返回：
            tuple: ((self.out_dim,), th.float32)
                - 输出张量的形状为 (self.out_dim,)
                - 输出数据类型为 torch.float32
        """
        return (self.out_dim,), th.float32

import torch as th
import numpy as np # type: ignore
from types import SimpleNamespace as SN

class EpisodeBatch:
    def __init__(self,
                 scheme,
                 groups,
                 batch_size,
                 max_seq_length,
                 data=None,
                 preprocess=None,
                 device="cpu"):
        """
        初始化 EpisodeBatch，用于存储一个回合（或多个回合）的数据

        参数：
            scheme (dict): 数据方案，描述各字段（如 "obs", "state", "actions" 等）的形状、数据类型等信息
            groups (dict): 分组信息（例如指定哪些字段属于智能体组）
            batch_size (int): 批次大小，通常表示回合数量
            max_seq_length (int): 单个回合的最大时间步数
            data (可选): 如果已有数据则直接使用，否则初始化为空的数据容器
            preprocess (dict, 可选): 预处理方法字典，用于对部分字段进行转换，例如 one-hot 编码
            device (str): 数据存储的设备（如 "cpu" 或 "cuda"）

        作用：
            根据 scheme 和 groups 初始化一个数据容器，包含两部分：
                - transition_data: 存储每个时间步的转移数据（形状为 (batch_size, max_seq_length, ...)）
                - episode_data: 存储每个回合不变的数据（形状为 (batch_size, ...)）
        """
        self.scheme = scheme.copy()
        self.groups = groups
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.preprocess = {} if preprocess is None else preprocess
        self.device = device

        if data is not None:
            self.data = data
        else:
            # 使用 SimpleNamespace 存储数据，包含 transition_data 和 episode_data 两部分
            self.data = SN()
            self.data.transition_data = {}
            self.data.episode_data = {}
            self._setup_data(self.scheme, self.groups, batch_size, max_seq_length, self.preprocess)

    def _setup_data(self, scheme, groups, batch_size, max_seq_length, preprocess):
        """
        根据 scheme、groups 等参数初始化数据存储区域

        参数：
            scheme (dict): 数据方案
            groups (dict): 分组信息
            batch_size (int): 批次大小
            max_seq_length (int): 最大时间步数
            preprocess (dict): 预处理方法

        作用：
            1. 如果对某个字段有预处理要求，则更新 scheme，添加预处理后新字段；
            2. 强制添加一个 "filled" 字段，用于标记数据是否有效（作为掩码）；
            3. 遍历 scheme 中的每个字段，根据是否为回合常量（episode_const）和所属组（group）创建相应的 tensor，
               分别存储在 episode_data 或 transition_data 中。
        """
        if preprocess is not None:
            for k in preprocess:
                # 确保预处理的字段在 scheme 中存在
                assert k in scheme
                new_k = preprocess[k][0]  # 新字段名
                transforms = preprocess[k][1]  # 一系列转换方法

                vshape = self.scheme[k]["vshape"]
                dtype = self.scheme[k]["dtype"]
                # 根据每个转换方法推断输出信息（形状和数据类型）
                for transform in transforms:
                    vshape, dtype = transform.infer_output_info(vshape, dtype)

                # 将转换后的信息添加到 scheme 中，新字段 new_k
                self.scheme[new_k] = {
                    "vshape": vshape,
                    "dtype": dtype
                }
                # 如果原字段有 group 或 episode_const 属性，复制到新字段上
                if "group" in self.scheme[k]:
                    self.scheme[new_k]["group"] = self.scheme[k]["group"]
                if "episode_const" in self.scheme[k]:
                    self.scheme[new_k]["episode_const"] = self.scheme[k]["episode_const"]

        # "filled" 为预留字段，用于掩码，不允许在 scheme 中预先定义
        assert "filled" not in scheme, '"filled" is a reserved key for masking.'
        scheme.update({
            "filled": {"vshape": (1,), "dtype": th.long},
        })

        # 遍历 scheme 中所有字段，初始化数据存储区域
        for field_key, field_info in scheme.items():
            assert "vshape" in field_info, "Scheme must define vshape for {}".format(field_key)
            vshape = field_info["vshape"]
            episode_const = field_info.get("episode_const", False)
            group = field_info.get("group", None)
            dtype = field_info.get("dtype", th.float32)

            if isinstance(vshape, int):
                vshape = (vshape,)

            if group:
                # 如果该字段属于某个 group，则其形状前面加上 group 中的成员数
                assert group in groups, "Group {} must have its number of members defined in _groups_".format(group)
                shape = (groups[group], *vshape)
            else:
                shape = vshape

            if episode_const:
                # 如果该字段为回合常量，则保存在 episode_data 中，形状为 (batch_size, *shape)
                self.data.episode_data[field_key] = th.zeros((batch_size, *shape), dtype=dtype, device=self.device)
            else:
                # 否则保存在 transition_data 中，形状为 (batch_size, max_seq_length, *shape)
                self.data.transition_data[field_key] = th.zeros((batch_size, max_seq_length, *shape), dtype=dtype, device=self.device)

    def extend(self, scheme, groups=None):
        """
        扩展 EpisodeBatch，添加新的字段

        参数：
            scheme (dict): 新的字段方案
            groups (dict, 可选): 新的分组信息；如果未提供，则使用当前的 groups

        作用：
            调用 _setup_data 方法，在已有数据结构上添加新的字段数据
        """
        self._setup_data(scheme, self.groups if groups is None else groups, self.batch_size, self.max_seq_length)

    def to(self, device):
        """
        将所有数据迁移到指定设备

        参数：
            device (str): 目标设备，如 "cpu" 或 "cuda"

        作用：
            遍历 transition_data 和 episode_data 中所有 tensor，将它们迁移到指定设备，
            并更新当前对象的 device 属性。
        """
        for k, v in self.data.transition_data.items():
            self.data.transition_data[k] = v.to(device)
        for k, v in self.data.episode_data.items():
            self.data.episode_data[k] = v.to(device)
        self.device = device

    def update(self, data, bs=slice(None), ts=slice(None), mark_filled=True):
        """
        更新 EpisodeBatch 中的数据

        参数：
            data (dict): 要更新的数据，键为字段名称，值为要更新的数据
            bs: 批次维度的索引或切片（默认为所有批次）
            ts: 时间步维度的索引或切片（默认为所有时间步）
            mark_filled (bool): 是否在更新时将 "filled" 字段标记为 1

        作用：
            根据给定索引，将 data 中的数据更新到 transition_data 或 episode_data 中，
            同时对预处理字段进行转换。如果 mark_filled 为 True，则将第一次更新时标记 "filled" 为 1。
        """
        slices = self._parse_slices((bs, ts))
        for k, v in data.items():
            if k in self.data.transition_data:
                target = self.data.transition_data
                if mark_filled:
                    target["filled"][slices] = 1
                    mark_filled = False  # 仅第一次更新标记
                _slices = slices
            elif k in self.data.episode_data:
                target = self.data.episode_data
                _slices = slices[0]  # episode_data 只按 batch 维度索引
            else:
                raise KeyError("{} not found in transition or episode data".format(k))

            dtype = self.scheme[k].get("dtype", th.float32)
            # 如果 v 是 tensor，则转到 CPU；否则转换为 numpy 数组
            if isinstance(v, th.Tensor):
                v = v.cpu()
            v = np.array(v)
            # 再次尝试将 v 转换为 tensor
            if isinstance(v, th.Tensor):
                v = v.clone().detach()
            else:
                v = th.tensor(v, dtype=dtype, device=self.device)

            # 检查 v 的形状是否安全地转换为目标形状
            self._check_safe_view(v, target[k][_slices])
            target[k][_slices] = v.view_as(target[k][_slices])

            # 如果该字段需要预处理，则对 v 应用预处理函数，并更新预处理后新字段的数据
            if k in self.preprocess:
                new_k = self.preprocess[k][0]
                v = target[k][_slices]
                for transform in self.preprocess[k][1]:
                    v = transform.transform(v)
                target[new_k][_slices] = v.view_as(target[new_k][_slices])

    def _check_safe_view(self, v, dest):
        """
        检查是否可以安全地将 tensor v 重塑为目标形状 dest

        参数：
            v (Tensor): 待重塑的 tensor
            dest (Tensor): 目标 tensor，用于检查其形状

        作用：
            逐步比较 v 的最后一维与 dest 的形状，如果不匹配且目标维度不为1，则抛出异常。
        """
        idx = len(v.shape) - 1
        for s in dest.shape[::-1]:
            if v.shape[idx] != s:
                if s != 1:
                    raise ValueError("Unsafe reshape of {} to {}".format(v.shape, dest.shape))
            else:
                idx -= 1

    def __getitem__(self, item):
        """
        支持下标索引操作

        参数：
            item: 可以是字符串、字符串元组或索引切片

        作用：
            如果 item 为字符串，则返回 episode_data 或 transition_data 中对应字段的全部数据；
            如果 item 为字符串元组，则返回仅包含这些字段的新 EpisodeBatch；
            否则，认为 item 是索引切片，返回在批次和时间步上切片后的新 EpisodeBatch。
        """
        if isinstance(item, str):
            if item in self.data.episode_data:
                return self.data.episode_data[item]
            elif item in self.data.transition_data:
                return self.data.transition_data[item]
            else:
                raise ValueError
        elif isinstance(item, tuple) and all([isinstance(it, str) for it in item]):
            new_data = self._new_data_sn()
            for key in item:
                if key in self.data.transition_data:
                    new_data.transition_data[key] = self.data.transition_data[key]
                elif key in self.data.episode_data:
                    new_data.episode_data[key] = self.data.episode_data[key]
                else:
                    raise KeyError("Unrecognised key {}".format(key))

            # 更新 scheme，仅保留请求的键
            new_scheme = {key: self.scheme[key] for key in item}
            new_groups = {self.scheme[key]["group"]: self.groups[self.scheme[key]["group"]]
                          for key in item if "group" in self.scheme[key]}
            ret = EpisodeBatch(new_scheme, new_groups, self.batch_size, self.max_seq_length, data=new_data, device=self.device)
            return ret
        else:
            item = self._parse_slices(item)
            new_data = self._new_data_sn()
            # 对 transition_data 按照给定切片索引
            for k, v in self.data.transition_data.items():
                new_data.transition_data[k] = v[item]
            # 对 episode_data 仅按照批次索引
            for k, v in self.data.episode_data.items():
                new_data.episode_data[k] = v[item[0]]

            ret_bs = self._get_num_items(item[0], self.batch_size)
            ret_max_t = self._get_num_items(item[1], self.max_seq_length)

            ret = EpisodeBatch(self.scheme, self.groups, ret_bs, ret_max_t, data=new_data, device=self.device)
            return ret

    def _get_num_items(self, indexing_item, max_size):
        """
        计算索引项对应的元素数量

        参数：
            indexing_item: 可以是 list、numpy 数组或 slice
            max_size (int): 对应维度的最大大小

        返回：
            int: 索引项包含的元素数量
        """
        if isinstance(indexing_item, list) or isinstance(indexing_item, np.ndarray):
            return len(indexing_item)
        elif isinstance(indexing_item, slice):
            _range = indexing_item.indices(max_size)
            return 1 + (_range[1] - _range[0] - 1) // _range[2]

    def _new_data_sn(self):
        """
        创建一个新的 SimpleNamespace 用于存储数据

        返回：
            SimpleNamespace: 包含空的 transition_data 和 episode_data 字典
        """
        new_data = SN()
        new_data.transition_data = {}
        new_data.episode_data = {}
        return new_data

    def _parse_slices(self, items):
        """
        解析索引项，将单个整数转换为 slice 对象

        参数：
            items: 索引项，可能为单个 slice、整数、list 或 numpy 数组

        返回：
            list: 解析后的索引项列表，保证每个维度都是 slice 对象
        """
        parsed = []
        # 如果只给定了批次索引，则自动添加时间步完整切片
        if (isinstance(items, slice)  # slice a:b
            or isinstance(items, int)  # int i
            or (isinstance(items, (list, np.ndarray, th.LongTensor, th.cuda.LongTensor)))  # [a,b,c]
            ):
            items = (items, slice(None))

        # 对时间步的索引必须是连续的，不支持 list 索引
        if isinstance(items[1], list):
            raise IndexError("Indexing across Time must be contiguous")

        for item in items:
            # 如果是单个整数，则转换为对应的 slice 对象
            if isinstance(item, int):
                parsed.append(slice(item, item+1))
            else:
                parsed.append(item)
        return parsed

    def max_t_filled(self):
        """
        返回 transition_data 中 "filled" 字段的最大有效时间步数

        作用：
            计算每个回合中 "filled" 字段的总和（即有效数据的数量），并返回其中的最大值，
            用于确定当前批次中最长的有效时间步数。
        """
        return th.sum(self.data.transition_data["filled"], 1).max(0)[0]

    def __repr__(self):
        """
        定义 EpisodeBatch 的字符串表示

        返回：
            str: 包含批次大小、最大时间步数、scheme 键和 groups 键的信息
        """
        return "EpisodeBatch. Batch Size:{} Max_seq_len:{} Keys:{} Groups:{}".format(
            self.batch_size,
            self.max_seq_length,
            self.scheme.keys(),
            self.groups.keys()
        )


class ReplayBuffer(EpisodeBatch):
    def __init__(self, scheme, groups, buffer_size, max_seq_length, preprocess=None, device="cpu"):
        """
        初始化 ReplayBuffer

        参数：
            scheme (dict): 数据方案
            groups (dict): 分组信息
            buffer_size (int): 回放缓冲区的大小（即最多存储多少回合）
            max_seq_length (int): 每个回合的最大时间步数
            preprocess (dict, 可选): 预处理设置
            device (str): 存储设备（例如 "cpu" 或 "cuda"）

        作用：
            ReplayBuffer 继承自 EpisodeBatch，用于存储多个回合的数据，
            并实现插入、采样等操作。
        """
        super(ReplayBuffer, self).__init__(scheme, groups, buffer_size, max_seq_length, preprocess=preprocess, device=device)
        self.buffer_size = buffer_size  # 与 self.batch_size 相同，但更明确表示为缓冲区大小
        self.buffer_index = 0  # 当前插入的索引位置
        self.episodes_in_buffer = 0  # 缓冲区中当前存储的回合数量

    def insert_episode_batch(self, ep_batch):
        """
        插入一个新的回合批次到缓冲区

        参数：
            ep_batch (EpisodeBatch): 待插入的回合批次

        作用：
            根据缓冲区剩余空间将 ep_batch 插入到缓冲区中，
            如果 ep_batch 超出当前缓冲区剩余空间，则递归拆分后插入。
        """
        if self.buffer_index + ep_batch.batch_size <= self.buffer_size:
            self.update(ep_batch.data.transition_data,
                        slice(self.buffer_index, self.buffer_index + ep_batch.batch_size),
                        slice(0, ep_batch.max_seq_length),
                        mark_filled=False)
            self.update(ep_batch.data.episode_data,
                        slice(self.buffer_index, self.buffer_index + ep_batch.batch_size))
            self.buffer_index = (self.buffer_index + ep_batch.batch_size)
            self.episodes_in_buffer = max(self.episodes_in_buffer, self.buffer_index)
            # 当缓冲区达到上限后，从头开始覆盖
            self.buffer_index = self.buffer_index % self.buffer_size
            assert self.buffer_index < self.buffer_size
        else:
            # 如果当前缓冲区剩余空间不足，则先插入一部分，再插入剩余部分
            buffer_left = self.buffer_size - self.buffer_index
            self.insert_episode_batch(ep_batch[0:buffer_left, :])
            self.insert_episode_batch(ep_batch[buffer_left:, :])

    def can_sample(self, batch_size):
        """
        判断是否可以从缓冲区采样

        参数：
            batch_size (int): 需要采样的回合数量

        返回：
            bool: 如果缓冲区中已有的回合数大于等于 batch_size，则返回 True，否则 False
        """
        return self.episodes_in_buffer >= batch_size

    def sample(self, batch_size):
        """
        从缓冲区中随机采样一个批次

        参数：
            batch_size (int): 采样的回合数量

        返回：
            EpisodeBatch: 采样得到的回合批次

        作用：
            当缓冲区中的回合数正好等于 batch_size 时直接返回，
            否则进行均匀随机采样，返回对应的回合批次。
        """
        assert self.can_sample(batch_size)
        if self.episodes_in_buffer == batch_size:
            return self[:batch_size]
        else:
            # 均匀随机采样，不放回
            ep_ids = np.random.choice(self.episodes_in_buffer, batch_size, replace=False)
            return self[ep_ids]

    def __repr__(self):
        """
        定义 ReplayBuffer 的字符串表示

        返回：
            str: 包含当前缓冲区中回合数、缓冲区总大小、scheme 键和 groups 键的信息
        """
        return "ReplayBuffer. {}/{} episodes. Keys:{} Groups:{}".format(
            self.episodes_in_buffer,
            self.buffer_size,
            self.scheme.keys(),
            self.groups.keys()
        )

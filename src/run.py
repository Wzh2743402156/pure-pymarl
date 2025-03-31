import datetime  # 处理日期和时间
import os        # 操作文件和目录
import pprint    # 用于格式化打印数据结构
import time      # 提供时间相关函数，例如计时
import threading # 多线程操作
import torch as th  # PyTorch，用于深度学习计算
from types import SimpleNamespace as SN  # 简化命名空间，便于将字典转换为对象属性访问
from utils.logging import Logger  # 自定义日志工具，用于记录和打印训练过程中的信息
from utils.timehelper import time_left, time_str  # 辅助函数，用于计算剩余时间和格式化时间字符串
from os.path import dirname, abspath  # 用于获取文件和目录的绝对路径

# 从各模块的注册表中导入对应的字典（注册的类或函数）
from learners import REGISTRY as le_REGISTRY       # 学习器（如 Q-learner）的注册表
from runners import REGISTRY as r_REGISTRY         # Runner 的注册表，负责组织环境和训练流程
from controllers import REGISTRY as mac_REGISTRY   # 控制器（Multi-Agent Controller）的注册表
from components.episode_buffer import ReplayBuffer  # 回放缓冲区，用于存储采样到的 episode 数据
from components.transforms import OneHot         # 用于将动作转为 one-hot 表示的转换函数


def run(_run, _config, _log):
    """
    主运行函数，负责整个训练/评估流程的搭建与执行
    参数：
      _run: Sacred 运行对象
      _config: 配置字典，包含所有实验参数
      _log: 日志对象
    """

    # 检查配置参数的合法性，可能对某些参数进行自动修正或警告
    _config = args_sanity_check(_config, _log)

    # 将配置字典转换成一个简单的命名空间，便于通过属性访问参数
    args = SN(**_config)
    # 根据是否使用 CUDA 设置设备：如果 use_cuda 为 True 则设备为 "cuda"，否则为 "cpu"
    args.device = "cuda" if args.use_cuda else "cpu"

    # 初始化自定义日志记录器
    logger = Logger(_log)

    # 打印整个实验的参数信息，使用 pprint 格式化输出
    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config, indent=4, width=1)
    _log.info("\n\n" + experiment_params + "\n")

    # 配置 TensorBoard 日志记录（如果启用了 tensorboard）
    # 这里使用环境名称（map_name）作为标识，默认使用 '2s3z'
    name = getattr(args, 'map_name', 'bottom')
    # 创建一个唯一标识符，格式为 "map_name__YYYY-MM-DD_HH-MM-SS"
    unique_token = "{}__{}".format(name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    args.unique_token = unique_token
    if args.use_tensorboard:
        # 构造 TensorBoard 日志目录路径
        tb_logs_direc = os.path.join(dirname(dirname(abspath(__file__))), "results", "tb_logs")
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        # 初始化 TensorBoard logger
        logger.setup_tb(tb_exp_direc)

    # 设置 Sacred 的日志输出（通常会将实验输出保存到 Sacred 观察器中）
    logger.setup_sacred(_run)

    # 运行训练/评估流程，采用顺序方式运行
    run_sequential(args=args, logger=logger)

    # 清理工作：训练结束后，输出提示并尝试停止所有线程
    try:
        print("Exiting Main")
        print("Stopping all threads")
    except OSError:
        pass

    # 遍历当前所有线程，除了主线程之外都进行 join 操作，等待它们退出
    for t in threading.enumerate():
        if t.name != "MainThread":
            try:
                print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            except OSError:
                pass
            t.join(timeout=1)
            try:
                print("Thread joined")
            except OSError:
                pass

    try:
        print("Exiting script")
    except OSError:
        pass

    # 强制退出程序，确保没有残留进程
    os._exit(os.EX_OK)


def evaluate_sequential(args, runner):
    """
    顺序评估函数，在测试模式下运行若干个 episode，并保存回放（如果设置了保存回放）
    参数：
      args: 配置参数
      runner: 环境与训练过程管理对象
    """
    for _ in range(args.test_nepisode):
        runner.run(test_mode=True)

    if args.save_replay:
        runner.save_replay()

    runner.close_env()


def run_sequential(args, logger):
    """
    顺序运行训练过程：初始化 Runner、控制器、学习器以及回放缓冲区，执行训练循环。
    参数：
      args: 实验参数
      logger: 日志记录器
    """
    # 初始化 Runner，通过注册表获取对应的 Runner 类实例
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    # 获取环境信息，用于设置一些基础参数，例如智能体数量、动作数量、状态维度
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]

    # 定义基础的数据 scheme，描述每个字段的维度、所属组以及数据类型
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    # 定义组信息，这里 "agents" 组的大小等于智能体数量
    groups = {
        "agents": args.n_agents
    }
    # 定义预处理操作，对动作数据进行 one-hot 编码
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }

    # 初始化回放缓冲区，用于存储采样到的 episodes 数据
    buffer = ReplayBuffer(
        scheme,
        groups,
        args.buffer_size,
        env_info["episode_limit"] + 1,
        preprocess=preprocess,
        device="cpu" if args.buffer_cpu_only else args.device
    )

    # 初始化多智能体控制器，通过注册表获取对应的控制器类
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)

    # 将控制器传递给 Runner，用于后续训练过程
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # 初始化学习器，通过注册表获取对应的学习器类（例如 Q-learner）
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)

    # 如果使用 GPU，则将学习器移动到 GPU
    if args.use_cuda:
        learner.cuda()

    # 如果指定了检查点路径，则加载对应模型
    if args.checkpoint_path != "":
        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(args.checkpoint_path):
            logger.console_logger.info("Checkpoint directiory {} doesn't exist".format(args.checkpoint_path))
            return

        # 遍历检查点目录中的所有文件夹（文件夹名称为数字代表训练步数）
        for name in os.listdir(args.checkpoint_path):
            full_name = os.path.join(args.checkpoint_path, name)
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))

        # 根据配置选择加载最新的检查点或者距离 load_step 最近的检查点
        if args.load_step == 0:
            timestep_to_load = max(timesteps)
        else:
            timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

        model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))
        logger.console_logger.info("Loading model from {}".format(model_path))
        learner.load_models(model_path)
        runner.t_env = timestep_to_load

        # 如果处于评估模式或需要保存回放，则直接进行评估并返回
        if args.evaluate or args.save_replay:
            evaluate_sequential(args, runner)
            return

    # 开始训练主循环
    episode = 0
    last_test_T = -args.test_interval - 1  # 上一次测试时的时间步，初始化为一个较小值
    last_log_T = 0  # 上一次记录日志的时间步
    model_save_time = 0  # 上一次保存模型的时间步

    start_time = time.time()  # 记录训练开始时间
    last_time = start_time

    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))

    # 主训练循环：直到累计时间步达到最大训练步数 t_max
    while runner.t_env <= args.t_max:
        # 执行一次完整的 episode，获得数据 batch
        episode_batch = runner.run(test_mode=False)
        # 将采集到的 episode 数据插入回放缓冲区
        buffer.insert_episode_batch(episode_batch)

        # 当缓冲区中的数据足够构成一个 batch 时，进行训练
        if buffer.can_sample(args.batch_size):
            episode_sample = buffer.sample(args.batch_size)

            # 截断 batch 到当前实际填充的时间步
            max_ep_t = episode_sample.max_t_filled()
            episode_sample = episode_sample[:, :max_ep_t]

            # 如果采样的数据不在目标设备上，则将其移动到目标设备
            if episode_sample.device != args.device:
                episode_sample.to(args.device)

            # 使用学习器对采样的 batch 进行训练，并传入当前时间步和 episode 数
            learner.train(episode_sample, runner.t_env, episode)

        # 定期执行测试：计算需要测试的次数（至少一次）
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:
            logger.console_logger.info("t_env: {} / {}".format(runner.t_env, args.t_max))
            logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                time_left(last_time, last_test_T, runner.t_env, args.t_max),
                time_str(time.time() - start_time)
            ))
            last_time = time.time()
            last_test_T = runner.t_env
            for _ in range(n_test_runs):
                runner.run(test_mode=True)

        # 定期保存模型：根据保存间隔或者首次保存
        if args.save_model and (runner.t_env - model_save_time >= args.save_model_interval or model_save_time == 0):
            model_save_time = runner.t_env
            save_path = os.path.join(args.local_results_path, "models", args.unique_token, str(runner.t_env))
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))
            # 委托学习器保存模型，可能包括 actor、critic、优化器状态等
            learner.save_models(save_path)

        # 更新 episode 数（根据每次运行的 batch 大小）
        episode += args.batch_size_run

        # 定期记录日志，输出训练统计信息
        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env

    # 训练结束后，关闭环境
    runner.close_env()
    logger.console_logger.info("Finished Training")


def args_sanity_check(config, _log):
    """
    检查并修正配置参数，确保参数合法且合理
    参数：
      config: 配置字典
      _log: 日志对象，用于打印警告信息
    返回：
      修正后的配置字典
    """
    # 如果设置了使用 CUDA，但实际上没有可用的 GPU，则关闭 use_cuda 并打印警告
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    # 确保测试 episode 数量至少等于 batch_size_run，如果不足则进行调整
    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        # 保证 test_nepisode 是 batch_size_run 的整数倍
        config["test_nepisode"] = (config["test_nepisode"] // config["batch_size_run"]) * config["batch_size_run"]

    return config

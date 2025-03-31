# 该项目只包含QMix算法
注：所有注释均有ChatGPT o3-mini-high生成，本人只是搬运工

## PYMARL源码地址
```
https://github.com/oxwhirl/pymarl
```

## 环境配置可参考
```
https://zhuanlan.zhihu.com/p/542727892
```

### 启动命令
```
python src/main.py --config=qmix --env-config=custom_env 
```

### 代码推荐阅读理解顺序
1. **入口文件：main.py**  
   这个文件是程序的启动点，主要负责加载配置、初始化 Sacred 实验、设置日志以及调用整个运行流程。通过阅读它，你能大致了解程序是如何组织和启动的。

2. **运行流程：run.py**  
   在 main.py 中，会调用 run.py 中的 run 函数来执行实验。这里面包含了训练/评估的主要流程逻辑，了解它有助于明白整个训练循环和数据流向。

3. **配置系统：config 目录**  
   - **default.yaml**：默认配置文件，定义了项目的基本参数。  
   - **algs/qmix.yaml 和 envs/custom_env.yaml**：分别为算法和环境提供具体的配置。  
   了解配置文件可以帮助你明白每个参数的含义和如何调试实验。

4. **环境部分：src/envs/**  
   - **multiagentenv.py**：定义了多智能体环境的基本接口。  
   - **custom_env.py**：自定义环境实现，阅读这里可以了解环境如何被构造、如何实现 reset/step 等基本方法。

5. **智能体和算法：src/modules 和 src/learners**  
   - 在 **modules/agents/** 中，可以查看智能体的实现（例如 rnn_agent.py）。  
   - 在 **learners/** 中，可以了解 Q-learning 的实现逻辑（例如 q_learner.py）。

6. **控制器和调度器：src/controllers 和 src/runners/**  
   - **controllers/**：通常负责协调各个智能体之间的动作选择等逻辑。  
   - **runners/episode_runner.py**：负责组织每个 episode 的运行（例如环境交互、数据采集、调用学习器更新等）。

7. **其他工具：src/utils/**  
   这里包含日志、时间辅助、数据转换等工具函数，能够帮助你理解项目的其他辅助功能。
# MACPO 环境接口分析文档

> 本文档基于对 MACPO 官方开源代码的逐文件阅读，总结了接入自定义环境所需的全部接口要求。
> 仓库地址：https://github.com/chauncygu/Multi-Agent-Constrained-Policy-Optimisation

---

## 1. MACPO 代码结构概览

```
MACPO/
├── macpo/
│   ├── __init__.py
│   ├── config.py                          # 全局超参数定义
│   ├── scripts/
│   │   └── train/
│   │       └── train_mujoco.py            # ★ 训练入口
│   ├── runner/
│   │   └── separated/
│   │       ├── base_runner_macpo.py        # ★ Runner 基类（MACPO 专用）
│   │       ├── mujoco_runner_macpo.py      # ★ MujocoRunner 继承基类，实现 run/warmup/collect/insert
│   │       ├── base_runner.py
│   │       └── mujoco_runner.py
│   ├── envs/
│   │   ├── env_wrappers.py                # ★ 向量化环境包装器（ShareSubprocVecEnv, ShareDummyVecEnv）
│   │   └── safety_ma_mujoco/
│   │       └── safety_multiagent_mujoco/
│   │           ├── multiagentenv.py        # 多智能体环境基类
│   │           ├── mujoco_multi.py         # ★ MujocoMulti 环境实现（参考样板）
│   │           └── ...
│   ├── algorithms/
│   │   └── r_mappo/
│   │       ├── r_macpo.py                 # ★ MACPO/MACTRPO 核心算法
│   │       └── algorithm/
│   │           ├── MACPPOPolicy.py        # ★ 策略包装（actor + critic + cost_critic）
│   │           └── r_actor_critic.py      # Actor / Critic 网络
│   └── utils/
│       ├── separated_buffer.py            # ★ SeparatedReplayBuffer
│       └── util.py
```

---

## 2. 训练入口文件

**文件路径**: `MACPO/macpo/scripts/train/train_mujoco.py`

### 训练流程:
1. `get_config()` 解析全局超参数
2. `parse_args()` 解析环境特定参数
3. `make_train_env()` → 创建向量化训练环境
4. `make_eval_env()` → 创建向量化评估环境
5. 实例化 `MujocoRunner(config)` → `runner.run()` 开始训练

### 关键 config 字典:
```python
config = {
    "all_args": all_args,      # 所有超参数
    "envs": envs,              # 向量化训练环境
    "eval_envs": eval_envs,    # 向量化评估环境
    "num_agents": num_agents,  # agent 数量
    "device": device,          # torch device
    "run_dir": run_dir         # 输出目录
}
```

---

## 3. 环境创建函数

**文件路径**: `train_mujoco.py` 中的 `make_train_env()` / `make_eval_env()`

```python
def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            env = MyEnvironment(...)  # 创建单个环境实例
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    
    if all_args.n_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])
```

---

## 4. Runner 与环境的交互（调用链）

**文件**: `mujoco_runner_macpo.py` 继承 `base_runner_macpo.py`

### 4.1 warmup()
```python
obs, share_obs, _ = self.envs.reset()
# obs: shape (n_rollout_threads, n_agents, obs_dim)
# share_obs: shape (n_rollout_threads, n_agents, share_obs_dim)
# _: available_actions, shape (n_rollout_threads, n_agents, n_actions)
```

### 4.2 collect(step)
- 对每个 agent 调用 `policy.get_actions(share_obs, obs, rnn_states, ...)`
- 返回: values, actions, action_log_probs, rnn_states, rnn_states_critic, cost_preds, rnn_states_cost
- actions 形状: (n_rollout_threads, n_agents, action_dim)

### 4.3 envs.step(actions)
```python
obs, share_obs, rewards, costs, dones, infos, _ = self.envs.step(actions)
# obs:       (n_rollout_threads, n_agents, obs_dim)
# share_obs: (n_rollout_threads, n_agents, share_obs_dim)
# rewards:   (n_rollout_threads, n_agents, 1)
# costs:     (n_rollout_threads, n_agents, 1)  ← 从 infos 中提取
# dones:     (n_rollout_threads, n_agents)
# infos:     tuple of dicts
# _:         available_actions
```

### 4.4 insert(data)
将 obs, share_obs, rewards, costs 等数据插入每个 agent 的 SeparatedReplayBuffer

### 4.5 compute()
计算 value returns 和 cost returns（GAE）

### 4.6 train()
对每个 agent 依次更新（MACPO 算法），包含 factor 传递

---

## 5. 单个环境必须提供的接口

基于 `ShareDummyVecEnv` 的 `shareworker()` 分析，环境需要：

### 必需属性:
```python
env.n_agents            # int, agent 数量
env.observation_space   # list of gym.Space, 长度 = n_agents
env.share_observation_space  # list of gym.Space, 长度 = n_agents
env.action_space        # tuple/list of gym.Space, 长度 = n_agents
```

### 必需方法:

#### reset()
```python
def reset(self) -> Tuple[obs, share_obs, available_actions]:
    """
    Returns:
        obs: list/array of shape (n_agents, obs_dim)
        share_obs: list/array of shape (n_agents, share_obs_dim)
        available_actions: array of shape (n_agents, n_actions)
    """
```

#### step(actions)
```python
def step(self, actions) -> Tuple[obs, share_obs, rewards, dones, infos, available_actions]:
    """
    Args:
        actions: 每个 agent 的动作，可以是 list/array，
                 对连续动作: actions[i] shape = (action_dim,)
    Returns:
        obs: list/array of shape (n_agents, obs_dim)
        share_obs: list/array of shape (n_agents, share_obs_dim)
        rewards: list of shape (n_agents, 1) ← 注意维度是 [[r]] * n_agents
        dones: list of shape (n_agents,) ← 布尔值
        infos: list of dicts, 长度 = n_agents
               ★ 每个 dict 必须包含 "cost" 键！
               infos[i]["cost"] 形状需为 (n_agents, 1) 或可被 ShareSubprocVecEnv 正确提取
        available_actions: array of shape (n_agents, n_actions)
    """
```

#### seed(seed)
```python
def seed(self, seed): pass
```

#### close()
```python
def close(self): pass
```

---

## 6. reset() / step() 返回值格式详解

### reset() 返回:
```
(obs_list, share_obs_list, avail_actions)
```
- `obs_list`: Python list，每个元素是 ndarray(obs_dim,)
- `share_obs_list`: Python list，每个元素是 ndarray(share_obs_dim,)
- `avail_actions`: ndarray(n_agents, n_actions)，连续动作时全部为 1

### step() 返回:
```
(obs_list, share_obs_list, rewards, dones, infos, avail_actions)
```
- `rewards`: `[[reward_n]] * n_agents` → 每个 agent 得到 shape (1,) 的 reward
- `dones`: `[done_n] * n_agents` → 布尔值列表
- `infos`: `[info_dict] * n_agents`，其中 info_dict 必须包含:
  - `"cost"`: `[[cost_value]] * n_agents` → 每个 agent 的 cost

---

## 7. VecEnv 类型

MACPO 使用两种向量化环境包装器（文件: `env_wrappers.py`）:

### ShareDummyVecEnv (n_rollout_threads=1)
- 单进程，直接调用环境
- reset/step 中自动 np.stack

### ShareSubprocVecEnv (n_rollout_threads>1)
- 多进程，每个进程一个环境副本
- 通过 `shareworker()` 进行进程间通信
- step 返回 7 个值: obs, share_obs, rews, costs, dones, infos, available_actions
- **cost 的提取方式**: `cost_x = np.array([item[0]['cost'] for item in infos])`
  - 即从每个环境的第一个 agent 的 info 中提取 cost
  - cost 应该是 shape (n_agents, 1) 的 list/array

---

## 8. Agent 维度排列顺序

在向量化环境外部（Runner 层面）:
```
第0维: n_rollout_threads (并行环境数)
第1维: n_agents (agent 数)
第2维: feature_dim (观测/动作/奖励维度)
```

在单个环境内部（step/reset 返回时）:
```
第0维: n_agents
第1维: feature_dim
```

---

## 9. 连续动作空间定义

```python
from gym.spaces import Box

# 每个 agent 的动作空间
action_space = tuple([
    Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)
    for _ in range(n_agents)
])
```

- 连续动作使用 `Box` 空间
- `SeparatedReplayBuffer` 中 `available_actions = None`（因为不是离散动作）
- 动作维度通过 `get_shape_from_act_space()` 获取

---

## 10. Reward / Cost / Done / Info 进入算法更新的路径

### Reward:
1. `env.step()` → rewards (n_agents, 1)
2. → `ShareDummyVecEnv.step_wait()` → np.stack → (n_threads, n_agents, 1)
3. → `MujocoRunner.insert()` → `buffer[agent_id].insert(rewards=rewards[:, agent_id])`
4. → `buffer.compute_returns()` 计算 GAE
5. → `R_MACTRPO_CPO.train()` 中使用 advantages

### Cost:
1. `env.step()` → infos[i]["cost"] = [[cost]] * n_agents
2. → `ShareSubprocVecEnv.step_wait()` 中: `cost_x = np.array([item[0]['cost'] for item in infos])`
3. → (n_threads, n_agents, 1) 形状
4. → `MujocoRunner.insert()` → `buffer[agent_id].insert(costs=costs[:, agent_id])`
5. → `buffer.compute_cost_returns()` 计算 cost GAE
6. → MACPO 中使用 cost advantage 进行约束优化

### Done:
1. `env.step()` → dones = [done_bool] * n_agents
2. → 在 Runner 中 `dones_env = np.all(dones, axis=1)` 判断 episode 是否结束
3. → 用于构造 masks 和 active_masks

### Info:
- 除 cost 外的自定义信息可通过 `infos` 传递
- 支持 `bad_transition` 标志

---

## 11. 新增环境的最小改动点

### 不需要修改的文件:
- `macpo/config.py` — 通过 `parse_known_args` 支持新参数
- `macpo/utils/separated_buffer.py` — 通用 buffer
- `macpo/runner/separated/base_runner_macpo.py` — 通用基类
- `macpo/algorithms/` — 算法代码完全不动

### 需要新增/修改的文件:

1. **新增环境文件** (`envs/fov_penetration/`)
   - 实现符合上述接口的环境类

2. **新增训练入口** (`scripts/train_fov_penetration_macpo.py`)
   - 参考 `train_mujoco.py`，替换环境创建逻辑

3. **新增 Runner** (可选，可复用 `mujoco_runner_macpo.py`)
   - 如果环境接口与 MujocoMulti 完全一致，可以直接复用
   - 建议新写一个 `fov_runner_macpo.py` 以便自定义日志

4. **新增配置** (`configs/`)
   - 环境特定参数

### 核心接口清单（必须实现）:
```python
class FOVPenetrationEnv:
    n_agents: int                           # = 4 (1 attacker + 3 escorts)
    observation_space: list[Box]            # 长度 n_agents
    share_observation_space: list[Box]      # 长度 n_agents
    action_space: tuple[Box]               # 长度 n_agents，每个 Box([-1,-1],[1,1])
    
    def reset(self) -> (obs, share_obs, avail_actions)
    def step(self, actions) -> (obs, share_obs, rewards, dones, infos, avail_actions)
    def seed(self, seed)
    def close(self)
```

### 最关键的 cost 接口:
```python
# 在 step() 返回的 infos 中:
info["cost"] = [[cost_value]] * self.n_agents
# 每个 agent 的 cost 是一个 (1,) 的列表
# cost_value 是标量 float
```

---

## 12. 总结：对齐策略

为了最小化对原始 MACPO 代码的修改:

1. 环境完全独立实现，放在 `envs/fov_penetration/` 目录
2. 训练脚本参考 `train_mujoco.py` 新写，只改环境创建部分
3. Runner 可直接复用 `mujoco_runner_macpo.py`，因为其 warmup/collect/insert/compute/train 逻辑与环境解耦
4. 唯一需要 patch 原仓库的点：如果需要在 `base_runner_macpo.py` 的 import 路径中支持新环境（实际上不需要，因为训练脚本独立）
5. 通过 `sys.path` 管理，使新代码可以引用 MACPO 的模块

**结论：零修改原始 MACPO 代码即可接入自定义环境。**

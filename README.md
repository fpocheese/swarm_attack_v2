# 多无人机协同突防 (Multi-UAV Cooperative Penetration)

> **4架进攻固定翼无人机 (RL训练)** vs **4架防御拦截器 (规则策略)** → **突破拦截，命中高价值目标(HVT)**
>
> 算法: MAPPO (Multi-Agent PPO with GRU-RNN) — 基于 MACPO 框架，安全约束已禁用
>
> 当前版本: **V37 "Fix Heading Drift"** (2026-04-06)

---

## 目录

- [1. 项目概述](#1-项目概述)
- [2. 快速开始](#2-快速开始)
- [3. 项目结构与关键文件](#3-项目结构与关键文件)
- [4. 环境设计](#4-环境设计)
- [5. 观测空间与动作空间](#5-观测空间与动作空间)
- [6. 奖励系统](#6-奖励系统)
- [7. 动力学模型](#7-动力学模型)
- [8. 防御方 AI（拦截器策略）](#8-防御方-ai拦截器策略)
- [9. 训练与评估](#9-训练与评估)
- [10. 算法接口说明](#10-算法接口说明)
- [11. 已知问题与核心挑战](#11-已知问题与核心挑战)
- [12. 版本演进历史](#12-版本演进历史)
- [13. 接手开发指南](#13-接手开发指南)

---

## 1. 项目概述

### 问题描述

本项目研究**多无人机集群协同突防**问题：4架进攻方固定翼无人机需要穿越4架防御方拦截器的拦截区域，命中地面高价值目标(HVT)。

- **进攻方**（4架，MARL训练对象）：低速(45m/s)、低机动(2.5G)，但通过协同战术突防
- **防御方**（4架，规则策略）：高速(55m/s)、高机动(5G)，采用3D比例导引(PN)拦截
- **HVT**：地面静止目标，位于 `(1200, 0, 0)`

### 核心设计理念

1. **同质智能体，角色涌现**：4架进攻无人机完全相同（同质），诱饵/突防角色通过奖励设计自然涌现
2. **不对称对抗**：防御方在速度和机动性上占优，进攻方靠数量和协同弥补
3. **V31 通过退化机制**：拦截器飞越目标后机动性大幅降低(20%)，为进攻方创造突防窗口
4. **死亡代价极低** (`killed_penalty=-0.5`)：鼓励勇敢突防，不惧牺牲
5. **命中奖励极大** (`hit_hvt_bonus=8000`)：一次成功突防的收益远超所有代价

### 技术栈

- **RL算法**: MAPPO (Multi-Agent PPO with GRU-RNN)，基于 `third_party/MACPO/` 框架
- **环境**: 自定义 OpenAI Gym 兼容多智能体环境
- **动力学**: 3D固定翼运动学，RK4积分，`dt=0.01s`
- **硬件需求**: NVIDIA GPU (推荐 RTX 3090 24GB+), conda环境 `rlgpu`
- **Python**: 3.8

---

## 2. 快速开始

### 环境配置

```bash
# 创建 conda 环境
conda create -n rlgpu python=3.8 -y
conda activate rlgpu

# 安装依赖
pip install -r requirements.txt

# 安装 MACPO (training framework)
cd third_party/MACPO
pip install -e .
cd ../..
```

### 训练

```bash
# 启动 V37 训练 (推荐)
bash scripts/run_v37_train.sh

# 或手动运行
conda run --no-capture-output -n rlgpu python scripts/train_fov_penetration_macpo.py \
    --env_name fov_penetration \
    --algorithm_name mappo \
    --experiment_name v37_heading_fix \
    --scenario scenario_1 \
    --n_rollout_threads 80 \
    --episode_length 8000 \
    --num_env_steps 200000000 \
    --hidden_size 256 \
    --layer_N 3 \
    --lr 3e-4 \
    --use_recurrent_policy \
    --use_feature_normalization
```

### 评估

```bash
# 评估最新 checkpoint（自动查找最新模型）
conda run --no-capture-output -n rlgpu python eval_v28_10episodes.py \
    --model_dir outputs/results/fov_penetration/mappo/v37_heading_fix/run1/ \
    --n_eval_episodes 10 \
    --render_gif
```

### 快速测试环境

```bash
conda run --no-capture-output -n rlgpu python tests/test_env.py
```

---

## 3. 项目结构与关键文件

```
muti_uav_attack_v16/
├── README.md                          # 本文件
├── requirements.txt                   # Python 依赖
│
├── envs/fov_penetration/              # ★★★ 核心环境包 ★★★
│   ├── config.py                      # ★ 所有参数配置（物理、奖励、场景）
│   ├── fov_penetration_env.py         # ★ 主环境类 (step/reset/obs)
│   ├── reward_cost.py                 # ★ 奖励函数设计
│   ├── dynamics.py                    # ★ 3D固定翼动力学 (RK4)
│   ├── entities.py                    #   实体类 (Aircraft, HVT)
│   ├── scenarios.py                   #   场景定义 (4v4, 4v6, 6v4)
│   ├── policies_interceptor.py        #   防御方AI策略 (PN导引)
│   └── render.py                      #   渲染可视化
│
├── scripts/
│   ├── train_fov_penetration_macpo.py # ★ 训练入口脚本
│   ├── run_v37_train.sh               #   V37 训练启动脚本
│   ├── diagnose_flight_deviation.py   #   航向漂移诊断工具
│   └── ...                            #   各版本历史训练/评估脚本
│
├── eval_v28_10episodes.py             # ★ 标准评估脚本 (10 episodes + GIF)
│
├── third_party/MACPO/                 # MACPO/MAPPO 算法框架 (第三方)
│   └── MACPO/macpo/
│       ├── algorithms/                #   PPO / MAPPO / MACPO 算法实现
│       ├── runner/                    #   MujocoRunner 训练循环
│       └── utils/                     #   向量化环境、Buffer等
│
├── outputs/
│   ├── results/fov_penetration/mappo/ #   训练输出 (模型checkpoint、TensorBoard日志)
│   └── gifs/                          #   评估生成的 GIF 动画
│
├── tests/                             #   环境测试
└── docs/                              #   开发文档
```

### ⚡ 如果你只有30分钟，优先阅读：

1. **`envs/fov_penetration/config.py`** — 理解所有参数设定
2. **`envs/fov_penetration/reward_cost.py`** — 理解奖励设计（最核心的调优对象）
3. **`envs/fov_penetration/fov_penetration_env.py`** — 理解环境接口 (`step`, `reset`, `_get_obs`)
4. **本 README 的「已知问题」章节** — 理解当前瓶颈和必须解决的问题

---

## 4. 环境设计

### 场景布局

```
       进攻方生成区                    防御方生成区              HVT
   (-1200, 0, 300)                (600, 0, 350)         (1200, 0, 0)
   spread: 150m XY                spread: 200m XY
          |                              |                     |
   ───→ ───→ ───→ ───→        ←──── ←──── ←──── ←────         ★
   4× offensive (45m/s)        4× defensive (55m/s)       ground target
          |                              |                     |
          |←─────── 约2400m 总攻击距离 ──────────────────────→|
```

- **地图大小**: 2000m × 2000m × 1000m (超界惩罚而非终止)
- **仿真步长**: `dt=0.01s`, 每 episode 最多 8000步 = 80秒
- **3个内置场景**: `scenario_1` (4v4 均衡), `scenario_2` (4v6 防御优势), `scenario_3` (6v4 进攻优势)

### 环境接口 (OpenAI Gym 兼容)

```python
env = FOVPenetrationEnv(config)

# Spaces
env.observation_space      # list of Box(shape=(37,))   × n_agents
env.share_observation_space  # list of Box(shape=(77,)) × n_agents  (for centralized critic)
env.action_space           # tuple of Box(low=-1, high=1, shape=(3,)) × n_agents
env.n_agents               # 4 (offensive only — defensive is rule-based)

# Reset
obs, share_obs, avail = env.reset()
# obs.shape     = (n_agents, 37)
# share_obs.shape = (n_agents, 77)

# Step
obs, share_obs, rewards, costs, dones, infos, avail = env.step(actions)
# actions.shape = (n_agents, 3)    — continuous [-1, 1]
# rewards.shape = (n_agents, 1)
# costs.shape   = (n_agents, 1)    — always 0 (costs disabled in V37)
# dones.shape   = (n_agents + 1,)  — last element is env_done
# infos         = list of dict per agent
```

### 终止条件

| 条件 | 说明 |
|------|------|
| **成功 (success)** | 任何进攻无人机进入 HVT 5m 范围 |
| **全灭 (all_killed)** | 所有4架进攻无人机被击杀 |
| **超时 (timeout)** | 达到 max_steps=8000 (80秒) |

### 击杀机制

- **FOV 锁定击杀**: 拦截器 FOV 内 3m → 进攻方被击杀
- **碰撞互杀 (CPA)**: 双方最近接近点 < 5m → 双方同时死亡
- **HVT 命中**: 进攻方与 HVT 距离 < 5m → 命中成功（基于 CPA 轨迹检测）

---

## 5. 观测空间与动作空间

### 观测空间 (37维, V35+ "Target-First" 设计)

| 组 | 维度 | 内容 | 说明 |
|----|------|------|------|
| **自身状态** | 10 | rel_goal_xyz(3), speed(1), heading(1), gamma(1), ax(1), ay(1), is_locked(1), locked_by_count(1) | 归一化到 [-1,1] |
| **HVT 制导** | 8 | rho(距离), closing(接近速度), omega(视线角速度), omega_dot, pn_hint, heading_error, dist_progress, teammates_drawing_fire | 类导弹制导量 |
| **Top-2 威胁** | 12 | 每个拦截器: rho, closing, bearing_error, in_fov, is_locking_me, omega_los × 2 | 最近2个拦截器 |
| **队伍态势** | 4 | n_locked, n_free, team_pen_score, alive_ratio | 全局协同信息 |
| **目标优先级** | 2 | P_pen(突防概率), P_hit(命中概率) | 解析先验计算 |
| **时间** | 1 | step / max_steps | 进度感知 |

### 共享观测 (77维, 用于 MAPPO Centralized Critic)

包含：所有进攻方状态 (10×4=40), 所有防御方简要状态 (7×4=28), HVT位置 (3), 全局指标 (5), 时间 (1)

### 动作空间 (3维 连续 [-1, 1])

| 维度 | 物理含义 | action=0 时 (trim) | 说明 |
|------|---------|-------------------|------|
| `action[0]` | 轴向加速度 ax | 0 m/s² (匀速) | 控制加减速; 映射到 [ax_min, ax_max] |
| `action[1]` | 法向加速度幅值 ay | G (重力补偿, 平飞) | 控制机动强度; 映射到 [0, ay_max] |
| `action[2]` | 法向加速度方向 μ (bank angle) | π/2 (竖直向上, 维持平飞) | 控制转弯/俯仰; **极其敏感** ⚠️ |

> **⚠️ 关键警告: `action[2]` (μ) 的物理敏感度问题**
>
> 偏航角速度公式: `ψ̇ = ay·cos(μ) / (v·cos(γ))`
>
> 在 trim 状态 (μ=π/2) 附近, ∂ψ̇/∂a₂ ≈ -0.685 rad/s per unit action。
> 神经网络输出仅 0.04 的系统性偏差 → 1.6°/s 持续航向漂移 → 80秒内偏转 128°。
>
> **这是当前项目最关键的已知问题**。V37 通过 `heading_error_penalty` 和 `mu_regularization` 尝试修复。
> 如果 V37 仍不够，应考虑修改 `dynamics.py` 中 μ 的映射增益 (当前 π → 建议尝试 π/2)。

---

## 6. 奖励系统

### V37 奖励设计 (reward_cost.py)

奖励分为 **每步奖励** 和 **终局奖励**。

#### 每步奖励 (per agent, per step)

| 组件 | 权重(λ) | 公式 | 目的 |
|------|---------|------|------|
| **Approach** | 30.0 | `λ × Δd / norm_dist`, close_range内放大10× | 接近 HVT |
| **Heading Align** | 0.5 | `λ × cos(heading_error)` | 朝向 HVT (**V37主导正向信号**) |
| **Heading Error Penalty** | 0.8 | `-λ × (err/π)²` | 惩罚航向偏差 (**V37新增**) ⭐ |
| **Closing Speed** | 0.8 | `λ × Vc / vel_range` | 接近速度奖励 |
| **Mu Regularization** | 0.15 | `-λ × \|action[2]\|` | 抑制不必要转弯 (**V37新增**) ⭐ |
| **Proximity Penalty** | 0.15 | `-λ × d / norm_dist` | 反绕圈 |
| **Boundary Penalty** | 2.0 | 超边界时梯度惩罚 | 限制活动范围 |
| **Ground Penalty** | 1.0 | 贴地时惩罚 | 防止撞地 |
| **Killed Penalty** | -0.5 | 被击杀时一次性 | 极小代价 → 鼓励勇敢 |
| **HVT Hit Bonus** | 8000 | 命中目标时，全队共享 | 超高终极激励 |
| **Step Penalty** | -0.003 | 每步固定 | 时间压力 |

#### 终局奖励 (episode 结束时)

| 组件 | 权重 | 说明 |
|------|------|------|
| Terminal Hit | 800 per hit | 命中计数奖励 |
| Terminal Distance | 500 × (1 - min_dist/2000) | 距离越近越好 |
| All Killed Penalty | -100 | 全灭惩罚 |

#### 典型奖励分析

在 1000m 处直飞向 HVT 时的每步奖励:
```
approach       = +0.225
heading_align  = +0.500  (主导正向信号)
heading_err    = -0.000  (无偏差时为0)
closing        = +0.300
mu_reg         = -0.000  (直飞时μ≈0)
proximity      = -0.100
step_penalty   = -0.003
─────────────────────────
NET            ≈ +0.922/step
```

偏航30°时: NET ≈ +0.751/step (**损失18.8%** → 强纠偏梯度)

#### 安全约束 (Costs)

**当前已全部禁用**。`safety_bound=200` 远大于自然 cost (~150)，使 MACPO 退化为无约束 MAPPO。如需启用，修改 `config.py` 中的 `cost` 节和 `safety_bound`。

---

## 7. 动力学模型

### 3D 固定翼运动学 (dynamics.py)

**状态向量**: `[x, y, z, v, ψ (heading), γ (flight path angle)]`

**运动方程**:

```
ẋ = v·cos(ψ)·cos(γ)
ẏ = v·sin(ψ)·cos(γ)
ż = v·sin(γ)
v̇ = ax - g·sin(γ)
ψ̇ = ay·cos(μ) / (v·cos(γ))
γ̇ = (ay·sin(μ) - g·cos(γ)) / v
```

### 控制映射 (action → control)

```python
# action ∈ [-1, 1]³ → (ax, ay, μ) 物理控制量
ax = (action[0] + 1) / 2 × (ax_max - ax_min) + ax_min   # 线性映射
ay = (action[1] + 1) / 2 × ay_max + G                     # 0 → G (平飞trim)
μ  = π/2 + action[2] × π                                   # 0 → π/2 (竖直trim)
```

> **Trim 状态 (action=[0,0,0])**: 匀速直线平飞。ax=0, ay=G(重力补偿), μ=π/2(竖直向上)→ψ̇=0, γ̇=0

### 速率限制

所有控制量有变化率限制：`dax_max=60 m/s³`, `day_max=120 m/s³`, `dmu_max=30 rad/s`

### 积分方法

RK4 (四阶 Runge-Kutta)，步长 `dt=0.01s`

---

## 8. 防御方 AI（拦截器策略）

### InterceptorPolicy (policies_interceptor.py)

拦截器采用**规则策略**（非RL学习），模拟现实中的防空拦截系统。

#### 状态机

```
INIT_GUIDE → SEARCH → LOCKED → MISSED → ABANDONED
```

- **INIT_GUIDE**: 初始阶段，按匈牙利分配导引向分配的目标
- **SEARCH**: 搜索模式（目标丢失时）
- **LOCKED**: FOV 锁定跟踪 + 3D 比例导引
- **MISSED**: 飞越目标后进入退化状态
- **ABANDONED**: 距离 >800m 放弃追踪

#### 3D 比例导引 (PN)

- 导引律: N=4 倍比例导引
- 双频信息更新: 直接测量 20Hz / 引导信息 5Hz / 后方追击 1Hz
- V26 外推: 0.2s 前向预测补偿导引延迟

#### V31 通过退化 (Pass-Through Degradation) ⭐

当拦截器飞越进攻无人机后（CPA<150m 然后距离增大>30m）:
- ay 降至 **20%** (0.20 × ay_max)
- 持续 500步 (5秒) 逐渐恢复
- 同时制动 (ax=-8 m/s²)
- 超过 800m 距离时彻底放弃

**这是进攻方突防的关键机制**：前方无人机吸引拦截器冲过，拦截器因退化无法回头，为后方无人机创造突防窗口。

---

## 9. 训练与评估

### 训练配置 (V37)

| 参数 | 值 | 说明 |
|------|------|------|
| algorithm | MAPPO | 无约束版 (MACPO with costs=0) |
| hidden_size | 256 | GRU-RNN 隐藏维度 |
| layer_N | 3 | 网络层数 |
| lr | 3e-4 | 学习率 |
| n_rollout_threads | 80 | 并行采集环境数 |
| episode_length | 8000 | 每回合步数 (80s) |
| ppo_epoch | 5 | PPO 更新轮数 |
| num_mini_batch | 40 | mini-batch 数 |
| clip_param | 0.2 | PPO clip |
| entropy_coef | 0.02 | 熵系数 |
| gamma | 0.99 | 折扣因子 |
| gae_lambda | 0.95 | GAE λ |
| max_grad_norm | 10.0 | 梯度裁剪 |
| num_env_steps | 200M | 总训练步数 |
| eval_interval | 5 | 每5次更新评估1次 |
| eval_episodes | 20 | 评估回合数 |

### 训练输出目录

```
outputs/results/fov_penetration/mappo/{experiment_name}/run{N}/
├── models/          # checkpoint文件 (actor_agent{i}.pt, critic_agent{i}.pt)
├── logs/            # TensorBoard 日志
│   ├── train_episode_rewards/    # 训练回合奖励
│   ├── eval_average_episode_rewards/  # 评估回合奖励
│   └── agent{i}/average_step_rewards/ # 每步平均奖励
└── wandb/           # WandB 日志 (如启用)
```

### 关键训练指标

| 指标 | 好的值 | 说明 |
|------|--------|------|
| `eval_average_episode_rewards` | > 4000 | 含终局奖励 |
| `average_step_rewards` | > 0.6 | 每步奖励 (直飞~0.9, 绕圈~0.2) |
| success_rate (eval) | > 20% | 至少部分突防成功 |
| avg_min_dist_to_hvt | < 200m | 最近接近距离 |

### 后台训练 & 恢复

```bash
# 后台启动训练 (推荐使用 screen 防止终端断开)
screen -dmS v37train bash -c "bash scripts/run_v37_train.sh; exec bash"

# 查看 screen 会话
screen -ls

# 进入 screen 查看训练
screen -r v37train

# 退出 screen (不中断训练): Ctrl+A, D

# 从 checkpoint 恢复中断的训练
bash scripts/run_v37_resume.sh

# 查看训练进度
tail -f outputs/v37_heading_fix.log

# TensorBoard 监控
tensorboard --logdir outputs/results/fov_penetration/mappo/v37_heading_fix/ --port 6006
```

> **⚠️ 如果机器重启**: screen 会话会丢失，但 checkpoint 会保留。
> 直接运行 `bash scripts/run_v37_resume.sh` 从最新 checkpoint 恢复训练。
> Resume 会创建新的 run 目录 (run2, run3...)，但加载 run1 的权重继续训练。

---

## 10. 算法接口说明

### MACPO/MAPPO 与环境的对接

训练入口: `scripts/train_fov_penetration_macpo.py`

```python
# 环境创建
from envs.fov_penetration import FOVPenetrationEnv, get_config

config = get_config(scenario_name="scenario_1")
env = FOVPenetrationEnv(config)

# 向量化 (MAPPO 需要多环境并行)
from MACPO.macpo.envs.env_wrappers import ShareSubprocVecEnv

envs = ShareSubprocVecEnv([make_env_fn(i) for i in range(n_threads)])

# Runner (训练循环)
from MACPO.macpo.runner.separated.mujoco_runner import MujocoRunner
runner = MujocoRunner(config)
runner.run()  # 主训练循环
```

### PatchedShareDummyVecEnv

由于 MACPO 框架假设 Mujoco 环境，需要 Patch 暴露 `n_agents`:

```python
class PatchedShareDummyVecEnv(ShareDummyVecEnv):
    @property
    def n_agents(self):
        return self.envs[0].n_agents
```

### 策略网络结构

- 每个 agent 独立的 Actor-Critic (Separated training)
- **Actor**: GRU-RNN, input=37, hidden=256, layers=3, output=6 (3个均值 + 3个标准差 for Gaussian policy)
- **Critic**: GRU-RNN, input=77 (share_obs), hidden=256, layers=3, output=1 (value)

### 加载训练好的模型进行评估

```python
from MACPO.macpo.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy

policy = R_MAPPOPolicy(
    args, obs_space, share_obs_space, act_space,
    device=torch.device("cuda:0")
)
state_dict = torch.load(f"models/actor_agent{i}.pt", map_location=device)
policy.actor.load_state_dict(state_dict)
```

详细使用方法参见 `eval_v28_10episodes.py` 中的 `load_policies()` 和 `evaluate()` 函数。

---

## 11. 已知问题与核心挑战

### 🔴 关键问题: 航向漂移 (Heading Drift)

**状态**: V37 正在修复中（训练进行中）

**现象**: 部分进攻无人机不朝 HVT 飞行，而是偏转甚至绕圈飞行

**根因**: `action[2]` (bank angle μ) 的物理敏感度极高。神经网络输出 0.04 的系统性偏差 → 1.6°/s 航向漂移。在 80 秒仿真中，漂移可累积至 360°+ (绕圈一周以上)。

**诊断数据** (V36 模型, 来自 `scripts/diagnose_flight_deviation.py`):

| Agent | 状态 | mu偏差 | 漂移速率 | 总漂移 | 最近距离 |
|-------|------|--------|---------|--------|---------|
| 0 | 被锁定 | +0.007 | +0.74°/s | 44° | 489m |
| 1 | 自由 | +0.043 | -1.68°/s | 53° | 1071m |
| 2 | 自由 | +0.202 | -7.4°/s | **444°** (多圈) | 2339m |
| 3 | 自由 | +0.087 | -3.4°/s | **203°** | 1961m |

**V37 修复方案**:
- `heading_error_penalty`: λ=0.8, (err/π)² 二次惩罚 → 强纠偏梯度
- `mu_regularization`: λ=0.15, |action[2]| → 直接抑制不必要转弯
- `heading_align`: 增强至 0.5 (从 V36 的 0.2)

**如果V37仍不enough，下一步方案**:
1. 修改 `dynamics.py` 中 μ 的映射增益 (π → π/2)，降低物理层面敏感度
2. 在 `action_to_control_3d` 中添加 dead zone
3. 切换到 heading rate 直接控制（绕过 bank angle）

### 🟡 待解决: Reward Collapse

V36 训练到 ~60M 步后出现 eval reward 剧烈波动并最终崩溃 (3953 → 568)。建议:
- 添加 learning rate decay schedule
- 尝试 PopArt value normalization
- 更保守的 clip_param (0.2 → 0.1)

### 🟡 待解决: 协同战术涌现不足

当前进攻无人机基本各自飞向 HVT，缺少明显的诱饵/掩护协同。可能需要:
- 团队奖励（如拉开拦截器的dispersal奖励）
- 通信机制（attention-based 智能体间通信）
- 更长时间训练（200M步可能不够）

### 🟢 已验证正常的机制

- V31 通过退化：拦截器飞越后确实弱化
- CPA 检测：命中/碰撞检测可靠
- FOV 锁定状态机：工作正常
- 匈牙利初始分配：目标分配正确
- 3D 比例导引：拦截器能有效追踪
- RK4 动力学积分：物理仿真准确

---

## 12. 版本演进历史

| 版本 | 代号 | 关键变更 | 结果 |
|------|------|---------|------|
| V1-V10 | 早期探索 | 基础环境搭建, 2D→3D, obs维度 47/54 | 基础功能验证 |
| V11-V15 | 奖励探索 | 多种奖励组合, 场景调整 | 训练不稳定 |
| V16-V20 | 安全约束 | MACPO约束实验, attack gate | 约束导致过保守 |
| V21-V25 | 解析先验 | cone cost, decoy game, 有效突防评估 | 观测空间复杂化 |
| V26-V30 | 拦截器升级 | 3D PN, 双频信息, 外推导引 | 拦截器更真实 |
| **V31** | **PassThrough** | **拦截器飞越退化(20%ay)**, obs=45dim | **关键突破**：创造突防窗口 |
| V33 | CloseRange | 近距离奖励放大 | 最后数百米强化 |
| V34 | RescaleReward | 奖励尺度统一 | 训练更稳定 |
| **V35** | **TargetFirst** | **obs 45→37维**, 3核心信号, risk全部归零 | 激进简化，有一定效果 |
| V36 | RewardScale | approach_norm↓, closing↓, episode 8000 | **退化**：暴露漂移问题 |
| **V37** | **HeadingFix** | **heading_err_penalty + mu_reg**, heading_align↑ | **进行中**: mu偏差10x↓, 无agent飞跑 |

### V37 训练进展

**V37/run1** (2026-04-06, 被重启中断):
- 训练到 update 8 / 312 (5.76M / 200M steps, 2.9%)
- step_reward: 0.72→0.40→0.67→0.62 (有波动但恢复)
- eval: u0=-2632 → u5=1658 (改善中)

**V37 eval (update 8, 5 episodes)**:
| 指标 | 值 |
|----|----|
| 成功率 | 0% |
| 平均最近距离 | 573.9m |
| 平均奖励 | -377.5 |
| timeout率 | 100% |

**V37 航向修复效果** (vs V36 诊断数据):
| 指标 | V36 | V37 (u8) | 改善 |
|------|-----|----------|------|
| mu bias 范围 | 0.007~0.202 | 0.000~0.022 | **10x↓** |
| 最大漂移 | -7.4°/s | -0.64°/s | **11x↓** |
| 飞跑agent数 | 3/4 | **0/4** | 全部修复 |

→ 航向修复有效，但训练时间太短。V37/run2 正在从 checkpoint 恢复训练。

### V35 → V36 为什么退化?

1. V36 弱化 closing_speed (3.0→1.5) → 失去前冲惯性，暴露 μ 漂移
2. V36 延长 episode (6000→8000) → 更多时间累积漂移偏差
3. V36 没有主动航向修正机制 → 漂移无法自愈
4. V35 的"暴力"奖励**意外地遮盖了问题**，V36 暴露了本质缺陷

---

## 13. 接手开发指南

### 在新服务器上继续开发

```bash
# 1. 克隆代码
git clone git@github.com:fpocheese/swarm_attack_v2.git
cd swarm_attack_v2

# 2. 配置 conda 环境
conda create -n rlgpu python=3.8 -y
conda activate rlgpu
pip install -r requirements.txt

# 3. 安装 MACPO 框架
cd third_party/MACPO && pip install -e . && cd ../..

# 4. 验证环境
python tests/test_env.py

# 5. 开始训练
bash scripts/run_v37_train.sh
```

### 开发优先级建议

1. **检查 V37 训练结果** → 如果航向漂移修复有效，在此基础上迭代
2. **如果 V37 仍然飞歪** → 修改 `dynamics.py` 中 μ 的映射增益 (π → π/2)，或切换到 heading rate 直接控制
3. **解决 Reward Collapse** → 添加 lr decay 或 PopArt normalization
4. **协同战术** → 单机飞行稳定后，再添加协同奖励组件
5. **场景泛化** → 在 scenario_2 (4v6) 和 scenario_3 (6v4) 上测试

### 调参指南

**主要调整**: `envs/fov_penetration/config.py` → `reward` 字典

```python
"reward": {
    "lambda_approach": 30.0,             # 接近奖励强度
    "approach_norm_dist": 60.0,          # 归一化距离
    "close_range_threshold": 800,        # 近距离放大阈值(m)
    "lambda_heading_align": 0.5,         # 航向对齐
    "lambda_heading_error_penalty": 0.8, # 航向偏差惩罚 (V37新增)
    "lambda_closing": 0.8,              # 接近速度奖励
    "lambda_mu_regularize": 0.15,        # μ正则化 (V37新增)
    "lambda_proximity": 0.15,            # 反绕圈
    "hit_hvt_bonus": 8000.0,             # 命中奖励
    "killed_penalty": -0.5,              # 死亡代价
}
```

**不要轻易修改的文件**:
- `dynamics.py` — 物理方程已验证正确（除非要改 μ 映射增益）
- `policies_interceptor.py` — 拦截器 AI 经过多版本调优
- `entities.py` — 基础实体类稳定
- V31 通过退化参数 — 这是突防可能性的关键机制

### 如何添加新的奖励组件

1. 在 `config.py` 的 `reward` 字典中添加权重参数
2. 在 `reward_cost.py` 的 `compute_rewards()` 中添加计算逻辑
3. 启动新训练 (用新的 `experiment_name`)
4. 用 `eval_v28_10episodes.py` 评估并生成 GIF

### 如何修改观测空间

1. 在 `fov_penetration_env.py` 的 `_compute_space_dims()` 中更新维度计算
2. 在 `_get_obs()` 中添加/修改观测数据
3. **⚠️ 必须同步更新 `observation_space` 的 shape**，否则 MAPPO 会报维度不匹配错误
4. 类似地更新 `_get_share_obs()` 和 `share_observation_space`

---

## 参考

- MACPO: [Multi-Agent Constrained Policy Optimisation](https://arxiv.org/abs/2110.02793)
- MAPPO: [The Surprising Effectiveness of PPO in Multi-Agent Settings](https://arxiv.org/abs/2103.01955)
- 3D Proportional Navigation: 经典导弹制导律
- 匈牙利算法: 目标分配 (scipy.optimize.linear_sum_assignment)

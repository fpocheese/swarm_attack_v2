# 多无人机协同突防 (Multi-UAV Cooperative Penetration) — V10# 基于 MACPO 的固定翼无人机集群协同突防仿真（V2）



> **4架进攻固定翼无人机** vs **4架防御拦截器** + **高价值目标(HVT)**> 多智能体约束策略优化 + 敌方视场角(FOV)限制的固定翼无人机集群协同突防三体问题

> 算法: MACPO (Multi-Agent Constrained Policy Optimization)

> 当前版本: **V10 "前方牺牲, 后方突防"**## 项目概述



---本项目将 **MACPO (Multi-Agent Constrained Policy Optimisation)** 算法接入一个自定义的多智能体协同突防场景，实现训练、测试和可视化动画。



## 1. 项目概述### 场景说明



本项目研究多无人机集群协同突防问题。进攻方4架固定翼无人机需要穿越防御方4架拦截器的FOV(视场角)拦截区域，到达高价值目标(HVT)附近。- **进攻集群（MARL训练对象）**：1 架 attacker + 3 架 escorts（4个智能体）

- **防御集群（规则策略）**：4 架 interceptors，比例导引 N=3 + 双频信息更新

### 核心设计理念- **高价值目标 (HVT)**：地面固定点

1. **进攻集群不知道拦截集群的目标分配方案** — 通过态势感知推断威胁

2. **前方飞机自然承担诱饵/牺牲角色，后方飞机突防** — 通过reward设计涌现**任务目标**：突防成功率最大化，attacker 进入 HVT 3m 范围即为成功

3. **防御方采用"死盯"策略** — 一次分配后不再切换目标（Hungarian + 比例导引）

## V2 核心改进

### 任务场景

```| 改进项 | V1 | V2 |

进攻方初始位置         防御方初始位置           高价值目标|--------|----|----|

  (-3000, 0)            (2000, 0)             (3000, 0, 0)| 动力学积分 | 欧拉法 | **RK4 四阶龙格-库塔** |

  4架, v=50m/s          4架, v=90m/s          hit_hvt_range=500m| 航向抖动 | 无处理 | **过载变化率限制** (dnx_max/dny_max) |

     ----→                ←----| 拦截器导引 | 简单追踪 | **比例导引 N=3 + 双频信息** (20Hz直测/2Hz指引) |

     heading→HVT          heading→进攻方| 击杀距离 | 多种 (hit_range等) | **统一 kill_range=3m** |

```| 护航击杀 | 不支持 | **escort可击杀interceptor** (3m) |

| 奖励设计 | 多项分散 | **突防导向** (hit_hvt=200, escort_kill=50) |

---| obs维度 | 33 | **47** (含nx/ny/alive flags) |

| share_obs | 40 | **54** (含intc_alive_ratio) |

## 2. 项目结构

### 过载变化率限制参数

```

muti_uav_attack/| 平台 | dnx_max (g/s) | dny_max (g/s) |

├── envs/fov_penetration/          # 核心环境|------|---------------|---------------|

│   ├── fov_penetration_env.py     # 主环境类 FOVPenetrationEnv| Attacker | 5.0 | 8.0 |

│   ├── config.py                  # 所有参数配置 (V10)| Escort | 5.0 | 8.0 |

│   ├── reward_cost.py             # 奖励函数 (V10)| Interceptor | 8.0 | 15.0 |

│   ├── dynamics.py                # 3D固定翼运动学 (RK4积分)

│   ├── entities.py                # Aircraft/HVT实体类### 速度与过载约束

│   ├── policies_interceptor.py    # 防御方策略 (比例导引+死盯)

│   ├── target_assignment.py       # Hungarian目标分配| 平台 | 速度范围 (m/s) | nx范围 | ny范围 |

│   ├── render.py                  # 3D可视化渲染|------|---------------|--------|--------|

│   └── scenarios.py               # 场景定义| Attacker | 35-60 | [-0.5g, 1.5g] | [-4g, 4g] |

├── scripts/| Escort | 40-65 | [-0.5g, 1.5g] | [-4g, 4g] |

│   ├── train_fov_penetration_macpo.py  # ⭐ 训练脚本| Interceptor | 70-120 | [-1.0g, 3.0g] | [-8g, 8g] |

│   ├── eval_v7_stats.py           # 多seed统计评估

│   ├── diagnose_episode.py        # 11图诊断工具## 项目结构

│   ├── render_trained_policy.py   # 渲染GIF

│   └── run_v*.sh                  # 各版本训练shell脚本```

├── third_party/MACPO/             # MACPO算法 (已修改适配).

│   └── MACPO/macpo/runner/separated/├── third_party/

│       └── mujoco_runner_macpo.py # ⭐ Runner (已修复lr_decay+step解包)│   └── MACPO/                          # 原始 MACPO 仓库（不修改）

├── outputs/                       # 训练输出 (gitignore, 不在仓库中)├── envs/

│   ├── results/                   # 模型+tensorboard日志│   └── fov_penetration/

│   └── gifs/                      # 可视化GIF│       ├── __init__.py

└── .gitignore│       ├── config.py                   # ★ 环境配置 (V2: kill_range, 过载率限制等)

```│       ├── dynamics.py                 # ★ 2D动力学 (V2: RK4 + 过载变化率限制)

│       ├── entities.py                 # 飞行器实体 (V2: nx_prev/ny_prev追踪)

---│       ├── fov_penetration_env.py      # ★ 主环境 (V2: obs=47, share_obs=54)

│       ├── reward_cost.py              # ★ 奖励/成本 (V2: 突防导向设计)

## 3. 环境安装│       ├── policies_interceptor.py     # ★ 拦截器策略 (V2: PN N=3, 双频信息)

│       ├── render.py                   # 可视化渲染

### 依赖环境│       └── scenarios.py                # 场景预设

- Python 3.8 (建议用conda)├── scripts/

- PyTorch (CPU即可, GPU更佳)│   ├── train_fov_penetration_macpo.py  # ★ 训练脚本 (V2: PatchedShareDummyVecEnv)

- numpy, scipy, gym, tensorboardX, wandb(可选)│   ├── train_fov_penetration_macpo.sh  # 训练启动 shell

│   ├── eval_fov_penetration_macpo.py   # ★ 评估脚本

### 安装步骤│   ├── render_random_episode.py        # 随机策略可视化

```bash│   └── render_trained_policy.py        # 训练模型可视化

# 1. 创建conda环境├── tests/

conda create -n rlgpu python=3.8 -y│   └── test_env.py                     # 环境测试 (7个测试全通过)

conda activate rlgpu├── docs/

├── run_training.sh                     # setsid长训练启动脚本

# 2. 安装PyTorch (CPU版)├── requirements.txt

pip install torch==1.13.1+cpu -f https://download.pytorch.org/whl/torch_stable.html└── README.md

# 或GPU版:```

# pip install torch==1.13.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html

## 安装

# 3. 安装其他依赖

pip install numpy scipy gym==0.21.0 tensorboardX setproctitle matplotlib imageio```bash

# 1. 克隆项目

# 4. 安装MACPO (本地)git clone https://github.com/fpocheese/muti_uav_attack.git

cd third_party/MACPO/MACPOcd muti_uav_attack

pip install -e .

```# 2. 创建conda环境

conda create -n rlgpu python=3.8

### 快速验证环境conda activate rlgpu

```bash

cd /path/to/muti_uav_attack# 3. 安装依赖

python -c "pip install -r requirements.txt

import sys; sys.path.insert(0, '.')

from envs.fov_penetration.fov_penetration_env import FOVPenetrationEnv# 4. 安装 MACPO（开发模式）

import numpy as npcd third_party/MACPO/MACPO

env = FOVPenetrationEnv({'scenario': 'scenario_1'})pip install -e .

obs, share_obs, avail = env.reset()cd ../../..

print(f'obs_dim={env.obs_dim}, share_obs_dim={env.share_obs_dim}')```

for _ in range(10):

    result = env.step(np.zeros((4,3)))## 快速开始

print('Environment OK!')

"### 1. 运行测试

# 期望输出: obs_dim=84, share_obs_dim=76

``````bash

export PYTHONPATH="$(pwd):$(pwd)/third_party/MACPO/MACPO:$PYTHONPATH"

---python tests/test_env.py

# 预期: 7个测试全部通过

## 4. 训练```



### 一键训练命令 (V10)### 2. 训练

```bash

cd /path/to/muti_uav_attack```bash

export PYTHONPATH="$(pwd):$(pwd)/third_party/MACPO/MACPO:$PYTHONPATH"

# 标准训练: 10M步, ~8-10小时(20核CPU), ~3-4小时(GPU)

PYTHONUNBUFFERED=1 nohup conda run --no-capture-output -n rlgpu \# ⚠️ 重要约束:

  python -u scripts/train_fov_penetration_macpo.py \# - 必须使用 --n_rollout_threads 1

  --env_name fov_penetration \# - 不要使用 --use_linear_lr_decay

  --algorithm_name macpo \

  --experiment_name v10_front_sacrifice \# 快速验证 (~1分钟)

  --scenario scenario_1 \python scripts/train_fov_penetration_macpo.py \

  --seed 1 \    --algorithm_name macpo \

  --n_rollout_threads 20 \    --experiment_name v2_test \

  --n_training_threads 1 \    --seed 42 \

  --num_mini_batch 40 \    --n_rollout_threads 1 \

  --num_env_steps 10000000 \    --num_env_steps 2000 \

  --episode_length 1500 \    --episode_length 500 \

  --ppo_epoch 5 \    --hidden_size 128 \

  --lr 3e-4 \    --layer_N 2

  --critic_lr 3e-4 \

  --entropy_coef 0.02 \# 完整训练 (2M步, ~10-22小时)

  --hidden_size 256 \# 推荐后台运行:

  --layer_N 3 \setsid bash run_training.sh &

  --use_eval \# 日志: outputs/train_v2_aggressive.log

  --eval_interval 5 \```

  --n_eval_rollout_threads 4 \

  --eval_episodes 8 \### 3. 评估

  --use_linear_lr_decay \

  --log_interval 1 \```bash

  --save_interval 20 \export PYTHONPATH="$(pwd):$(pwd)/third_party/MACPO/MACPO:$PYTHONPATH"

  > outputs/v10_train.log 2>&1 &

python scripts/eval_fov_penetration_macpo.py \

echo "Training started, PID: $!"    --model_dir outputs/results/fov_penetration/macpo/v2_aggressive/run1/models \

```    --n_episodes 20 \

    --hidden_size 128 \

### 关键超参数说明    --layer_N 2 \

| 参数 | 值 | 说明 |    --save_gif

|------|-----|------|```

| `episode_length` | **1500** | **必须≥1200!** (v=50m/s, dist=6000m, dt=0.1, 需1200步到达) |

| `num_env_steps` | 10000000 | 总训练步数 |### 4. 可视化

| `n_rollout_threads` | 20 | 并行环境数 (根据CPU核数调整) |

| `hidden_size` | 256 | 网络隐藏层大小 |```bash

| `layer_N` | 3 | 网络层数 |export PYTHONPATH="$(pwd):$(pwd)/third_party/MACPO/MACPO:$PYTHONPATH"

| `entropy_coef` | 0.02 | 探索系数 |

python scripts/render_trained_policy.py \

> ⚠️ **episode_length 必须与 config.py 中的 max_steps 一致 (当前都是1500)**    --model_dir outputs/results/fov_penetration/macpo/v2_aggressive/run1/models \

> 如果 episode_length < 1200, 飞机永远无法到达HVT, 训练必定失败!    --hidden_size 128 \

    --layer_N 2

### 监控训练```

```bash

# 实时查看日志## 已知约束

tail -f outputs/v10_train.log

1. **必须 `n_rollout_threads=1`**：MACPO 的 ShareSubprocVecEnv 只解包6个返回值，env.step()返回7个

# 查看进程2. **不能用 `--use_linear_lr_decay`**：separated policy 下 trainer 是 list

ps aux | grep macpo-fov3. **hidden_size=128, layer_N=2**：V2默认网络配置，评估/渲染时必须匹配



# tensorboard (如果可用)## 参考文献

tensorboard --logdir outputs/results/fov_penetration/macpo/v10_front_sacrifice/

```- [MACPO](https://github.com/chauncygu/Multi-Agent-Constrained-Policy-Optimisation)

- Gu, Shangding, et al. "Multi-Agent Constrained Policy Optimisation." arXiv 2110.02793 (2021).

### 训练收敛判断
- **eval_average_episode_rewards > 500**: 良好 (零动作直飞基准约+515)
- **eval_average_episode_rewards > 1500**: 学会命中HVT
- **eval_average_episode_rewards 持续下降到负数**: 训练坍塌, 需要调参

---

## 5. 测试与评估

### 快速基准测试 (零动作直飞)
```bash
python -c "
import sys; sys.path.insert(0, '.')
from envs.fov_penetration.fov_penetration_env import FOVPenetrationEnv
import numpy as np

hits, total_rews = 0, []
for seed in range(20):
    env = FOVPenetrationEnv({'scenario': 'scenario_1'})
    env.seed(seed * 100)
    obs, share_obs, avail = env.reset()
    ep_rew = np.zeros(4)
    for step in range(1500):
        result = env.step(np.zeros((4, 3)))
        ep_rew += np.array(result[2]).flatten()
        if np.all(result[4]): break
    n_hit = sum(o.hit_hvt for o in env.offensives)
    hits += n_hit
    total_rews.append(np.mean(ep_rew))
    print(f'seed={seed}: hit={n_hit}, rew={np.mean(ep_rew):.1f}')
print(f'Hit rate: {hits}/{20*4}={hits/(20*4)*100:.1f}%, Mean rew: {np.mean(total_rews):.1f}')
"
# 期望: 命中率~12.5%, 平均reward~1160
```

### 多seed统计评估 (训练好的模型)
修改 `scripts/eval_v7_stats.py` 中的 `model_dir` 指向训练好的模型目录，然后:
```bash
python scripts/eval_v7_stats.py
```

### 生成诊断图
修改 `scripts/diagnose_episode.py` 中的 `model_dir`，然后:
```bash
python scripts/diagnose_episode.py
# 输出11张诊断图到 outputs/diagnostics/
```

---

## 6. 环境技术细节

### 状态空间 (obs_dim=84)
每个进攻方agent的观测向量:
```
[9]    自身状态: x,y,z,v,heading,gamma (归一化) + dist_hvt, bearing_hvt, z/z_range
[3]    HVT相对位置: dx,dy,dz (归一化)
[9×4=36] 防御方信息(按距离排序): dx,dy,dz,dv,dheading,dgamma,alive,threat_heading,closing_speed
[9×3=27] 队友信息(按距离排序): dx,dy,dz,v,heading,gamma,alive,mate_dist_hvt,front_rank
[3]    全局: team_min_dist, n_alive_off, n_alive_def
[2]    时间: step_ratio, step_ratio²
[4]    协同感知: my_front_rank, n_threats_on_me, team_min_dist, def_alive_ratio
```
**注意**: 观测中**没有**lock_on_map (敌方分配方案), 通过threat_heading和closing_speed推断威胁。

### 共享状态空间 (share_obs_dim=76)
```
[10×4=40] 所有进攻方: x,y,z,v,heading,gamma,alive,dist_hvt,detected,n_det
[7×4=28]  所有防御方: x,y,z,v,heading,gamma,alive
[3]       全局: team_min_dist, n_alive_off, n_alive_def
[5]       额外: step_ratio, hit_count, def_alive_ratio, mean_off_dist, std_off_dist
```

### 动作空间 (3维连续, Box[-1,1])
```
action[0] → nx (纵向过载, 控制加减速)
action[1] → ny (侧向过载, 控制转弯)
action[2] → nz (法向过载, 控制爬升/俯冲)
```
**关键**: `action[2]=0` 映射到 `nz=1.0` (平飞), **不是nz=0!**
这是在 `dynamics.py` 的 `action_to_overload_3d()` 中通过偏置piecewise映射实现的。

### 动力学模型
3D固定翼运动学, RK4积分, dt=0.1s:
```
dx/dt = v × cos(gamma) × cos(heading)
dy/dt = v × cos(gamma) × sin(heading)
dz/dt = v × sin(gamma)
dv/dt = g × (nx - sin(gamma))
dheading/dt = g × ny / (v × cos(gamma))
dgamma/dt = g × (nz - cos(gamma)) / v
```

### 防御方策略 (非学习, 规则)
- **目标分配**: Hungarian算法, 一次分配不再切换 (`reassign=False`)
- **导引律**: 3D比例导引 (PN), gain=3
- **死盯**: 如果目标死亡, 该拦截器返回None (不切换到其他目标)
- **速度**: 90m/s (进攻方50m/s), 机动性也更强

### 奖励函数 V10 (reward_cost.py)
| 分量 | 系数 | 说明 |
|------|------|------|
| hit_hvt_bonus | 1000 | 命中HVT, **全队**每人+1000 |
| approach_hvt_coef | 500 | 个人接近HVT的距离差 (**主导**) |
| closest_bonus_coef | 200 | 只给队内最近HVT那1架额外奖金 |
| progress_coef | 0.3 | 靠近HVT持续正奖 |
| retreat_penalty | -0.15 | 远离HVT惩罚 |
| mutual_kill_team_bonus | 80 | 同归于尽后存活队友获奖 |
| killed_penalty | -10 | 被杀 (仅触发一次) |
| altitude_penalty_coef | 8 | 低高度/高高度保护 |
| spread_bonus_coef | 0.01 | 分散阵型 |
| step_penalty | -0.01 | 步惩罚 |

---

## 7. 版本迭代历史与已知问题

### 版本历史
| 版本 | 关键改动 | 结果 |
|------|---------|------|
| V3 | 初始环境搭建 | nz映射bug, 飞机撞地 |
| V4-V5 | reward平衡迭代 | 开始学习接近HVT |
| V6 | 修复killed_penalty每步触发bug | reward不再崩塌 |
| V7 | 2M步训练, reward+132 | 首次正reward |
| V8 | 5M步, approach_coef=500 | eval reward+445 |
| V9 | 协同突防, 加lock_on_map | ❌ 失败: 信息泄露+蹭分, 坍塌到-247 |
| **V10** | **去lock_on_map, 态势感知, max_steps=1500, hit_range=500** | **⏳ 待训练** |

### V10的关键修复 (相对V9)
1. ✅ `max_steps`: 500→**1500** (之前500步=50s, 只飞2500m, 到不了6000m外的HVT!)
2. ✅ `hit_hvt_range`: 3→**500m** (dt=0.1s, v=50m/s, 每步飞5m, 3m范围会跳过)
3. ✅ `lr_decay` bug修复 (separated模式下self.trainer是list)
4. ✅ 观测中去掉lock_on_map, 改用态势感知 (threat_heading, closing_speed)
5. ✅ 去掉team_progress (防蹭分) 和 decoy_lure (防远离HVT反而得分)

### 当前已知问题与改进方向
1. **零动作直飞命中率只有12.5%** — 初始heading有噪声, 模型需学习航向修正
2. **V10还未正式跑完完整训练** — 需要10M步完整训练
3. **可能的后续改进** (如果训练效果不好):
   - 训练坍塌 → 减小approach_hvt_coef, 加大hit_hvt_bonus
   - 飞机不分工 → 加入front_rank差异化奖励
   - 撞地严重 → 增大altitude_penalty_coef
   - 学不到命中 → 增大hit_hvt_range或减小初始距离
   - 考虑 curriculum learning

---

## 8. 分析与改进指南 (给AI助手)

### 训练完成后的标准分析流程

#### Step 1: 检查训练曲线
```bash
grep "average rewards:" outputs/v10_train.log | awk -F'average rewards: ' '{print NR, $2}' | awk -F',' '{print $1}'
grep "eval_average" outputs/v10_train.log
```
- 正常: 前期负数 → 逐渐上升 → 稳定在正数
- 异常: 先升后降 = 坍塌; 一直负数 = 学不到

#### Step 2: 跑基准对比
```bash
# 零动作基准 (期望: ~1160)
# 随机动作基准 (期望: ~-200)
# 训练模型 (期望: >1160 才算有效学习)
```

#### Step 3: 诊断单episode行为
关注:
- 飞机是否朝HVT方向飞? (dist_hvt曲线应递减)
- 高度是否稳定? (z应在200-1500m)
- 是否有分工? (前方飞机被杀时间 < 后方)
- 是否命中HVT? (dist_hvt < 500)

#### Step 4: 判断问题并修改
| 症状 | 原因 | 修改文件 | 具体改法 |
|------|------|---------|---------|
| reward持续下降 | 奖励信号弱/矛盾 | reward_cost.py | 增大approach_coef |
| 飞机远离HVT | retreat_penalty不够 | config.py reward | 增大retreat_penalty |
| 全部撞地 | altitude_penalty不够 | config.py reward | 增大altitude_penalty_coef |
| 不命中HVT | hit_hvt_range太小 | config.py | 增大hit_hvt_range |
| 所有飞机相同行为 | 缺乏分工信号 | reward_cost.py | 加front_rank差异化奖励 |
| 3架停飞1架冲 | 蹭分 | reward_cost.py | 确保无team_progress, 检查closest_bonus |

---

## 9. 停止训练 / 清理进程

```bash
# 停止所有训练进程
pkill -9 -f "macpo-fov"

# 确认清理干净
ps aux | grep macpo | grep -v grep

# 如有残留子进程
pkill -9 -f "train_fov_penetration"
```

---

## 10. 第三方库修改清单

以下是对 `third_party/MACPO/` 的修改，如果需要重装MACPO需重新apply:

### mujoco_runner_macpo.py
- **Line ~33**: `lr_decay` fix — separated模式下 `self.trainer` 是list, 需遍历
- **Line ~44**: `step()` 返回值解包 — 改为7个值 (加了costs和avail)
- **Line ~262**: `eval step()` 解包 — 同上改为7个值

### env_wrappers.py
- 适配FOV环境的 `step()` 返回 `(obs, share_obs, rewards, costs, dones, infos, avail)` 7元组

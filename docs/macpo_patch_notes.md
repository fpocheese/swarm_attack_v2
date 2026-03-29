# MACPO 补丁说明

## 目标

将 FOVPenetrationEnv（基于敌方视场角限制的固定翼无人机集群协同突防环境）接入 MACPO 算法训练。

## 修改策略

**零修改原始 MACPO 代码。**

本项目采用"外部适配"策略：
1. 新环境完全独立实现在 `envs/fov_penetration/` 目录
2. 训练脚本独立编写，通过 `sys.path` 引用 MACPO 模块
3. 直接复用 MACPO 的 `mujoco_runner_macpo.py` 作为 Runner
4. 不修改 MACPO 的任何算法代码、配置解析器或包装器

## 接口对齐方式

新环境 `FOVPenetrationEnv` 严格实现了 MACPO 要求的接口：

| 接口 | 实现方式 |
|------|---------|
| `n_agents` | 属性，= 4（1 attacker + 3 escorts）|
| `observation_space` | `list[Box]`，长度 4 |
| `share_observation_space` | `list[Box]`，长度 4 |
| `action_space` | `tuple[Box]`，每个 `Box([-1,-1],[1,1])` |
| `reset()` | 返回 `(obs, share_obs, avail_actions)` |
| `step(actions)` | 返回 `(obs, share_obs, rewards, dones, infos, avail_actions)` |
| `infos[i]["cost"]` | `[[cost_val]] * n_agents`，符合 ShareSubprocVecEnv 提取格式 |
| `seed(seed)` | 设置随机种子 |
| `close()` | 清理 |

## 原始 MACPO 文件列表（未修改）

- `macpo/config.py` — 使用 `parse_known_args` 天然支持额外参数
- `macpo/envs/env_wrappers.py` — `ShareDummyVecEnv` / `ShareSubprocVecEnv` 通用包装器
- `macpo/runner/separated/base_runner_macpo.py` — Runner 基类
- `macpo/runner/separated/mujoco_runner_macpo.py` — 具体 Runner
- `macpo/algorithms/` — 算法代码
- `macpo/utils/separated_buffer.py` — Replay Buffer

## 潜在兼容性注意事项

1. MACPO 原代码中 `eval()` 方法有一个 bug：引用了未定义的 `eval_costs`，实际评估时可能报错。本项目的独立评估脚本 (`eval_fov_penetration_macpo.py`) 绕过了这个问题。

2. `ShareSubprocVecEnv` 提取 cost 的方式是 `cost_x = np.array([item[0]['cost'] for item in infos])`，即取每个环境的第 0 个 agent 的 info 中的 `"cost"` 字段。我们的环境确保所有 agent 的 info 中 cost 格式一致。

# MAPPO V38 交接 README（给后续编程 AI）

更新时间：2026-04-17  
适用代码库：`muti_uav_attack_v16`

---

## 1. 这个项目在做什么（先看这个）

这是一个**多无人机协同突防**的多智能体强化学习项目：

- 进攻方（RL 控制）：4 架固定翼无人机（只训练这 4 个 agent）
- 防御方（规则策略）：4 架拦截器（3D PN 导引 + 锁定状态机）
- 目标：至少 1 架进攻方突破并命中 HVT（高价值地面目标）

技术上是：

- 环境：自定义 `FOVPenetrationEnv`
- 算法：MAPPO（使用 `third_party/MACPO` 框架中的 MAPPO runner）
- 训练脚本：`scripts/train_fov_penetration_mappo.py`

---

## 2. V38 当前核心变化（相对旧版本）

V38 的本质是把动力学和拦截逻辑从旧表示切换到更物理一致的表示，并修正命中判定尺度。

### 2.1 动力学控制量改造（V5）

从旧控制：

- `(ax, ay, mu)`

改为新控制：

- `(ax, an_pitch, an_yaw)`

含义：

- `ax`：轴向加速度
- `an_pitch`：俯仰平面法向加速度
- `an_yaw`：偏航平面法向加速度

关键语义：`action=[0,0,0]` 对应平飞 trim（近似不掉高、不转向）。

主要落地文件：

- `envs/fov_penetration/dynamics.py`
- `envs/fov_penetration/entities.py`
- `envs/fov_penetration/policies_interceptor.py`
- `envs/fov_penetration/analytic_priors/*.py`

### 2.2 锁定机制：排他锁定

V38 增加“已锁定目标集合”，避免多个拦截器重复锁定同一个进攻方，逻辑在环境 step 中维护并传给拦截器策略。

主要文件：

- `envs/fov_penetration/fov_penetration_env.py`
- `envs/fov_penetration/policies_interceptor.py`

### 2.3 HVT 命中阈值收紧

- `point_target.hit_threshold` 从 `50m` 收紧到 `5m`
- 与碰撞杀伤尺度对齐，命中定义更严格

主要文件：

- `envs/fov_penetration/config.py`

---

## 3. 代码结构（只列接手最常用入口）

- 环境主类：`envs/fov_penetration/fov_penetration_env.py`
- 配置中心：`envs/fov_penetration/config.py`
- 奖励/成本：`envs/fov_penetration/reward_cost.py`
- 进攻动力学：`envs/fov_penetration/dynamics.py`
- 防御策略（PN+锁定）：`envs/fov_penetration/policies_interceptor.py`
- 训练入口（MAPPO）：`scripts/train_fov_penetration_mappo.py`
- V38 启动脚本：`scripts/run_v38_train.sh`
- 快速冒烟：`scripts/smoke_v38.py`
- PN 对 PN 验证：`scripts/test_pn_vs_pn.py`
- 快速评估导 GIF：`scripts/eval_fast_to_gifs.py`
- 10 回合评估：`eval_v28_10episodes.py`

---

## 4. 环境接口约定（非常重要）

`env.step(actions)` 返回 7 元组：

1. `obs`
2. `share_obs`
3. `rewards`
4. `costs`
5. `dones`
6. `infos`
7. `avail_actions`

说明：

- 当前 MAPPO 训练路径里 `costs` 基本不参与优化（可视为兼容位）
- 训练 runner 侧已经按这个 7 元组适配

相关适配文件：

- `third_party/MACPO/MACPO/macpo/runner/separated/mujoco_runner.py`

---

## 5. 训练与评估最短路径

## 5.1 训练（V38）

推荐直接用：

```bash
bash scripts/run_v38_train.sh
```

它会调用：

```bash
python -u scripts/train_fov_penetration_mappo.py --algorithm_name mappo ...
```

关键训练参数（脚本内）：

- `experiment_name=v38_inertial_accel`
- `ap_config=v28`
- `episode_length=8000`
- `num_env_steps=200000000`
- `hidden_size=256`, `layer_N=3`
- `use_recurrent_policy`

日志：

- `outputs/v38_inertial_accel.log`

模型默认输出：

- `outputs/results/fov_penetration/mappo/v38_inertial_accel/<run>/models/`

## 5.2 冒烟检查

```bash
python scripts/smoke_v38.py
```

目的：检查环境可 reset/step、动作维度和阈值配置。

## 5.3 评估（GIF+曲线）

```bash
python scripts/eval_fast_to_gifs.py \
  --model_dir outputs/results/fov_penetration/mappo/v38_inertial_accel/run3/models \
  --out_dir outputs/gifs/v38_run3_eval_fast
```

该脚本会输出：

- episode GIF
- telemetry CSV
- telemetry 图
- 事件 JSON（死亡、距离等）

---

## 6. Analytic Priors（AP）开关说明

训练脚本 `train_fov_penetration_mappo.py` 支持：

- `--ap_config none`
- `--ap_config v22`
- `--ap_config v22_full`
- `--ap_config v28`（V38 默认）

映射逻辑在 `_get_ap_override()`，会把对应模块写入 `analytic_priors` 配置。

接手时如果想做消融实验，优先改这个函数，不要分散改多个文件。

---

## 7. 目前你最可能遇到的问题

1. **评估脚本网络超参不匹配**  
   训练和评估的 `hidden_size/layer_N` 必须一致，否则加载 actor 可能异常或表现异常。

2. **路径不一致导致找不到模型**  
   `model_dir` 要指向具体 `models` 目录，里面应有 `actor_agent0.pt` 等文件。

3. **运行环境未装好第三方框架**  
   需要确保 `third_party/MACPO/MACPO` 已 `pip install -e .`，且 `PYTHONPATH`/脚本内 `sys.path` 生效。

4. **把 V38 当成 MACPO 约束训练**  
   当前是 MAPPO 主路径，约束成本不是主要优化目标。

---

## 8. 给后续 AI 的建议工作流

当你要改逻辑时，建议按这个顺序：

1. 先跑 `scripts/smoke_v38.py`，确认环境基线 OK
2. 小改动后先跑 `scripts/test_pn_vs_pn.py`（看动力学/锁定是否破坏）
3. 再跑短评估（`eval_fast_to_gifs.py --n_episodes 1`）看轨迹
4. 最后再开长训练

这样能避免“训练了很久才发现物理或接口崩了”。

---

## 9. 一句话结论

V38 是一个已经切换到**惯性系加速度控制 + 排他锁定 + 5m 严格命中阈值**的 MAPPO 版本；接手开发时请围绕 `train_fov_penetration_mappo.py + fov_penetration_env.py + dynamics.py + policies_interceptor.py` 这四个核心文件推进。

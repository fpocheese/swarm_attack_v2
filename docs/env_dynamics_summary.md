# 突防论文代码的环境与飞行器动力学说明

> 适用版本：`swarm_attack_v2`
> 主要源文件：
> 
> - 配置：[envs/fov_penetration/config.py](../envs/fov_penetration/config.py)
> - 动力学：[envs/fov_penetration/dynamics.py](../envs/fov_penetration/dynamics.py)
> - 环境主体：[envs/fov_penetration/fov_penetration_env.py](../envs/fov_penetration/fov_penetration_env.py)

---

## 1. 任务场景概览

三维同构集群协同**突防 / 打击**任务：

- **进攻方 (offensive)**：`n_offensive = 4` 架固定翼无人机，目标是穿越拦截区抵达高价值目标 HVT。
- **防御方 (defensive)**：`n_defensive = 4` 架拦截器，使用 PN（比例导引）+ Hungarian 分配 + FOV 触发锁定 的脚本策略，对进攻方进行拦截。
- **HVT**：固定地面点目标，位置 `[1200, 0, 0]` m。

| 项目                            | 值                                            | 说明            |
| ----------------------------- | -------------------------------------------- | ------------- |
| 战场尺寸 `map_size`               | 2000 m                                       | XY 平面正方形      |
| 高度范围 `[z_min, z_max]`         | `[0, 1000]` m                                | 允许打地面目标       |
| 仿真步长 `dt`                     | 0.01 s                                       | 每步约 0.45 m 位移 |
| 最大步数 `max_steps`              | 8000                                         | 总时长 80 s      |
| 进攻方初始位置                       | 中心 `(-1200, 0, 300)`，xy 散布 ±150 m，z 散布 ±30 m | 朝向 HVT        |
| 防御方初始位置                       | 中心 `(600, 0, 350)`，xy 散布 ±200 m，z 散布 ±50 m   | 朝向进攻方         |
| HVT 命中阈值 `hit_hvt_range`      | 5 m                                          | 点目标命中         |
| 碰撞双杀阈值 `collision_kill_range` | 5 m                                          | 拦截器/进攻方 CPA   |
| FOV 锁定击杀 `kill_range`         | 3 m                                          | 拦截器锁定 + 距离阈值  |
| 拦截器 FOV 半角                    | 30°                                          |               |
| 探测距离 `detection_range`        | 2500 m                                       | 覆盖整个战场        |

**对称性说明**：进攻命中 HVT 与防御拦截进攻方均为 5 m CPA 阈值，攻防对称。

---

## 2. 飞行器动力学方程（V5：惯性系加速度控制）

两类飞行器使用**完全相同形式**的三维质点动力学，仅参数（速度/加速度上限）不同。

### 2.1 状态量

$$
\mathbf{s} = [\,x,\; y,\; z,\; v,\; \psi,\; \gamma\,]
$$

- $x, y, z$：惯性系位置 (m)
- $v$：空速 (m/s)
- $\psi$：偏航角 / heading (rad)
- $\gamma$：俯仰 / 航迹倾角 (rad)

### 2.2 控制量

$$
\mathbf{u} = [\,a_x,\; a_{n,\text{pitch}},\; a_{n,\text{yaw}}\,]
$$

- $a_x$：**轴向加速度**（沿速度方向，m/s²）
- $a_{n,\text{pitch}}$：**俯仰平面法向加速度**（含重力补偿，正值=抬头，m/s²）
- $a_{n,\text{yaw}}$：**偏航平面法向加速度**（正值=左转，m/s²）

### 2.3 运动方程

$$
\begin{aligned}
\dot{x} &= v \cos\gamma \cos\psi \\
\dot{y} &= v \cos\gamma \sin\psi \\
\dot{z} &= v \sin\gamma \\
\dot{v} &= a_x - g \sin\gamma \\
\dot{\psi} &= \dfrac{a_{n,\text{yaw}}}{v \cos\gamma} \\
\dot{\gamma} &= \dfrac{a_{n,\text{pitch}} - g \cos\gamma}{v}
\end{aligned}
$$

其中 $g = 9.81\,\text{m/s}^2$。积分采用四阶 **Runge-Kutta (RK4)**。

### 2.4 平飞 (trim) 条件

$$
a_x = 0,\quad a_{n,\text{pitch}} = g,\quad a_{n,\text{yaw}} = 0 \;\;\Longrightarrow\;\; \dot v = 0,\;\dot\psi = 0,\;\dot\gamma = 0
$$

动作映射特别设计：`action = [0, 0, 0]` 自动对应平飞（俯仰通道做了 +g 偏置）。

### 2.5 动作空间

每个 agent 的动作 $\mathbf{a} \in [-1, 1]^3$（连续 Box）：

| 通道                            | 映射方式            | `a=0` | `a=+1`                    | `a=-1`                     |
| ----------------------------- | --------------- | ----- | ------------------------- | -------------------------- |
| `a[0]` → $a_x$                | 偏置线性            | $0$   | $a_{x,\max}$              | $a_{x,\min}$               |
| `a[1]` → $a_{n,\text{pitch}}$ | 偏置线性 (中心 = $g$) | $g$   | $a_{n,\text{pitch},\max}$ | $-a_{n,\text{pitch},\max}$ |
| `a[2]` → $a_{n,\text{yaw}}$   | 对称线性            | $0$   | $a_{n,\text{yaw},\max}$   | $-a_{n,\text{yaw},\max}$   |

---

## 3. 加速度与运动学约束（核心）

### 3.1 进攻方 (Offensive)

| 参数                                       | 值                         | 说明              |
| ---------------------------------------- | ------------------------- | --------------- |
| 速度范围 $[v_{\min}, v_{\max}]$              | `[40, 50]` m/s ｜ 标称 45    | 较慢              |
| 轴向加速度 $a_x \in [a_{x,\min}, a_{x,\max}]$ | **`[-5, +20]` m/s²**      | 减速 ≈0.5g，加速 ≈2g |
| 俯仰法向加速度 $\|a_{n,\text{pitch}}\| \le$     | **`2.5 g ≈ 24.525` m/s²** | $n_{\max}=2.5$  |
| 偏航法向加速度 $\|a_{n,\text{yaw}}\| \le$       | **`2.5 g ≈ 24.525` m/s²** | $n_{\max}=2.5$  |
| 轴向加加速度 `dax_max`                         | 60 m/s³                   | 控制变化率限制         |
| 俯仰加加速度 `dan_pitch_max`                   | 120 m/s³                  |                 |
| 偏航加加速度 `dan_yaw_max`                     | 120 m/s³                  |                 |
| 航迹倾角范围 $\gamma$                          | $[-15°, +15°]$            | 较保守             |
| `action_scale`                           | 1.0                       | 全机动能力           |



### 3.2 防御方 / 拦截器 (Defensive)

| 参数                                       | 值                        | 说明                         |
| ---------------------------------------- | ------------------------ | -------------------------- |
| 速度范围 $[v_{\min}, v_{\max}]$              | `[50, 60]` m/s ｜ 标称 55   | 比进攻方快约 20%                 |
| 轴向加速度 $a_x \in [a_{x,\min}, a_{x,\max}]$ | **`[-10, +30]` m/s²**    | 减速 ≈1g，加速 ≈3g              |
| 俯仰法向加速度 $\|a_{n,\text{pitch}}\| \le$     | **`5.0 g ≈ 49.05` m/s²** | $n_{\max}=5.0$，**约为进攻方两倍** |
| 偏航法向加速度 $\|a_{n,\text{yaw}}\| \le$       | **`5.0 g ≈ 49.05` m/s²** | $n_{\max}=5.0$             |
| 轴向加加速度 `dax_max`                         | 80 m/s³                  |                            |
| 俯仰加加速度 `dan_pitch_max`                   | 150 m/s³                 |                            |
| 偏航加加速度 `dan_yaw_max`                     | 150 m/s³                 |                            |
| 航迹倾角范围 $\gamma$                          | $[-45°, +45°]$           | 比进攻方激进 3 倍                 |



### 3.3 攻防机动能力对比

| 维度                | 进攻方     | 防御方     | 比值（防/攻）   |
| ----------------- | ------- | ------- | --------- |
| 最大速度              | 50 m/s  | 60 m/s  | 1.20×     |
| 法向过载 $n_{\max}$   | 2.5 g   | 5.0 g   | **2.00×** |
| 最大轴向加速度           | 20 m/s² | 30 m/s² | 1.50×     |
| 俯仰自由度 $\gamma$ 上限 | ±15°    | ±45°    | 3.00×     |

**核心博弈含义**：拦截器在速度、过载、机动包线上**全面优于**进攻方，进攻方必须依靠**协同（诱饵/分散注意力）+ 突防时机**才能命中 HVT，单机正面突防不可行。

### 

---

## 4. 观测 / 动作空间汇总

| 接口                           | 形状                                 | 说明                                                |
| ---------------------------- | ---------------------------------- | ------------------------------------------------- |
| `action_space[i]`            | `Box(-1, 1, (3,))`                 | $[a_x, a_{n,\text{pitch}}, a_{n,\text{yaw}}]$ 归一化 |
| `observation_space[i]`       | `Box(-inf, inf, (obs_dim,))`       | 个体观测                                              |
| `share_observation_space[i]` | `Box(-inf, inf, (share_obs_dim,))` | 中心化 critic 观测                                     |

`obs_dim` / `share_obs_dim` 由 `_compute_space_dims()` 动态计算（含 analytic priors / HVT guidance / decoy game 等可选拼接）。

---

## 

# FOV Penetration 当前版本审阅（动作 / 观测 / 奖励）

本文基于当前代码状态梳理：
- 进攻方动作是否来自 RL 输出
- 动作如何映射到动力学控制量
- 观测空间每一维含义
- 奖励/成本各项构成与数值量级
- 对“进攻机回头跑掉、向地面掉”的结构性原因分析

---

## 1. 结论先行

1. **进攻方控制确实由 RL 输出动作驱动**，不是 PN 接管。  
   环境每步直接执行 `actions[i] -> off.step_with_action(...)`。

2. 你看到“动作下发很大，但实际加速度很小”是因为有**控制变化率约束**（rate limit），不是动作被覆盖。

3. 当前奖励结构里，**“回头行为”只受到很弱惩罚**（`retreat_penalty=-0.15`），但存在多项与“生存/牵制”相关的正奖励，容易出现“回头牵制、保命、不打点”的策略局部最优。

4. 当前高度相关设计并不会强烈阻止低空飞行（`z_min=0` 时低空惩罚项基本失效），所以“向地面贴飞”并不奇怪。

---

## 2. 动作空间（Action）

### 2.1 RL 动作定义

- 每个进攻体动作维度：`3`
- 动作范围：`[-1, 1]^3`
- 含义：
  - `a[0]`：轴向加速度通道（`ax`）
  - `a[1]`：法向加速度幅值通道（`ay`）
  - `a[2]`：法向方向角通道（`mu`）

### 2.2 动作到控制量映射（action_to_control_3d）

- `ax`：分段线性映射，`a[0]=0 -> ax=0`
- `ay`：`ay = (a[1] + 1) * 0.5 * ay_max`
- `mu`：`mu = a[2] * pi`

当前参数（offensive）：
- `ax in [-5, 20] m/s^2`
- `ay in [0, 2.5g] = [0, 24.53] m/s^2`
- `mu in [-pi, pi]`

### 2.3 为什么实测 actual 比 cmd 小

动力学里存在变化率约束（`dt=0.01`）：
- `dax_max=60 -> 每步最多变化 0.6 m/s^2`
- `day_max=120 -> 每步最多变化 1.2 m/s^2`
- `dmu_max=30 -> 每步最多变化 0.3 rad`

所以首步即使命令 `ay_cmd=24.5`，实际 `ay` 也只会先到 `1.2`，这是正常机制。

### 2.4 进攻方动作来源链路

环境步进逻辑：
1. `action = np.array(actions[i])`
2. `off.step_with_action(action, dt)`
3. `step_with_action -> action_to_control_3d -> step_dynamics_3d`

防守方才使用 `InterceptorPolicy.get_action(...)`。

---

## 3. 观测空间（Observation）

### 3.1 当前维度

当前 `scenario_1`（4v4）且 analytic obs 打开时：
- `obs_dim = 108`
- `share_obs_dim = 77`

推导：
- base: `9 + 5 + 11*K + 10*(n_off-1) + 3 + 2 + 5`
- 对 4v4，`K=min(4,4)=4`：`9+5+44+30+3+2+5 = 98`
- analytic obs: `4 + 6 = 10`
- 总计：`98 + 10 = 108`

### 3.2 单智能体局部观测 108 维分块

1. **自身状态 9 维**  
   `x,y,z,v,heading,gamma,ax,ay,mu`（归一化）

2. **HVT 相对信息 5 维**  
   `rel_x, rel_y, rel_z, los_rate_az, los_rate_el`

3. **最近 K=4 个防守体，每个11维，共44维**  
   相对位置/速度/姿态、生存标志、威胁朝向、闭合速度、
   `is_chasing_me`、`is_locked`

4. **队友信息 30维（3个队友 × 10）**  
   相对状态 + 队友到 HVT 距离 + 前后排序 + 队友被追击强度

5. **暴露状态 3 维**  
   `detected`、`detected_by_count`、`continuous_exposure`

6. **全局进度 2 维**  
   自身到 HVT 归一化距离、时间进度 `current_step/max_steps`

7. **协同态势 5 维**  
   前后排位、威胁数量、队伍最小距 HVT、防守存活比、自身被追击强度

8. **解析先验附加 10 维**  
   `Z_tilde, psi_agg, M_tilde_norm, Xi_max` +
   `rho, closing, omega, omega_dot, pn_hint, penetration_score`

### 3.3 全局共享观测 77 维

- 所有进攻体：`10 * n_off = 40`
- 所有防守体：`7 * n_def = 28`
- HVT 位置：`3`
- 全局统计：`5`
- penetration share extra：`1`

合计 `40+28+3+5+1 = 77`。

---

## 4. 奖励结构（Reward）

基础奖励由 `compute_rewards()` 给出，之后在 `env.step()` 中还会叠加 analytic priors 模块奖励。

## 4.1 基础奖励项（reward_cost.py）

记 `r_i` 为第 i 架进攻体每步奖励：

1. **命中 HVT 全队大奖**  
   `+ hit_hvt_bonus * n_hits`（当前 `6000`）

2. **个人接近奖励（主项）**  
   `approach_hvt_coef * (prev_dist - cur_dist) / init_dist`，并带近距放大

3. **持续进度奖励**  
   `+ progress_coef * (1 - dist/init_dist)`

4. **持续 proximity 奖励**  
   `+ proximity_coef * (1 - dist/init_dist)^3`

5. **里程碑奖励**（首次突破 1500/1000/500/200m）

6. **最近突防者额外奖励**（closest bonus）

7. **后退惩罚**  
   若本步 `dist` 变大：`+ retreat_penalty`（当前 `-0.15`）

8. **被击杀惩罚**（当前 `-3`）

9. **同归于尽/诱饵牺牲奖励**（有攻防互杀时）

10. **持续诱饵奖励（V25）**  
    被拦截器追击会给队友分发正奖励；诱饵自身也有奖励

11. **被探测惩罚**（当前 `-0.005`）

12. **高度与俯仰相关项**
    - 低空惩罚（依赖 `z_min_safe`）
    - 高空惩罚
    - 俯冲/爬升惩罚
    - 安全高度小奖励

13. **步惩罚 + 动作平滑**
    - `step_penalty=-0.005`
    - `smooth_action_coef * (ay/g)`（当前系数 `-0.002`）

14. **队形分散奖励**

### 4.2 Analytic Priors 叠加项（env.step 内）

在基础奖励后还会叠加：
- decoy game reward
- effective penetration reward
- HVT guidance shaping + attack gate reward
- （以及 cone cost 加入 `cost`）

这部分会显著影响策略偏好，不是“纯基础 reward”。

### 4.3 timeout 终止附加

若超时：
- 全体加 `timeout_penalty`
- 存活且未命中者按距离再扣分
- 存活再扣 `timeout_alive_penalty`

---

## 5. 成本结构（Cost）

每步成本 `c_i` 主要包括：
- 被任一防守体 FOV 覆盖：`fov_exposure`
- 接近防守体 300m 内危险区成本
- 越界成本
- 近地面风险成本
- 友机碰撞成本
- 另有 cone cost（analytic priors）叠加

---

## 6. 为什么会出现“回头跑掉 + 贴地”

## 6.1 回头跑掉：结构上惩罚弱、替代回报强

- 对“回头”只有**间接惩罚**：`retreat_penalty=-0.15`（每步很小）
- 同时存在多类“生存/牵制”正反馈（尤其 decoy / attack-gate / 部分协同项）
- 在未学会稳定命中前，策略可能停在局部最优：
  - 不冒险强冲点
  - 通过机动和牵制获得中等回报
  - 避免死亡与超时大罚

## 6.2 贴地趋势：低空惩罚在当前配置下基本不强

- 当前 `z_min=0`
- 低空惩罚项使用 `z_min_safe = z_min*2 = 0`，实际几乎不会触发
- `off.z < 0` 才会被地面判死
- 所以“低空飞行”不会受到强烈负反馈，策略会把它当作可行规避行为

> 这也解释了你担心“不能简单加低高度惩罚”的点：目标在地面，过强低空惩罚会与终端打击目标冲突。

---

## 7. 合理性审阅（针对你当前目标）

你的目标是“进攻方不要无意义回头，要稳定朝地面 HVT 突防”。

从结构上看，当前版本存在以下不匹配：

1. **目标导向不够“单调”**：后退只轻罚，导致回头可被其他正奖励抵消。  
2. **协同牵制奖励偏强**：在某些阶段会鼓励“牵制价值”而非“命中价值”。  
3. **低空行为边界过宽**：没有细化“低空接近目标”与“低空逃逸跑偏”的区别。  

---

## 8. 你现在最该改哪一类项（不直接改代码，仅建议）

若你只想先解决“回头跑掉”，优先级建议：

1. **加强后退惩罚（或做累计后退惩罚）**，而非简单一步 `-0.15`。  
2. **减弱 decoy/attack-gate 在中后期的权重**，避免压过“接近HVT”主目标。  
3. 给“持续朝向 HVT 的航向一致性”增加奖励/约束（比直接低高度惩罚更贴合你的诉求）。

---

## 9. 本文对应代码位置

- 动作映射与动力学：
  - `envs/fov_penetration/dynamics.py`
  - `envs/fov_penetration/entities.py`
- 步进与动作执行：
  - `envs/fov_penetration/fov_penetration_env.py`
- 观测构造：
  - `envs/fov_penetration/fov_penetration_env.py` (`_get_obs`, `_get_share_obs`)
- 基础奖励/成本：
  - `envs/fov_penetration/reward_cost.py`
- 超参与权重：
  - `envs/fov_penetration/config.py`

---

（生成时间：2026-04-02）

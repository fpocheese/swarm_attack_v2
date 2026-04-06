# MAPPO 版本观测空间与奖励函数最终落地规格（精简但机制完整）

> 目标：把当前版本改成一个**更博弈、更机制化、更适合 MAPPO** 的版本。  
> 原则：
> 1. 观测不要继续堆很多对象原始信息  
> 2. 观测以“关键威胁 + 群体博弈状态 + 终端几何”组织  
> 3. 原 MACPO 的 cost 全部并入 MAPPO reward 惩罚项  
> 4. 奖励主线必须清晰：  
>    - **安全突防**
>    - **主动诱饵 / 注意力重定向**
>    - **局部近距逃逸**
>    - **有效突防数量最大化**
>    - **点目标命中（脱靶量 ≤ 3 m）**
>
> 当前旧版存在：观测 108 维、HVT 与敌方对象块较重、奖励项碎且有 decoy / attack-gate / shaping 叠加，容易形成“规避/牵制但不打点”的局部最优。:contentReference[oaicite:0]{index=0}

---

# 1. 记号统一

## 1.1 集合与索引
- 进攻集群：
  \[
  \mathcal A=\{1,\dots,N_A\}
  \]
- 拦截集群：
  \[
  \mathcal I=\{1,\dots,N_I\}
  \]
- 当前讨论的进攻智能体索引：\(i\)
- 拦截器索引：\(j\)

---

## 1.2 基本状态量

### 进攻飞行器 \(i\)
- 位置：
  \[
  p_i=[x_i,y_i,z_i]^\top
  \]
- 速度向量：
  \[
  v_i
  \]
- 速度标量：
  \[
  V_i=\|v_i\|
  \]
- 航向角 / 俯仰角：
  \[
  \psi_i,\ \gamma_i
  \]
- 实际轴向/法向加速度：
  \[
  a_{x,i},\ a_{y,i}
  \]

### 拦截器 \(j\)
- 位置：
  \[
  p_j=[x_j,y_j,z_j]^\top
  \]
- 速度向量：
  \[
  v_j
  \]
- 雷达锥轴单位向量：
  \[
  b_j
  \]
- 雷达半锥角：
  \[
  \alpha_j
  \]

### 高价值目标 HVT
- 固定点目标坐标：
  \[
  p_H
  \]

---

## 1.3 相对量

### 相对于拦截器
\[
r_{ij}=p_i-p_j
\]
\[
\rho_{ij}=\|r_{ij}\|
\]
\[
v_{ij}=v_i-v_j
\]

### 相对于 HVT
\[
r_{iH}=p_H-p_i
\]
\[
\rho_{iH}=\|r_{iH}\|
\]

### 对 HVT 的闭合速度
\[
V_{c,iH}
=
-\frac{r_{iH}^\top v_i}{\|r_{iH}\|}
\]

### 对拦截器的闭合速度
\[
V_{c,ij}
=
-\frac{r_{ij}^\top (v_i-v_j)}{\|r_{ij}\|}
\]

---

# 2. 最终局部观测空间（单个进攻智能体）

> 设计原则：  
> - 只对“最危险的 2 个拦截器”给详细量  
> - 其余敌方给摘要  
> - HVT 只给关键 LOS 几何  
> - 队友只给博弈角色摘要  
> - 保留少量解析先验汇总变量

---

## 2.1 观测总结构

单机局部观测由 5 组组成：

1. `Self State`：自身机动状态  
2. `HVT Guidance State`：对点目标命中的终端几何  
3. `Top-2 Threat Interceptors`：两个最危险拦截器的详细威胁状态  
4. `Team / Game Summary`：群体博弈与角色摘要  
5. `Analytic Prior Summary`：解析先验汇总量

---

## 2.2 Self State（自身状态）

建议变量：

### 1. `self_rel_goal_x`
**含义**：自身相对 HVT 的 x 方向相对位置  
**定义**：
\[
(self\_rel\_goal\_x)=x_H-x_i
\]

### 2. `self_rel_goal_y`
**含义**：自身相对 HVT 的 y 方向相对位置  
**定义**：
\[
(self\_rel\_goal\_y)=y_H-y_i
\]

### 3. `self_rel_goal_z`
**含义**：自身相对 HVT 的 z 方向相对位置（二维环境可删除）  
**定义**：
\[
(self\_rel\_goal\_z)=z_H-z_i
\]

### 4. `self_speed`
**含义**：自身速度大小  
**定义**：
\[
self\_speed=V_i=\|v_i\|
\]

### 5. `self_heading`
**含义**：自身航向角  
**定义**：
\[
self\_heading=\psi_i
\]

### 6. `self_gamma`
**含义**：自身俯仰角（二维可删除）  
**定义**：
\[
self\_gamma=\gamma_i
\]

### 7. `self_ax`
**含义**：自身实际轴向加速度  
**定义**：
\[
self\_ax=a_{x,i}
\]

### 8. `self_ay`
**含义**：自身实际法向加速度幅值  
**定义**：
\[
self\_ay=a_{y,i}
\]

### 9. `is_locked`
**含义**：当前是否被至少一个拦截器锁定  
**定义**：
\[
is\_locked=
\mathbf 1\{locked\_by\_count_i > 0\}
\]

### 10. `locked_by_count`
**含义**：当前有多少个拦截器锁定自己  
**定义**：
\[
locked\_by\_count_i
=
|\{j\in\mathcal I:\ current\_locked\_target\_idx[j]=i\}|
\]

---

## 2.3 HVT Guidance State（终端任务几何）

> 这里不要再堆很多 HVT 原始信息。  
> 只保留最关键的“点目标命中几何”。

### 1. `rho_to_hvt`
**含义**：自身到 HVT 点目标的距离  
**定义**：
\[
rho\_to\_hvt=\rho_{iH}=\|p_H-p_i\|
\]

### 2. `closing_speed_to_hvt`
**含义**：自身对 HVT 的闭合速度  
**定义**：
\[
closing\_speed\_to\_hvt
=
V_{c,iH}
=
-\frac{r_{iH}^\top v_i}{\|r_{iH}\|}
\]

### 3. `omega_hvt_los`
**含义**：相对于 HVT 的 LOS 角速度  
**定义**：
\[
\omega_{iH}^{LOS}
=
\frac{r_{iH}\times v_i}{\|r_{iH}\|^2}
\]
二维环境可退化为标量角速度。

### 4. `omega_hvt_los_dot`
**含义**：相对于 HVT 的 LOS 角速度变化率  
**定义**：
\[
\dot\omega_{iH}^{LOS}
\approx
\frac{\omega_{iH}^{LOS}(t)-\omega_{iH}^{LOS}(t-\Delta t)}{\Delta t}
\]

### 5. `pn_hint_to_hvt`（可选）
**含义**：基于比例导引思想构造的启发量  
**定义**：
\[
u_{iH}^{PN}=N_H\,V_{c,iH}\,\omega_{iH}^{LOS}
\]
其中 \(N_H\) 为经验比例导引增益。  
**用途**：辅助策略学会终端收敛，不直接接管控制。

---

## 2.4 Top-2 Threat Interceptors（两个最危险拦截器）

> 不再给 4 个拦截器都铺满一堆字段。  
> 只对最危险的 2 个给详细量。

### 2.4.1 如何选“最危险”
对每个拦截器计算局部威胁分数，例如：
\[
Threat_{ij}
=
\lambda_\rho \frac{1}{\rho_{ij}+\epsilon}
+
\lambda_q [q_{ij}]_+
+
\lambda_L \mathbf 1\{j\text{锁定}i\}
\]
取最大的两个拦截器作为 `top1_threat` 和 `top2_threat`。

---

### 对每个 Threat \(j^\star\)，给以下变量：

#### 1. `rho_ij`
**含义**：与该拦截器的距离  
\[
rho_{ij}=\|p_i-p_j\|
\]

#### 2. `closing_speed_ij`
**含义**：与该拦截器的闭合速度  
\[
V_{c,ij}
=
-\frac{r_{ij}^\top(v_i-v_j)}{\|r_{ij}\|}
\]

#### 3. `bearing_error_ij`
**含义**：该拦截器锥轴与自身方向的夹角误差  
**定义**：
\[
\theta_{ij}
=
\arccos\left(\frac{b_j^\top r_{ij}}{\|r_{ij}\|}\right)
\]
可以用
\[
bearing\_error_{ij}=\theta_{ij}-\alpha_j
\]
也可以直接用 \(q_{ij}\)。

#### 4. `in_fov_ij`
**含义**：是否在该拦截器视场内  
**定义**：
\[
in\_fov_{ij}
=
\mathbf 1\{q_{ij}>0\}
\]
其中
\[
q_{ij}
=
\frac{b_j^\top r_{ij}}{\|r_{ij}\|}-\cos\alpha_j
\]

#### 5. `is_locking_me`
**含义**：该拦截器是否当前锁定自己  
**定义**：
\[
is\_locking\_me
=
\mathbf 1\{current\_locked\_target\_idx[j]=i\}
\]

#### 6. `omega_los_ij`
**含义**：相对于该拦截器的 LOS 角速度  
**定义**：
\[
\omega_{ij}^{LOS}
=
\frac{r_{ij}\times(v_i-v_j)}{\|r_{ij}\|^2}
\]

#### 7. `Gamma_ij`
**含义**：近距突然机动后，LOS 角速度超出敌方跟踪上限的裕度  
**定义**：
\[
\Gamma_{ij}
=
\|\omega_{ij}^{LOS,+}\|
-\omega_{j,\max}^{trk}
\]
其中
\[
\|\omega_{ij}^{LOS,+}\|
\approx
\frac{\|v_{t,ij}+a_{t,ij}^{cmd}\Delta t\|}{\rho_{ij}}
\]

#### 8. `Xi_ij`
**含义**：近距逃逸触发量  
**定义**：
\[
\Xi_{ij}
=
G_{ij}^{near}\,[\Gamma_{ij}]_+
\]
其中
\[
G_{ij}^{near}
=
\sigma(\kappa_\rho(\rho_0-\rho_{ij}))
\]

#### 9. `local_Z_ij`
**含义**：相对于该拦截器的局部 zero-effort cone-margin  
**定义**：
\[
Z_{ij}
=
Y_{1,ij}(t_{go})(q_{ij}+t_{go}\dot q_{ij})
+
Y_{3,ij}(t_{go})a_{I,j}
+
Y_{4,ij}(t_{go})a_{A,i}
\]

---

## 2.5 Team / Game Summary（群体博弈摘要）

> 这一块很重要。  
> 它比堆很多队友原始状态更“博弈”。

### 1. `my_decoy_value`
**含义**：自己当前承担主动诱饵角色的局部价值  
**定义**：
\[
U_i^{decoy}
=
B_i^{attn}+B_i^{H}-C_i^{self}
\]
其中：
- \(B_i^{attn}\)：吸走敌方注意力后给队友带来的收益
- \(B_i^H\)：对队友突破 HVT 的收益
- \(C_i^{self}\)：自己暴露和被击杀的代价

### 2. `my_decoy_prob`
**含义**：自己当前成为诱饵角色的概率 / 倾向  
**定义**：
\[
\pi_i^{D}
=
\frac{\exp(\eta U_i^D)}
{\exp(\eta U_i^D)+\exp(\eta U_i^P)+\exp(\eta U_i^S)}
\]

### 3. `my_penetrator_prob`
**含义**：自己当前成为主攻突破者的概率 / 倾向  
**定义**：
\[
\pi_i^{P}
=
\frac{\exp(\eta U_i^P)}
{\exp(\eta U_i^D)+\exp(\eta U_i^P)+\exp(\eta U_i^S)}
\]

### 4. `num_teammates_locked`
**含义**：当前被任意拦截器锁定的队友数量  
**定义**：
\[
num\_teammates\_locked
=
\sum_{k\in\mathcal A,\ k\neq i}
\mathbf 1\{locked\_by\_count_k>0\}
\]

### 5. `num_teammates_free`
**含义**：当前未被任何拦截器锁定的队友数量  
**定义**：
\[
num\_teammates\_free
=
\sum_{k\in\mathcal A,\ k\neq i}
\mathbf 1\{locked\_by\_count_k=0\}
\]

### 6. `team_effective_penetration_score`
**含义**：全队当前有效突防能力的连续评分  
**定义**：
\[
team\_effective\_penetration\_score
=
\sum_{k\in\mathcal A} e_k
\]
其中
\[
e_k=P_k^{pen}P_k^{hit}
\]

---

## 2.6 Analytic Prior Summary（解析先验汇总）

### 1. `Z_tilde_i`
**含义**：自己在多拦截器联合锥约束下的聚合脱锥风险  
**定义**：
\[
\widetilde Z_i
=
\frac{1}{\beta_Z}
\log\left(
\sum_{j\in\mathcal I}\exp(\beta_Z Z_{ij})
\right)
\]

### 2. `cone_cost_i`
**含义**：单机聚合锥风险  
**定义**：
\[
cone\_cost_i
=
[\widetilde Z_i+M_c]_+
\]

### 3. `E_i_esc`
**含义**：单机局部近距逃逸能力  
**定义**：
\[
E_i^{esc}
=
\max_{j\in\mathcal I}\Xi_{ij}
\]

### 4. `P_i_pen`
**含义**：单机突防成功度  
**定义**：
\[
P_i^{pen}
=
\sigma(-\kappa_Z(\widetilde Z_i+M_c))
\cdot
\sigma(\kappa_d(\rho_i^{threat}-\rho_s))
\cdot
\sigma(\kappa_m m_i^{local})
\cdot
\sigma(\kappa_E E_i^{esc})
\]

### 5. `P_i_hit`
**含义**：单机点目标命中可行度  
**定义**：
\[
P_i^{hit}
=
\sigma(\kappa_h(3.0-\rho_{iH}))
\cdot
\sigma(\kappa_c V_{c,iH})
\cdot
\sigma(-\kappa_\omega\|\omega_{iH}^{LOS}\|)
\]

---

# 3. MAPPO 最终奖励函数结构

> 由于改成 MAPPO，原本所有 cost 都要并入 reward 惩罚。  
> 所以最终 reward 统一写成：

\[
r_t^{total}
=
r_t^{task}
+
r_t^{game}
+
r_t^{escape}
-
r_t^{risk}
\]

终端时刻另加：

\[
r_T^{terminal}
\]

---

## 3.1 第一类：任务主线奖励 `r_task`

### 3.1.1 `reward_penetration`
**含义**：鼓励有效突防能力提升  
**定义**：
\[
reward\_penetration
=
\lambda_P \sum_{i\in\mathcal A} P_i^{pen}(t)
\]

---

### 3.1.2 `reward_hit_geometry`
**含义**：鼓励形成对 HVT 点目标的收敛命中几何  
**定义**：
\[
reward\_hit\_geometry
=
\lambda_\rho \sum_i \big(\rho_{iH}(t)-\rho_{iH}(t+\Delta t)\big)
+
\lambda_c \sum_i [V_{c,iH}]_+
-
\lambda_\omega \sum_i \|\omega_{iH}^{LOS}\|
\]

其中三项分别表示：
1. 距 HVT 更近
2. 对 HVT 闭合速度为正
3. HVT LOS 几何更稳定

---

### 3.1.3 `reward_no_retreat`
**含义**：惩罚远离 HVT、回头跑  
**定义**：
推荐不用旧的单步 `retreat_penalty=-0.15`，改成：
\[
reward\_no\_retreat
=
-\lambda_{ret}\sum_i [-V_{c,iH}]_+
\]

解释：
- 若对 HVT 的闭合速度为负，说明在远离目标
- 直接按“远离目标的速度”惩罚，比单步距离差更合理

---

## 3.2 第二类：主动诱饵博弈奖励 `r_game`

### 3.2.1 `reward_decoy_value`
**含义**：鼓励在合适时机涌现主动诱饵  
**定义**：
\[
reward\_decoy\_value
=
\lambda_D \sum_{i\in\mathcal A} U_i^{decoy}
\]

---

### 3.2.2 `reward_decoy_potential`
**含义**：鼓励群体势函数沿有利方向变化  
**定义**：
\[
reward\_decoy\_potential
=
\lambda_\Phi\big(\Phi_{decoy}(t+\Delta t)-\Phi_{decoy}(t)\big)
\]

其中
\[
\Phi_{decoy}
=
\sum_i
(\pi_i^D U_i^D+\pi_i^P U_i^P+\pi_i^S U_i^S)
-
\frac{\tau_\pi}{\eta}\sum_i\sum_{r\in\{D,P,S\}}\pi_i^r\ln\pi_i^r
\]

---

### 3.2.3 `reward_attention_redirect`（可选）
**含义**：鼓励敌方锁定更多落在诱饵或高诱饵价值个体上，而不是落在突破者上  
**定义**：
可用一个简化实现：
\[
reward\_attention\_redirect
=
\lambda_R
\sum_{j\in\mathcal I}
\Big(
U_{locked(j)}^{decoy}
-
U_{locked(j)}^{penetrate}
\Big)
\]
其中 `locked(j)` 表示当前被拦截器 \(j\) 锁定的进攻个体。

---

## 3.3 第三类：局部逃逸奖励 `r_escape`

### 3.3.1 `reward_escape`
**含义**：鼓励在近距条件下形成局部甩脱关键拦截器的能力  
**定义**：
\[
reward\_escape
=
\lambda_E \sum_{i\in\mathcal A} E_i^{esc}
\]

其中
\[
E_i^{esc}=\max_{j\in\mathcal I}\Xi_{ij}
\]

---

### 3.3.2 `reward_escape_progress`（可选）
**含义**：鼓励近距逃逸能力提升，而不是只看绝对值  
**定义**：
\[
reward\_escape\_progress
=
\lambda_{dE}\sum_i\big(E_i^{esc}(t+\Delta t)-E_i^{esc}(t)\big)
\]

---

## 3.4 第四类：风险惩罚 `r_risk`

> 所有原 cost 直接折进 reward 惩罚项

### 3.4.1 `penalty_cone`
**含义**：多拦截器未来终端暴露风险惩罚  
**定义**：
\[
penalty\_cone
=
\lambda_{cone}\sum_{i\in\mathcal A} [\widetilde Z_i+M_c]_+
\]

---

### 3.4.2 `penalty_fov`
**含义**：当前被敌方视场覆盖的惩罚  
**定义**：
\[
penalty\_fov
=
\lambda_{fov}\sum_i c_i^{fov}
\]
其中 `c_i^{fov}` 为环境中已有的 FOV exposure 成本。

---

### 3.4.3 `penalty_danger`
**含义**：进入拦截器危险近距区的惩罚  
**定义**：
\[
penalty\_danger
=
\lambda_{danger}\sum_i c_i^{danger}
\]

---

### 3.4.4 `penalty_boundary`
**含义**：越界惩罚  
**定义**：
\[
penalty\_boundary
=
\lambda_{boundary}\sum_i c_i^{boundary}
\]

---

### 3.4.5 `penalty_ground`
**含义**：不合理贴地 / 地面风险惩罚  
**定义**：
\[
penalty\_ground
=
\lambda_{ground}\sum_i c_i^{ground}
\]

注意：
- 不要把“低空”一律强罚
- 要区分“终端合理低空逼近目标”和“无意义贴地逃逸”
- 若现有 `z_min=0` 设计导致这项几乎无效，可引入“远离 HVT 且低空”的联合惩罚

例如：
\[
c_i^{ground\_bad}
=
\mathbf 1\{\rho_{iH}>d_{near}\}\cdot \mathbf 1\{z_i<z_{low}\}
\]

---

### 3.4.6 `penalty_collision`
**含义**：友机碰撞风险惩罚  
**定义**：
\[
penalty\_collision
=
\lambda_{collision}\sum_i c_i^{collision}
\]

---

## 3.5 总步奖励公式

最终每步奖励建议写成：

\[
r_t^{total}
=
\underbrace{reward\_penetration + reward\_hit\_geometry + reward\_no\_retreat}_{r_t^{task}}
+
\underbrace{reward\_decoy\_value + reward\_decoy\_potential + reward\_attention\_redirect}_{r_t^{game}}
+
\underbrace{reward\_escape + reward\_escape\_progress}_{r_t^{escape}}
-
\underbrace{(penalty\_cone + penalty\_fov + penalty\_danger + penalty\_boundary + penalty\_ground + penalty\_collision)}_{r_t^{risk}}
\]

---

# 4. 终端奖励

---

## 4.1 终端硬命中判定

### 点目标命中条件
对每架飞行器 \(i\)，若
\[
d_{iH}(T)=\|p_i(T)-p_H\|\le 3.0\ \text{m}
\]
则认为该飞行器成功命中 HVT。

---

## 4.2 终端统计量定义

### 1. `N_hit`
**含义**：真实点目标命中数量  
**定义**：
\[
N_{hit}
=
\sum_{i\in\mathcal A}\mathbf 1\{d_{iH}(T)\le 3.0\}
\]

### 2. `N_eff`
**含义**：有效突防数量  
**定义**：
\[
N_{eff}
=
\sum_{i\in\mathcal A} e_i
=
\sum_{i\in\mathcal A} P_i^{pen}P_i^{hit}
\]

### 3. `N_loss`
**含义**：总损失数量  
**定义**：
\[
N_{loss}
=
\sum_{i\in\mathcal A} \ell_i
\]
其中 \(\ell_i=1\) 表示该飞行器被毁。

### 4. `N_waste`
**含义**：无效牺牲数量  
**定义**：
\[
N_{waste}
=
\sum_{i\in\mathcal A} U_i^{waste}
\]
其中
\[
U_i^{waste}
=
\ell_i\cdot \chi_i^{salvage}
\]

### 5. `terminal_synergy`
**含义**：有效突防成员带来的群体协同增益  
**定义**：
\[
terminal\_synergy
=
N_{eff}^2
\]

---

## 4.3 最终终端奖励

\[
r_T^{terminal}
=
\lambda_{eff} N_{eff}
+
\lambda_{hit} N_{hit}
+
\lambda_{syn} N_{eff}^2
-
\lambda_{loss} N_{loss}
-
\lambda_{waste} N_{waste}
\]

---

# 5. 建议删除 / 明显减弱的旧奖励项

当前版本有：
- milestone rewards
- closest bonus
- proximity 奖励
- 中后期较强的 decoy / attack-gate 叠加
- 单步 retreat_penalty 很弱

建议：

## 删除或明显减弱
- `closest bonus`
- 过多 `milestone rewards`
- `proximity^3` 这类过细 shaping
- 中后期可能压过主线的 `attack_gate_reward`

## 替换为
- `reward_no_retreat`
- `reward_hit_geometry`
- `reward_escape`
- `reward_decoy_potential`

这样奖励主线更清晰。

---

# 6. 建议的最终 observation 清单（代码实现版）

## Self
- `self_rel_goal_x`
- `self_rel_goal_y`
- `self_rel_goal_z`
- `self_speed`
- `self_heading`
- `self_gamma`
- `self_ax`
- `self_ay`
- `is_locked`
- `locked_by_count`

## HVT
- `rho_to_hvt`
- `closing_speed_to_hvt`
- `omega_hvt_los`
- `omega_hvt_los_dot`
- `pn_hint_to_hvt`

## Top1 Threat
- `top1_rho`
- `top1_closing`
- `top1_bearing_error`
- `top1_in_fov`
- `top1_is_locking_me`
- `top1_omega_los`
- `top1_Gamma`
- `top1_Xi`
- `top1_local_Z`

## Top2 Threat
- `top2_rho`
- `top2_closing`
- `top2_bearing_error`
- `top2_in_fov`
- `top2_is_locking_me`
- `top2_omega_los`
- `top2_Gamma`
- `top2_Xi`
- `top2_local_Z`

## Team/Game
- `my_decoy_value`
- `my_decoy_prob`
- `my_penetrator_prob`
- `num_teammates_locked`
- `num_teammates_free`
- `team_effective_penetration_score`

## Priors
- `Z_tilde_i`
- `cone_cost_i`
- `E_i_esc`
- `P_i_pen`
- `P_i_hit`

---

# 7. 建议的最终 reward 清单（代码实现版）

## Task
- `reward_penetration`
- `reward_hit_geometry`
- `reward_no_retreat`

## Game
- `reward_decoy_value`
- `reward_decoy_potential`
- `reward_attention_redirect`（optional）

## Escape
- `reward_escape`
- `reward_escape_progress`（optional）

## Risk penalties
- `penalty_cone`
- `penalty_fov`
- `penalty_danger`
- `penalty_boundary`
- `penalty_ground`
- `penalty_collision`

## Terminal
- `terminal_effective_penetration`
- `terminal_hvt_hit`
- `terminal_synergy`
- `terminal_loss_penalty`
- `terminal_waste_penalty`

---

# 8. 代码实现优先级建议

## 第一优先级
先把 reward 主线改清楚：
- cost 全部并入 reward
- 删除过多碎项
- 明确四层奖励结构

## 第二优先级
压缩 HVT 观测：
- 只保留 LOS 几何量

## 第三优先级
敌方观测改成：
- top-2 threat + 摘要

## 第四优先级
队友观测改成：
- 角色概率 / 锁定摘要 / team penetration summary

---

# 9. 你复现后我重点帮你检查什么

你按这个实现以后，我会重点帮你检查：

1. 各 observation 维度是否有重复信息
2. `P_i_pen` 是否真的把第一/二/三部分串起来了
3. `reward_hit_geometry` 是否足以替代旧的 proximity / milestone
4. `reward_no_retreat` 是否足以压制回头跑
5. risk 惩罚权重是否压过 task 主线
6. 终端 `N_eff / N_hit / N_waste` 是否定义清楚
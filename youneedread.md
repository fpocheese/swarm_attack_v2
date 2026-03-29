# 任务：基于三个研究内容重构 `swarm_attack_v2` 代码（终极完整版本）

你现在是我的高级强化学习工程助手。  
请你在我当前仓库 `swarm_attack_v2` 的现有代码基础上，完成一次**面向论文研究内容的系统性重构**。

## 总目标

当前代码已经实现了一部分解析先验与训练流程，但还**不符合我现在最终确定的三部分研究内容**。  
我现在要你做的不是小修小补，而是：

1. **保留并规范化第一部分**
2. **彻底重构第二部分**
3. **重新整理第三部分，使其和第一、第二部分逻辑闭环**
4. **把环境机制修改成符合我现在论文设定的新敌方规则**
5. **把环境、观测、奖励、cost、日志、测试脚本统一调整**
6. **不要重写 MACPO 算法主体**
7. **优先修改环境层、解析先验层、训练脚本和评估脚本**
8. **所有修改必须模块化、可配置、可开关、可做 ablation**
9. **最终命中判定必须修改成“点目标命中、脱靶量 3m 内才算成功”**

---

# 一、当前论文对应的三个研究内容（必须严格按这个理解改代码）

## 第一部分：多拦截器锥约束下的聚合脱锥风险建模与终端安全突防机理
这一部分保留，作为底层几何/安全约束层。

代码目标：
- 保留并规范化 `q_ij -> Z_ij -> Z_tilde_i -> cone_cost`
- 它主要作为 **安全 cost** 和部分辅助观测
- 不要让它主导整个任务，只负责“未来终端暴露风险”的统一建模

---

## 第二部分：基于“先入视场即锁定”规则的主动诱饵突防博弈机理与群体决策方法
这一部分是**重构重点**。

旧版第二模块是：
- 固定分配
- mismatch reward
- 诱导固定目标分配失效

这已经**不符合我现在的敌方规则设定**，需要重写。

### 新敌方规则
敌方规则改为：

1. 初始阶段，拦截器依靠目指信息进行初始目标分配，并按指定方向飞行
2. 一旦某架进攻飞行器率先进入某拦截器视场
3. 该拦截器立即放弃原有目标分配
4. 直接锁定并攻击该“先入视场目标”

注意：
- 这是一个**事件触发式锁定重定向机制**
- 不再是“死盯原目标”
- 也不是“周期性重分配”
- 更不是“全覆盖保证 + 持续追击”

### 我方任务设定
我方进攻集群不预先固定“诱饵机”和“主攻机”身份。  
而是希望通过集群内部协同，动态涌现出：

- 有的成员主动暴露，去吸引拦截器锁定
- 有的成员利用诱饵创造出的低风险窗口去突破

所以第二部分在代码层面应该变成：

## 主动诱饵博弈 / 注意力重定向机制

你要实现的不是“固定谁是诱饵”，而是：

- 敌方注意力 / 锁定状态的环境建模
- 个体成为诱饵的局部价值量
- 群体层面的注意力重定向收益
- 这些量可作为 observation / reward shaping / logging 信号

---

## 第三部分：面向有效突防数量最大化的局部逃逸机理与群体价值优化方法
这一部分也要重构，但不是推翻。

第三部分要明确分成两层：

### 局部层
研究单架飞行器在窗口已经被造出来之后，是否能真正利用**近距 LOS 角速度突增**甩掉关键拦截器。

### 群体层
把这些局部成功突破叠加成：

- 更多有效突防成员
- 更少无效牺牲
- 更高群体终端收益

所以第三部分一定要同时包含：

### (1) 局部逃逸机理
也就是之前推导出来的：
- `Gamma_ij`
- `Xi_ij`

### (2) 群体有效突防数量
也就是：
- `P_i_pen`
- `P_i_hit`
- `e_i`
- `N_eff`
- `N_waste`

---

# 二、必须先做的第一件事：重新审查并修改敌方拦截逻辑

## 2.1 废弃当前不符合新设定的敌方逻辑

请你先在代码中找出并修改以下旧逻辑：

- 持续追击原始分配目标
- 不因事件放弃目标
- 周期性重分配
- full coverage / ensure_full_coverage 这种“保证每个进攻方都被拦”的分配逻辑

这些都不符合现在的论文设定。

### 新的环境规则应改为：

#### 初始阶段
- 拦截器可依靠目指信息获得一个初始目标或初始航向
- 但这个目标只作为“起始飞行参考”
- 不是后续必须持续追击的硬目标

#### 视场触发阶段
- 对每个拦截器 `j`
- 一旦某个进攻飞行器 `i` 率先满足“进入该拦截器视场”的条件
- 立刻触发：
  - 该拦截器进入 `LOCKED` 状态
  - 锁定对象变为该飞行器 `i`
  - 放弃原始分配目标

### 需要新增的环境状态
请在环境中为每个拦截器新增至少以下状态：

- `initial_assigned_target_idx`
- `current_locked_target_idx`
- `lock_mode` / `tracking_mode`
  - `INIT_GUIDE`
  - `SEARCH`
  - `LOCKED`
  - `MISSED`
  - `ABANDONED`（可选）
- `first_lock_time`
- `has_ever_locked`

### 需要新增的事件记录
- 哪个拦截器在什么时刻锁定了谁
- 哪个进攻飞行器是“先入视场目标”
- 锁定是否被打断
- 锁定后是否发生逃逸

---

## 2.2 环境中必须显式体现“谁锁定了谁”
这是我特别强调的补充要求：

### 代码层面必须支持：
进攻集群知道**哪个拦截器已经锁定了谁开始拦截了**。

这不是论文里的设定要求，而是代码实现层面的要求，目的是让集群内部有信息去实现：
- 谁去当诱饵
- 谁正在被敌人追
- 谁更适合趁机突破

### 你必须新增两个映射

#### defender -> offensive
例如：
- `locked_target_by_defender[j] = i or None`

#### offensive -> defenders
例如：
- `locked_by_map[i] = [j1, j2, ...]`

并把这些信息：
- 写入 `info`
- 必要时写入 observation
- 写入诊断日志

注意：
这和当前代码里的 `lock_on_map` 不是一回事。  
我要的是**环境事件触发后的真实锁定状态映射**。

---

# 三、第一部分代码修改要求：聚合 Z / cone cost 规范化

第一部分不是重写，而是规范化整理。

## 3.1 保留并检查当前已有模块
请检查并保留：
- `analytic_priors/y_system.py`
- `analytic_priors/cone_margin.py`

## 3.2 第一部分要保留的核心链条
对每个 `(i, j)`：
- `q_ij`
- `q_dot_ij`
- `Z_ij`

对每个进攻飞行器 `i`：
- `Z_tilde_i`

对整个集群：
- `cone_cost`

## 3.3 第一部分在训练中的定位
第一部分主要作为：
- **安全 cost**
- 少量辅助 observation
- 少量风险改善 reward（如果已有且合理）

不要让它单独决定策略目标。

## 3.4 需要你做的检查和修改
1. 检查 `cone_cost` 数值尺度是否合理  
2. 检查 `Z_tilde_i` 是否已经正确写入 observation  
3. 检查 `cone_cost` 是否只约束“未来暴露风险”，而没有压制所有前进行为  
4. 若必要，重做归一化 / clipping / weight 调整  
5. 把第一部分的日志字段统一整理清楚

---

# 四、第二部分代码修改要求：把旧的 assignment mismatch 模块改造成“主动诱饵博弈模块”

这是本次修改的重点。

---

## 4.1 删除或下线旧逻辑
旧的第二模块基于：
- 固定目标分配
- mismatch reward
- 敌方不改分配

这和新设定冲突。

请不要简单保留旧模块名字继续用旧逻辑。

### 两种可接受做法：

#### 做法 A（推荐）
直接新建模块，例如：
- `analytic_priors/decoy_game.py`
- `analytic_priors/attention_redirect.py`

#### 做法 B
保留原文件名 `assignment_mismatch.py`
但彻底重写其内部逻辑，并在文档/注释中明确说明：  
它现在不再表示旧的固定分配失配，而是表示**视场触发下的锁定注意力重定向博弈量**。

---

## 4.2 第二部分新的理论对应代码目标
你要实现的是：

### (1) 视场占用函数
对每个 `(i,j)` 定义：
- `q_ij`
- `s_ij = sigmoid(k_q * q_ij)`

这个 `s_ij` 表示：
- 目标是否进入该拦截器视场
- 进入程度多大

### (2) 锁定吸引强度
实现：
- `eta_ij`

建议至少依赖：
- `s_ij`
- `rho_ij`
- `V_c_ij`

### (3) soft lock 概率
对每个拦截器 `j` 实现：
- `p_lock_ij`

它不一定直接驱动环境锁定，但必须作为**解析先验量**存在，用于：
- 价值函数
- reward
- 日志
- 观测辅助量

### (4) 主动诱饵局部价值
实现个体诱饵价值函数：
- `U_i_decoy`

其结构应至少包含：
- `self_cost`
- `attention_benefit`
- `team_penetration_benefit`

### (5) 角色概率 / 角色倾向
不要预先固定“诱饵机”“主攻机”。

请实现软角色量，例如：
- `pi_i_decoy`
- `pi_i_penetrate`
- `pi_i_stealth`

或者等价的角色 score。

### (6) 群体势函数
实现一个群体势函数，例如：
- `Phi_decoy`

它应该体现：
- 有人主动暴露带来的群体收益
- 诱饵的局部代价
- 其他成员威胁下降带来的收益

---

## 4.3 第二部分应该如何接入环境

### Observation
必须考虑把以下量中的一部分加入 observation：
- `currently_locked_by_count`
- `current_locked_target_idx` 的局部编码
- `decoy_value` 或 `U_i_decoy`
- `pi_i_decoy`
- 队友中谁正在被锁定的信息摘要
- 哪些拦截器当前处于 `LOCKED` 状态

### Reward
第二部分不要再用旧的 `mismatch_reward`。  
改成新的 reward，例如：
- `reward_decoy`
- `reward_attention_redirect`
- `reward_team_cover`

推荐使用势函数增量：
- `Phi_decoy(t+dt) - Phi_decoy(t)`

### Logging
必须记录：
- 每个拦截器当前锁定谁
- 每个进攻飞行器被多少拦截器锁定
- 每个个体 `U_i_decoy`
- `pi_i_decoy`
- `Phi_decoy`

---

# 五、第三部分代码修改要求：把 LOS 近距逃逸机制正式并入“有效突防数量最大化”

第三部分要重新整理，不是只保留旧的 `escape reward` 就够了。

---

## 5.1 第三部分新的核心定位
第三部分不是单纯：
- 打 HVT
- 或末段打击控制

第三部分真正要做的是：

### 在第二部分造出来的局部突破窗口基础上
研究：
- 哪些个体真的能突破最后一层拦截
- 哪些牺牲是值得的
- 怎样让更多成员真正穿过去

所以第三部分一定要同时包含：

### (1) 局部逃逸机理
也就是之前推导出来的：
- `omega_los_ij`
- `omega_track_max_j`
- `Gamma_ij`
- `Xi_ij`

### (2) 群体有效突防数量
也就是：
- `P_i_pen`
- `P_i_hit`
- `e_i`
- `N_eff`
- `N_waste`

---

## 5.2 之前的 LOS escape 模块需要升级
请检查并升级：
- `analytic_priors/los_escape.py`

要求：

### 局部层
对每个 `(i,j)` 计算：
- `omega_los_ij`
- `omega_track_max_j`
- `Gamma_ij`
- `Xi_ij`

### 单机层
对每个进攻飞行器 `i` 定义：
- `E_i_esc = max_j Xi_ij`

这一步非常关键。  
不要只把 `Xi_ij` 当成一个额外 reward 就结束了。  
你必须把它正式并入后面的 `penetration success`。

---

## 5.3 penetration success score 必须重写
当前代码里已经有：
- `compute_penetration_success_score(...)`

但现在必须按新的论文逻辑重构。

### 新要求
`P_i_pen` 至少应显式包含四项：

1. `cone_safety`
2. `threat_distance`
3. `local_threat_weakening / redirected_attention`
4. `local_escape_ability E_i_esc`

也就是：
- 第一部分提供底层脱锥安全
- 第二部分提供局部窗口/注意力转移
- 第三部分前半提供近距最终甩脱能力

然后三者共同决定：
- 这架机当前是否真的“有望成为有效突防成员”

---

## 5.4 第三部分终端价值要改成“有效突防数量最大化”
请重新整理第三部分的终端收益，不要把重点写成“打击控制”，而要写成：

### 主目标
- `N_eff`: 有效突防数量

### 惩罚
- `N_loss`: 总损失
- `N_waste`: 无效牺牲

### 可选协同增强项
- `N_eff^2` 或其他饱和协同项

推荐实现一个清晰的终端项：
- `terminal_group_value`

并在代码中明确拆出：
- `effective_penetration_count`
- `wasted_loss_count`
- `terminal_synergy_bonus`

---

## 5.5 HVT guidance 逻辑的位置要调整
当前代码里已经有：
- `compute_hvt_guidance_features`
- `compute_penetration_success_score`
- HVT guidance obs
- attack gate reward

这些逻辑不要直接删掉，但要重新定位：

### 正确定位
HVT 相关量不是第三部分的主体，  
而是第三部分里“有效突防成员进一步形成终端任务收益”的一小部分。

所以：
- 保留 HVT guidance
- 但不要让它压过“有效突防数量”主线
- 优先保证第三部分仍然是“突防问题”

---

# 六、额外重要补充：终端任务判定必须改成“点目标命中”，不能再用目标区域/目标球

## 6.1 当前问题
现在环境里进攻飞行器对高价值目标 HVT 的“打击成功”判定逻辑不对。  
当前实现更像是：

- 只要进入以 HVT 为球心的某个圆/球形区域
- 或进入某个攻击半径范围
- 就算成功

这不符合我现在的场景设定。

## 6.2 正确设定
我要的不是“进入目标附近范围就算命中”，而是：

> **高价值目标 HVT 是一个点目标。**  
> **进攻飞行器最终必须命中这个点目标本身。**  
> **只有最终脱靶量不超过 3 m，才算成功命中。**

也就是说：
- HVT 不是一个区域目标
- 不是一个攻击球
- 不是一个容许大范围命中的区域
- 它就是一个固定坐标点
- 终端脱靶量 `miss_distance <= 3m` 才记为成功

---

## 6.3 HVT 命中判定逻辑
请你在环境中找到当前 HVT 命中 / 攻击成功 / 任务完成的判定逻辑，并改成：

### 定义
设进攻飞行器 `i` 的当前位置为：
- `p_i`

HVT 固定坐标为：
- `p_H`

则终端脱靶量定义为：

\[
d_{iH}(t)=\|p_i(t)-p_H\|
\]

### 成功命中条件
\[
d_{iH}(t)\le 3.0\ \text{m}
\]

只有满足这个条件，才算该飞行器成功命中 HVT。

---

## 6.4 删除/下线旧的“区域攻击成功”逻辑
请检查代码里所有类似下面的逻辑并修改：

- 进入某个 attack radius 就算 hit
- 进入 HVT 周围圆形区域就算 success
- 进入某个球形区域就算成功打击
- 靠近到一个较大半径就记为 mission success

这些都不符合现在的设定。

### 正确要求
- 保留“接近 HVT”的中间奖励可以有
- 但**最终成功判定**必须严格使用：
  - 点目标
  - 脱靶量
  - `<= 3m`

---

## 6.5 命中判定要和 episode 结束逻辑统一

### 对单架飞行器
如果某架进攻飞行器满足：
- `miss_distance_to_HVT <= 3m`

则标记：
- `hit_hvt = True`
- `effective_penetrator = True`

### 对整个 episode
请根据当前任务设定决定 episode 是否结束：

#### 推荐做法
- 若任意进攻飞行器成功命中 HVT，则可以判定任务成功并结束 episode
- 同时保留统计：
  - 哪一架命中了
  - 命中时刻
  - 命中前是否已被锁定
  - 命中前的脱靶量变化轨迹

如果你认为当前代码结构更适合“继续仿真到结束再统计”，也可以不立即结束，但必须明确：
- 任务成功判定以“命中点目标，脱靶量 <= 3m”为准

---

## 6.6 必须记录真实脱靶量
请对每架进攻飞行器逐步记录：
- `miss_distance_to_HVT`

并加入：
- 日志
- 诊断脚本
- rollout 保存数据
- 绘图脚本

---

## 6.7 终端统计必须新增
请在评估统计里新增：

- `num_hit_hvt`
- `first_hit_time`
- `hit_agent_id`
- `terminal_miss_distance_min`
- `terminal_miss_distance_per_agent`

---

## 6.8 reward 中的 HVT 相关项也要检查
当前如果有：
- `hvt_progress_reward`
- `attack_gate_reward`
- `P_i_hit`
- `HVT guidance`

请统一改成围绕“点目标命中”来设计。

### 正确方向
可以保留中间过程奖励，例如：
- 距离 HVT 缩短
- 闭合速度为正
- LOS 收敛更稳定

但这些只是中间 shaping，不能代替最终 hit 判定。

---

## 6.9 `P_i_hit` 的定义也要同步修改
当前第三部分里若有 `P_i_hit`，请改成更符合点目标打击的形式。

建议定义仍然保留连续软形式，但必须围绕“点目标脱靶量”构造，例如：

\[
P_i^{\mathrm{hit}}(t)
=
\sigma\!\bigl(\kappa_h(3.0-\rho_{iH}(t))\bigr)
\cdot
\sigma\!\bigl(\kappa_c V_{c,iH}(t)\bigr)
\cdot
\sigma\!\bigl(-\kappa_{\omega}\|\omega_{iH}^{LOS}(t)\|\bigr)
\]

其中：
- 第一项中的阈值改成 `3.0 m`
- 明确表示这是围绕点目标命中构造的软命中可行度
- 但最终成功仍以硬条件 `miss_distance <= 3m` 为准

---

## 6.10 可视化和诊断必须同步修改

### 必须画的曲线
- `miss_distance_to_HVT - time`
- `closing_speed_to_HVT - time`
- `omega_HVT_LOS - time`

### 轨迹图要求
轨迹图中请把 HVT 画成：
- 一个明确的点
- 不是圆形攻击区

如果为了可视化需要，可额外画一个极小的参考圆表示 `3m miss tolerance`，但必须明确说明：
- 这不是攻击区域
- 只是命中判定误差圈

---

# 七、必须新增的环境信息：进攻集群知道谁锁定了谁

这个是专门加的一条代码要求，必须实现。

## 7.1 环境中必须有显式字段

### defender 视角
- `current_locked_target_idx[j]`
- `lock_mode[j]`
- `lock_time[j]`

### offensive 视角
- `locked_by_map[i]`
- `locked_by_count[i]`
- `locked_defender_ids[i]`

## 7.2 observation 中必须体现
至少为每个进攻 agent 加入：
- `locked_by_count`
- `is_currently_locked`
- `is_primary_decoy_candidate`（可由局部价值判断）
- 最近若干拦截器中，谁锁定了自己 / 队友 的摘要特征

## 7.3 info / 日志中必须体现
每步记录：
- 哪个拦截器锁定了哪个进攻机
- 哪些进攻机当前无人锁定
- 哪些进攻机被多机锁定
- 锁定发生的时间和持续时间

---

# 八、具体文件级修改要求

请你先完整阅读当前项目，再按下面方向修改。

## 8.1 必须重点阅读并修改的文件

### 环境层
- `envs/fov_penetration/fov_penetration_env.py`
- `envs/fov_penetration/reward_cost.py`
- `envs/fov_penetration/config.py`
- `envs/fov_penetration/policies_interceptor.py`
- `envs/fov_penetration/target_assignment.py`

### 解析先验层
- `envs/fov_penetration/analytic_priors/cone_margin.py`
- `envs/fov_penetration/analytic_priors/assignment_mismatch.py`（重构或替换）
- `envs/fov_penetration/analytic_priors/los_escape.py`
- `envs/fov_penetration/analytic_priors/hvt_guidance.py`
- `envs/fov_penetration/analytic_priors/penetration_phase.py`
- `envs/fov_penetration/analytic_priors/y_system.py`

### 训练 / 评估 / 诊断层
- `scripts/train_fov_penetration_macpo.py`
- `scripts/eval_fov_penetration_macpo.py`
- `scripts/diagnose_episode.py`
- `scripts/run_diagnostic.py`
- 如有 reward breakdown / monitor / render 脚本，也一起同步调整

---

# 九、具体实施顺序（必须按顺序做）

## 第一步：读代码并输出改动计划
先不要写代码。  
先阅读项目并回复我：

1. 当前敌方锁定逻辑具体在哪些文件里
2. 当前 assignment mismatch 模块哪些地方必须删/改
3. 第二部分新的 decoy game 模块应该放在哪个文件
4. 第三部分 LOS 逃逸和有效突防数量逻辑怎么接入当前代码
5. 哪些 observation 字段需要新增
6. 哪些日志字段需要新增
7. 哪些 config 需要新增或修改
8. 当前命中判定逻辑在哪些文件里
9. 旧的“攻击区域/攻击球”逻辑在哪里要删除或替换

---

## 第二步：先改环境规则
优先完成：
- “先入视场即锁定”敌方机制
- 显式锁定关系映射
- 去掉与新设定冲突的周期性重分配 / 持续死追逻辑

---

## 第三步：重构第二部分
实现：
- decoy game / attention redirect
- decoy value
- soft role probability
- group potential

---

## 第四步：重构第三部分
实现：
- `E_i_esc`
- 新版 `P_i_pen`
- `N_eff`
- `N_waste`
- 终端群体价值

---

## 第五步：修改点目标命中逻辑
实现：
- `miss_distance_to_HVT`
- `hit_hvt` 的 3m 判定
- 删除旧区域攻击成功逻辑
- 更新 `P_i_hit`
- 更新 episode done 条件
- 更新统计与诊断

---

## 第六步：统一 reward / cost / obs / info
把三部分内容统一接入环境与日志。

---

## 第七步：补诊断和测试
你必须同步修改诊断脚本，使其能验证：
- 谁锁定了谁
- decoy 机制有没有涌现
- 局部逃逸机制有没有发生
- 有效突防数量是否提升
- 哪些牺牲是无效牺牲
- 是否出现真实的 3m 内命中

---

# 十、你最终交付我的内容

完成后请按下面格式回复我：

1. 修改后的项目结构树
2. 修改和新增的文件列表
3. 每个文件具体做了什么
4. 第二部分新的环境锁定机制在哪里实现
5. 第二部分的 decoy value / role probability / potential function 在哪里实现
6. 第三部分的 `E_i_esc / P_i_pen / N_eff / N_waste` 在哪里实现
7. observation 新增了哪些字段
8. reward / cost 新增了哪些项
9. 当前命中判定逻辑改在了哪些文件
10. 旧的“攻击区域/攻击球”逻辑在哪里被删除或替换了
11. 现在 `hit_hvt` 的唯一判定条件是什么
12. 评估统计里新增了哪些命中相关指标
13. 诊断图里新增了哪些脱靶量相关曲线
14. 如何运行训练
15. 如何运行诊断
16. 你做了哪些近似处理
17. 后续还可以怎么继续增强

---

# 十一、最后强调

## 这次修改的关键不是：
- 再加几个 reward
- 再堆几个 observation

## 而是：
- 把环境机制真正改成论文设定
- 把第一、第二、第三部分的理论结构落到代码里
- 让策略能够学会：
  - 聚合脱锥风险约束下的安全突防
  - 动态诱饵
  - 注意力重定向
  - 局部近距甩脱
  - 群体有效突防数量最大化
  - 最终对点目标的精确命中

现在开始第一步：  
**请先完整阅读项目代码，然后给我实施计划，不要立刻生成大段代码。**
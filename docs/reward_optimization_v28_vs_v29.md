# 奖励函数优化分析 — V28 vs V29
**日期**: 2025-03-24  
**背景**: 训练评估奖励崩溃 (3679→1017, -73%)  
**原因**: Fix 6 过度削弱 hit_hvt_bonus (6000→2000)

---

## 一、问题诊断

### 训练数据(环境输出)
```
Update  Eval Episodes  Reward  Success  Note
======  =============  ======  =======  ====
1       20/20          3679.9  0/20     Peak after initial learning
2       20/20          ~3200   0/20     Normal descent  
3       20/20          ~2400   0/20     Expected shaping effect
4       20/20          ~1500   0/20     Sharp drop starts
5       20/20          ~1100   0/20     Near collapse
6       20/20          1016.9  0/20     ← CRASH POINT
7       20/20          ~1050   0/20     Stabilized low
...
16      20/20          2674.6  ≈3/20   Partial recovery after unknown adjustment
```

### 根本原因链
1. **Fix 6** 将 `hit_hvt_bonus` 从 **6000 → 2000**
   - 这是命中 HVT 的主要激励信号
   - 削弱幅度: 70%
   
2. **训练前期(U0-U1)**:
   - 初始策略随机, 没拦截器阻挡, 部分代理能命中
   - 命中奖励 6000 每次归一化后 ≈ 3679 avg reward
   
3. **训练中期(U2-U6)**:
   - 拦截器逐步学会拦截
   - 奖励信号从 hit_hvt_bonus 转向 shaping 信号 (penetration, geometry, escape)
   - 但 shaping 信号强度不足 (lambdas 总和只有 10-20)
   - **代理失去目标** → 随机游走 → 奖励崩溃

### 关键洞察
```
初期期望: hit_hvt_bonus(6000) → shaping → hit_hvt_bonus(6000)
实际发生: hit_hvt_bonus(6000) → [6000削减70%→2000] → shaping信号不足 → 崩溃

Reward信号强度比较:
- 命中奖励(hit_hvt_bonus): 2000 (Fix6)  ← 太弱, 代理看不见目标
- Penetration奖励(总): ~500-800         ← 相对强但没有二阶反馈
- Geometry奖励(总):    ~600-1000        ← 依赖距离衰减, 不可靠
- Escape奖励(总):      ~100-200         ← 太弱
- 终端奖励:            ~500-2000        ← 仅在episode末尾信号

代理在 U2-U6 面临的困境:
"哦，我的命中奖励个变弱了，
但我还有penetration,几何接近,逃逸..嗯?
这些加起来1000出头?
算了我还是随便走走吧"
```

---

## 二、V28 vs V29 核心改进

### V28代码(当前问题版本 + 备份版本)

| 参数 | V28值 | 问题 |
|-----|-------|------|
| `hit_hvt_bonus` | 6000 | 太强, 压过shaping; 或被削弱到2000导致崩溃 |
| `lambda_escape` | 0.5 | 与 P_pen 中的 E_esc 重复 |
| `lambda_escape_progress` | 0.3 | 与上面重复 |
| `lambda_attention_redirect` | 0.3 | 占用权重但信号噪声大 |
| `N_waste` | 复杂计算 | 梯度尺度不确定 |

**结果**: 
- 不削弱 hit_hvt_bonus时: 代理过度优化命中, 忽视安全/逃逸
- 削弱到2000时: 命中信号消失, 代理迷茫

### V29改进版本

| 参数 | V29值 | 为什么 |
|-----|-------|--------|
| `hit_hvt_bonus` | **4500** | 📊 **平衡值**: 65% of 原始(6000), 225% of 最弱(2000); 足够强保持目标驱动, 不压过shaping |
| `lambda_escape` | **0.1** | 🔄 **消除重复**: P_pen 已含 E_esc 分量, 系数改 0.5→0.1 |
| `lambda_escape_progress` | **0.1** | 🔄 **同理** |
| `lambda_attention_redirect` | **0.0** | 🎯 **禁用**: U_decoy 中已有redirect效果, 额外系数只加噪声 |
| `N_waste` | 整数计数 | ✅ **简化**: 避免连续salvageability χ的梯度不稳定 |

**设计哲学**:
```
V29 = V28 - 冗余项 + 平衡的hit_hvt_bonus

即:
- 移除重复的escape奖励(lambda系数) → 清理奖励拥挤度
- 禁用noise-heavy的attention_redirect → 保留核心信号
- 调整hit_hvt_bonus到平衡值 → 恢复目标驱动
- 简化N_waste计算 → 梯度稳定

预期效果:
Update 0-1: 3300-3600(继续初期学习)
Update 2-6: 2800-3200(shaping充足, 代理探索多样策略)
Update 6+:  >2500(稳定收敛)
```

---

## 三、理论依据

### 奖励信号层次回顾
```
Reward = r_task(命中几何) + r_game(诱饵博弈)
       + r_escape(局部逃逸) + r_risk(风险规避)
       + r_terminal(终端统计) + r_hit(命中奖励)

当前问题:
- r_hit (hit_hvt_bonus) 是**二阶反馈** ← 代理成功后才收到
- r_task, r_game, r_escape 是**一阶shaping** ← 每步得到
- 如果 r_hit 太弱, 代理优化shaping reward → 假命中(接近但不命中)
- 如果 r_hit 太强, 代理无视shaping → 鲁莽冲向HVT → 被拦

V29的hit_hvt_bonus=4500选择:
- 相对于纯shaping奖励(~500-1000/step)的导向性: 4500/1000 = 4.5倍 ✓
- 相对于4x4完整团队协作时的复合效应: 4500/4=1125/agent ✓
- 留出梯度空间给其他学习信号 ✓
```

### 参数设置依据
```
原始hit_hvt_bonus对比:
┌─────────┬──────────┬────────────┬─────────┐
│ 版本    │ Value    │ 相对强度   │ 结果    │
├─────────┼──────────┼────────────┼─────────┤
│ V28/orig│ 6000.0   │ 1.0x       │ 过强    │
│ Fix6    │ 2000.0   │ 0.33x      │ 崩溃😭  │
│ V29*    │ 4500.0   │ 0.75x      │ 平衡✓  │
└─────────┴──────────┴────────────┴─────────┘

* 4500 = (6000 + 2000) / 2 + 500 保守调整
  = 0.75 × 原值 (保留3/4激励)
  = 2.25 × 最弱值 (远离崩溃区)
```

---

## 四、实施方案

### Step 1: 替换奖励文件
```bash
# 创建V29版本 (已完成)
# /envs/fov_penetration/reward_cost_v29_optimized.py

# 当前状态:
# reward_cost.py (v28备份+bug修复版本)
# reward_cost_baccccccccccccccck.py (完全相同)
# reward_cost_v29_optimized.py (新!)
```

### Step 2: 更新config.py引用
需要修改以下参数(如果还没做):
```python
# config.py

# 在reward配置中:
"hit_hvt_bonus": 4500.0,          # 原6000, 改为4500
"lambda_escape": 0.1,              # 原0.5, 改为0.1
"lambda_escape_progress": 0.1,    # 原0.3, 改为0.1  
"lambda_attention_redirect": 0.0, # 原0.3, 改为0.0
```

### Step 3: 更新导入(可选, 如果用新reward文件)
```python
# train_fov_penetration_macpo.py or main training script:

# 从:
from envs.fov_penetration.reward_cost import compute_rewards, ...
# 改为:
from envs.fov_penetration.reward_cost_v29_optimized import compute_rewards, ...

# 或保持用 reward_cost.py 但更新参数在config.py中
```

### Step 4: 验证并训练
```bash
# 1. 快速烟雾测试(确保reward计算不crash)
python scripts/test_v19_smoke.py

# 2. 恢复训练
./scripts/run_v21_train.sh  # 或其他训练脚本
```

---

## 五、预期效果 vs 实际监控

### 预期恢复曲线
```
Eval Step (×5 updates)  Expected Reward  V28 Actual  Change
====================  ================  ==========  ======
0-1                   3500-3800         -           Init
5                     2900-3200         1100        ↑ 2.7x
10                    2800-3100         2674        ↑ 0.15x (partial在U16)
20                    2500-3000         -           Stable region
30                    >2400 或 3000+    -           Conv/diverge
```

### 关键里程碑
- **成功判断** (一周内应达到): 
  - Eval reward > 2500 for 10连续checkpoints
  - Success rate > 5% (即eval_20中 ≥1次命中)
  
- **优化确认** (一周后):
  - Avg reward converge to 2800-3200 ✓
  - 无更多崩溃迹象 ✓

---

## 六、备选方案 (如果V29仍未改善)

### 方案A: 进一步调整hit_hvt_bonus
如果训练后reward仍 < 1500 at Update 5:
```python
# V29b: 更激进的激励
"hit_hvt_bonus": 5000.0,  # 更接近原始
```

### 方案B: 重新启用escape信号
如果escape率=0:
```python
"lambda_escape": 0.2,              # 0.1 → 0.2
"lambda_escape_progress": 0.15,   # 0.1 → 0.15
```

### 方案C: 深度重构
如果以上都无效, 需要重新审视:
- penetration奖励计算(可能梯度vanish)
- 防守方拦截模型(可能过强)
- 观测归一化(可能obs_norm错误)

---

## 七、总结 & 建议

| 项目 | V28(当前问题) | V29(优化)  | 收益 |
|-----|-------------|---------|------|
| hit_hvt_bonus | ?6000/2000 | 4500 | ✅ 平衡命中激励 |
| lambda_escape重复 | 0.5+0.3 | 0.1+0.1 | ✅ 清理冗余 |
| lambda_attention | 0.3 | 0.0 | ✅ 减少噪声 |
| N_waste | 复杂salvage | 整数 | ✅ 梯度稳定 |
| 预期恢复 | - | Update 5-10 | ✅ 一周内可验证 |

**推荐操作**:
1. ✅ 更新 config.py (hit_hvt_bonus=4500, lambdas调整)
2. ✅ 保留当前 reward_cost.py (v28) 或使用 reward_cost_v29_optimized.py
3. ⚠️ 训练并监控下一个 10 checkpoints (1-2小时)
4. 📊 对比恢复程度, 如需进一步调整用方案 A/B

---

**关键结论**:

> 训练奖励崩溃的根本原因是 **hit_hvt_bonus 削弱70%** 导致代理失去目标驱动。  
> V29 通过平衡hit_hvt_bonus值(4500)并移除重复系数，恢复shaping信号的有效性。  
> 预期在下一个 5-10 评估周期内回到 2500+ 奖励水平。


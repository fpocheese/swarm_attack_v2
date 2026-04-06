# 当前版本最重要的修改清单（精简版）

> 这几条是**必须先改**的，先不要分散精力做太多优化。  
> 先把这几条修好，再继续训练。

---

## 1. 统一 HVT 命中阈值为 5m
### 必改原因
你现在配置和环境里的命中阈值来源不统一：

- `point_target.hit_threshold`
- `hit_hvt_range`

而且环境实际判定还在读旧字段。

### 要求
全代码统一成：

\[
\|p_i-p_H\| \le 5.0\text{ m}
\]

### 只保留一个权威字段
建议只保留：
```python
"point_target": {
    "hit_threshold": 5.0
}
```

---

## 2. 修正 `P_i_hit` 的公式尺度
### 必改原因
你现在 `P_i_hit` 里把距离写成了类似 `rho / 100.0`，这会把“命中可行度”的阈值错误放大到百米量级，和点目标命中逻辑不一致。

### 要求
`P_i_hit` 只能围绕“点目标命中几何”来定义。

### 推荐写法
硬命中仍然是 5m，soft shaping 可以稍大一点，例如 10m：
\[
P_i^{hit}
=
\sigma(\kappa_h(d_{soft}-\rho_{iH}))
\cdot
\sigma(\kappa_c V_{c,iH})
\cdot
\sigma(-\kappa_\omega |\omega_{iH}^{LOS}|)
\]
其中：
- `d_soft = 10m`
- 最终成功判定仍然只认 `5m`

---

## 3. 修复 `reward_decoy_potential` 的 bug
### 必改原因
你现在大概率是先更新了 `Phi_decoy`，再把它当作 `prev_Phi_decoy` 传给 reward。这样会导致：

\[
\Phi(t+\Delta t)-\Phi(t)=0
\]

也就是说 `reward_decoy_potential` 基本失效。

### 要求
顺序必须改成：

1. 先保存旧的 `prev_phi`
2. 再计算新的 `Phi_decoy`
3. 把旧值传进 reward
4. 最后才更新缓存

---

## 4. 重构 `N_waste`
### 必改原因
你现在的 `N_waste` 本质上还是：

- 死了
- 且没命中

这不符合论文里“无效牺牲”的定义。

### 要求
要改成：

> 本来还有较大机会突破/保全，却被白白消耗掉

### 最简实现建议
至少用这三个量构造：
- `Z_tilde_i`
- `E_i_esc`
- `threat_distance_i`

定义一个 `salvageability`，再算：

\[
N_{waste} = \sum_i \ell_i \cdot \chi_i^{salvage}
\]

不要再直接用整数计数。

---

## 5. 收一下重复奖励
### 必改原因
你现在有几处明显重复：

#### 重复 1
- `reward_escape`
- `reward_penetration`

因为 `P_i_pen` 里已经包含 `E_i_esc`

#### 重复 2
- `reward_decoy_value`
- `reward_attention_redirect`

因为 `U_i_decoy` 本身已经在反映吸引火力收益

### 建议
先做一个简洁版：

- 降低 `lambda_escape`
- 暂时把 `lambda_attention_redirect = 0.0`
- 保留：
  - `reward_penetration`
  - `reward_decoy_value`
  - `reward_decoy_potential`

---

## 6. 下调 `hit_hvt_bonus`
### 必改原因
你现在 `hit_hvt_bonus = 6000` 太大，容易让单次命中压过其他所有 shaping 和终端项。

### 建议
先降到：
```python
"hit_hvt_bonus": 2000.0
```

---

# 最后一句
## 先只改这 6 条，再继续训练：
1. 统一 5m 命中阈值  
2. 修正 `P_i_hit`  
3. 修复 `reward_decoy_potential`  
4. 重构 `N_waste`  
5. 收缩重复奖励  
6. 下调 `hit_hvt_bonus`

"""
FOV Penetration Environment — 防御方目标分配 V21
=================================================
基于威胁度与拦截代价的目标分配模块

V21 改动 (2026-03-28):
  - 新增 ensure_full_coverage(): 保证每个存活进攻方至少有一架拦截器
  - 多余的拦截器分配给威胁度最高的进攻方(多机围攻)
  - 配合 config.reassign=True 周期性重分配, 进攻方死亡后释放拦截器

实现:
  1. 为每个防御飞行器分配一个优先拦截目标
  2. 构造代价矩阵: C[i,j] = w_threat × threat(j) + w_cost × intercept_cost(i,j)
  3. 使用匈牙利算法或贪心算法最小化总代价
  4. ensure_full_coverage 保证全覆盖

威胁度计算:
  threat(j) = 1.0 - dist(j, HVT) / max_dist  (越近 HVT 威胁越大)
  + 0.2 × cos(heading_to_HVT)                (航向指向 HVT 额外威胁)

拦截代价计算:
  intercept_cost(i,j) = dist(i,j) / max_dist  (距离越远代价越大)
  + 0.3 × (1 - V_c / V_max)                  (接近速度低代价高)
"""

import numpy as np


def compute_threat_score(offensive, hvt, max_dist):
    """
    计算单个进攻飞行器的威胁度

    Args:
        offensive: Aircraft 实例
        hvt: HVT 实例
        max_dist: 归一化距离

    Returns:
        threat: [0, ~1.5] 越大越危险
    """
    if not offensive.alive:
        return 0.0

    # 1. 距离 HVT (越近越危险)
    dist_to_hvt = offensive.distance_to(hvt.x, hvt.y, hvt.z)
    dist_threat = 1.0 - np.clip(dist_to_hvt / max_dist, 0, 1)

    # 2. 航向指向 HVT (越正对越危险)
    dx = hvt.x - offensive.x
    dy = hvt.y - offensive.y
    angle_to_hvt = np.arctan2(dy, dx)
    heading_diff = angle_to_hvt - offensive.heading
    heading_diff = np.arctan2(np.sin(heading_diff), np.cos(heading_diff))
    heading_threat = 0.2 * max(np.cos(heading_diff), 0.0)

    # 3. 速度 (越快越危险)
    speed_threat = 0.1 * offensive.v / 100.0

    return dist_threat + heading_threat + speed_threat


def compute_intercept_cost(defensive, offensive, max_dist):
    """
    计算单个防御飞行器拦截单个进攻飞行器的代价

    Args:
        defensive: Aircraft (防御方)
        offensive: Aircraft (进攻方)
        max_dist: 归一化距离

    Returns:
        cost: [0, ~1.5] 越大越难拦截
    """
    if not defensive.alive or not offensive.alive:
        return 999.0  # 不可用

    # 1. 3D 距离代价
    dist = defensive.distance_3d(offensive)
    dist_cost = np.clip(dist / max_dist, 0, 1)

    # 2. 接近速度代价 (接近速度低 → 代价高)
    dx = offensive.x - defensive.x
    dy = offensive.y - defensive.y
    dz = offensive.z - defensive.z
    r = max(np.sqrt(dx**2 + dy**2 + dz**2), 1.0)

    # 防御方速度分量
    cos_g_d = np.cos(defensive.gamma)
    vx_d = defensive.v * cos_g_d * np.cos(defensive.heading)
    vy_d = defensive.v * cos_g_d * np.sin(defensive.heading)
    vz_d = defensive.v * np.sin(defensive.gamma)

    # 进攻方速度分量
    cos_g_o = np.cos(offensive.gamma)
    vx_o = offensive.v * cos_g_o * np.cos(offensive.heading)
    vy_o = offensive.v * cos_g_o * np.sin(offensive.heading)
    vz_o = offensive.v * np.sin(offensive.gamma)

    # 相对速度在连线方向的投影 (接近速度)
    dvx = vx_o - vx_d
    dvy = vy_o - vy_d
    dvz = vz_o - vz_d
    closing_vel = -(dx * dvx + dy * dvy + dz * dvz) / r
    approach_cost = 0.3 * (1.0 - np.clip(closing_vel / 120.0, -0.5, 1.0))

    return dist_cost + approach_cost


def build_cost_matrix(defensives, offensives, hvt, config):
    """
    构造代价矩阵 C[n_def × n_off]

    C[i,j] = w_threat × (1 - threat(j)) + w_cost × intercept_cost(i,j)

    注: 威胁度高的目标, 我们希望优先分配 → 代价用 (1 - threat) 使高威胁 = 低代价

    Args:
        defensives: list of Aircraft (防御方)
        offensives: list of Aircraft (进攻方)
        hvt: HVT
        config: 环境配置

    Returns:
        cost_matrix: ndarray shape (n_def, n_off)
    """
    assign_cfg = config["assignment"]
    w_threat = assign_cfg["threat_weight"]
    w_cost = assign_cfg["intercept_cost_weight"]
    max_dist = config["map_size"] * 2.0

    n_def = len(defensives)
    n_off = len(offensives)
    C = np.full((n_def, n_off), 999.0)

    # 预计算威胁度
    threats = np.array([compute_threat_score(o, hvt, max_dist) for o in offensives])

    for i, d in enumerate(defensives):
        if not d.alive:
            continue
        for j, o in enumerate(offensives):
            if not o.alive:
                continue
            ic = compute_intercept_cost(d, o, max_dist)
            # 高威胁 → 低代价 (优先分配)
            C[i, j] = w_threat * (1.0 - threats[j]) + w_cost * ic

    return C


def hungarian_assignment(cost_matrix):
    """
    匈牙利算法求解最优分配

    处理非方阵: 如果 n_def ≠ n_off, 填充虚拟行/列

    Args:
        cost_matrix: ndarray shape (n_def, n_off)

    Returns:
        assignments: dict {def_idx: off_idx}
    """
    try:
        from scipy.optimize import linear_sum_assignment
        n_def, n_off = cost_matrix.shape
        # 扩展为方阵
        size = max(n_def, n_off)
        padded = np.full((size, size), 999.0)
        padded[:n_def, :n_off] = cost_matrix

        row_ind, col_ind = linear_sum_assignment(padded)

        assignments = {}
        for r, c in zip(row_ind, col_ind):
            if r < n_def and c < n_off and cost_matrix[r, c] < 900:
                assignments[r] = c
        return assignments

    except ImportError:
        # scipy 不可用 → 回退到贪心
        return greedy_assignment(cost_matrix)


def greedy_assignment(cost_matrix):
    """
    稳定贪心分配 (scipy不可用时的备选)

    按代价从小到大贪心分配, 尽量避免重复
    """
    n_def, n_off = cost_matrix.shape
    assignments = {}
    assigned_off = set()

    # 收集所有 (cost, def_idx, off_idx)
    entries = []
    for i in range(n_def):
        for j in range(n_off):
            if cost_matrix[i, j] < 900:
                entries.append((cost_matrix[i, j], i, j))

    entries.sort()

    for _, d_idx, o_idx in entries:
        if d_idx in assignments:
            continue
        if o_idx in assigned_off and len(assigned_off) < n_off:
            # 还有未分配目标, 跳过已分配的
            continue
        assignments[d_idx] = o_idx
        assigned_off.add(o_idx)

    # 未分配的防御机 → 分配威胁最高的未覆盖目标 (允许重复)
    for i in range(n_def):
        if i not in assignments:
            best_j = np.argmin(cost_matrix[i])
            if cost_matrix[i, best_j] < 900:
                assignments[i] = best_j

    return assignments


def assign_targets(defensives, offensives, hvt, config):
    """
    目标分配主接口

    Args:
        defensives: list of Aircraft (防御方)
        offensives: list of Aircraft (进攻方)
        hvt: HVT
        config: 环境配置

    Returns:
        assignments: dict {def_idx: off_idx}
        cost_matrix: ndarray shape (n_def, n_off)
        threat_scores: ndarray shape (n_off,)
    """
    max_dist = config["map_size"] * 2.0
    threat_scores = np.array([
        compute_threat_score(o, hvt, max_dist) for o in offensives
    ])

    cost_matrix = build_cost_matrix(defensives, offensives, hvt, config)

    method = config["assignment"]["method"]
    if method == "hungarian":
        assignments = hungarian_assignment(cost_matrix)
    elif method == "greedy":
        assignments = greedy_assignment(cost_matrix)
    else:
        raise ValueError(f"Unknown assignment method: {method}")

    # V4: 保证全覆盖 — 每个存活进攻方至少一架拦截器, 多余拦截器围攻高威胁目标
    assignments = ensure_full_coverage(
        assignments, defensives, offensives, hvt, config, threat_scores)

    return assignments, cost_matrix, threat_scores


def ensure_full_coverage(assignments, defensives, offensives, hvt, config,
                         threat_scores=None):
    """
    保证全覆盖: 每个存活进攻方至少一架拦截器分配

    规则:
      1. 如果存活进攻方未被任何拦截器覆盖, 优先分配最近的空闲拦截器
      2. 如果没有空闲拦截器, 则从已分配到死亡目标的拦截器中抢夺
      3. 多余的拦截器(比进攻方多)分配给威胁度最高的存活进攻方

    Args:
        assignments: dict {def_idx: off_idx} — 初始分配结果
        defensives: list of Aircraft
        offensives: list of Aircraft
        hvt: HVT
        config: 环境配置
        threat_scores: ndarray or None

    Returns:
        assignments: dict {def_idx: off_idx} — 修正后的全覆盖分配
    """
    n_def = len(defensives)
    n_off = len(offensives)
    max_dist = config["map_size"] * 2.0

    # 存活进攻方集合
    alive_off = set(j for j in range(n_off) if offensives[j].alive)
    # 存活拦截器集合
    alive_def = set(i for i in range(n_def) if defensives[i].alive)

    if not alive_off or not alive_def:
        return assignments

    # 当前被覆盖的进攻方
    covered_off = set()
    for d_idx, o_idx in assignments.items():
        if d_idx in alive_def and o_idx in alive_off:
            covered_off.add(o_idx)

    # 清理无效分配(拦截器或目标已死亡)
    valid_assignments = {}
    for d_idx, o_idx in assignments.items():
        if d_idx in alive_def and o_idx in alive_off:
            valid_assignments[d_idx] = o_idx
    assignments = valid_assignments

    # 未被覆盖的进攻方
    uncovered_off = alive_off - covered_off

    # 未分配的拦截器
    assigned_def = set(assignments.keys())
    unassigned_def = alive_def - assigned_def

    # --- 第一步: 将空闲拦截器分配给未覆盖的进攻方 ---
    if uncovered_off and unassigned_def:
        # 按威胁度排序未覆盖进攻方(高威胁优先)
        if threat_scores is not None:
            sorted_uncovered = sorted(uncovered_off,
                                      key=lambda j: -threat_scores[j])
        else:
            sorted_uncovered = sorted(uncovered_off,
                                      key=lambda j: -compute_threat_score(
                                          offensives[j], hvt, max_dist))

        for off_idx in sorted_uncovered:
            if not unassigned_def:
                break
            # 找最近的空闲拦截器
            best_def = min(unassigned_def,
                           key=lambda d: defensives[d].distance_3d(offensives[off_idx]))
            assignments[best_def] = off_idx
            unassigned_def.discard(best_def)
            uncovered_off.discard(off_idx)
            covered_off.add(off_idx)

    # --- 第二步: 仍有未覆盖进攻方, 从多余覆盖中抢拦截器 ---
    if uncovered_off:
        # 统计每个进攻方被几架拦截器覆盖
        coverage_count = {}
        for d_idx, o_idx in assignments.items():
            coverage_count[o_idx] = coverage_count.get(o_idx, 0) + 1

        # 从覆盖数>1的进攻方中抢拦截器
        for off_idx in sorted(uncovered_off):
            # 找可以抢的: 被覆盖数>1的进攻方对应的拦截器(取最远的)
            stealable = []
            for d_idx, o_idx in list(assignments.items()):
                if coverage_count.get(o_idx, 0) > 1:
                    stealable.append((d_idx, o_idx))
            if not stealable:
                break
            # 选离新目标最近的可抢拦截器
            best_steal = min(stealable,
                             key=lambda x: defensives[x[0]].distance_3d(
                                 offensives[off_idx]))
            steal_d, steal_from_o = best_steal
            assignments[steal_d] = off_idx
            coverage_count[steal_from_o] -= 1
            coverage_count[off_idx] = coverage_count.get(off_idx, 0) + 1
            uncovered_off.discard(off_idx)

    # --- 第三步: 多余拦截器分配给威胁度最高的进攻方(围攻) ---
    assigned_def = set(assignments.keys())
    remaining_def = alive_def - assigned_def
    if remaining_def and alive_off:
        if threat_scores is not None:
            sorted_alive = sorted(alive_off, key=lambda j: -threat_scores[j])
        else:
            sorted_alive = sorted(alive_off,
                                  key=lambda j: -compute_threat_score(
                                      offensives[j], hvt, max_dist))

        for d_idx in remaining_def:
            # 分配给威胁最高的(允许多机围攻同一目标)
            best_off = min(sorted_alive,
                           key=lambda j: defensives[d_idx].distance_3d(
                               offensives[j]))
            assignments[d_idx] = best_off

    return assignments

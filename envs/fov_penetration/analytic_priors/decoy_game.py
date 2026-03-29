"""
Module 2b: Decoy Game — 主动诱饵博弈模块 (V22)
====================================================
替代旧的 assignment_mismatch 模块。

基于事件触发式 FOV 锁定, 建模进攻集群中的诱饵-突防博弈。

核心量:
  (1) s_ij        — 视场占用  sigmoid(k_q · q_ij)
  (2) eta_ij      — 锁定吸引强度 (s, rho, V_c 加权)
  (3) p_lock_ij   — soft lock 概率 (softmax over i for each j)
  (4) U_i_decoy   — 个体诱饵价值
  (5) pi_i_{role} — 角色倾向 (decoy / penetrate / stealth)
  (6) Phi_decoy   — 群体势函数 (用于势函数增量奖励)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional

G = 9.81


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + np.exp(-x))
    ex = np.exp(x)
    return ex / (1.0 + ex)


def _softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Numerically stable softmax."""
    x_scaled = x / max(temperature, 1e-8)
    x_shifted = x_scaled - np.max(x_scaled)
    e = np.exp(x_shifted)
    return e / (np.sum(e) + 1e-12)


def _closing_speed(off, defn) -> float:
    """Compute closing speed V_c between interceptor j and attacker i."""
    dx = off.x - defn.x
    dy = off.y - defn.y
    dz = off.z - defn.z
    rho = np.sqrt(dx**2 + dy**2 + dz**2)
    if rho < 1e-6:
        return 0.0

    cos_ga = np.cos(off.gamma)
    vx_a = off.v * cos_ga * np.cos(off.heading)
    vy_a = off.v * cos_ga * np.sin(off.heading)
    vz_a = off.v * np.sin(off.gamma)

    cos_gd = np.cos(defn.gamma)
    vx_d = defn.v * cos_gd * np.cos(defn.heading)
    vy_d = defn.v * cos_gd * np.sin(defn.heading)
    vz_d = defn.v * np.sin(defn.gamma)

    dvx = vx_a - vx_d
    dvy = vy_a - vy_d
    dvz = vz_a - vz_d

    rx, ry, rz = dx / rho, dy / rho, dz / rho
    return float(-(rx * dvx + ry * dvy + rz * dvz))


def _off_axis_angle(defn, off) -> float:
    """Off-axis angle from interceptor j's bore to target i (rad)."""
    dx = off.x - defn.x
    dy = off.y - defn.y
    dz = off.z - defn.z
    dist = np.sqrt(dx**2 + dy**2 + dz**2)
    if dist < 1e-6:
        return 0.0
    cg = np.cos(defn.gamma)
    bx = cg * np.cos(defn.heading)
    by = cg * np.sin(defn.heading)
    bz = np.sin(defn.gamma)
    cos_theta = np.clip((bx * dx + by * dy + bz * dz) / dist, -1.0, 1.0)
    return float(np.arccos(cos_theta))


# -----------------------------------------------------------------------
# (1) 视场占用函数 s_ij
# -----------------------------------------------------------------------

def compute_fov_occupancy(off, defn, fov_half: float,
                          k_q: float = 8.0) -> Tuple[float, float]:
    """
    q_ij: 归一化视场偏差 = 1 - theta_ij / fov_half (在视场内>0)
    s_ij: sigmoid(k_q * q_ij)

    Returns: (q_ij, s_ij)
    """
    theta = _off_axis_angle(defn, off)
    q_ij = 1.0 - theta / max(fov_half, 1e-6)
    s_ij = _sigmoid(k_q * q_ij)
    return float(q_ij), float(s_ij)


# -----------------------------------------------------------------------
# (2) 锁定吸引强度 eta_ij
# -----------------------------------------------------------------------

def compute_lock_attraction(
    off, defn,
    s_ij: float,
    rho_ref: float = 3000.0,
    vc_ref: float = 120.0,
    w_s: float = 0.4,
    w_rho: float = 0.3,
    w_vc: float = 0.3,
) -> float:
    """
    eta_ij = w_s * s_ij + w_rho * (1 - rho/rho_ref) + w_vc * (V_c/vc_ref)

    越大 → 拦截器越容易锁定该目标
    """
    dx = off.x - defn.x
    dy = off.y - defn.y
    dz = off.z - defn.z
    rho = np.sqrt(dx**2 + dy**2 + dz**2)
    rho_norm = np.clip(rho / max(rho_ref, 1.0), 0.0, 2.0)

    V_c = _closing_speed(off, defn)
    vc_norm = np.clip(V_c / max(vc_ref, 1.0), -0.5, 1.5)

    eta = w_s * s_ij + w_rho * (1.0 - rho_norm) + w_vc * vc_norm
    return float(np.clip(eta, -1.0, 2.0))


# -----------------------------------------------------------------------
# (3) soft lock 概率 p_lock_ij
# -----------------------------------------------------------------------

def compute_soft_lock_probs(
    eta_matrix: np.ndarray,
    alive_off: List[bool],
    alive_def: List[bool],
    temperature: float = 5.0,
) -> np.ndarray:
    """
    对每个拦截器 j: p_lock_ij = softmax_i(eta_ij / tau)

    Returns: (n_off, n_def) matrix of p_lock_ij
    """
    n_off, n_def = eta_matrix.shape
    p_lock = np.zeros((n_off, n_def))

    for j in range(n_def):
        if not alive_def[j]:
            continue
        eta_col = np.full(n_off, -100.0)
        for i in range(n_off):
            if alive_off[i]:
                eta_col[i] = eta_matrix[i, j]
        p_lock[:, j] = _softmax(eta_col, temperature)

    return p_lock


# -----------------------------------------------------------------------
# (4) 个体诱饵价值 U_i_decoy
# -----------------------------------------------------------------------

def compute_individual_decoy_value(
    off_idx: int,
    offensives: list,
    defensives: list,
    p_lock: np.ndarray,
    locked_by_map: Dict[int, list],
    hvt,
    config: dict,
    decoy_cfg: dict,
) -> Tuple[float, Dict]:
    """
    U_i_decoy = -w_self * self_cost + w_attn * attention_benefit + w_team * team_benefit

    self_cost: 被锁定的风险 (有多少拦截器盯着自己)
    attention_benefit: 自己吸引拦截器注意力, 减轻队友被锁定
    team_benefit: 因为自己当诱饵, 队友离 HVT 更近 / 更安全
    """
    off = offensives[off_idx]
    if not off.alive:
        return 0.0, {}

    n_off = len(offensives)
    n_def = len(defensives)
    w_self = decoy_cfg.get("self_cost_weight", 0.3)
    w_attn = decoy_cfg.get("attention_benefit_weight", 0.5)
    w_team = decoy_cfg.get("team_penetration_benefit_weight", 0.2)
    obs_range = max(config.get("obs_range", 5000.0), 1.0)

    # --- self_cost: 被锁定概率之和 ---
    lock_pressure = 0.0
    for j in range(n_def):
        if defensives[j].alive:
            lock_pressure += p_lock[off_idx, j]
    # 加上实际锁定
    actual_lock_count = len(locked_by_map.get(off_idx, []))
    self_cost = lock_pressure + 2.0 * actual_lock_count

    # --- attention_benefit: 自己吸引的锁定概率总量 ---
    attention = lock_pressure  # 自己吸引的注意力

    # --- team_benefit: 队友因此获得的突防机会 ---
    # 简化: 对每个队友, 其被锁定概率如果因为自己的高锁定吸引而降低
    # 近似: 队友的 min_dist_to_hvt 加权
    team_benefit = 0.0
    n_alive_teammates = 0
    for i, o in enumerate(offensives):
        if i == off_idx or not o.alive or o.hit_hvt:
            continue
        n_alive_teammates += 1
        d_hvt = o.distance_to(hvt.x, hvt.y, hvt.z)
        proximity = 1.0 - np.clip(d_hvt / obs_range, 0.0, 1.0)
        # 队友被锁定少 → benefit 高
        mate_lock_prob = sum(p_lock[i, j] for j in range(n_def) if defensives[j].alive)
        freed_pressure = max(0.0, attention - mate_lock_prob)
        team_benefit += proximity * freed_pressure

    if n_alive_teammates > 0:
        team_benefit /= n_alive_teammates

    U_decoy = -w_self * self_cost + w_attn * attention + w_team * team_benefit

    detail = {
        "self_cost": self_cost,
        "attention_benefit": attention,
        "team_benefit": team_benefit,
        "actual_lock_count": actual_lock_count,
    }
    return float(U_decoy), detail


# -----------------------------------------------------------------------
# (5) 角色倾向
# -----------------------------------------------------------------------

def compute_role_scores(
    off_idx: int,
    offensives: list,
    defensives: list,
    p_lock: np.ndarray,
    locked_by_map: Dict[int, list],
    hvt,
    config: dict,
    role_temperature: float = 2.0,
) -> Tuple[float, float, float]:
    """
    计算 soft 角色倾向: (pi_decoy, pi_penetrate, pi_stealth)
    基于当前状态自动推断, 不做硬分配。

    Returns: (pi_decoy, pi_penetrate, pi_stealth) 归一化到 [0,1] 和为1
    """
    off = offensives[off_idx]
    if not off.alive:
        return 0.0, 0.0, 0.0

    n_def = len(defensives)
    obs_range = max(config.get("obs_range", 5000.0), 1.0)

    # Decoy score: 被多少拦截器注意 / 锁定
    lock_pressure = sum(p_lock[off_idx, j] for j in range(n_def) if defensives[j].alive)
    actual_locks = len(locked_by_map.get(off_idx, []))
    decoy_raw = lock_pressure + 2.0 * actual_locks

    # Penetrate score: 离 HVT 近 + 被锁定少
    d_hvt = off.distance_to(hvt.x, hvt.y, hvt.z)
    proximity = 1.0 - np.clip(d_hvt / obs_range, 0.0, 1.0)
    penetrate_raw = proximity * max(0.1, 1.0 - lock_pressure)

    # Stealth score: 不被探测 + 不被锁定
    detected_cost = 1.0 if off.detected else 0.0
    stealth_raw = max(0.0, 1.0 - lock_pressure - detected_cost)

    # Softmax
    scores = np.array([decoy_raw, penetrate_raw, stealth_raw])
    probs = _softmax(scores, role_temperature)
    return float(probs[0]), float(probs[1]), float(probs[2])


# -----------------------------------------------------------------------
# (6) 群体势函数 Phi_decoy
# -----------------------------------------------------------------------

def compute_group_decoy_potential(
    offensives: list,
    defensives: list,
    p_lock: np.ndarray,
    locked_by_map: Dict[int, list],
    U_decoy_per_agent: List[float],
    hvt,
    config: dict,
    decoy_cfg: dict,
) -> float:
    """
    Phi_decoy = sum_i [ U_i_decoy * w_i ]

    w_i 权重: 离 HVT 更近的 agent 权重更高 (突防价值更大)

    体现:
      - 有人主动暴露带来的群体收益
      - 诱饵的局部代价
      - 其他成员威胁下降带来的收益
    """
    obs_range = max(config.get("obs_range", 5000.0), 1.0)
    phi = 0.0
    total_weight = 0.0

    for i, off in enumerate(offensives):
        if not off.alive or off.hit_hvt:
            continue
        d_hvt = off.distance_to(hvt.x, hvt.y, hvt.z)
        w_i = 1.0 + (1.0 - np.clip(d_hvt / obs_range, 0.0, 1.0))
        phi += U_decoy_per_agent[i] * w_i
        total_weight += w_i

    if total_weight > 0:
        phi /= total_weight

    return float(phi)


# -----------------------------------------------------------------------
# 主接口
# -----------------------------------------------------------------------

def compute_decoy_game(
    offensives: list,
    defensives: list,
    hvt,
    config: dict,
    ap_config: dict,
    locked_by_map: Dict[int, list],
    prev_Phi_decoy: Optional[float] = None,
) -> Tuple[float, List[float], float, Dict]:
    """
    计算诱饵博弈全部量。

    Args:
        offensives: list of offensive Aircraft
        defensives: list of defensive Aircraft
        hvt: HVT
        config: env config
        ap_config: analytic_priors config
        locked_by_map: {off_idx: [def_idx, ...]}
        prev_Phi_decoy: 上一步的 Phi_decoy (用于势函数增量)

    Returns:
        decoy_reward:     scalar  phi_weight * (Phi(t) - Phi(t-1))
        per_agent_reward: per-agent decoy reward
        Phi_decoy:        current potential (to cache)
        info:             logging dict
    """
    decoy_cfg = ap_config.get("decoy_game", {})
    fov_half = config["fov_half_angle"]
    k_q = decoy_cfg.get("k_q_sigmoid", 8.0)
    w_s = decoy_cfg.get("eta_w_s", 0.4)
    w_rho = decoy_cfg.get("eta_w_rho", 0.3)
    w_vc = decoy_cfg.get("eta_w_vc", 0.3)
    lock_temp = decoy_cfg.get("lock_prob_temperature", 5.0)
    role_temp = decoy_cfg.get("role_temperature", 2.0)
    phi_weight = decoy_cfg.get("phi_decoy_weight", 0.5)
    rho_ref = config.get("obs_range", 5000.0)
    vc_ref = config.get("vel_range", 120.0)

    n_off = len(offensives)
    n_def = len(defensives)
    alive_off = [o.alive and not o.hit_hvt for o in offensives]
    alive_def = [d.alive for d in defensives]

    # --- (1) s_ij 矩阵 ---
    s_matrix = np.zeros((n_off, n_def))
    q_matrix = np.zeros((n_off, n_def))
    for i, off in enumerate(offensives):
        if not alive_off[i]:
            continue
        for j, defn in enumerate(defensives):
            if not alive_def[j]:
                continue
            q_ij, s_ij = compute_fov_occupancy(off, defn, fov_half, k_q)
            q_matrix[i, j] = q_ij
            s_matrix[i, j] = s_ij

    # --- (2) eta_ij 矩阵 ---
    eta_matrix = np.zeros((n_off, n_def))
    for i, off in enumerate(offensives):
        if not alive_off[i]:
            continue
        for j, defn in enumerate(defensives):
            if not alive_def[j]:
                continue
            eta_matrix[i, j] = compute_lock_attraction(
                off, defn, s_matrix[i, j],
                rho_ref=rho_ref, vc_ref=vc_ref,
                w_s=w_s, w_rho=w_rho, w_vc=w_vc)

    # --- (3) p_lock_ij ---
    p_lock = compute_soft_lock_probs(eta_matrix, alive_off, alive_def, lock_temp)

    # --- (4) U_i_decoy ---
    U_decoy_list = []
    decoy_details = []
    for i in range(n_off):
        if alive_off[i]:
            u, detail = compute_individual_decoy_value(
                i, offensives, defensives, p_lock, locked_by_map,
                hvt, config, decoy_cfg)
        else:
            u, detail = 0.0, {}
        U_decoy_list.append(u)
        decoy_details.append(detail)

    # --- (5) 角色倾向 ---
    role_decoy = []
    role_penetrate = []
    role_stealth = []
    for i in range(n_off):
        if alive_off[i]:
            d, p, s = compute_role_scores(
                i, offensives, defensives, p_lock, locked_by_map,
                hvt, config, role_temp)
        else:
            d, p, s = 0.0, 0.0, 0.0
        role_decoy.append(d)
        role_penetrate.append(p)
        role_stealth.append(s)

    # --- (6) Phi_decoy ---
    Phi_decoy = compute_group_decoy_potential(
        offensives, defensives, p_lock, locked_by_map,
        U_decoy_list, hvt, config, decoy_cfg)

    # --- 势函数增量奖励 ---
    if prev_Phi_decoy is not None:
        delta_Phi = Phi_decoy - prev_Phi_decoy
    else:
        delta_Phi = 0.0
    decoy_reward = phi_weight * delta_Phi

    # Per-agent reward: 个体贡献 + 团队共享
    per_agent_reward = [0.0] * n_off
    n_alive = max(sum(1 for a in alive_off if a), 1)
    ind_w = ap_config.get("cooperative_decoy", {}).get("individual_weight", 0.6) \
        if isinstance(ap_config.get("cooperative_decoy"), dict) else 0.6
    team_w = 1.0 - ind_w
    team_share = decoy_reward * team_w / n_alive

    total_u = sum(abs(U_decoy_list[i]) for i in range(n_off) if alive_off[i]) + 1e-8
    for i in range(n_off):
        if alive_off[i]:
            ind_share = decoy_reward * ind_w * abs(U_decoy_list[i]) / total_u
            per_agent_reward[i] = ind_share + team_share

    # --- info ---
    # Per-agent lock counts for obs
    locked_by_count = [len(locked_by_map.get(i, [])) for i in range(n_off)]
    lock_pressure_per_agent = [
        sum(p_lock[i, j] for j in range(n_def) if alive_def[j])
        for i in range(n_off)
    ]

    info = {
        "Phi_decoy": Phi_decoy,
        "delta_Phi_decoy": delta_Phi,
        "decoy_reward": decoy_reward,
        "U_decoy_per_agent": U_decoy_list,
        "role_decoy_per_agent": role_decoy,
        "role_penetrate_per_agent": role_penetrate,
        "role_stealth_per_agent": role_stealth,
        "lock_pressure_per_agent": lock_pressure_per_agent,
        "locked_by_count_per_agent": locked_by_count,
        "p_lock_matrix": p_lock,
        "eta_matrix_mean": float(np.mean(eta_matrix[np.array(alive_off)][:, np.array(alive_def)]))
            if any(alive_off) and any(alive_def) else 0.0,
        "s_matrix_mean": float(np.mean(s_matrix[np.array(alive_off)][:, np.array(alive_def)]))
            if any(alive_off) and any(alive_def) else 0.0,
    }
    return decoy_reward, per_agent_reward, Phi_decoy, info

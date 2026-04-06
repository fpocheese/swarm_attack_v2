"""FOV Penetration Environment - Reward & Cost V28
===================================================
V28 根据 swarmrl_reward_obser.md 全面重构:
  - 删除旧的 milestone / closest / proximity / 碎片 shaping
  - 四层奖励: task + game + escape - risk
  - 所有 cost 并入 reward 惩罚项 (MAPPO, 无独立 cost 通道)
  - 终端奖励: N_eff / N_hit / synergy / loss / waste
"""

import numpy as np
from .config import G


# ======================================================================
# Helper: sigmoid
# ======================================================================
def _sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + np.exp(-x))
    ex = np.exp(x)
    return ex / (1.0 + ex)


# ======================================================================
# Helper: velocity vector
# ======================================================================
def _vel3d(entity):
    cg = np.cos(entity.gamma)
    return np.array([
        entity.v * cg * np.cos(entity.heading),
        entity.v * cg * np.sin(entity.heading),
        entity.v * np.sin(entity.gamma),
    ])


# ======================================================================
# Helper: closing speed (positive = approaching)
# ======================================================================
def _closing_speed_to_point(off, px, py, pz):
    """进攻方对定点的闭合速度 V_{c,iH} = -r_iH^T v_i / |r_iH|"""
    r = np.array([px - off.x, py - off.y, pz - off.z])
    rho = np.linalg.norm(r)
    if rho < 1e-6:
        return 0.0
    v = _vel3d(off)
    return float(-np.dot(r, v) / rho)    # 注意: r_iH = p_H - p_i


def _closing_speed_to_point_correct(off, px, py, pz):
    """V_{c,iH} = -r_{iH}^T v_i / |r_{iH}|, where r_{iH} = p_H - p_i"""
    r_iH = np.array([px - off.x, py - off.y, pz - off.z])
    rho = np.linalg.norm(r_iH)
    if rho < 1e-6:
        return 0.0
    v_i = _vel3d(off)
    # 注意: 按照文档定义, 闭合速度 = - r_iH^T v_i / |r_iH|
    # 当 v_i 朝向 HVT 时, r_iH 方向与 v_i 大致相同 → dot > 0 → Vc 为负
    # 实际上应该是: Vc = +r_iH^T v_i / |r_iH| (v_i 朝向目标时 Vc > 0)
    return float(np.dot(r_iH, v_i) / rho)


# ======================================================================
# Main: compute_rewards (V28 全面重构)
# ======================================================================
def compute_rewards(offensives, defensives, hvt, config,
                    prev_dists_to_hvt, hit_events, current_step,
                    just_killed=None, just_killed_def=None,
                    lock_on_map=None, prev_team_min_dist=None,
                    escape_events=None, miss_events=None,
                    defensive_policies=None,
                    # V28 新参数: analytic priors 传入
                    ap_data=None):
    """
    V28 奖励计算.

    ap_data: dict, 由 env.step 中的 analytic priors 模块计算后传入, 包含:
        P_pen_per_agent:  list[float]   个体突防概率
        P_hit_per_agent:  list[float]   个体点目标命中可行度
        E_esc_per_agent:  list[float]   个体近距逃逸能力
        U_decoy_per_agent: list[float]  个体诱饵价值
        Phi_decoy:         float        群体诱饵势函数
        prev_Phi_decoy:    float        上步群体诱饵势函数
        Z_tilde_per_agent: list[float]  聚合脱锥风险
        cone_cost_per_agent: list[float] 单机锥风险成本
        locked_target_by_defender: dict  拦截器锁定映射
    """
    rc = config["reward"]
    n_off = len(offensives)
    n_def = len(defensives)
    obs_range = config["obs_range"]
    rewards = [0.0] * n_off
    reward_info = {}

    if just_killed is None:
        just_killed = [False] * n_off
    if just_killed_def is None:
        just_killed_def = [False] * n_def
    if escape_events is None:
        escape_events = []
    if miss_events is None:
        miss_events = []
    if ap_data is None:
        ap_data = {}

    hvt_x, hvt_y, hvt_z = hvt.x, hvt.y, hvt.z

    # 当前各架到 HVT 的距离
    cur_dists = []
    for i, off in enumerate(offensives):
        if off.alive and not off.hit_hvt:
            cur_dists.append(off.distance_to(hvt_x, hvt_y, hvt_z))
        else:
            cur_dists.append(float('inf'))

    # ==========================================================
    # 第一类: 任务主线奖励 r_task
    # ==========================================================

    # --- 3.1.1 reward_penetration: 鼓励有效突防能力提升 ---
    lambda_P = rc.get("lambda_penetration", 2.0)
    P_pen_list = ap_data.get("P_pen_per_agent", [0.0] * n_off)
    total_pen_reward = 0.0
    for i, off in enumerate(offensives):
        if off.alive and not off.hit_hvt:
            r_pen = lambda_P * P_pen_list[i]
            rewards[i] += r_pen
            total_pen_reward += r_pen
    reward_info["reward_penetration"] = total_pen_reward

    # --- 3.1.2 reward_hit_geometry: 鼓励对HVT点目标收敛命中几何 ---
    lambda_rho = rc.get("lambda_hit_approach", 8.0)
    lambda_c = rc.get("lambda_hit_closing", 1.5)
    lambda_omega = rc.get("lambda_hit_los_rate", 0.5)
    total_hit_geo = 0.0
    # V33: 近距指数放大参数
    close_range_threshold = rc.get("close_range_threshold", 500.0)
    close_range_max_mult = rc.get("close_range_max_multiplier", 20.0)
    for i, off in enumerate(offensives):
        if off.alive and not off.hit_hvt and prev_dists_to_hvt[i] < float('inf'):
            # (a) 距 HVT 更近
            delta_rho = prev_dists_to_hvt[i] - cur_dists[i]
            init_dist = max(config.get("obs_range", 2500.0), 1.0)
            approach_r = lambda_rho * delta_rho / init_dist
            # V33: 近距指数放大 — 越近HVT奖励越强
            # 500m以内开始指数放大, 50m以内达到最大倍率
            d = cur_dists[i]
            if d < close_range_threshold:
                # 指数增长: d从500→0时, 倍率从1→close_range_max_mult
                t = max(1.0 - d / close_range_threshold, 0.0)
                multiplier = 1.0 + (close_range_max_mult - 1.0) * (t ** 2)
            else:
                # 远距: 原有线性放大
                dist_ratio = max(1.0 - d / init_dist, 0.0)
                multiplier = 1.0 + 3.0 * dist_ratio
            approach_r *= multiplier

            # (b) 对 HVT 闭合速度为正
            Vc_iH = _closing_speed_to_point_correct(off, hvt_x, hvt_y, hvt_z)
            closing_r = lambda_c * max(Vc_iH, 0.0) / max(config.get("vel_range", 120.0), 1.0)
            # V33: 近距也放大闭合速度奖励
            if d < close_range_threshold:
                closing_r *= (1.0 + 3.0 * max(1.0 - d / close_range_threshold, 0.0))

            # (c) HVT LOS 角速度越小越好 (命中几何更稳定)
            omega_los = ap_data.get("hvt_omega_los_per_agent", [0.0] * n_off)[i]
            los_penalty = lambda_omega * abs(omega_los)

            hit_geo = approach_r + closing_r - los_penalty
            rewards[i] += hit_geo
            total_hit_geo += hit_geo
    reward_info["reward_hit_geometry"] = total_hit_geo

    # --- 3.1.3 reward_no_retreat: 惩罚远离HVT/回头跑 ---
    lambda_ret = rc.get("lambda_no_retreat", 3.0)
    total_retreat_pen = 0.0
    for i, off in enumerate(offensives):
        if off.alive and not off.hit_hvt:
            Vc_iH = _closing_speed_to_point_correct(off, hvt_x, hvt_y, hvt_z)
            if Vc_iH < 0:
                retreat_pen = -lambda_ret * abs(Vc_iH) / max(config.get("vel_range", 120.0), 1.0)
                rewards[i] += retreat_pen
                total_retreat_pen += retreat_pen
    reward_info["reward_no_retreat"] = total_retreat_pen

    # ==========================================================
    # 第二类: 主动诱饵博弈奖励 r_game
    # ==========================================================

    # --- 3.2.1 reward_decoy_value: 鼓励主动诱饵 ---
    lambda_D = rc.get("lambda_decoy_value", 0.5)
    U_decoy_list = ap_data.get("U_decoy_per_agent", [0.0] * n_off)
    total_decoy_val = 0.0
    for i, off in enumerate(offensives):
        if off.alive and not off.hit_hvt:
            r_decoy = lambda_D * U_decoy_list[i]
            rewards[i] += r_decoy
            total_decoy_val += r_decoy
    reward_info["reward_decoy_value"] = total_decoy_val

    # --- 3.2.2 reward_decoy_potential: 鼓励群体势函数沿有利方向变化 ---
    lambda_Phi = rc.get("lambda_decoy_potential", 1.0)
    Phi_decoy = ap_data.get("Phi_decoy", 0.0)
    prev_Phi = ap_data.get("prev_Phi_decoy", None)
    total_decoy_pot = 0.0
    if prev_Phi is not None:
        delta_Phi = Phi_decoy - prev_Phi
        n_alive = max(sum(1 for o in offensives if o.alive and not o.hit_hvt), 1)
        pot_r = lambda_Phi * delta_Phi / n_alive
        for i, off in enumerate(offensives):
            if off.alive and not off.hit_hvt:
                rewards[i] += pot_r
                total_decoy_pot += pot_r
    reward_info["reward_decoy_potential"] = total_decoy_pot

    # --- 3.2.3 reward_attention_redirect: 鼓励敌方锁定落在诱饵上 ---
    lambda_R = rc.get("lambda_attention_redirect", 0.3)
    locked_target_by_defender = ap_data.get("locked_target_by_defender", {})
    total_attn_redirect = 0.0
    if lambda_R > 0 and locked_target_by_defender:
        for j, off_idx in locked_target_by_defender.items():
            if off_idx is not None and off_idx < n_off:
                # 被锁定的是诱饵价值高的 → 好
                u_decoy_locked = U_decoy_list[off_idx]
                u_pen_locked = P_pen_list[off_idx]
                redirect_r = lambda_R * (u_decoy_locked - u_pen_locked)
                # 全队共享
                n_alive = max(sum(1 for o in offensives if o.alive and not o.hit_hvt), 1)
                per_r = redirect_r / n_alive
                for i, off in enumerate(offensives):
                    if off.alive and not off.hit_hvt:
                        rewards[i] += per_r
                        total_attn_redirect += per_r
    reward_info["reward_attention_redirect"] = total_attn_redirect

    # ==========================================================
    # 第三类: 局部逃逸奖励 r_escape
    # ==========================================================

    # --- 3.3.1 reward_escape: 鼓励近距甩脱拦截器 ---
    lambda_E = rc.get("lambda_escape", 0.5)
    E_esc_list = ap_data.get("E_esc_per_agent", [0.0] * n_off)
    total_escape = 0.0
    for i, off in enumerate(offensives):
        if off.alive and not off.hit_hvt:
            r_esc = lambda_E * E_esc_list[i]
            rewards[i] += r_esc
            total_escape += r_esc
    reward_info["reward_escape"] = total_escape

    # --- 3.3.2 reward_escape_progress: optional, 逃逸能力提升增量 ---
    lambda_dE = rc.get("lambda_escape_progress", 0.3)
    prev_E_esc_list = ap_data.get("prev_E_esc_per_agent", None)
    total_esc_prog = 0.0
    if prev_E_esc_list is not None and lambda_dE > 0:
        for i, off in enumerate(offensives):
            if off.alive and not off.hit_hvt:
                delta_E = E_esc_list[i] - prev_E_esc_list[i]
                r_prog = lambda_dE * delta_E
                rewards[i] += r_prog
                total_esc_prog += r_prog
    reward_info["reward_escape_progress"] = total_esc_prog

    # ==========================================================
    # 第四类: 风险惩罚 r_risk (原 cost 全部并入)
    # ==========================================================

    # --- 3.4.1 penalty_cone: 多拦截器未来终端暴露风险 ---
    lambda_cone = rc.get("lambda_penalty_cone", 0.5)
    cone_cost_list = ap_data.get("cone_cost_per_agent", [0.0] * n_off)
    total_cone_pen = 0.0
    for i, off in enumerate(offensives):
        if off.alive and not off.hit_hvt:
            pen = lambda_cone * cone_cost_list[i]
            rewards[i] -= pen
            total_cone_pen += pen
    reward_info["penalty_cone"] = total_cone_pen

    # --- 3.4.2 penalty_fov: 当前被敌方视场覆盖 ---
    lambda_fov = rc.get("lambda_penalty_fov", 0.1)
    fov_half = config["fov_half_angle"]
    det_range = config["detection_range"]
    total_fov_pen = 0.0
    for i, off in enumerate(offensives):
        if not off.alive:
            continue
        c_fov = 0.0
        for d in defensives:
            if d.alive and d.is_in_fov(off.x, off.y, off.z, fov_half, det_range):
                c_fov += 1.0
        pen = lambda_fov * c_fov
        rewards[i] -= pen
        total_fov_pen += pen
    reward_info["penalty_fov"] = total_fov_pen

    # --- 3.4.3 penalty_danger: 进入拦截器危险近距区 ---
    lambda_danger = rc.get("lambda_penalty_danger", 0.3)
    danger_radius = rc.get("danger_radius", 300.0)
    total_danger_pen = 0.0
    for i, off in enumerate(offensives):
        if not off.alive:
            continue
        c_danger = 0.0
        for d in defensives:
            if d.alive:
                dist = off.distance_3d(d)
                if dist < danger_radius:
                    c_danger += 1.0 - dist / danger_radius
        pen = lambda_danger * c_danger
        rewards[i] -= pen
        total_danger_pen += pen
    reward_info["penalty_danger"] = total_danger_pen

    # --- 3.4.4 penalty_boundary: 越界 ---
    lambda_boundary = rc.get("lambda_penalty_boundary", 2.0)
    map_size = config["map_size"]
    total_boundary_pen = 0.0
    for i, off in enumerate(offensives):
        if not off.alive:
            continue
        if abs(off.x) > map_size * 0.9 or abs(off.y) > map_size * 0.9:
            rewards[i] -= lambda_boundary
            total_boundary_pen += lambda_boundary
    reward_info["penalty_boundary"] = total_boundary_pen

    # --- 3.4.5 penalty_ground: 不合理低空/贴地 ---
    lambda_ground = rc.get("lambda_penalty_ground", 1.0)
    z_low = rc.get("ground_z_low", 20.0)
    d_near = rc.get("ground_d_near", 500.0)
    total_ground_pen = 0.0
    for i, off in enumerate(offensives):
        if not off.alive:
            continue
        # 联合条件: 远离 HVT 且低空 → 惩罚 (区分终端合理低空 vs 无意义贴地逃逸)
        rho_iH = cur_dists[i]
        if rho_iH > d_near and off.z < z_low:
            pen = lambda_ground * (1.0 - off.z / max(z_low, 1.0))
            rewards[i] -= pen
            total_ground_pen += pen
        # 极低高度惩罚 (无条件)
        z_min = config.get("z_min", 0.0)
        z_min_safe = max(z_min * 2.0, 5.0)
        if off.z < z_min_safe:
            ratio = (z_min_safe - off.z) / max(z_min_safe, 1.0)
            pen = lambda_ground * 2.0 * ratio * ratio
            rewards[i] -= pen
            total_ground_pen += pen
    reward_info["penalty_ground"] = total_ground_pen

    # --- 3.4.6 penalty_collision: 友机碰撞 ---
    lambda_collision = rc.get("lambda_penalty_collision", 2.0)
    collision_range = config.get("collision_range", 5.0)
    total_collision_pen = 0.0
    alive_idx = [i for i, off in enumerate(offensives) if off.alive and not off.hit_hvt]
    for ii in range(len(alive_idx)):
        for jj in range(ii + 1, len(alive_idx)):
            d_ij = offensives[alive_idx[ii]].distance_3d(offensives[alive_idx[jj]])
            if d_ij < collision_range * 4:  # 20m 内
                pen = lambda_collision * max(0.0, 1.0 - d_ij / (collision_range * 4))
                rewards[alive_idx[ii]] -= pen
                rewards[alive_idx[jj]] -= pen
                total_collision_pen += 2 * pen
    reward_info["penalty_collision"] = total_collision_pen

    # ==========================================================
    # 被击杀惩罚 (保留, 不属于四类但对训练必要)
    # ==========================================================
    killed_pen = rc.get("killed_penalty", -3.0)
    for i in range(n_off):
        if just_killed[i] and not offensives[i].hit_hvt:
            rewards[i] += killed_pen

    # ==========================================================
    # 命中 HVT — 终极目标, 全队共享
    # ==========================================================
    if hit_events:
        hit_bonus = rc.get("hit_hvt_bonus", 6000.0) * len(hit_events)
        for i in range(n_off):
            rewards[i] += hit_bonus
        reward_info["hit_hvt"] = hit_bonus

    # ==========================================================
    # 步惩罚 (微弱, 鼓励尽快完成)
    # ==========================================================
    step_pen = rc.get("step_penalty", -0.005)
    for i, off in enumerate(offensives):
        if off.alive:
            rewards[i] += step_pen

    return rewards, reward_info


# ======================================================================
# compute_costs: V28 全部并入 reward, costs 返回零
# ======================================================================
def compute_costs(offensives, defensives, config):
    """V28: 所有 cost 已并入 reward, 这里返回全零以保持接口兼容"""
    n_off = len(offensives)
    costs = [0.0] * n_off
    return costs, {}


# ======================================================================
# Terminal Reward (V28 新增)
# ======================================================================
def compute_terminal_rewards(offensives, hvt, config, ap_data=None):
    """
    终端奖励 (episode 结束时调用)

    r_T = lambda_eff * N_eff + lambda_hit * N_hit + lambda_syn * N_eff^2
          - lambda_loss * N_loss - lambda_waste * N_waste
    """
    rc = config["reward"]
    n_off = len(offensives)
    if ap_data is None:
        ap_data = {}

    # 终端统计量
    hit_threshold = config.get("point_target", {}).get("hit_threshold", 3.0)

    # N_hit: 真实命中数
    N_hit = sum(1 for off in offensives if off.hit_hvt)

    # N_loss: 总损失
    N_loss = sum(1 for off in offensives if not off.alive)

    # P_pen, P_hit 用于计算 N_eff
    P_pen_list = ap_data.get("P_pen_per_agent", [0.0] * n_off)
    P_hit_list = ap_data.get("P_hit_per_agent", [0.0] * n_off)

    # N_eff = sum(P_pen * P_hit)
    N_eff = sum(P_pen_list[i] * P_hit_list[i]
                for i in range(n_off)
                if offensives[i].alive and not offensives[i].hit_hvt)
    # 已命中的也算有效突防
    N_eff += N_hit

    # N_waste: 被毁且未命中
    N_waste = sum(1 for off in offensives if not off.alive and not off.hit_hvt)

    # 终端奖励系数
    lambda_eff = rc.get("lambda_terminal_eff", 50.0)
    lambda_hit = rc.get("lambda_terminal_hit", 200.0)
    lambda_syn = rc.get("lambda_terminal_synergy", 20.0)
    lambda_loss = rc.get("lambda_terminal_loss", 10.0)
    lambda_waste = rc.get("lambda_terminal_waste", 15.0)

    # timeout额外惩罚
    lambda_timeout = rc.get("timeout_penalty", -120.0)
    lambda_timeout_dist = rc.get("timeout_distance_penalty_coef", 800.0)

    terminal_r = (lambda_eff * N_eff
                  + lambda_hit * N_hit
                  + lambda_syn * N_eff ** 2
                  - lambda_loss * N_loss
                  - lambda_waste * N_waste)

    # 每个 agent 分配
    rewards = [terminal_r / max(n_off, 1)] * n_off

    info = {
        "terminal_N_eff": N_eff,
        "terminal_N_hit": N_hit,
        "terminal_N_loss": N_loss,
        "terminal_N_waste": N_waste,
        "terminal_reward": terminal_r,
    }
    return rewards, info

"""FOV Penetration Environment - Reward & Cost V37
===================================================
V37: 修复 '飞歪' 问题 — 强化航向修正, 抑制mu偏置

V36诊断 (飞行轨迹诊断脚本确认):
  Agent mu偏置0.04 → 航向漂移1.6°/s → 50s偏80°
  Agent mu偏置0.20 → 航向漂移7.4°/s → 60s偏444° (转了好几圈!)
  根因: cos(heading_err)小角度梯度≈0, closing奖励独大
  自由agent没有足够的航向纠正信号

V37修复:
  1. +heading_error_penalty: λ*(err/π)² (30°偏航 → -0.022/step, 强梯度)
  2. +mu_regularization: λ*|a[2]| (防止无意义转弯)
  3. heading_align: 0.2→0.5 (2.5x增强)
  4. closing: 1.5→0.8 (不再主导)
  5. close_range_threshold: 500→800 (更早放大)

V37各信号每步量级 (@1000m直飞, v=45m/s):
  approach: 0.225, heading_align: +0.500, heading_pen: 0.000
  closing: +0.300, mu_reg: 0.000, proximity: -0.100
  NET: +0.925/step (直飞最优)
 @1000m 30°偏航: NET=+0.751 (-18.8%, V36仅-14%)
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
def _closing_speed_to_point_correct(off, px, py, pz):
    r_iH = np.array([px - off.x, py - off.y, pz - off.z])
    rho = np.linalg.norm(r_iH)
    if rho < 1e-6:
        return 0.0
    v_i = _vel3d(off)
    return float(np.dot(r_iH, v_i) / rho)


# ======================================================================
# Main: compute_rewards V35 — 极简目标优先
# ======================================================================
def compute_rewards(offensives, defensives, hvt, config,
                    prev_dists_to_hvt, hit_events, current_step,
                    just_killed=None, just_killed_def=None,
                    lock_on_map=None, prev_team_min_dist=None,
                    escape_events=None, miss_events=None,
                    defensive_policies=None,
                    ap_data=None,
                    raw_actions=None):
    """
    V35 极简奖励: 只关心 '接近目标' 和 '面朝目标'.
    """
    rc = config["reward"]
    n_off = len(offensives)
    n_def = len(defensives)
    rewards = [0.0] * n_off
    reward_info = {}

    if just_killed is None:
        just_killed = [False] * n_off
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
    # 核心奖励 1: 距离减少奖励 (approach_reward)
    # ==========================================================
    # delta_rho / norm_dist * lambda, 归一化距离小 → 奖励信号大
    lambda_approach = rc.get("lambda_approach", 30.0)
    approach_norm = rc.get("approach_norm_dist", 300.0)
    close_range_threshold = rc.get("close_range_threshold", 500.0)
    close_range_max_mult = rc.get("close_range_max_multiplier", 15.0)
    total_approach = 0.0
    for i, off in enumerate(offensives):
        if off.alive and not off.hit_hvt and prev_dists_to_hvt[i] < float('inf'):
            delta_rho = prev_dists_to_hvt[i] - cur_dists[i]  # 正=接近
            r_app = lambda_approach * delta_rho / approach_norm

            # 近距放大
            d = cur_dists[i]
            if d < close_range_threshold:
                t = max(1.0 - d / close_range_threshold, 0.0)
                mult = 1.0 + (close_range_max_mult - 1.0) * (t ** 2)
            else:
                mult = 1.0
            r_app *= mult

            rewards[i] += r_app
            total_approach += r_app
    reward_info["reward_approach"] = total_approach

    # ==========================================================
    # 核心奖励 2: 航向对准HVT (heading_alignment)
    # ==========================================================
    # 机头朝向目标 → 正奖励, 背对 → 惩罚
    lambda_heading = rc.get("lambda_heading_align", 0.15)
    total_heading = 0.0
    for i, off in enumerate(offensives):
        if off.alive and not off.hit_hvt:
            dx = hvt_x - off.x
            dy = hvt_y - off.y
            bearing = np.arctan2(dy, dx)
            heading_err = bearing - off.heading
            heading_err = (heading_err + np.pi) % (2 * np.pi) - np.pi
            # cos(heading_err): 1=完全对准, -1=完全背对
            alignment = np.cos(heading_err)
            r_head = lambda_heading * alignment
            rewards[i] += r_head
            total_heading += r_head
    reward_info["reward_heading_align"] = total_heading

    # ==========================================================
    # V37新增: 航向误差平方惩罚 (heading_error_penalty)
    # ==========================================================
    # cos(err)在小角度梯度≈0, 而(err/π)²在任何角度都有强梯度
    # 30°偏航: penalty = 0.8*(30/180)² = 0.022/step
    # 90°偏航: penalty = 0.8*(90/180)² = 0.200/step (!很重)
    lambda_heading_pen = rc.get("lambda_heading_error_penalty", 0.8)
    total_heading_pen = 0.0
    for i, off in enumerate(offensives):
        if off.alive and not off.hit_hvt:
            dx = hvt_x - off.x
            dy = hvt_y - off.y
            bearing = np.arctan2(dy, dx)
            heading_err = bearing - off.heading
            heading_err = (heading_err + np.pi) % (2 * np.pi) - np.pi
            pen = lambda_heading_pen * (heading_err / np.pi) ** 2
            rewards[i] -= pen
            total_heading_pen += pen
    reward_info["penalty_heading_error"] = total_heading_pen

    # ==========================================================
    # 核心奖励 3: 闭合速度奖励 (closing_speed)
    # ==========================================================
    lambda_closing = rc.get("lambda_closing", 3.0)
    vel_range = max(config.get("vel_range", 120.0), 1.0)
    total_closing = 0.0
    for i, off in enumerate(offensives):
        if off.alive and not off.hit_hvt:
            Vc = _closing_speed_to_point_correct(off, hvt_x, hvt_y, hvt_z)
            # 正闭合速度 → 奖励, 负闭合速度 → 惩罚
            r_close = lambda_closing * Vc / vel_range
            rewards[i] += r_close
            total_closing += r_close
    reward_info["reward_closing_speed"] = total_closing

    # ==========================================================
    # 核心惩罚: 距离惩罚 (proximity_penalty)
    # ==========================================================
    # 远离目标 → 持续惩罚, 越远越重
    lambda_proximity = rc.get("lambda_proximity", 0.4)
    proximity_norm = rc.get("proximity_norm_dist", 1500.0)
    total_prox = 0.0
    for i, off in enumerate(offensives):
        if off.alive and not off.hit_hvt:
            prox_pen = lambda_proximity * (cur_dists[i] / proximity_norm)
            rewards[i] -= prox_pen
            total_prox += prox_pen
    reward_info["penalty_proximity"] = total_prox

    # ==========================================================
    # V37/V5: 偏航加速度正则化 (防止无意义转弯)
    # ==========================================================
    # 惩罚|action[2]| — 鼓励策略输出近零的偏航指令
    lambda_mu_reg = rc.get("lambda_mu_regularize", 0.0)
    total_mu_reg = 0.0
    if lambda_mu_reg > 0 and raw_actions is not None:
        for i, off in enumerate(offensives):
            if off.alive and not off.hit_hvt:
                act = np.array(raw_actions[i], dtype=np.float32).flatten()
                if len(act) >= 3:
                    yaw_pen = lambda_mu_reg * abs(act[2])
                    rewards[i] -= yaw_pen
                    total_mu_reg += yaw_pen
    reward_info["penalty_yaw_reg"] = total_mu_reg

    # ==========================================================
    # 安全惩罚 (仅保留物理安全: boundary + ground)
    # ==========================================================
    lambda_boundary = rc.get("lambda_penalty_boundary", 2.0)
    map_size = config["map_size"]
    total_boundary = 0.0
    for i, off in enumerate(offensives):
        if not off.alive:
            continue
        if abs(off.x) > map_size * 0.9 or abs(off.y) > map_size * 0.9:
            rewards[i] -= lambda_boundary
            total_boundary += lambda_boundary
    reward_info["penalty_boundary"] = total_boundary

    lambda_ground = rc.get("lambda_penalty_ground", 1.0)
    total_ground = 0.0
    for i, off in enumerate(offensives):
        if not off.alive:
            continue
        z_min = config.get("z_min", 0.0)
        z_min_safe = max(z_min * 2.0, 5.0)
        if off.z < z_min_safe:
            ratio = (z_min_safe - off.z) / max(z_min_safe, 1.0)
            pen = lambda_ground * 2.0 * ratio * ratio
            rewards[i] -= pen
            total_ground += pen
    reward_info["penalty_ground"] = total_ground

    # ==========================================================
    # 被击杀惩罚 (很小, 鼓励勇敢突入不怕死)
    # ==========================================================
    killed_pen = rc.get("killed_penalty", -0.5)
    for i in range(n_off):
        if just_killed[i] and not offensives[i].hit_hvt:
            rewards[i] += killed_pen

    # ==========================================================
    # 命中 HVT — 巨额奖励, 全队共享
    # ==========================================================
    if hit_events:
        hit_bonus = rc.get("hit_hvt_bonus", 8000.0) * len(hit_events)
        for i in range(n_off):
            rewards[i] += hit_bonus
        reward_info["hit_hvt"] = hit_bonus

    # ==========================================================
    # 步惩罚
    # ==========================================================
    step_pen = rc.get("step_penalty", -0.003)
    for i, off in enumerate(offensives):
        if off.alive:
            rewards[i] += step_pen

    return rewards, reward_info


# ======================================================================
# compute_costs: 全部并入 reward, costs 返回零
# ======================================================================
def compute_costs(offensives, defensives, config):
    n_off = len(offensives)
    return [0.0] * n_off, {}


# ======================================================================
# Terminal Reward V35 — 简化: 主要看命中 + 距离
# ======================================================================
def compute_terminal_rewards(offensives, hvt, config, ap_data=None):
    """
    V35终端奖励 — 极简:
      命中奖 + 距离奖/罚 - 全军覆没惩罚
    """
    rc = config["reward"]
    n_off = len(offensives)

    N_hit = sum(1 for off in offensives if off.hit_hvt)
    N_alive = sum(1 for off in offensives if off.alive)

    # 存活agent到HVT的最小距离
    min_dist = float('inf')
    sum_dist = 0.0
    n_valid = 0
    for off in offensives:
        if off.alive and not off.hit_hvt:
            d = off.distance_to(hvt.x, hvt.y, hvt.z)
            min_dist = min(min_dist, d)
            sum_dist += d
            n_valid += 1

    if min_dist == float('inf'):
        min_dist = 2000.0

    # 终端奖励
    lambda_hit = rc.get("lambda_terminal_hit", 800.0)
    lambda_dist = rc.get("lambda_terminal_dist", 500.0)
    max_dist_ref = 2000.0

    # 命中奖励
    terminal_r = lambda_hit * N_hit

    # 距离奖励: 越近越好 (即使没命中, 靠得近也给奖)
    dist_reward = lambda_dist * max(1.0 - min_dist / max_dist_ref, 0.0)
    terminal_r += dist_reward

    # 全军覆没额外惩罚
    if N_alive == 0 and N_hit == 0:
        terminal_r -= 100.0

    rewards = [terminal_r / max(n_off, 1)] * n_off

    info = {
        "terminal_N_hit": N_hit,
        "terminal_N_alive": N_alive,
        "terminal_min_dist": min_dist,
        "terminal_reward": terminal_r,
    }
    return rewards, info

"""
FOV Penetration Environment - Reward & Cost V3
================================================
围绕隐身突防设计, 删除所有旧护卫逻辑
"""

import numpy as np


def compute_rewards(offensives, defensives, hvt, config,
                    prev_dists_to_hvt, hit_events, current_step):
    rc = config["reward"]
    n_off = len(offensives)
    max_steps = config["max_steps"]
    obs_range = config["obs_range"]
    rewards = [0.0] * n_off
    reward_info = {}

    # 1. 命中 HVT (全队共享, 多命中递减)
    if hit_events:
        n_hits = len(hit_events)
        bonus = rc["hit_hvt_bonus"]
        dim = rc["hit_hvt_diminishing"]
        total_hit_reward = sum(bonus * (dim ** k) for k in range(n_hits))
        for i in range(n_off):
            rewards[i] += total_hit_reward
        reward_info["hit_hvt_total"] = total_hit_reward
        reward_info["n_hits"] = n_hits

    # 2. 接近 HVT
    approach_total = 0.0
    for i, off in enumerate(offensives):
        if off.alive and not off.hit_hvt:
            dist = off.distance_to(hvt.x, hvt.y, hvt.z)
            prev_d = prev_dists_to_hvt[i]
            delta = prev_d - dist
            approach_r = rc["approach_hvt_coef"] * delta / obs_range
            if not off.detected:
                approach_r += rc["stealth_approach_coef"] * max(delta, 0) / obs_range
            rewards[i] += approach_r
            approach_total += approach_r
            for j in range(n_off):
                if j != i:
                    rewards[j] += approach_r * 0.15
    reward_info["approach_total"] = approach_total

    # 3. 探测与暴露惩罚
    exposure_penalty_total = 0.0
    for i, off in enumerate(offensives):
        if not off.alive:
            continue
        if off.detected:
            rewards[i] += rc["detected_penalty"]
            exposure_penalty_total += rc["detected_penalty"]
            if off.detected_by_count >= 2:
                multi_p = rc["multi_detected_penalty"] * (off.detected_by_count - 1)
                rewards[i] += multi_p
                exposure_penalty_total += multi_p
            cont_p = rc["continuous_exposure_penalty"] * off.continuous_exposure
            rewards[i] += cont_p
            exposure_penalty_total += cont_p
        if (off.first_detected_step >= 0 and
                off.first_detected_step == current_step and
                current_step < max_steps * 0.2):
            rewards[i] += rc["early_detection_penalty"]
            exposure_penalty_total += rc["early_detection_penalty"]
    reward_info["exposure_penalty"] = exposure_penalty_total

    # 4. 被击杀惩罚
    for i, off in enumerate(offensives):
        if not off.alive and not off.hit_hvt:
            rewards[i] += rc["killed_penalty"]
            for j in range(n_off):
                if j != i and offensives[j].alive:
                    rewards[j] += rc["killed_penalty"] * 0.1

    # 5. 步惩罚 + 动作平滑 + 高度保持与绕飞(Flanking)奖励
    z_min_safe = config.get("z_min", 100.0) * 1.5
    z_mid = config.get("z_max", 2000.0) * 0.5  # 理想巡航高度区间中值
    
    # 计算当前存活的进攻方距离
    alive_idx = [i for i, off in enumerate(offensives) if off.alive and not off.hit_hvt]
    
    for i, off in enumerate(offensives):
        if off.alive:
            rewards[i] += rc["step_penalty"]
            overload_mag = np.sqrt(off.ny**2 + off.nz**2)
            rewards[i] += rc["smooth_action_coef"] * overload_mag
            
            # 高度惩罚：不要贴地飞行（梯度惩罚，越低惩越重）
            if off.z < z_min_safe:
                pen = rc.get("altitude_penalty_coef", 5.0) * ((z_min_safe - off.z) / z_min_safe)
                rewards[i] -= pen
            
            # gamma 俯仰角引导：惩罚持续大负gamma（持续俯冲）
            gamma_deg = np.degrees(off.gamma)
            if gamma_deg < -15.0:
                # gamma在-15°~-30°时线性惩罚
                dive_pen = rc.get("altitude_penalty_coef", 5.0) * 0.3 * ((-gamma_deg - 15.0) / 15.0)
                rewards[i] -= dive_pen
                
    # 绕飞/分散阵型奖励
    spread_coef = rc.get("spread_bonus_coef", 0.05)
    if len(alive_idx) >= 2 and spread_coef > 0:
        for idx_i in alive_idx:
            for idx_j in alive_idx:
                if idx_i < idx_j:
                    dist_ij = offensives[idx_i].distance_3d(offensives[idx_j])
                    if dist_ij > 300:
                        # 距离在 300~2000 给予奖励，促进编队拉开
                        spread_r = spread_coef * min((dist_ij - 300) / 1700.0, 1.0)
                        rewards[idx_i] += spread_r
                        rewards[idx_j] += spread_r

    return rewards, reward_info


def compute_costs(offensives, defensives, config):
    cc = config["cost"]
    n_off = len(offensives)
    fov_half = config["fov_half_angle"]
    det_range = config["detection_range"]
    z_min = config.get("z_min", 100.0)
    map_size = config["map_size"]
    costs = [0.0] * n_off
    cost_info = {}

    for i, off in enumerate(offensives):
        if not off.alive:
            continue
        fov_count = 0
        for d in defensives:
            if d.alive and d.is_in_fov(off.x, off.y, off.z, fov_half, det_range):
                fov_count += 1
        if fov_count > 0:
            costs[i] += cc["fov_exposure"] * fov_count
        for d in defensives:
            if d.alive:
                dist = off.distance_3d(d)
                if dist < 300.0:
                    costs[i] += cc["danger_zone"] * (1.0 - dist / 300.0)
        if abs(off.x) > map_size * 0.9 or abs(off.y) > map_size * 0.9:
            costs[i] += cc["boundary"]
        if off.z < z_min * 1.5:
            costs[i] += cc["ground_crash"] * (1.0 - off.z / (z_min * 1.5))
        for j, other in enumerate(offensives):
            if j != i and other.alive:
                if off.distance_3d(other) < config["collision_range"]:
                    costs[i] += cc["collision"]

    return costs, cost_info

"""
FOV Penetration Environment - Reward & Cost V4
================================================
核心设计原则:
  1. 活着飞向目标是最高优先级 (接近HVT奖励碾压其他一切)
  2. 被探测是小惩罚，被击杀才是大惩罚 (绝不能让"死掉比活着划算")
  3. 大机动脱离FOV给予即时奖励 (鼓励利用锥形视场死角)
  4. 牺牲掩护队友给予共享奖励 (碰撞消灭拦截器 = 为队友开路)
  5. 保持高度是生存基础 (极强的高度保护)

物理机理:
  - 拦截器视场是锥形 (半角30°), 锥形下半部分是盲区
  - 迎面飞时做大机动可以快速穿过锥形底部/侧面脱离
  - 拦截器感知更新频率有限, 脱离后有时间窗口加速突防
  - 碰撞同归于尽: 牺牲1架换掉1个拦截器, 为队友开路
"""

import numpy as np


def compute_rewards(offensives, defensives, hvt, config,
                    prev_dists_to_hvt, hit_events, current_step):
    rc = config["reward"]
    n_off = len(offensives)
    max_steps = config["max_steps"]
    obs_range = config["obs_range"]
    z_min = config.get("z_min", 100.0)
    z_min_safe = z_min * 2.0
    rewards = [0.0] * n_off
    reward_info = {}

    # ======================================================
    # 1. 命中 HVT — 终极目标, 巨额奖励 (全队共享)
    # ======================================================
    if hit_events:
        n_hits = len(hit_events)
        bonus = rc["hit_hvt_bonus"]
        total_hit_reward = bonus * n_hits
        for i in range(n_off):
            rewards[i] += total_hit_reward
        reward_info["hit_hvt_total"] = total_hit_reward

    # ======================================================
    # 2. 接近 HVT — 主导奖励, 必须是最强的持续正信号
    # ======================================================
    approach_total = 0.0
    for i, off in enumerate(offensives):
        if off.alive and not off.hit_hvt:
            dist = off.distance_to(hvt.x, hvt.y, hvt.z)
            prev_d = prev_dists_to_hvt[i]
            delta = prev_d - dist  # 正值=接近

            # 基础接近奖励 (不管是否被探测, 接近就给!)
            approach_r = rc["approach_hvt_coef"] * delta / obs_range

            # 额外: 隐身接近加成 (锦上添花, 不是主体)
            if not off.detected:
                approach_r += rc["stealth_approach_coef"] * max(delta, 0) / obs_range

            # 距离越近, 奖励越大 (指数加速, 逼近时爆发)
            dist_ratio = max(1.0 - dist / obs_range, 0.0)
            approach_r *= (1.0 + 2.0 * dist_ratio)

            rewards[i] += approach_r
            approach_total += approach_r

            # 队友共享 (弱)
            for j in range(n_off):
                if j != i and offensives[j].alive:
                    rewards[j] += approach_r * 0.1
    reward_info["approach_total"] = approach_total

    # ======================================================
    # 3. 探测惩罚 — 必须很轻! 只是"提醒"
    # ======================================================
    for i, off in enumerate(offensives):
        if not off.alive:
            continue
        if off.detected:
            rewards[i] += rc["detected_penalty"]  # -0.01

    # ======================================================
    # 4. FOV脱离奖励 — 鼓励大机动利用锥形盲区
    # ======================================================
    for i, off in enumerate(offensives):
        if not off.alive:
            continue
        if not off.detected and off.continuous_exposure == 0 and off.total_exposure_steps > 0:
            rewards[i] += rc.get("fov_escape_bonus", 3.0)

    # ======================================================
    # 5. 被击杀惩罚 — 必须很重!
    # ======================================================
    n_def_alive = sum(1 for d in defensives if d.alive)
    for i, off in enumerate(offensives):
        if not off.alive and not off.hit_hvt:
            rewards[i] += rc["killed_penalty"]  # -100

    # ======================================================
    # 6. 牺牲掩护奖励 — 碰撞消灭拦截器 = 为队友开路
    # ======================================================
    sacrifice_bonus = rc.get("sacrifice_cover_bonus", 15.0)
    for i, off in enumerate(offensives):
        if not off.alive and not off.hit_hvt:
            teammates_alive = sum(1 for j, o in enumerate(offensives)
                                if j != i and o.alive and not o.hit_hvt)
            if teammates_alive > 0 and n_def_alive < len(defensives):
                rewards[i] += sacrifice_bonus * 0.5
                for j, o in enumerate(offensives):
                    if j != i and o.alive and not o.hit_hvt:
                        rewards[j] += sacrifice_bonus / teammates_alive

    # ======================================================
    # 7. 高度保护 — 极强! 绝对不允许撞地
    # ======================================================
    for i, off in enumerate(offensives):
        if not off.alive:
            continue
        if off.z < z_min_safe:
            ratio = (z_min_safe - off.z) / z_min_safe
            pen = rc.get("altitude_penalty_coef", 10.0) * ratio * ratio
            rewards[i] -= pen
        gamma_deg = np.degrees(off.gamma)
        if gamma_deg < -10.0:
            dive_severity = min((-gamma_deg - 10.0) / 20.0, 1.0)
            rewards[i] -= rc.get("altitude_penalty_coef", 10.0) * 0.5 * dive_severity

    # ======================================================
    # 8. 步惩罚 + 动作平滑
    # ======================================================
    for i, off in enumerate(offensives):
        if off.alive:
            rewards[i] += rc["step_penalty"]  # -0.01
            overload_mag = np.sqrt(off.ny**2 + off.nz**2)
            rewards[i] += rc["smooth_action_coef"] * overload_mag

    # ======================================================
    # 9. 分散阵型奖励
    # ======================================================
    alive_idx = [i for i, off in enumerate(offensives) if off.alive and not off.hit_hvt]
    spread_coef = rc.get("spread_bonus_coef", 0.02)
    if len(alive_idx) >= 2 and spread_coef > 0:
        for idx_i in alive_idx:
            for idx_j in alive_idx:
                if idx_i < idx_j:
                    dist_ij = offensives[idx_i].distance_3d(offensives[idx_j])
                    if dist_ij > 200:
                        spread_r = spread_coef * min((dist_ij - 200) / 1800.0, 1.0)
                        rewards[idx_i] += spread_r
                        rewards[idx_j] += spread_r

    return rewards, reward_info


def compute_costs(offensives, defensives, config):
    """约束成本函数 (MACPO的安全约束通道)"""
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

        # FOV暴露成本
        fov_count = 0
        for d in defensives:
            if d.alive and d.is_in_fov(off.x, off.y, off.z, fov_half, det_range):
                fov_count += 1
        if fov_count > 0:
            costs[i] += cc["fov_exposure"] * fov_count

        # 危险接近成本
        for d in defensives:
            if d.alive:
                dist = off.distance_3d(d)
                if dist < 300.0:
                    costs[i] += cc["danger_zone"] * (1.0 - dist / 300.0)

        # 边界成本
        if abs(off.x) > map_size * 0.9 or abs(off.y) > map_size * 0.9:
            costs[i] += cc["boundary"]

        # 触地成本 (极高)
        if off.z < z_min * 2.0:
            costs[i] += cc["ground_crash"] * (1.0 - off.z / (z_min * 2.0))

        # 碰撞成本
        for j, other in enumerate(offensives):
            if j != i and other.alive:
                if off.distance_3d(other) < config["collision_range"]:
                    costs[i] += cc["collision"]

    return costs, cost_info

"""
FOV Penetration Environment - Reward & Cost V5
================================================
V4问题诊断:
  - 飞机学会了爬高躲避 (approach累计+25 vs killed一次-100, 不进攻更安全)
  - 接近HVT奖励太弱: 50 * 5m/5000m = +0.05/步, 500步只有+25
  - killed惩罚太重: -100一次 > 500步approach总和
  
V5核心修正 (数量级验证):
  approach_hvt:  200 * 5/5000 = +0.20/步 → 500步 = +100
  progress_bonus: ~+0.15/步(近) → 500步 = +75
  retreat_penalty: -0.10/步(不接近时) → ~250步 = -25
  killed_penalty: -20 (一次)
  → 冲向目标: +100+75-20 = +155 (即使被杀也赚)
  → 苟活躲避: -25+0 = -25 (不进攻亏)
  结论: 必须冲过去！

物理机理:
  - 拦截器视场是锥形 (半角30°), 锥形侧/后方是盲区
  - 迎面飞时做大机动可以穿过锥底/侧面脱离
  - 碰撞同归于尽: 牺牲1架换掉1个拦截器, 为队友开路
"""

import numpy as np


def compute_rewards(offensives, defensives, hvt, config,
                    prev_dists_to_hvt, hit_events, current_step,
                    just_killed=None):
    rc = config["reward"]
    n_off = len(offensives)
    obs_range = config["obs_range"]
    z_min = config.get("z_min", 100.0)
    z_min_safe = z_min * 2.0
    rewards = [0.0] * n_off
    reward_info = {}

    # 初始距离 (用于归一化progress)
    init_dist = 6000.0  # HVT在[3000,0,0], 初始在[-3000,0,500] → ~6000m

    # ======================================================
    # 1. 命中 HVT — 终极目标 (全队共享)
    # ======================================================
    if hit_events:
        n_hits = len(hit_events)
        bonus = rc["hit_hvt_bonus"]   # +500
        total_hit_reward = bonus * n_hits
        for i in range(n_off):
            rewards[i] += total_hit_reward
        reward_info["hit_hvt_total"] = total_hit_reward

    # ======================================================
    # 2. 接近 HVT — 压倒性主导奖励
    #    approach_r ≈ 200 * 5/5000 * (1~3) = 0.20~0.60/步
    # ======================================================
    approach_total = 0.0
    for i, off in enumerate(offensives):
        if off.alive and not off.hit_hvt:
            dist = off.distance_to(hvt.x, hvt.y, hvt.z)
            prev_d = prev_dists_to_hvt[i]
            delta = prev_d - dist  # 正=接近

            # 基础接近奖励
            approach_r = rc["approach_hvt_coef"] * delta / obs_range

            # 距离越近, 奖励指数增长 (逼近时爆发到3x)
            dist_ratio = max(1.0 - dist / init_dist, 0.0)
            approach_r *= (1.0 + 2.0 * dist_ratio)

            rewards[i] += approach_r
            approach_total += approach_r

    reward_info["approach_total"] = approach_total

    # ======================================================
    # 3. 进度奖励 (distance-based potential shaping)
    #    靠近HVT本身就给持续正奖励 (不管delta)
    #    progress ≈ 0.15~0.40/步 (取决于距离)
    # ======================================================
    progress_total = 0.0
    for i, off in enumerate(offensives):
        if off.alive and not off.hit_hvt:
            dist = off.distance_to(hvt.x, hvt.y, hvt.z)
            progress = max(1.0 - dist / init_dist, 0.0)  # 0~1, 越近越高
            progress_r = rc.get("progress_coef", 0.3) * progress
            rewards[i] += progress_r
            progress_total += progress_r
    reward_info["progress_total"] = progress_total

    # ======================================================
    # 4. 后退惩罚 — 不接近就扣分!
    #    确保"躲避"策略不再安全
    # ======================================================
    for i, off in enumerate(offensives):
        if off.alive and not off.hit_hvt:
            dist = off.distance_to(hvt.x, hvt.y, hvt.z)
            prev_d = prev_dists_to_hvt[i]
            delta = prev_d - dist
            if delta < 0:  # 在远离HVT
                retreat_pen = rc.get("retreat_penalty", -0.10)
                rewards[i] += retreat_pen

    # ======================================================
    # 5. 探测惩罚 — 极轻提醒
    # ======================================================
    for i, off in enumerate(offensives):
        if not off.alive:
            continue
        if off.detected:
            rewards[i] += rc["detected_penalty"]  # -0.01

    # ======================================================
    # 6. FOV脱离奖励 — 鼓励大机动利用锥形盲区
    # ======================================================
    for i, off in enumerate(offensives):
        if not off.alive:
            continue
        if not off.detected and off.continuous_exposure == 0 and off.total_exposure_steps > 0:
            rewards[i] += rc.get("fov_escape_bonus", 2.0)

    # ======================================================
    # 7. 被击杀惩罚 — 只在死亡那一步触发一次!
    #    (之前bug: 每步都对死亡飞机给-20, 400步=-8000!)
    # ======================================================
    if just_killed is None:
        just_killed = [False] * n_off
    n_def_alive = sum(1 for d in defensives if d.alive)
    for i, off in enumerate(offensives):
        if just_killed[i] and not off.hit_hvt:
            rewards[i] += rc["killed_penalty"]  # -20, 只触发一次

    # ======================================================
    # 8. 牺牲掩护奖励 — 只在死亡那一步触发
    # ======================================================
    sacrifice_bonus = rc.get("sacrifice_cover_bonus", 10.0)
    for i, off in enumerate(offensives):
        if just_killed[i] and not off.hit_hvt:
            teammates_alive = sum(1 for j, o in enumerate(offensives)
                                if j != i and o.alive and not o.hit_hvt)
            if teammates_alive > 0 and n_def_alive < len(defensives):
                rewards[i] += sacrifice_bonus * 0.5
                for j, o in enumerate(offensives):
                    if j != i and o.alive and not o.hit_hvt:
                        rewards[j] += sacrifice_bonus / teammates_alive

    # ======================================================
    # 9. 高度保护 — 防止撞地 + 防止无限爬升
    # ======================================================
    for i, off in enumerate(offensives):
        if not off.alive:
            continue
        # 低高度惩罚
        if off.z < z_min_safe:
            ratio = (z_min_safe - off.z) / z_min_safe
            pen = rc.get("altitude_penalty_coef", 8.0) * ratio * ratio
            rewards[i] -= pen

        # 过高惩罚 (新增! 防止无限爬升)
        z_max = config.get("z_max", 2000.0)
        z_high_thresh = z_max * 0.7  # 1400m以上开始惩罚
        if off.z > z_high_thresh:
            ratio_high = (off.z - z_high_thresh) / (z_max - z_high_thresh)
            pen_high = rc.get("high_alt_penalty_coef", 3.0) * ratio_high * ratio_high
            rewards[i] -= pen_high

        # 俯冲惩罚
        gamma_deg = np.degrees(off.gamma)
        if gamma_deg < -10.0:
            dive_severity = min((-gamma_deg - 10.0) / 20.0, 1.0)
            rewards[i] -= rc.get("altitude_penalty_coef", 8.0) * 0.3 * dive_severity

    # ======================================================
    # 10. 步惩罚 + 动作平滑
    # ======================================================
    for i, off in enumerate(offensives):
        if off.alive:
            rewards[i] += rc["step_penalty"]  # -0.01
            overload_mag = np.sqrt(off.ny**2 + off.nz**2)
            rewards[i] += rc["smooth_action_coef"] * overload_mag

    # ======================================================
    # 11. 分散阵型奖励 (轻微)
    # ======================================================
    alive_idx = [i for i, off in enumerate(offensives) if off.alive and not off.hit_hvt]
    spread_coef = rc.get("spread_bonus_coef", 0.01)
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
    """约束成本函数 (MACPO安全约束通道)"""
    cc = config["cost"]
    n_off = len(offensives)
    fov_half = config["fov_half_angle"]
    det_range = config["detection_range"]
    z_min = config.get("z_min", 100.0)
    map_size = config["map_size"]
    costs = [0.0] * n_off

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

        # 触地成本
        if off.z < z_min * 2.0:
            costs[i] += cc["ground_crash"] * (1.0 - off.z / (z_min * 2.0))

        # 碰撞成本
        for j, other in enumerate(offensives):
            if j != i and other.alive:
                if off.distance_3d(other) < config["collision_range"]:
                    costs[i] += cc["collision"]

    return costs, {}

"""
FOV Penetration Environment - Reward & Cost V9
================================================
协同突防核心设计:

理念: 4v4对抗, 防守方一次分配后死盯不放(比例导引)
      -> 进攻方需要自己学会"诱饵+突防"分工
      -> 不指定角色, 通过reward激励让模型自己涌现分工

三大核心激励:
  1. 全队最近距离进步奖励 (team_progress)
     - 不管是谁靠近HVT, 全队都得分
     - 让个体愿意为团队最优做牺牲
  2. 诱饵吸引奖励 (decoy_attraction)
     - 如果我被拦截器锁定且正在远离最近突防者的路径
     - -> 我在替队友拉走拦截器, 给正奖励
  3. 同归于尽全队共享 (mutual_kill_team_bonus)
     - 碰撞消灭拦截器后, 存活队友获得大额奖励
     - 等价于"牺牲开路", 让队友面前少一道防线

数量级设计 (500步):
  个人接近HVT:  500 * 5/5000 * 1.5 = +0.75/步 -> +375 (活到最后直飞)
  全队进步:     500 * 0.5 = +250 (如果全队最佳持续推进)
  诱饵吸引:     被锁定期间 ~200步 * 0.3 = +60
  同归于尽:     一次性 +100 给存活队友
  命中HVT:      +1000 全队
  死亡惩罚:     -10 一次
  结论: 冲HVT >> 躲避, 帮队友也有回报
"""

import numpy as np


def compute_rewards(offensives, defensives, hvt, config,
                    prev_dists_to_hvt, hit_events, current_step,
                    just_killed=None, just_killed_def=None,
                    lock_on_map=None, prev_team_min_dist=None):
    rc = config["reward"]
    n_off = len(offensives)
    n_def = len(defensives)
    obs_range = config["obs_range"]
    z_min = config.get("z_min", 100.0)
    z_min_safe = z_min * 2.0
    rewards = [0.0] * n_off
    reward_info = {}
    init_dist = 6000.0

    if just_killed is None:
        just_killed = [False] * n_off
    if just_killed_def is None:
        just_killed_def = [False] * n_def
    if lock_on_map is None:
        lock_on_map = {i: [] for i in range(n_off)}

    # 当前各架到HVT的距离
    cur_dists = []
    for i, off in enumerate(offensives):
        if off.alive and not off.hit_hvt:
            cur_dists.append(off.distance_to(hvt.x, hvt.y, hvt.z))
        else:
            cur_dists.append(float('inf'))

    # 全队当前最近距离
    alive_dists = [d for d in cur_dists if d < float('inf')]
    team_min_dist = min(alive_dists) if alive_dists else init_dist
    # 谁是当前"最近突防者"
    closest_idx = min(range(n_off), key=lambda i: cur_dists[i]) if alive_dists else -1

    # ======================================================
    # 1. 命中 HVT -- 终极目标, 全队共享超级奖励
    # ======================================================
    if hit_events:
        n_hits = len(hit_events)
        bonus = rc.get("hit_hvt_bonus", 1000.0)
        total_hit_reward = bonus * n_hits
        for i in range(n_off):
            rewards[i] += total_hit_reward
        reward_info["hit_hvt_total"] = total_hit_reward

    # ======================================================
    # 2. 个人接近 HVT -- 每个活着的飞机自己的接近奖励
    # ======================================================
    approach_total = 0.0
    for i, off in enumerate(offensives):
        if off.alive and not off.hit_hvt:
            dist = cur_dists[i]
            prev_d = prev_dists_to_hvt[i]
            if prev_d < float('inf'):
                delta = prev_d - dist

                approach_r = rc.get("approach_hvt_coef", 500.0) * delta / obs_range
                dist_ratio = max(1.0 - dist / init_dist, 0.0)
                approach_r *= (1.0 + 2.0 * dist_ratio)
                rewards[i] += approach_r
                approach_total += approach_r
    reward_info["approach_total"] = approach_total

    # ======================================================
    # 3. 全队最近距离进步奖励 -- 核心协同激励!
    #    不管是谁推进, 全队都得分 -> 激励"帮队友突防"
    # ======================================================
    team_progress_r = 0.0
    if prev_team_min_dist is not None and alive_dists:
        team_delta = prev_team_min_dist - team_min_dist  # 正=更近
        team_progress_r = rc.get("team_progress_coef", 300.0) * team_delta / obs_range
        # 距离越近, 奖励越大
        team_ratio = max(1.0 - team_min_dist / init_dist, 0.0)
        team_progress_r *= (1.0 + 3.0 * team_ratio)
        for i in range(n_off):
            if offensives[i].alive:
                rewards[i] += team_progress_r
    reward_info["team_progress"] = team_progress_r

    # ======================================================
    # 4. 进度奖励 -- 靠近HVT本身就给持续正奖励
    # ======================================================
    for i, off in enumerate(offensives):
        if off.alive and not off.hit_hvt:
            dist = cur_dists[i]
            progress = max(1.0 - dist / init_dist, 0.0)
            progress_r = rc.get("progress_coef", 0.3) * progress
            rewards[i] += progress_r

    # ======================================================
    # 5. 诱饵吸引奖励 -- 被锁定时拉走拦截器为队友开路
    #    条件: 我被锁定 + 离最近突防者有距离 + 拦截器跟我走
    # ======================================================
    decoy_coef = rc.get("decoy_lure_coef", 0.3)
    for i, off in enumerate(offensives):
        if not off.alive or off.hit_hvt:
            continue
        locked_defs = lock_on_map.get(i, [])
        if len(locked_defs) == 0:
            continue
        # 我被锁定了, 检查我是否在帮队友
        if closest_idx >= 0 and closest_idx != i:
            # 计算我吸引的拦截器到最近突防者的距离
            closest_off = offensives[closest_idx]
            for def_idx in locked_defs:
                d = defensives[def_idx]
                if not d.alive:
                    continue
                # 拦截器到突防者的距离
                d_to_closest = d.distance_3d(closest_off)
                # 如果拦截器被我拉离了突防者 (距离>某个阈值)
                if d_to_closest > 1500.0:
                    lure_r = decoy_coef * min(d_to_closest / 3000.0, 1.0)
                    rewards[i] += lure_r
    reward_info["decoy_active"] = sum(1 for i in range(n_off)
                                       if offensives[i].alive and len(lock_on_map.get(i, [])) > 0
                                       and i != closest_idx)

    # ======================================================
    # 6. 后退惩罚 -- 远离HVT就扣分
    # ======================================================
    for i, off in enumerate(offensives):
        if off.alive and not off.hit_hvt:
            prev_d = prev_dists_to_hvt[i]
            if prev_d < float('inf'):
                delta = prev_d - cur_dists[i]
                if delta < 0:
                    rewards[i] += rc.get("retreat_penalty", -0.15)

    # ======================================================
    # 7. 被击杀惩罚 -- 轻! 只在死亡那步触发一次
    # ======================================================
    for i in range(n_off):
        if just_killed[i] and not offensives[i].hit_hvt:
            rewards[i] += rc.get("killed_penalty", -10.0)

    # ======================================================
    # 8. 同归于尽全队共享 -- 核心协同!
    #    牺牲消灭拦截器 -> 存活队友获得巨额奖励
    # ======================================================
    mutual_kill_bonus = rc.get("mutual_kill_team_bonus", 100.0)
    for i in range(n_off):
        if just_killed[i]:
            n_def_killed_this_step = sum(1 for jk in just_killed_def if jk)
            if n_def_killed_this_step > 0:
                alive_teammates = [j for j in range(n_off)
                                   if offensives[j].alive and not offensives[j].hit_hvt]
                if alive_teammates:
                    per_mate = mutual_kill_bonus / len(alive_teammates)
                    for j in alive_teammates:
                        rewards[j] += per_mate
                    rewards[i] += mutual_kill_bonus * 0.3

    # ======================================================
    # 9. 探测惩罚 -- 极轻
    # ======================================================
    for i, off in enumerate(offensives):
        if off.alive and off.detected:
            rewards[i] += rc.get("detected_penalty", -0.01)

    # ======================================================
    # 10. FOV脱离奖励
    # ======================================================
    for i, off in enumerate(offensives):
        if not off.alive:
            continue
        if not off.detected and off.continuous_exposure == 0 and off.total_exposure_steps > 0:
            rewards[i] += rc.get("fov_escape_bonus", 2.0)

    # ======================================================
    # 11. 高度保护
    # ======================================================
    for i, off in enumerate(offensives):
        if not off.alive:
            continue
        if off.z < z_min_safe:
            ratio = (z_min_safe - off.z) / z_min_safe
            pen = rc.get("altitude_penalty_coef", 8.0) * ratio * ratio
            rewards[i] -= pen
        z_max = config.get("z_max", 2000.0)
        z_high_thresh = z_max * 0.7
        if off.z > z_high_thresh:
            ratio_high = (off.z - z_high_thresh) / (z_max - z_high_thresh)
            pen_high = rc.get("high_alt_penalty_coef", 3.0) * ratio_high * ratio_high
            rewards[i] -= pen_high
        gamma_deg = np.degrees(off.gamma)
        if gamma_deg < -10.0:
            dive_severity = min((-gamma_deg - 10.0) / 20.0, 1.0)
            rewards[i] -= rc.get("altitude_penalty_coef", 8.0) * 0.3 * dive_severity

    # ======================================================
    # 12. 步惩罚 + 动作平滑
    # ======================================================
    for i, off in enumerate(offensives):
        if off.alive:
            rewards[i] += rc.get("step_penalty", -0.01)
            overload_mag = np.sqrt(off.ny**2 + off.nz**2)
            rewards[i] += rc.get("smooth_action_coef", -0.003) * overload_mag

    # ======================================================
    # 13. 分散阵型奖励 (轻微, 防止扎堆)
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
        if off.z < z_min * 2.0:
            costs[i] += cc["ground_crash"] * (1.0 - off.z / (z_min * 2.0))
        for j, other in enumerate(offensives):
            if j != i and other.alive:
                if off.distance_3d(other) < config["collision_range"]:
                    costs[i] += cc["collision"]

    return costs, {}

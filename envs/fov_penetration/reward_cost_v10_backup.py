"""
FOV Penetration Environment - Reward & Cost V10
=================================================
核心设计: "前方牺牲, 后方突防"

用户需求:
  1. 不暴露敌方分配方案, 通过态势推断
  2. 偏向让靠后的飞机(离拦截器更远)成为被保护的突防者
     前方飞机自然承担诱饵/牺牲角色

V9失败分析:
  - team_progress全队蹭分 -> 3架苟活不动也有奖励
  - lock_on_map信息泄露 -> 知道谁追我就躲
  - 诱饵奖励过强 -> 远离HVT反而得分

V10策略:
  1. 个人接近HVT是绝对主导 (+0.50/步 * 500步 = +250)
  2. 全队最近者额外奖金 (只给"最近那架"少量加分, 不给其他人)
  3. 同归于尽后队友奖励 (不依赖lock_on_map, 只看"有没有拦截器死")
  4. 去掉诱饵吸引奖励 (让模型自己通过approach_hvt涌现分工)
  5. 后退惩罚与前后位置无关 (所有人远离都受罚)

关键洞察: 只要approach_hvt足够强, 模型自然会学到:
  - 前方飞机先遇到拦截器 -> 被拦截/同归于尽 -> 为后方清路
  - 后方飞机晚遇到拦截器 -> 存活更久 -> 更可能突防成功
  不需要显式指定角色!
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

    # 当前各架到HVT的距离
    cur_dists = []
    for i, off in enumerate(offensives):
        if off.alive and not off.hit_hvt:
            cur_dists.append(off.distance_to(hvt.x, hvt.y, hvt.z))
        else:
            cur_dists.append(float('inf'))

    alive_dists = [d for d in cur_dists if d < float('inf')]
    team_min_dist = min(alive_dists) if alive_dists else init_dist
    closest_idx = min(range(n_off), key=lambda i: cur_dists[i]) if alive_dists else -1

    # ======================================================
    # 1. 命中 HVT --- 终极目标, 全队共享超级奖励
    # ======================================================
    if hit_events:
        bonus = rc.get("hit_hvt_bonus", 1000.0) * len(hit_events)
        for i in range(n_off):
            rewards[i] += bonus
        reward_info["hit_hvt"] = bonus

    # ======================================================
    # 2. 个人接近 HVT --- 绝对主导奖励, 每架都要冲!
    #    ~500 * 5/5000 * 1.5 = 0.75/步 -> 375 total
    # ======================================================
    approach_coef = rc.get("approach_hvt_coef", 500.0)
    for i, off in enumerate(offensives):
        if off.alive and not off.hit_hvt and prev_dists_to_hvt[i] < float('inf'):
            delta = prev_dists_to_hvt[i] - cur_dists[i]
            approach_r = approach_coef * delta / obs_range
            dist_ratio = max(1.0 - cur_dists[i] / init_dist, 0.0)
            approach_r *= (1.0 + 2.0 * dist_ratio)
            rewards[i] += approach_r

    # ======================================================
    # 3. 进度奖励 --- 靠近HVT本身就给持续正奖励
    # ======================================================
    progress_coef = rc.get("progress_coef", 0.3)
    for i, off in enumerate(offensives):
        if off.alive and not off.hit_hvt:
            progress = max(1.0 - cur_dists[i] / init_dist, 0.0)
            rewards[i] += progress_coef * progress

    # ======================================================
    # 4. 最近突防者额外奖金 --- 只给最近的那1架
    #    这激励至少有1架持续推进, 但不让其他人蹭分
    # ======================================================
    if closest_idx >= 0 and prev_team_min_dist is not None:
        team_delta = prev_team_min_dist - team_min_dist
        closest_bonus = rc.get("closest_bonus_coef", 200.0) * team_delta / obs_range
        dist_ratio = max(1.0 - team_min_dist / init_dist, 0.0)
        closest_bonus *= (1.0 + 3.0 * dist_ratio)
        rewards[closest_idx] += closest_bonus

    # ======================================================
    # 5. 后退惩罚 --- 远离HVT就扣分 (所有人)
    # ======================================================
    retreat_pen = rc.get("retreat_penalty", -0.15)
    for i, off in enumerate(offensives):
        if off.alive and not off.hit_hvt and prev_dists_to_hvt[i] < float('inf'):
            if prev_dists_to_hvt[i] - cur_dists[i] < 0:
                rewards[i] += retreat_pen

    # ======================================================
    # 6. 被击杀惩罚 --- 轻, 只在死亡那步触发一次
    # ======================================================
    killed_pen = rc.get("killed_penalty", -10.0)
    for i in range(n_off):
        if just_killed[i] and not offensives[i].hit_hvt:
            rewards[i] += killed_pen

    # ======================================================
    # 7. 同归于尽全队共享 --- 不依赖lock_on_map!
    #    只要"有进攻方死了 + 有拦截器也死了" = 同归于尽
    #    存活队友获得大额奖励 (为他们清了路)
    # ======================================================
    n_off_killed = sum(1 for jk in just_killed if jk)
    n_def_killed = sum(1 for jk in just_killed_def if jk)
    if n_off_killed > 0 and n_def_killed > 0:
        mutual_bonus = rc.get("mutual_kill_team_bonus", 80.0) * n_def_killed
        alive_mates = [j for j in range(n_off) if offensives[j].alive and not offensives[j].hit_hvt]
        if alive_mates:
            per_mate = mutual_bonus / len(alive_mates)
            for j in alive_mates:
                rewards[j] += per_mate
        # 牺牲者也得到认可
        for i in range(n_off):
            if just_killed[i]:
                rewards[i] += rc.get("mutual_kill_team_bonus", 80.0) * 0.3

    # ======================================================
    # 8. 探测惩罚 --- 极轻
    # ======================================================
    det_pen = rc.get("detected_penalty", -0.01)
    for i, off in enumerate(offensives):
        if off.alive and off.detected:
            rewards[i] += det_pen

    # ======================================================
    # 9. 高度保护
    # ======================================================
    alt_coef = rc.get("altitude_penalty_coef", 8.0)
    high_coef = rc.get("high_alt_penalty_coef", 3.0)
    z_max = config.get("z_max", 2000.0)
    z_high_thresh = z_max * 0.7
    for i, off in enumerate(offensives):
        if not off.alive:
            continue
        if off.z < z_min_safe:
            ratio = (z_min_safe - off.z) / z_min_safe
            rewards[i] -= alt_coef * ratio * ratio
        if off.z > z_high_thresh:
            ratio_h = (off.z - z_high_thresh) / (z_max - z_high_thresh)
            rewards[i] -= high_coef * ratio_h * ratio_h
        gamma_deg = np.degrees(off.gamma)
        if gamma_deg < -10.0:
            dive = min((-gamma_deg - 10.0) / 20.0, 1.0)
            rewards[i] -= alt_coef * 0.3 * dive

    # ======================================================
    # 10. 步惩罚 + 动作平滑
    # ======================================================
    step_pen = rc.get("step_penalty", -0.01)
    smooth_coef = rc.get("smooth_action_coef", -0.003)
    for i, off in enumerate(offensives):
        if off.alive:
            rewards[i] += step_pen
            rewards[i] += smooth_coef * np.sqrt(off.ny**2 + off.nz**2)

    # ======================================================
    # 11. 分散阵型 (轻微)
    # ======================================================
    alive_idx = [i for i, off in enumerate(offensives) if off.alive and not off.hit_hvt]
    spread_coef = rc.get("spread_bonus_coef", 0.01)
    if len(alive_idx) >= 2 and spread_coef > 0:
        for ii in range(len(alive_idx)):
            for jj in range(ii+1, len(alive_idx)):
                d_ij = offensives[alive_idx[ii]].distance_3d(offensives[alive_idx[jj]])
                if d_ij > 200:
                    sr = spread_coef * min((d_ij - 200) / 1800.0, 1.0)
                    rewards[alive_idx[ii]] += sr
                    rewards[alive_idx[jj]] += sr

    return rewards, reward_info


def compute_costs(offensives, defensives, config):
    """约束成本函数"""
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
        for d in defensives:
            if d.alive and d.is_in_fov(off.x, off.y, off.z, fov_half, det_range):
                costs[i] += cc["fov_exposure"]
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

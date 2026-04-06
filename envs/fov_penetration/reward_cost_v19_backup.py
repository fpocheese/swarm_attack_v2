"""FOV Penetration Environment - Reward & Cost V19
===================================================
V19 关键改进 (解决v18 cost-reward失衡问题):
  1. 接近奖励从除以obs_range改为除以init_dist, 信号强度提升2.5x
  2. proximity_reward 幂次从2→3, 近距离信号更锐利
  3. 新增距离里程碑奖励: 首次到达3000/2000/1000/500m时给大额奖励
  4. 降低高度/gamma惩罚系数, 减少与接近信号的对冲
  5. 降低被探测/被击杀惩罚, 让策略敢于接近
  6. 强化timeout惩罚, 英雄不问出处只问前进
"""

import numpy as np
from .config import G


def compute_rewards(offensives, defensives, hvt, config,
                    prev_dists_to_hvt, hit_events, current_step,
                    just_killed=None, just_killed_def=None,
                    lock_on_map=None, prev_team_min_dist=None,
                    escape_events=None, miss_events=None,
                    defensive_policies=None):
    """
    计算奖励

    新参数:
        escape_events: list of dict, 本步发生的逃逸事件
            [{"off_idx": i, "def_idx": j, "type": "fov_break"|"pass_through"}, ...]
        miss_events: list of dict, 本步发生的脱靶事件
            [{"off_idx": i, "def_idx": j, "reason": "..."}, ...]
    """
    rc = config["reward"]
    n_off = len(offensives)
    n_def = len(defensives)
    obs_range = config["obs_range"]
    z_min = config.get("z_min", 100.0)
    z_min_safe = z_min * 2.0
    rewards = [0.0] * n_off
    reward_info = {}
    init_dist = 2500.0  # V24fix: 5000→2500, 匹配V23实际进攻→HVT距离~2400m

    if just_killed is None:
        just_killed = [False] * n_off
    if just_killed_def is None:
        just_killed_def = [False] * n_def
    if escape_events is None:
        escape_events = []
    if miss_events is None:
        miss_events = []

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
    # 1. 命中 HVT — 终极目标, 全队共享超级奖励
    # ======================================================
    if hit_events:
        bonus = rc.get("hit_hvt_bonus", 2000.0) * len(hit_events)
        for i in range(n_off):
            rewards[i] += bonus
        reward_info["hit_hvt"] = bonus

    # ======================================================
    # 2. 个人接近 HVT — 绝对主导奖励
    # ======================================================
    approach_coef = rc.get("approach_hvt_coef", 800.0)

    for i, off in enumerate(offensives):
        if off.alive and not off.hit_hvt and prev_dists_to_hvt[i] < float('inf'):
            delta = prev_dists_to_hvt[i] - cur_dists[i]
            approach_r = approach_coef * delta / init_dist
            dist_ratio = max(1.0 - cur_dists[i] / init_dist, 0.0)
            approach_r *= (1.0 + 3.0 * dist_ratio)
            rewards[i] += approach_r

    # ======================================================
    # 3. 进度奖励 — 靠近HVT本身就给持续正奖励
    # ======================================================
    progress_coef = rc.get("progress_coef", 0.5)
    for i, off in enumerate(offensives):
        if off.alive and not off.hit_hvt:
            progress = max(1.0 - cur_dists[i] / init_dist, 0.0)
            rewards[i] += progress_coef * progress

    # ======================================================
    # 3.5 V15新增: 持续距离奖励 (proximity reward)
    #     距HVT越近, 每步基础奖励越大 (平方放大效果)
    #     这为策略提供了始终如一的"距离梯度信号"
    # ======================================================
    proximity_coef = rc.get("proximity_reward_coef", 6.0)
    for i, off in enumerate(offensives):
        if off.alive and not off.hit_hvt:
            d_norm = cur_dists[i] / init_dist  # 0~1
            proximity = max(1.0 - d_norm, 0.0)  # 越近越大
            # V19: 幂次提高到3次方, 近距离信号更锐利
            rewards[i] += proximity_coef * proximity * proximity * proximity

    # ======================================================
    # 3.8 V19新增: 距离里程碑奖励
    #     首次有agent到达某个距离HVT的门槛时,全队获得一次性奖励
    # ======================================================
    milestones = rc.get("milestone_bonuses", {})
    if milestones and alive_dists:
        for threshold, bonus_val in milestones.items():
            threshold = float(threshold)
            # 检查是否本步首次突破该门槛
            prev_within = any(d < threshold for d in prev_dists_to_hvt if d < float('inf'))
            cur_within = any(d < threshold for d in alive_dists)
            if cur_within and not prev_within:
                for i in range(n_off):
                    if offensives[i].alive:
                        rewards[i] += bonus_val
                reward_info[f"milestone_{int(threshold)}m"] = bonus_val

    # ======================================================
    # 4. 最近突防者额外奖金
    # ======================================================
    if closest_idx >= 0 and prev_team_min_dist is not None:
        team_delta = prev_team_min_dist - team_min_dist
        closest_bonus = rc.get("closest_bonus_coef", 200.0) * team_delta / obs_range
        dist_ratio = max(1.0 - team_min_dist / init_dist, 0.0)
        closest_bonus *= (1.0 + 3.0 * dist_ratio)
        rewards[closest_idx] += closest_bonus

    # ======================================================
    # 5. 后退惩罚
    # ======================================================
    retreat_pen = rc.get("retreat_penalty", -0.15)
    for i, off in enumerate(offensives):
        if off.alive and not off.hit_hvt and prev_dists_to_hvt[i] < float('inf'):
            if prev_dists_to_hvt[i] - cur_dists[i] < 0:
                rewards[i] += retreat_pen

    # ======================================================
    # 6. 被击杀惩罚
    # ======================================================
    killed_pen = rc.get("killed_penalty", -5.0)
    for i in range(n_off):
        if just_killed[i] and not offensives[i].hit_hvt:
            rewards[i] += killed_pen

    # ======================================================
    # 7. 同归于尽全队共享 + 诱饵牺牲奖励(合并)
    # ======================================================
    decoy_sacrifice_bonus = rc.get("decoy_sacrifice_bonus", 800.0)
    n_off_killed = sum(1 for jk in just_killed if jk)
    n_def_killed = sum(1 for jk in just_killed_def if jk)
    if n_off_killed > 0 and n_def_killed > 0:
        # 碰撞双杀 → 存活队友分享牺牲奖励
        alive_mates = [j for j in range(n_off) if offensives[j].alive and not offensives[j].hit_hvt]
        if alive_mates:
            per_mate = decoy_sacrifice_bonus / len(alive_mates)
            for j in alive_mates:
                rewards[j] += per_mate
        # 被杀者也获得部分奖励(鼓励牺牲换取团队收益)
        for i in range(n_off):
            if just_killed[i]:
                rewards[i] += decoy_sacrifice_bonus * 0.3
        reward_info["decoy_sacrifice"] = n_off_killed

    # ======================================================
    # 8. ★ V25: 持续诱饵奖励 ★
    #    通过defensive_policies准确判断谁在追谁
    #    被拦截器追击的agent是诱饵 → 队友获奖 (通道更安全)
    #    关键: 诱饵应远离HVT(靠近拦截器)引诱,
    #          不被追的队友趁机接近HVT
    # ======================================================
    decoy_attract_coef = rc.get("decoy_attract_coef", 2.0)
    decoy_front_bonus = rc.get("decoy_front_bonus_coef", 1.5)

    # 用defensive_policies精确统计每个进攻方被追击数
    chasing_counts = [0] * n_off  # chasing_counts[i] = 追击第i架进攻方的拦截器数
    if defensive_policies is not None:
        for dp in defensive_policies:
            if dp.interceptor.alive and dp.target is not None:
                for i, off in enumerate(offensives):
                    if dp.target is off:
                        chasing_counts[i] += 1
                        break

    for i, off in enumerate(offensives):
        if not off.alive or off.hit_hvt:
            continue
        n_chasing = chasing_counts[i]
        if n_chasing > 0:
            # 自己吸引了拦截器 → 团队收益
            # 计算我离拦截器集群的远近: 离拦截器越近=诱饵效果越好
            # 用距HVT的距离判断: 离HVT越远(离拦截器越近) → 更好的诱饵
            n_farther = 0  # 有多少队友比我离HVT更远
            for j, other in enumerate(offensives):
                if j != i and other.alive and not other.hit_hvt:
                    if cur_dists[j] > cur_dists[i]:
                        n_farther += 1
            # 如果没有队友比我更远离HVT → 我在最前方面对拦截器 → 诱饵价值最大
            is_frontline = (n_farther == 0)
            front_mult = decoy_front_bonus if is_frontline else 1.0

            decoy_r = decoy_attract_coef * n_chasing * front_mult
            # 不被追的队友获得"诱饵掩护"奖励
            alive_mates = [j for j in range(n_off)
                          if j != i and offensives[j].alive and not offensives[j].hit_hvt]
            if alive_mates:
                per_mate = decoy_r / len(alive_mates)
                for j in alive_mates:
                    rewards[j] += per_mate
            # 诱饵自己获得小额奖励(鼓励吸引行为)
            rewards[i] += decoy_r * 0.3

    # ======================================================
    # 10. 探测惩罚
    # ======================================================
    det_pen = rc.get("detected_penalty", -0.01)
    for i, off in enumerate(offensives):
        if off.alive and off.detected:
            rewards[i] += det_pen

    # ======================================================
    # 11. 高度保护 (V19: 放宽惩罚阈值, 降低gamma惩罚, 给更多空间)
    # ======================================================
    alt_coef = rc.get("altitude_penalty_coef", 12.0)   # V19: 18→12
    high_coef = rc.get("high_alt_penalty_coef", 5.0)    # V19: 8→5
    z_max = config.get("z_max", 2000.0)
    z_high_thresh = z_max * 0.6       # V19: 0.5→0.6 (1200m才惩罚, 给更多空间)
    z_safe_low = z_min * 2.5          # 250m
    z_safe_high = 1000.0              # V19: 800→1000m (放宽安全区)
    for i, off in enumerate(offensives):
        if not off.alive:
            continue
        # 低空保护 (z < 200m)
        if off.z < z_min_safe:
            ratio = (z_min_safe - off.z) / z_min_safe
            rewards[i] -= alt_coef * ratio * ratio
        # 高空惩罚 (z > 1200m)
        if off.z > z_high_thresh:
            ratio_h = (off.z - z_high_thresh) / (z_max - z_high_thresh)
            rewards[i] -= high_coef * ratio_h * ratio_h
        gamma_deg = np.degrees(off.gamma)
        # V19: 俯冲惩罚 (gamma < -5°, 从V15的-3°放宽)
        if gamma_deg < -5.0:
            dive = min((-gamma_deg - 5.0) / 12.0, 1.0)
            rewards[i] -= alt_coef * 0.4 * dive        # V19: 0.6→0.4
        # V19: 爬升惩罚 (gamma > 12°, 从V15的8°放宽)
        if gamma_deg > 12.0:
            climb = min((gamma_deg - 12.0) / 15.0, 1.0)
            rewards[i] -= high_coef * 0.3 * climb       # V19: 0.5→0.3
        # 安全高度奖励 (250-1000m)
        if z_safe_low <= off.z <= z_safe_high:
            rewards[i] += 0.02

    # ======================================================
    # 12. 步惩罚 + 动作平滑
    # ======================================================
    step_pen = rc.get("step_penalty", -0.01)
    smooth_coef = rc.get("smooth_action_coef", -0.005)
    for i, off in enumerate(offensives):
        if off.alive:
            rewards[i] += step_pen
            rewards[i] += smooth_coef * (off.ay / G)

    # ======================================================
    # 13. 分散阵型
    # ======================================================
    alive_idx = [i for i, off in enumerate(offensives) if off.alive and not off.hit_hvt]
    spread_coef = rc.get("spread_bonus_coef", 0.01)
    if len(alive_idx) >= 2 and spread_coef > 0:
        for ii in range(len(alive_idx)):
            for jj in range(ii + 1, len(alive_idx)):
                d_ij = offensives[alive_idx[ii]].distance_3d(offensives[alive_idx[jj]])
                if d_ij > 200:
                    sr = spread_coef * min((d_ij - 200) / 1800.0, 1.0)
                    rewards[alive_idx[ii]] += sr
                    rewards[alive_idx[jj]] += sr

    return rewards, reward_info


def compute_costs(offensives, defensives, config):
    """约束成本函数 (基本不变)"""
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

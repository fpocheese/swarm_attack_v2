"""
FOV Penetration Environment - 奖励与成本 V2
=============================================
围绕进攻集群突防成功目标设计:

核心目标: 进攻飞行器成功打击HVT (距离<3m)
护卫目标: 护卫飞行器打击拦截器, 为进攻飞行器创造突防窗口
约束: 规避敌方FOV, 避免被拦截

奖励设计原则:
  1. 接近HVT的正向引导 (approach)
  2. 成功命中HVT的大额奖励 (全队共享)
  3. FOV规避奖励 — 不在敌方探测范围内
  4. 护卫击杀拦截器的奖励
  5. 护卫牵制/接近拦截器的引导
  6. 被击杀的惩罚
  7. 过载平滑奖励 (防抖头)
  8. 超时惩罚 (鼓励快速突防)
"""

import numpy as np


def compute_rewards(attacker, escorts, interceptors, hvt, config,
                    prev_dist_to_hvt, escort_kill_events=None):
    """
    计算所有训练 agent 的奖励

    Args:
        attacker: Aircraft
        escorts: list of Aircraft
        interceptors: list of Aircraft
        hvt: HVT
        config: 配置
        prev_dist_to_hvt: 上一步 attacker 到 HVT 距离
        escort_kill_events: 本步护卫击杀拦截器事件列表

    Returns:
        rewards: list of float
        reward_info: dict
    """
    rc = config["reward"]
    n_agents = 1 + len(escorts)
    rewards = [0.0] * n_agents
    reward_info = {}

    kill_range = config["kill_range"]

    # ------ 1. attacker 接近 HVT ------
    if attacker.alive:
        dist_to_hvt = attacker.distance_to(hvt.x, hvt.y)
        # 归一化距离变化, 正值=接近
        delta_dist = (prev_dist_to_hvt - dist_to_hvt)
        approach_r = rc["approach_hvt_coef"] * delta_dist / config["obs_range"]
        rewards[0] += approach_r
        # 全队也获得小额接近奖励 (协同)
        for i in range(1, n_agents):
            rewards[i] += approach_r * 0.3
        reward_info["approach_hvt"] = approach_r
    else:
        dist_to_hvt = float('inf')

    # ------ 2. 成功命中 HVT (全队大奖) ------
    if attacker.alive and dist_to_hvt <= kill_range:
        for i in range(n_agents):
            rewards[i] += rc["hit_hvt_bonus"]
        reward_info["hit_hvt"] = rc["hit_hvt_bonus"]

    # ------ 3. FOV 规避 ------
    if attacker.alive:
        fov_count = 0
        for intc in interceptors:
            if intc.alive and intc.is_in_fov(
                    attacker.x, attacker.y,
                    config["fov_half_angle"], config["detection_range"]):
                fov_count += 1

        if fov_count == 0:
            # 不在任何FOV中 → 奖励
            rewards[0] += rc["fov_evasion_coef"]
            reward_info["fov_evasion"] = rc["fov_evasion_coef"]
        else:
            # 在FOV中 → 惩罚
            rewards[0] += rc["in_fov_penalty"] * fov_count
            reward_info["in_fov_penalty"] = rc["in_fov_penalty"] * fov_count

        if fov_count >= 2:
            multi_p = rc["multi_fov_penalty_coef"] * fov_count
            rewards[0] += multi_p
            reward_info["multi_fov_penalty"] = multi_p

    # ------ 4. 护卫击杀拦截器 ------
    if escort_kill_events:
        for event in escort_kill_events:
            esc_idx = event["escort_idx"]
            rewards[1 + esc_idx] += rc["escort_kill_interceptor_bonus"]
            # 全队也获得奖励
            for i in range(n_agents):
                if i != 1 + esc_idx:
                    rewards[i] += rc["escort_kill_interceptor_bonus"] * 0.3
        reward_info["escort_kills"] = len(escort_kill_events)

    # ------ 5. 护卫接近拦截器 (引导攻击) ------
    escort_approach_total = 0.0
    for ei, esc in enumerate(escorts):
        if not esc.alive:
            continue
        for intc in interceptors:
            if not intc.alive:
                continue
            d = esc.distance_to(intc.x, intc.y)
            if d < config["detection_range"]:
                # 接近拦截器的奖励 (越近越大)
                approach_intc_r = rc["escort_approach_intc_coef"] * (1.0 - d / config["detection_range"])
                rewards[1 + ei] += approach_intc_r
                escort_approach_total += approach_intc_r

                # 如果在拦截器FOV中, 额外牵制奖励
                if intc.is_in_fov(esc.x, esc.y, config["fov_half_angle"], config["detection_range"]):
                    divert_r = rc["escort_divert_coef"] * (1.0 - d / config["detection_range"])
                    rewards[1 + ei] += divert_r
                    escort_approach_total += divert_r
    reward_info["escort_approach"] = escort_approach_total

    # ------ 6. 存活奖励 ------
    alive_count = int(attacker.alive) + sum(1 for e in escorts if e.alive)
    survival_r = rc["survival_coef"] * alive_count / n_agents
    for i in range(n_agents):
        rewards[i] += survival_r
    reward_info["survival"] = survival_r

    # ------ 7. 团队分散协同 ------
    all_agents = [attacker] + escorts
    alive_agents = [a for a in all_agents if a.alive]
    if len(alive_agents) >= 2:
        positions = np.array([[a.x, a.y] for a in alive_agents])
        center = positions.mean(axis=0)
        spread = np.mean(np.linalg.norm(positions - center, axis=1))
        ideal_spread = 500.0
        spread_r = rc["team_spread_coef"] * np.exp(-((spread - ideal_spread) / ideal_spread) ** 2)
        for i in range(n_agents):
            rewards[i] += spread_r
        reward_info["team_spread"] = spread_r

    # ------ 8. 被击杀惩罚 ------
    if not attacker.alive:
        for i in range(n_agents):
            rewards[i] += rc["attacker_killed_penalty"]
        reward_info["attacker_killed"] = rc["attacker_killed_penalty"]

    for ei, esc in enumerate(escorts):
        if not esc.alive:
            rewards[1 + ei] += rc["escort_killed_penalty"]

    # ------ 9. 过载平滑惩罚 ------
    smooth_penalty = 0.0
    for i, agent in enumerate(all_agents):
        if agent.alive:
            # 用当前过载的绝对值作为平滑度指标 (过载越大越不平滑)
            overload_mag = abs(agent.ny)
            sp = rc["smooth_action_coef"] * overload_mag
            rewards[i] += sp
            smooth_penalty += sp
    reward_info["smooth_penalty"] = smooth_penalty

    # ------ 10. 每步惩罚 ------
    for i in range(n_agents):
        rewards[i] += rc["step_penalty"]

    return rewards, reward_info


def compute_costs(attacker, escorts, interceptors, config):
    """计算安全约束 cost"""
    cc = config["cost"]
    n_agents = 1 + len(escorts)
    costs = [0.0] * n_agents
    cost_info = {}

    all_trained = [attacker] + escorts
    kill_range = config["kill_range"]

    # 1. attacker FOV 暴露
    fov_cost = 0.0
    if attacker.alive:
        for intc in interceptors:
            if intc.alive and intc.is_in_fov(
                    attacker.x, attacker.y,
                    config["fov_half_angle"], config["detection_range"]):
                fov_cost += cc["fov_exposure"]
    costs[0] += fov_cost
    cost_info["attacker_fov"] = fov_cost

    # 2. 近距危险 (attacker 离拦截器太近)
    danger_range = 300.0  # 危险距离
    danger_cost = 0.0
    if attacker.alive:
        for intc in interceptors:
            if intc.alive and attacker.distance_to(intc.x, intc.y) < danger_range:
                danger_cost += cc["danger_zone"]
    costs[0] += danger_cost
    cost_info["attacker_danger"] = danger_cost

    # 3. escorts 危险
    for ei, esc in enumerate(escorts):
        if esc.alive:
            for intc in interceptors:
                if intc.alive and esc.distance_to(intc.x, intc.y) < danger_range:
                    costs[1 + ei] += cc["escort_danger"]

    # 4. 碰撞 (友军间)
    collision_total = 0.0
    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            ai, aj = all_trained[i], all_trained[j]
            if ai.alive and aj.alive and ai.distance_to(aj.x, aj.y) < config["collision_range"]:
                costs[i] += cc["collision"]
                costs[j] += cc["collision"]
                collision_total += cc["collision"]
    cost_info["collision"] = collision_total

    # 5. 边界
    map_size = config["map_size"]
    for i, agent in enumerate(all_trained):
        if agent.alive and (abs(agent.x) > map_size or abs(agent.y) > map_size):
            costs[i] += cc["boundary"]

    # 6. 速度越界
    for i, agent in enumerate(all_trained):
        if agent.alive:
            if agent.v <= agent.params["v_min"] * 1.01 or agent.v >= agent.params["v_max"] * 0.99:
                costs[i] += cc["speed_violation"]

    return costs, cost_info

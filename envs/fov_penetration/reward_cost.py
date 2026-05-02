"""FOV Penetration Environment - Reward & Cost V38
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
# Main: compute_rewards V38 — 目标优先 + 终端命中友好
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
        V38 奖励:
            1) 强化朝向HVT的 2D/3D 几何一致性
            2) 显式惩罚回头远离目标
            3) 增加团队最小距离进度奖励, 直接优化“至少一机突防”
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
    closing_speeds = [0.0] * n_off

    # ==========================================================
    # V58: 分阶段门控缓存 + 主攻索引缓存
    # ==========================================================
    stage_far_dist = float(rc.get("stage_far_dist", 900.0))
    stage_near_dist = float(rc.get("stage_near_dist", 180.0))
    if stage_near_dist >= stage_far_dist:
        stage_near_dist = max(stage_far_dist - 1.0, 1.0)
    stage_span = max(stage_far_dist - stage_near_dist, 1.0)

    heading_errs = [0.0] * n_off
    gamma_errs = [0.0] * n_off
    heading_aligns = [0.0] * n_off
    gamma_aligns = [0.0] * n_off
    stage_far_w = [0.0] * n_off
    stage_mid_w = [0.0] * n_off
    stage_near_w = [0.0] * n_off

    def _stage_weights(d):
        near_w = float(np.clip((stage_near_dist - d) / max(stage_near_dist, 1.0), 0.0, 1.0))
        far_w = float(np.clip((d - stage_far_dist) / stage_span, 0.0, 1.0))
        mid_w = float(np.clip(1.0 - near_w - far_w, 0.0, 1.0))
        s = far_w + mid_w + near_w
        if s > 1e-8:
            far_w /= s
            mid_w /= s
            near_w /= s
        return far_w, mid_w, near_w

    primary_idx = -1
    primary_dist = float("inf")
    alive_def_hvt_dists = [
        d.distance_to(hvt_x, hvt_y, hvt_z)
        for d in defensives
        if d.alive
    ]
    nearest_def_hvt_dist = min(alive_def_hvt_dists) if alive_def_hvt_dists else float("inf")
    phase_terminal = [False] * n_off
    for i, off in enumerate(offensives):
        if not (off.alive and not off.hit_hvt):
            continue
        dx = hvt_x - off.x
        dy = hvt_y - off.y
        horiz = max(np.sqrt(dx * dx + dy * dy), 1e-3)

        bearing = np.arctan2(dy, dx)
        heading_err = (bearing - off.heading + np.pi) % (2 * np.pi) - np.pi
        desired_gamma = np.arctan2(hvt_z - off.z, horiz)
        gamma_err = (desired_gamma - off.gamma + np.pi) % (2 * np.pi) - np.pi

        heading_errs[i] = heading_err
        gamma_errs[i] = gamma_err
        heading_aligns[i] = np.cos(heading_err)
        gamma_aligns[i] = np.cos(gamma_err)

        fw, mw, nw = _stage_weights(cur_dists[i])
        stage_far_w[i], stage_mid_w[i], stage_near_w[i] = fw, mw, nw

        if cur_dists[i] < primary_dist:
            primary_dist = cur_dists[i]
            primary_idx = i
        phase_terminal[i] = bool(cur_dists[i] <= nearest_def_hvt_dist)

    def _stage_scale(prefix, i):
        return (
            stage_far_w[i] * float(rc.get(f"{prefix}_far", 1.0))
            + stage_mid_w[i] * float(rc.get(f"{prefix}_mid", 1.0))
            + stage_near_w[i] * float(rc.get(f"{prefix}_near", 1.0))
        )

    def _phase_component_scale(prefix, i):
        """Optional hard split between penetration rewards and terminal rewards.

        Defaults are neutral. Profiles can set component-specific keys like
        ``approach_phase_scale_terminal`` or fall back to
        ``legacy_phase_scale_terminal``.
        """
        phase_name = "terminal" if phase_terminal[i] else "penetration"
        return float(rc.get(
            f"{prefix}_phase_scale_{phase_name}",
            rc.get(f"legacy_phase_scale_{phase_name}", 1.0),
        ))

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
            r_app *= _stage_scale("approach_stage_scale", i)
            r_app *= _phase_component_scale("approach", i)

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
            # cos(heading_err): 1=完全对准, -1=完全背对
            r_head = lambda_heading * heading_aligns[i]
            r_head *= _stage_scale("heading_stage_scale", i)
            r_head *= _phase_component_scale("heading", i)
            rewards[i] += r_head
            total_heading += r_head
    reward_info["reward_heading_align"] = total_heading

    # ==========================================================
    # 核心奖励 2b: 俯仰对准HVT (gamma_alignment)
    # ==========================================================
    lambda_gamma = rc.get("lambda_gamma_align", 0.25)
    total_gamma = 0.0
    for i, off in enumerate(offensives):
        if off.alive and not off.hit_hvt:
            r_gamma = lambda_gamma * gamma_aligns[i]
            r_gamma *= _stage_scale("gamma_stage_scale", i)
            r_gamma *= _phase_component_scale("gamma", i)
            rewards[i] += r_gamma
            total_gamma += r_gamma
    reward_info["reward_gamma_align"] = total_gamma

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
            pen = lambda_heading_pen * (heading_errs[i] / np.pi) ** 2
            pen *= _stage_scale("heading_penalty_stage_scale", i)
            pen *= _phase_component_scale("heading_penalty", i)
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
            closing_speeds[i] = Vc
            # 正闭合速度 → 奖励, 负闭合速度 → 惩罚
            r_close = lambda_closing * Vc / vel_range
            r_close *= _stage_scale("closing_stage_scale", i)
            r_close *= _phase_component_scale("closing", i)
            rewards[i] += r_close
            total_closing += r_close
    reward_info["reward_closing_speed"] = total_closing

    # ==========================================================
    # 核心惩罚: 反回头惩罚 (no_retreat)
    # ==========================================================
    # 用负闭合速度直接惩罚“远离目标”，比距离差分更稳定
    lambda_no_retreat = rc.get("lambda_no_retreat", 1.2)
    retreat_speed_ref = max(rc.get("retreat_speed_ref", 35.0), 1.0)
    total_retreat = 0.0
    for i, off in enumerate(offensives):
        if off.alive and not off.hit_hvt:
            retreat_pen = lambda_no_retreat * max(-closing_speeds[i], 0.0) / retreat_speed_ref
            retreat_pen *= _stage_scale("no_retreat_stage_scale", i)
            retreat_pen *= _phase_component_scale("no_retreat", i)
            rewards[i] -= retreat_pen
            total_retreat += retreat_pen
    reward_info["penalty_no_retreat"] = total_retreat

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
            prox_pen *= _stage_scale("proximity_stage_scale", i)
            prox_pen *= _phase_component_scale("proximity", i)
            rewards[i] -= prox_pen
            total_prox += prox_pen
    reward_info["penalty_proximity"] = total_prox

    # ==========================================================
    # V44 新增 A: 过冲惩罚 (overshoot_penalty)
    # ==========================================================
    # 当 d < overshoot_trigger 且闭合速度为负 (在远离HVT) 时，重罚
    # 针对 trim 轨迹“掠过HVT后不回头”的问题 — 远比 closing<0 的一般远离要重
    lambda_overshoot = rc.get("lambda_overshoot", 5.0)
    overshoot_trigger = rc.get("overshoot_trigger_dist", 800.0)
    overshoot_near_boost = rc.get("overshoot_near_boost", 1.0)
    total_overshoot = 0.0
    for i, off in enumerate(offensives):
        if off.alive and not off.hit_hvt and cur_dists[i] < overshoot_trigger:
            if closing_speeds[i] < 0:
                # 越近越重： (1 - d/trigger) 放大
                near_factor = max(1.0 - cur_dists[i] / overshoot_trigger, 0.0)
                pen = lambda_overshoot * (-closing_speeds[i]) / vel_range * (1.0 + 4.0 * near_factor)
                pen *= (1.0 + (overshoot_near_boost - 1.0) * stage_near_w[i])
                pen *= _phase_component_scale("overshoot", i)
                rewards[i] -= pen
                total_overshoot += pen
    reward_info["penalty_overshoot"] = total_overshoot

    # ==========================================================
    # V44 新增 B: 稠密近靶指引 (proximity_dense_bonus)
    # ==========================================================
    # exp(-d/sigma) 越接近越大; sigma=200m 时: d=5m→0.975, 100m→0.61, 300m→0.22, 800m→0.018
    # 这个信号在 trim 直飞下总量~0（d 一直>=200），只有末端机动收紧 d 才能拿到大头
    # 相比 approach 增量，它是纯粵度信号，actor 必须主动向 d=0 优化
    lambda_dense = rc.get("lambda_proximity_dense", 8.0)
    sigma_dense = rc.get("proximity_dense_sigma", 200.0)
    total_dense = 0.0
    for i, off in enumerate(offensives):
        if off.alive and not off.hit_hvt:
            r_dense = lambda_dense * np.exp(-cur_dists[i] / sigma_dense)
            r_dense *= _stage_scale("dense_stage_scale", i)
            r_dense *= _phase_component_scale("dense", i)
            rewards[i] += r_dense
            total_dense += r_dense
    reward_info["reward_proximity_dense"] = total_dense

    # ==========================================================
    # V56A 新增: 近距尖峰奖励 (near_strike_bonus)
    # ==========================================================
    # exp(-(d/sigma)^2): d=200m→~0, 50m→0.062, 20m→0.64, 5m→0.97
    # default lambda=0 → 不影响历史 baseline; v56A 由 FOV_REWARD_PROFILE 打开为 20.
    lambda_near = rc.get("lambda_near_strike", 0.0)
    sigma_near = rc.get("near_strike_sigma", 30.0)
    near_active_dist = rc.get("near_strike_active_dist", 1e9)
    near_min_closing = rc.get("near_strike_min_closing", -1e9)
    near_negative_scale = rc.get("near_strike_negative_scale", 1.0)
    total_near = 0.0
    if lambda_near > 0.0 and sigma_near > 0.0:
        for i, off in enumerate(offensives):
            if off.alive and not off.hit_hvt and cur_dists[i] <= near_active_dist:
                ratio = cur_dists[i] / sigma_near
                r_near = lambda_near * float(np.exp(-ratio * ratio))
                if closing_speeds[i] < near_min_closing:
                    r_near *= near_negative_scale
                r_near *= _stage_scale("near_strike_stage_scale", i)
                r_near *= _phase_component_scale("near_strike", i)
                rewards[i] += r_near
                total_near += r_near
    reward_info["reward_near_strike"] = total_near

    # ==========================================================
    # V58 新增: 近距横偏惩罚 (lateral_miss_penalty)
    # ==========================================================
    lambda_lateral = rc.get("lambda_lateral_miss", 0.0)
    lateral_active_dist = max(rc.get("lateral_miss_active_dist", 300.0), 1.0)
    total_lateral = 0.0
    if lambda_lateral > 0.0:
        for i, off in enumerate(offensives):
            if not (off.alive and not off.hit_hvt):
                continue
            if cur_dists[i] >= lateral_active_dist:
                continue
            near_factor = (lateral_active_dist - cur_dists[i]) / lateral_active_dist
            miss_ratio = abs(np.sin(heading_errs[i])) + 0.5 * abs(np.sin(gamma_errs[i]))
            pen = lambda_lateral * near_factor * miss_ratio
            pen *= _stage_scale("lateral_miss_stage_scale", i)
            pen *= _phase_component_scale("lateral_miss", i)
            rewards[i] -= pen
            total_lateral += pen
    reward_info["penalty_lateral_miss"] = total_lateral

    # ==========================================================
    # 团队奖励: 队伍最小距离进度 (team_min_progress)
    # ==========================================================
    # 直接奖励“至少有人持续接近目标”，减少全队绕飞局部最优
    lambda_team_progress = rc.get("lambda_team_min_progress", 0.0)
    team_primary_scale = rc.get("team_primary_scale", 1.0)
    team_support_scale = rc.get("team_support_scale", 1.0)
    total_team_progress = 0.0
    team_dists = [d for d in cur_dists if np.isfinite(d)]
    if lambda_team_progress > 0.0 and prev_team_min_dist is not None and team_dists:
        team_min_dist = min(team_dists)
        delta_team = prev_team_min_dist - team_min_dist
        team_r = lambda_team_progress * delta_team / max(approach_norm, 1.0)
        for i, off in enumerate(offensives):
            if off.alive and not off.hit_hvt:
                coeff = team_primary_scale if i == primary_idx else team_support_scale
                r_i = team_r * coeff
                r_i *= _phase_component_scale("team_progress", i)
                rewards[i] += r_i
                total_team_progress += r_i
    reward_info["reward_team_min_progress"] = total_team_progress

    # ==========================================================
    # V58 新增: 主攻机个体进度奖励 (primary_delta_progress)
    # ==========================================================
    lambda_primary_delta = rc.get("lambda_primary_delta_progress", 0.0)
    primary_regress_scale = rc.get("primary_regress_scale", 1.5)
    r_primary_delta = 0.0
    if (
        lambda_primary_delta > 0.0
        and primary_idx >= 0
        and prev_dists_to_hvt[primary_idx] < float("inf")
        and np.isfinite(cur_dists[primary_idx])
    ):
        delta_primary = prev_dists_to_hvt[primary_idx] - cur_dists[primary_idx]
        if delta_primary >= 0.0:
            r_primary_delta = lambda_primary_delta * delta_primary / max(approach_norm, 1.0)
            r_primary_delta *= _phase_component_scale("primary_delta", primary_idx)
        else:
            r_primary_delta = lambda_primary_delta * primary_regress_scale * delta_primary / max(approach_norm, 1.0)
            r_primary_delta *= _phase_component_scale("primary_delta", primary_idx)
        rewards[primary_idx] += r_primary_delta
    reward_info["reward_primary_delta_progress"] = r_primary_delta
    reward_info["primary_attacker_idx"] = primary_idx

    # ==========================================================
    # V60: phase-role shaping for decoy cover and PN-like terminal attack
    # ==========================================================
    lambda_phase_los = rc.get("lambda_phase_terminal_los", 0.0)
    phase_los_sigma = max(rc.get("phase_terminal_los_sigma", 0.08), 1e-4)
    lambda_phase_progress = rc.get("lambda_phase_terminal_progress", 0.0)
    lambda_phase_pn_action = rc.get("lambda_phase_terminal_pn_action", 0.0)
    phase_pn_gain = rc.get("phase_terminal_pn_gain", 4.0)
    phase_pn_obs_gain = rc.get("phase_terminal_pn_obs_gain", 0.0)
    phase_pn_obs_sign = float(rc.get("phase_terminal_pn_obs_sign", 1.0))
    phase_pn_err_scale = rc.get("phase_terminal_pn_err_scale", 1.0)
    phase_pn_align_scale = rc.get("phase_terminal_pn_align_scale", 1.0)
    phase_pn_active_dist = rc.get("phase_terminal_pn_active_dist", 1400.0)
    phase_pn_min_gate = rc.get("phase_terminal_pn_min_gate", 0.20)
    phase_pn_max_action = rc.get("phase_terminal_pn_max_action", 0.90)
    phase_pn_deadband = rc.get("phase_terminal_pn_deadband", 0.005)
    phase_pn_score_clip = rc.get("phase_terminal_pn_score_clip", 1.0)
    phase_pn_closing_ref = max(rc.get("phase_terminal_pn_closing_ref", 45.0), 1.0)
    phase_nonprimary_scale = rc.get("phase_nonprimary_terminal_scale", 0.15)
    lambda_decoy_lock = rc.get("lambda_phase_decoy_lock", 0.0)
    lambda_primary_lock_pen = rc.get("lambda_phase_primary_lock_penalty", 0.0)

    total_phase_los = 0.0
    total_phase_progress = 0.0
    total_phase_pn_action = 0.0
    total_decoy_lock = 0.0
    total_primary_lock_pen = 0.0
    if (
        lambda_phase_los > 0.0
        or lambda_phase_progress > 0.0
        or lambda_phase_pn_action > 0.0
        or lambda_decoy_lock > 0.0
        or lambda_primary_lock_pen > 0.0
    ):
        for i, off in enumerate(offensives):
            if not (off.alive and not off.hit_hvt):
                continue

            is_primary = (i == primary_idx)
            terminal_scale = 1.0 if is_primary else phase_nonprimary_scale

            if phase_terminal[i] and terminal_scale > 0.0:
                r_vec = np.array([hvt_x - off.x, hvt_y - off.y, hvt_z - off.z], dtype=np.float64)
                rho = max(float(np.linalg.norm(r_vec)), 1e-6)
                v_vec = _vel3d(off)
                omega_los = float(np.linalg.norm(np.cross(r_vec, v_vec)) / max(rho * rho, 1e-6))
                closing_gate = max(closing_speeds[i], 0.0) / vel_range
                los_bonus = (
                    lambda_phase_los
                    * terminal_scale
                    * float(np.exp(-((omega_los / phase_los_sigma) ** 2)))
                    * closing_gate
                )
                rewards[i] += los_bonus
                total_phase_los += los_bonus

                if prev_dists_to_hvt[i] < float("inf"):
                    delta_i = prev_dists_to_hvt[i] - cur_dists[i]
                    if delta_i >= 0.0:
                        progress_bonus = lambda_phase_progress * terminal_scale * delta_i / max(approach_norm, 1.0)
                    else:
                        progress_bonus = 2.0 * lambda_phase_progress * terminal_scale * delta_i / max(approach_norm, 1.0)
                    rewards[i] += progress_bonus
                    total_phase_progress += progress_bonus

                if lambda_phase_pn_action > 0.0 and raw_actions is not None and rho <= phase_pn_active_dist:
                    act = np.array(raw_actions[i], dtype=np.float32).flatten()
                    if len(act) >= 3:
                        rx, ry, rz = r_vec
                        vx, vy, vz = v_vec
                        rho_h_sq = rx * rx + ry * ry
                        rho_h = float(np.sqrt(max(rho_h_sq, 1e-6)))
                        d_az = float((rx * vy - ry * vx) / max(rho_h_sq, 1e-6))
                        d_el = float((vz * rho_h_sq - rz * (rx * vx + ry * vy)) /
                                     (max(rho * rho, 1e-6) * rho_h))
                        if abs(d_az) < phase_pn_deadband:
                            d_az = 0.0
                        if abs(d_el) < phase_pn_deadband:
                            d_el = 0.0

                        params = getattr(off, "params", config.get("offensive", {}))
                        an_pitch_max = max(float(params.get("an_pitch_max", 2.5 * G)), G + 1e-3)
                        an_yaw_max = max(float(params.get("an_yaw_max", 2.5 * G)), 1e-3)
                        closing_pos = max(closing_speeds[i], 0.0)
                        if phase_pn_obs_gain > 0.0:
                            omega_norm = max(float(config.get("phase_terminal_obs_omega_norm", 0.5)), 1e-3)
                            desired_yaw = phase_pn_obs_sign * phase_pn_obs_gain * d_az / omega_norm
                            desired_pitch = phase_pn_obs_sign * phase_pn_obs_gain * d_el / omega_norm
                        else:
                            desired_an_yaw = phase_pn_gain * closing_pos * d_az
                            desired_an_pitch = G * np.cos(off.gamma) + phase_pn_gain * closing_pos * d_el
                            desired_yaw = desired_an_yaw / an_yaw_max
                            if desired_an_pitch >= G:
                                desired_pitch = (desired_an_pitch - G) / max(an_pitch_max - G, 1e-3)
                            else:
                                desired_pitch = (desired_an_pitch - G) / max(an_pitch_max + G, 1e-3)
                        desired_yaw = float(np.clip(desired_yaw, -phase_pn_max_action, phase_pn_max_action))
                        desired_pitch = float(np.clip(desired_pitch, -phase_pn_max_action, phase_pn_max_action))

                        desired_norm = desired_pitch * desired_pitch + desired_yaw * desired_yaw
                        err = (float(act[1]) - desired_pitch) ** 2 + (float(act[2]) - desired_yaw) ** 2
                        align = desired_pitch * float(act[1]) + desired_yaw * float(act[2])
                        score = phase_pn_align_scale * align - phase_pn_err_scale * err
                        if phase_pn_obs_gain <= 0.0:
                            score += desired_norm
                        score = float(np.clip(score, -phase_pn_score_clip, phase_pn_score_clip))
                        range_gate = max(phase_pn_min_gate, 1.0 - rho / max(phase_pn_active_dist, 1.0))
                        closing_gate = float(np.clip(closing_pos / phase_pn_closing_ref, 0.0, 1.0))
                        pn_action_bonus = lambda_phase_pn_action * terminal_scale * range_gate * closing_gate * score
                        rewards[i] += pn_action_bonus
                        total_phase_pn_action += pn_action_bonus

            if not phase_terminal[i]:
                lock_count = float(min(getattr(off, "locked_by_count", 0), max(len(defensives), 1)))
                lock_norm = lock_count / max(len(defensives), 1)
                if is_primary:
                    primary_pen = lambda_primary_lock_pen * lock_norm
                    rewards[i] -= primary_pen
                    total_primary_lock_pen += primary_pen
                else:
                    decoy_bonus = lambda_decoy_lock * lock_norm
                    rewards[i] += decoy_bonus
                    total_decoy_lock += decoy_bonus

    reward_info["reward_phase_terminal_los"] = total_phase_los
    reward_info["reward_phase_terminal_progress"] = total_phase_progress
    reward_info["reward_phase_terminal_pn_action"] = total_phase_pn_action
    reward_info["reward_phase_decoy_lock"] = total_decoy_lock
    reward_info["penalty_phase_primary_lock"] = total_primary_lock_pen
    reward_info["phase_terminal_count"] = sum(1 for x in phase_terminal if x)
    reward_info["phase_nearest_def_hvt_dist"] = nearest_def_hvt_dist

    # ==========================================================
    # V67: terminal-only collision-course CPA precision shaping
    # ==========================================================
    lambda_terminal_cpa = rc.get("lambda_phase_terminal_cpa", 0.0)
    terminal_cpa_sigma = max(rc.get("phase_terminal_cpa_sigma", 25.0), 1e-3)
    terminal_cpa_active_dist = rc.get("phase_terminal_cpa_active_dist", 1200.0)
    terminal_cpa_min_gate = rc.get("phase_terminal_cpa_min_gate", 0.25)
    terminal_cpa_min_closing = rc.get("phase_terminal_cpa_min_closing", 0.0)
    terminal_cpa_closing_ref = max(rc.get("phase_terminal_cpa_closing_ref", 45.0), 1.0)
    total_terminal_cpa = 0.0
    if lambda_terminal_cpa > 0.0:
        for i, off in enumerate(offensives):
            if not (off.alive and not off.hit_hvt and phase_terminal[i]):
                continue
            if cur_dists[i] > terminal_cpa_active_dist:
                continue
            closing_pos = max(closing_speeds[i], 0.0)
            if closing_pos < terminal_cpa_min_closing:
                continue
            r_vec = np.array([hvt_x - off.x, hvt_y - off.y, hvt_z - off.z], dtype=np.float64)
            v_vec = _vel3d(off)
            speed = max(float(np.linalg.norm(v_vec)), 1e-6)
            time_to_cpa = float(np.dot(r_vec, v_vec) / max(speed * speed, 1e-6))
            if time_to_cpa <= 0.0:
                continue
            cpa_miss = float(np.linalg.norm(np.cross(r_vec, v_vec)) / speed)
            range_gate = max(terminal_cpa_min_gate, 1.0 - cur_dists[i] / max(terminal_cpa_active_dist, 1.0))
            closing_gate = float(np.clip(closing_pos / terminal_cpa_closing_ref, 0.0, 1.0))
            terminal_scale = 1.0 if i == primary_idx else phase_nonprimary_scale
            r_cpa = (
                lambda_terminal_cpa
                * terminal_scale
                * range_gate
                * closing_gate
                * float(np.exp(-((cpa_miss / terminal_cpa_sigma) ** 2)))
            )
            r_cpa *= _phase_component_scale("terminal_cpa", i)
            rewards[i] += r_cpa
            total_terminal_cpa += r_cpa
    reward_info["reward_phase_terminal_cpa"] = total_terminal_cpa

    # ==========================================================
    # V37/V5: 偏航加速度正则化 (防止无意义转弯)
    # ==========================================================
    # 惩罚|action[2]| — 鼓励策略输出近零的偏航指令
    lambda_mu_reg = rc.get("lambda_mu_regularize", 0.0)
    yaw_reg_relax_dist = max(rc.get("yaw_reg_relax_dist", 350.0), 1.0)
    yaw_reg_near_factor = np.clip(rc.get("yaw_reg_near_factor", 0.2), 0.0, 1.0)
    total_mu_reg = 0.0
    if lambda_mu_reg > 0 and raw_actions is not None:
        for i, off in enumerate(offensives):
            if off.alive and not off.hit_hvt:
                act = np.array(raw_actions[i], dtype=np.float32).flatten()
                if len(act) >= 3:
                    dist_scale = np.clip(cur_dists[i] / yaw_reg_relax_dist,
                                         yaw_reg_near_factor, 1.0)
                    yaw_pen = lambda_mu_reg * abs(act[2]) * dist_scale
                    yaw_pen *= _phase_component_scale("yaw_reg", i)
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
        hit_bonus = rc.get("hit_hvt_bonus", 6000.0) * len(hit_events)
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

    # ==========================================================
    # 终端奖励
    # V44 重塑: lambda_terminal_dist 从“线性衰减到 2000m=0”改为“指数衰减 exp(-min_d/sigma)”
    #         默认 sigma=80m: min_d=5m→0.94, 50m→0.54, 200m→0.082, 500m→0.002
    #         这样 actor 必须把 min_d 从 200m 推到 50m 才能拿到实质变化，逆逗末端机动
    lambda_hit = rc.get("lambda_terminal_hit", 800.0)
    lambda_dist = rc.get("lambda_terminal_dist", 500.0)
    sigma_terminal = rc.get("terminal_dist_sigma", 80.0)

    # 命中奖励
    terminal_r = lambda_hit * N_hit

    # 距离奖励: V44 指数衰减—末端越近越陈险, 逆逗 actor 从 200m 供给到 5m
    #            同时加一个“严重脱靶”线性项: min_d>500m 时线性扣分 (抖不出反馈)
    dist_reward = lambda_dist * np.exp(-min_dist / sigma_terminal)
    if min_dist > 500.0:
        dist_reward -= lambda_dist * 0.5 * (min_dist - 500.0) / 1500.0
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

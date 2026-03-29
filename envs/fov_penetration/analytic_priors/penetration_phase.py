"""
Module 3b: Effective Penetration — 有效突防数量最大化 (V22)
=============================================================
替代旧的 penetration_success_score 单一评估。

核心改动 (youneedread 5.3-5.4):
  P_i_pen:  个体突防概率 (四项构成)
    1. cone_safety    (第一部分: 脱锥安全)
    2. threat_distance (第二部分: 局部威胁距离弱化)
    3. redirected_attention (第二部分: 注意力转移)
    4. E_i_esc        (第三部分: 近距逃逸能力)

  N_eff:    有效突防数量 = sum(P_i_pen)
  N_loss:   损失数量
  N_waste:  无效牺牲 = killed - effective_sacrifice
  terminal_group_value: N_eff 为主目标的终端收益

保留旧接口 compute_penetration_success_score 向后兼容。
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + np.exp(-x))
    ex = np.exp(x)
    return ex / (1.0 + ex)


# -----------------------------------------------------------------------
# 旧接口 (向后兼容, 保留)
# -----------------------------------------------------------------------

def compute_penetration_success_score(
    rho_norm: float,
    closing_norm: float,
    omega_norm: float,
    omega_dot_norm: float,
    pn_hint_norm: float,
    cone_risk_norm: float,
    mismatch_norm: float,
    detected_norm: float,
    score_bias: float = -0.35,
    score_scale: float = 2.2,
) -> float:
    """旧接口保留 — 单一综合评分 (V21 兼容)"""
    rho_term = 1.0 - np.clip(rho_norm, 0.0, 1.0)
    closing_term = max(np.clip(closing_norm, -1.0, 1.0), 0.0)
    omega_term = np.clip(omega_norm, 0.0, 1.0)
    omega_dot_term = 1.0 - min(abs(np.clip(omega_dot_norm, -1.0, 1.0)), 1.0)
    pn_term = max(np.clip(pn_hint_norm, -1.0, 1.0), 0.0)
    cone_term = np.clip(cone_risk_norm, 0.0, 1.0)
    mismatch_term = np.clip(mismatch_norm, 0.0, 1.0)
    detect_term = np.clip(detected_norm, 0.0, 1.0)

    raw = (
        score_bias
        + 1.6 * rho_term
        + 1.2 * closing_term
        + 0.6 * omega_dot_term
        + 0.6 * pn_term
        - 0.3 * omega_term
        - 1.0 * cone_term
        - 0.6 * mismatch_term
        - 0.8 * detect_term
    )
    score = _sigmoid(score_scale * raw)
    return float(np.clip(score, 0.0, 1.0))


# -----------------------------------------------------------------------
# V22: P_i_pen — 个体突防概率
# -----------------------------------------------------------------------

def compute_P_pen(
    off_idx: int,
    off,
    hvt,
    config: dict,
    pen_cfg: dict,
    cone_risk: float = 0.0,
    lock_pressure: float = 0.0,
    n_locked_by: int = 0,
    E_i_esc: float = 0.0,
) -> Tuple[float, Dict]:
    """
    P_i_pen = sigma( kappa * (
        w_cone * cone_safety
      + w_threat * threat_dist_factor
      + w_redirect * redirect_factor
      + w_escape * E_i_esc
      - c ) )

    Args:
        off_idx: agent index
        off: offensive Aircraft
        hvt: HVT
        config: env config
        pen_cfg: effective_penetration config
        cone_risk: psi_i from Module 1
        lock_pressure: soft lock probability sum from Module 2
        n_locked_by: how many interceptors actually locked onto this agent
        E_i_esc: escape ability from Module 3a

    Returns: (P_pen, detail_dict)
    """
    if not off.alive or off.hit_hvt:
        return 0.0, {}

    w_cone = pen_cfg.get("P_pen_cone_weight", 0.25)
    w_threat = pen_cfg.get("P_pen_threat_weight", 0.25)
    w_redirect = pen_cfg.get("P_pen_redirect_weight", 0.25)
    w_escape = pen_cfg.get("P_pen_escape_weight", 0.25)
    kappa = pen_cfg.get("kappa_h", 3.0)
    c_bias = pen_cfg.get("kappa_c", 0.3)
    obs_range = max(config.get("obs_range", 5000.0), 1.0)

    # (1) cone_safety: 脱锥安全 (高 → 安全)
    cone_safety = 1.0 - np.clip(cone_risk, 0.0, 2.0) / 2.0

    # (2) threat_distance: 离最近拦截器越远越安全
    #     近似: 没有拦截器锁定 + lock_pressure低 → 安全
    threat_dist_factor = max(0.0, 1.0 - lock_pressure - 0.5 * n_locked_by)
    threat_dist_factor = np.clip(threat_dist_factor, 0.0, 1.0)

    # (3) redirected_attention: 注意力被转移 → 对自己有利
    #     lock_pressure低说明注意力被诱饵吸走
    redirect_factor = 1.0 - np.clip(lock_pressure, 0.0, 1.5) / 1.5

    # (4) E_i_esc: 近距逃逸能力
    escape_factor = np.clip(E_i_esc, 0.0, 2.0) / 2.0

    raw = (w_cone * cone_safety
           + w_threat * threat_dist_factor
           + w_redirect * redirect_factor
           + w_escape * escape_factor
           - c_bias)

    P_pen = _sigmoid(kappa * raw)

    detail = {
        "cone_safety": cone_safety,
        "threat_dist_factor": threat_dist_factor,
        "redirect_factor": redirect_factor,
        "escape_factor": escape_factor,
        "P_pen_raw": raw,
    }
    return float(P_pen), detail


# -----------------------------------------------------------------------
# V22: N_eff, N_loss, N_waste, terminal_group_value
# -----------------------------------------------------------------------

def compute_effective_penetration(
    offensives: list,
    defensives: list,
    hvt,
    config: dict,
    ap_config: dict,
    cone_risk_per_agent: Optional[List[float]] = None,
    lock_pressure_per_agent: Optional[List[float]] = None,
    locked_by_count_per_agent: Optional[List[int]] = None,
    E_i_esc_per_agent: Optional[List[float]] = None,
    prev_N_eff: Optional[float] = None,
) -> Tuple[float, List[float], float, Dict]:
    """
    计算有效突防相关量。

    Returns:
        pen_reward:      scalar  N_eff增量奖励
        per_agent_reward: per-agent penetration reward
        N_eff:           current effective count (to cache)
        info:            logging dict
    """
    pen_cfg = ap_config.get("effective_penetration", {})
    n_off = len(offensives)
    n_def = len(defensives)
    synergy_exp = pen_cfg.get("synergy_exponent", 1.5)
    N_eff_weight = pen_cfg.get("N_eff_reward_weight", 0.8)
    waste_penalty = pen_cfg.get("N_waste_penalty_weight", 0.3)

    P_pen_list = []
    details = []
    for i, off in enumerate(offensives):
        cone_r = cone_risk_per_agent[i] if cone_risk_per_agent else 0.0
        lock_p = lock_pressure_per_agent[i] if lock_pressure_per_agent else 0.0
        n_lock = locked_by_count_per_agent[i] if locked_by_count_per_agent else 0
        e_esc = E_i_esc_per_agent[i] if E_i_esc_per_agent else 0.0

        P_pen, detail = compute_P_pen(
            i, off, hvt, config, pen_cfg,
            cone_risk=cone_r,
            lock_pressure=lock_p,
            n_locked_by=n_lock,
            E_i_esc=e_esc)
        P_pen_list.append(P_pen)
        details.append(detail)

    # N_eff = sum of P_pen for alive agents
    N_eff = sum(P_pen_list[i] for i in range(n_off)
                if offensives[i].alive and not offensives[i].hit_hvt)

    # N_loss = killed count
    N_loss = sum(1 for off in offensives if not off.alive)

    # N_waste = ineffective losses (killed but low P_pen at death)
    N_waste = 0
    for i, off in enumerate(offensives):
        if not off.alive and not off.hit_hvt:
            # Was not an effective penetrator
            N_waste += 1

    # 协同增强: N_eff^synergy_exp
    synergy_term = N_eff ** synergy_exp if N_eff > 0 else 0.0

    # terminal_group_value
    terminal_group_value = synergy_term - waste_penalty * N_waste

    # 增量奖励
    if prev_N_eff is not None:
        delta_N_eff = N_eff - prev_N_eff
    else:
        delta_N_eff = 0.0

    pen_reward = N_eff_weight * delta_N_eff

    # Per-agent reward: 按 P_pen 权重分配
    per_agent_reward = [0.0] * n_off
    n_alive = max(sum(1 for o in offensives if o.alive and not o.hit_hvt), 1)
    total_P = sum(P_pen_list[i] for i in range(n_off)
                  if offensives[i].alive and not offensives[i].hit_hvt) + 1e-8

    for i in range(n_off):
        if offensives[i].alive and not offensives[i].hit_hvt:
            # 个体贡献: P_pen 占比
            ind_share = pen_reward * 0.6 * P_pen_list[i] / total_P
            team_share = pen_reward * 0.4 / n_alive
            per_agent_reward[i] = ind_share + team_share

    info = {
        "N_eff": N_eff,
        "N_loss": N_loss,
        "N_waste": N_waste,
        "delta_N_eff": delta_N_eff,
        "synergy_term": synergy_term,
        "terminal_group_value": terminal_group_value,
        "pen_reward": pen_reward,
        "P_pen_per_agent": P_pen_list,
        "P_pen_mean": float(np.mean([p for i, p in enumerate(P_pen_list)
                                      if offensives[i].alive])) if any(o.alive for o in offensives) else 0.0,
        "P_pen_max": float(np.max(P_pen_list)) if P_pen_list else 0.0,
    }
    return pen_reward, per_agent_reward, N_eff, info

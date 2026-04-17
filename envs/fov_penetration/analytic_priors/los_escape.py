"""
Module 3a: Near-Distance Escape via LOS Angular Rate Surge (V22)
==================================================================
When an attacker performs a sudden lateral maneuver at close range,
the resulting Line-of-Sight (LOS) angular rate may exceed the
interceptor's maximum tracking capability, causing tracking failure.

V22 升级 (2026-03-29):
  - 新增 E_i_esc = max_j Xi_ij 作为正式输出 → 并入 penetration_phase
  - 新增 per-pair Gamma_ij, Xi_ij 矩阵输出
  - 新增 omega_track_max 矩阵输出
  - 保留旧接口向后兼容

Key quantities:
  omega_los       – current LOS angular rate ||r × v|| / ||r||^2
  omega_track_max – max trackable LOS rate for interceptor j
  omega_los_plus  – predicted LOS rate after applying current action
  Gamma_ij        – tracking mismatch margin  (omega_los_plus - omega_track_max)
  G_near_ij       – near-distance gate  sigmoid(k_rho * (rho_0 - rho))
  Xi_ij           – escape trigger  G_near * [Gamma]+
  E_i_esc         – max_j Xi_ij  (单机层逃逸能力)
"""

import numpy as np
from typing import Dict, List, Tuple

G = 9.81


# -----------------------------------------------------------------------
# Per-pair LOS rate computation
# -----------------------------------------------------------------------

def _velocity_3d(entity) -> np.ndarray:
    """Full 3D velocity vector."""
    cg = np.cos(entity.gamma)
    return np.array([entity.v * cg * np.cos(entity.heading),
                     entity.v * cg * np.sin(entity.heading),
                     entity.v * np.sin(entity.gamma)])


def compute_los_rate(off, defn) -> Tuple[float, np.ndarray, float]:
    """Compute LOS angular rate for pair (i, j).

    Returns:
        omega_los:  scalar LOS angular rate (rad/s)
        v_t_ij:     relative tangential velocity vector (3D)
        rho:        range between pair
    """
    r = np.array([off.x - defn.x, off.y - defn.y, off.z - defn.z])
    rho = np.linalg.norm(r)
    if rho < 1e-3:
        return 0.0, np.zeros(3), rho

    v_off = _velocity_3d(off)
    v_def = _velocity_3d(defn)
    v_rel = v_off - v_def

    # Cross product  r × v
    cross = np.cross(r, v_rel)
    omega_los = np.linalg.norm(cross) / (rho ** 2)

    # Tangential velocity component:  v_t = v_rel - (v_rel . r_hat) * r_hat
    r_hat = r / rho
    v_r = np.dot(v_rel, r_hat) * r_hat
    v_t = v_rel - v_r

    return float(omega_los), v_t, float(rho)


def compute_tracking_limit(defn, omega_sens_max: float = 1e6) -> float:
    """Max trackable LOS angular rate for interceptor j.

    omega_trk_max = min( a_perp_max / V_j,  omega_sens_max )

    a_perp_max is the interceptor's maximum lateral acceleration.
    """
    ny_max = abs(defn.params.get("ny_max", 8.0))
    a_perp_max = ny_max * G
    omega_trk = a_perp_max / max(defn.v, 1.0)
    return min(omega_trk, omega_sens_max)


def compute_escape_margin(
    off, defn,
    v_t_ij: np.ndarray,
    rho: float,
    dt_trigger: float,
    omega_trk_max: float,
) -> float:
    """Compute tracking mismatch margin Gamma_ij.

    Gamma_ij = ||omega_los^+|| - omega_trk_max

    where omega_los^+ ≈ ||v_t + a_t_cmd * dt|| / rho

    a_t_cmd is the component of the attacker's current commanded
    acceleration in the LOS tangential plane.
    """
    if rho < 1e-3:
        return 0.0

    # V5动力学: 从 (ax, an_pitch, an_yaw) 重建惯性系加速度
    # an_yaw: 偏航平面法向加速度 (causing heading change)
    # an_pitch - g*cos(gamma): 俯仰平面净法向加速度 (causing gamma change)
    cg = np.cos(off.gamma)
    sh = np.sin(off.heading)
    ch = np.cos(off.heading)
    sg = np.sin(off.gamma)

    # 偏航法向加速度 → 垂直于速度方向在XY平面内 (旋转90°)
    a_horiz = off.an_yaw
    a_lat_x = -a_horiz * sh
    a_lat_y = a_horiz * ch
    a_lat_z = 0.0

    # 俯仰法向加速度 (扣除重力补偿, 仅保留机动分量)
    a_vert_net = off.an_pitch - G * np.cos(off.gamma)
    a_vert_x = -a_vert_net * sg * ch
    a_vert_y = -a_vert_net * sg * sh
    a_vert_z = a_vert_net * cg

    a_cmd = np.array([a_lat_x + a_vert_x,
                      a_lat_y + a_vert_y,
                      a_lat_z + a_vert_z])

    # Project acceleration onto tangential plane
    r = np.array([off.x - defn.x, off.y - defn.y, off.z - defn.z])
    r_hat = r / max(rho, 1e-6)
    a_r = np.dot(a_cmd, r_hat) * r_hat
    a_t = a_cmd - a_r  # tangential component of commanded acceleration

    # Predicted tangential velocity after dt
    v_t_plus = v_t_ij + a_t * dt_trigger
    omega_los_plus = np.linalg.norm(v_t_plus) / rho

    Gamma = omega_los_plus - omega_trk_max
    return float(Gamma)


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + np.exp(-x))
    else:
        ex = np.exp(x)
        return ex / (1.0 + ex)


# -----------------------------------------------------------------------
# Full escape reward computation
# -----------------------------------------------------------------------

def compute_escape_reward(
    offensives: list,
    defensives: list,
    config: dict,
    ap_config: dict,
) -> Tuple[float, List[float], Dict]:
    """Compute the near-distance escape reward for all attackers.

    V22: 新增 E_i_esc, Gamma_matrix, Xi_matrix, omega_track_max_per_def

    Args:
        offensives:  list of offensive Aircraft
        defensives:  list of defensive Aircraft
        config:      env config
        ap_config:   analytic_priors config

    Returns:
        total_escape_reward: scalar
        per_agent_reward:    list of per-attacker escape reward
        info:                logging dict (含 E_i_esc, Gamma/Xi矩阵)
    """
    dt = config["dt"]
    lambda_E = ap_config.get("escape_reward_weight", 0.2)
    rho_0 = ap_config.get("rho_trigger", 150.0)
    k_rho = ap_config.get("k_rho", 0.1)
    dt_trigger = ap_config.get("dt_trigger", None)
    if dt_trigger is None:
        dt_trigger = dt

    n_off = len(offensives)
    n_def = len(defensives)

    per_agent_reward = [0.0] * n_off
    E_i_esc = [0.0] * n_off          # V22: 单机层逃逸能力
    Gamma_all = []
    Xi_all = []
    near_triggers = 0
    max_threat_indices = []

    # V22: 矩阵输出
    Gamma_matrix = np.zeros((n_off, n_def))
    Xi_matrix = np.zeros((n_off, n_def))
    omega_los_matrix = np.zeros((n_off, n_def))
    omega_track_max_per_def = np.zeros(n_def)

    # 预计算拦截器最大跟踪角速度
    for j, defn in enumerate(defensives):
        if defn.alive:
            omega_track_max_per_def[j] = compute_tracking_limit(defn)

    for i, off in enumerate(offensives):
        if not off.alive:
            max_threat_indices.append(-1)
            continue

        Xi_max = 0.0
        max_j = -1

        for j, defn in enumerate(defensives):
            if not defn.alive:
                continue

            # LOS rate
            omega_los, v_t_ij, rho = compute_los_rate(off, defn)
            omega_los_matrix[i, j] = omega_los

            # Tracking limit
            omega_trk_max = omega_track_max_per_def[j]

            # Escape margin
            Gamma = compute_escape_margin(off, defn, v_t_ij, rho,
                                          dt_trigger, omega_trk_max)
            Gamma_matrix[i, j] = Gamma
            Gamma_all.append(Gamma)

            # Near-distance gate
            G_near = _sigmoid(k_rho * (rho_0 - rho))

            # Escape trigger
            Xi = G_near * max(0.0, Gamma)
            Xi_matrix[i, j] = Xi
            Xi_all.append(Xi)

            if Xi > 0:
                near_triggers += 1

            if Xi > Xi_max:
                Xi_max = Xi
                max_j = j

        # V22: E_i_esc = max_j Xi_ij
        E_i_esc[i] = Xi_max
        per_agent_reward[i] = lambda_E * Xi_max
        max_threat_indices.append(max_j)

    total_escape_reward = sum(per_agent_reward)

    # Per-agent Xi_max values (before lambda_E scaling) for obs
    per_agent_Xi_max = [per_agent_reward[i] / max(lambda_E, 1e-8) for i in range(n_off)]

    info = {
        "escape_reward": total_escape_reward,
        "Gamma_mean": float(np.mean(Gamma_all)) if Gamma_all else 0.0,
        "Gamma_max": float(np.max(Gamma_all)) if Gamma_all else 0.0,
        "Xi_mean": float(np.mean(Xi_all)) if Xi_all else 0.0,
        "Xi_max": float(np.max(Xi_all)) if Xi_all else 0.0,
        "near_trigger_count": near_triggers,
        "max_threat_interceptor_per_agent": max_threat_indices,
        "per_agent_escape_reward": per_agent_reward,
        "_per_agent_Xi_max": per_agent_Xi_max,
        # V22 新增
        "E_i_esc": E_i_esc,
        "_Gamma_matrix": Gamma_matrix,
        "_Xi_matrix": Xi_matrix,
        "_omega_los_matrix": omega_los_matrix,
        "_omega_track_max_per_def": omega_track_max_per_def,
    }
    return total_escape_reward, per_agent_reward, info

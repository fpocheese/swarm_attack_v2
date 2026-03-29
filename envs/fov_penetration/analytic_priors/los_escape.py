"""
Module 3: Near-Distance Escape via LOS Angular Rate Surge
============================================================
When an attacker performs a sudden lateral maneuver at close range,
the resulting Line-of-Sight (LOS) angular rate may exceed the
interceptor's maximum tracking capability, causing tracking failure.

This module evaluates whether the *current action* triggers such
a tracking-loss condition and rewards the agent accordingly.

Key quantities:
  omega_los       – current LOS angular rate ||r × v|| / ||r||^2
  omega_track_max – max trackable LOS rate for interceptor j
  omega_los_plus  – predicted LOS rate after applying current action
  Gamma_ij        – tracking mismatch margin  (omega_los_plus - omega_track_max)
  G_near_ij       – near-distance gate  sigmoid(k_rho * (rho_0 - rho))
  Xi_ij           – escape trigger  G_near * [Gamma]+
  escape_reward   – sum over attackers of max_j Xi_ij

Important: This module does NOT output actions.  It only evaluates
whether the action chosen by the RL policy triggers a favorable
near-distance escape mechanism.
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

    # Attacker's current commanded acceleration
    # ny and nz produce lateral and vertical forces
    # We approximate the full 3D acceleration from overload
    a_x = off.nx * G  # along velocity direction (approx)
    # Lateral acceleration in body frame → need to project to inertial
    cg = np.cos(off.gamma)
    sh = np.sin(off.heading)
    ch = np.cos(off.heading)
    sg = np.sin(off.gamma)

    # Body-frame lateral accel (ny*g) → perpendicular to velocity in XY plane
    # Rotated 90deg from heading
    a_lat_x = -off.ny * G * sh
    a_lat_y = off.ny * G * ch
    a_lat_z = 0.0

    # Body-frame vertical accel  (nz - cos(gamma))*g  → perpendicular to velocity in vertical plane
    a_vert_x = -(off.nz - np.cos(off.gamma)) * G * sg * ch
    a_vert_y = -(off.nz - np.cos(off.gamma)) * G * sg * sh
    a_vert_z = (off.nz - np.cos(off.gamma)) * G * cg

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

    Args:
        offensives:  list of offensive Aircraft
        defensives:  list of defensive Aircraft
        config:      env config
        ap_config:   analytic_priors config

    Returns:
        total_escape_reward: scalar
        per_agent_reward:    list of per-attacker escape reward
        info:                logging dict
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
    Gamma_all = []
    Xi_all = []
    near_triggers = 0
    max_threat_indices = []

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

            # Tracking limit
            omega_trk_max = compute_tracking_limit(defn)

            # Escape margin
            Gamma = compute_escape_margin(off, defn, v_t_ij, rho,
                                          dt_trigger, omega_trk_max)
            Gamma_all.append(Gamma)

            # Near-distance gate
            G_near = _sigmoid(k_rho * (rho_0 - rho))

            # Escape trigger
            Xi = G_near * max(0.0, Gamma)
            Xi_all.append(Xi)

            if Xi > 0:
                near_triggers += 1

            if Xi > Xi_max:
                Xi_max = Xi
                max_j = j

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
    }
    return total_escape_reward, per_agent_reward, info

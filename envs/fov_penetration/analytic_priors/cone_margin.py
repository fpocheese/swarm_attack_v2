"""
Module 1: Multi-Interceptor Cone-Margin Aggregated Cost
=========================================================
Computes the "zero-effort cone-margin" risk for each attacker
against all interceptors, and produces a MACPO safety cost signal.

Key quantities:
  q_ij       – instantaneous cone-boundary margin for pair (i,j)
  q_dot_ij   – time derivative (finite-difference)
  Z_ij       – zero-effort predicted cone margin at terminal time
  Z_tilde_i  – smooth-max aggregation across interceptors
  Psi_i_agg  – dead-zone risk  [Z_tilde_i + M_c]_+
  cone_cost  – sum of Psi_i_agg over all attackers

Approximations used:
  - The interceptor's "radar axis" b_j is taken as its velocity
    unit vector (same as FOV axis in this env).
  - q_dot_ij is estimated via finite difference from stored
    previous-step q values.
  - Equivalent normal accelerations a_I and a_A are approximated
    from the current overload commands (ny, nz) * g.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from .y_system import YSystemCache

G = 9.81


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _velocity_unit_vector(heading: float, gamma: float) -> np.ndarray:
    """Returns the 3D unit velocity vector [vx, vy, vz] / v."""
    cg = np.cos(gamma)
    return np.array([cg * np.cos(heading),
                     cg * np.sin(heading),
                     np.sin(gamma)])


def _relative_vec(off, defn) -> np.ndarray:
    """r_ij = p_i - p_j  (attacker minus interceptor)."""
    return np.array([off.x - defn.x,
                     off.y - defn.y,
                     off.z - defn.z])


# -----------------------------------------------------------------------
# Per-pair computations
# -----------------------------------------------------------------------

def compute_q_ij(off, defn, fov_half_angle: float) -> float:
    """Cone-boundary margin q_ij = (b_j^T r_ij) / ||r_ij|| - cos(alpha_j).

    q_ij > 0  → target inside cone.
    q_ij < 0  → target outside cone.
    """
    r = _relative_vec(off, defn)
    rho = np.linalg.norm(r)
    if rho < 1e-6:
        return 1.0  # effectively inside
    b_j = _velocity_unit_vector(defn.heading, defn.gamma)
    cos_alpha_j = np.cos(fov_half_angle)
    return float(np.dot(b_j, r) / rho - cos_alpha_j)


def compute_q_dot_ij(q_now: float, q_prev: float, dt: float) -> float:
    """Finite-difference estimate of q_dot."""
    if dt <= 0:
        return 0.0
    return (q_now - q_prev) / dt


def compute_Z_ij(q_ij: float, q_dot_ij: float, t_go: float,
                 a_I_j: float, a_A_i: float,
                 y_cache: YSystemCache) -> float:
    """Zero-effort cone-margin prediction.

    Z_ij = Y1 * (q_ij + t_go * q_dot_ij) + Y3 * a_I_j + Y4 * a_A_i
    """
    Y1, Y3, Y4 = y_cache.query(t_go)
    return Y1 * (q_ij + t_go * q_dot_ij) + Y3 * a_I_j + Y4 * a_A_i


def compute_equivalent_normal_accel(entity) -> float:
    """Approximate equivalent normal acceleration magnitude (m/s^2).

    V5动力学: entity.an_pitch 和 entity.an_yaw 分别是俯仰和偏航法向加速度,
    合成为等效法向加速度大小。
    """
    return np.sqrt(entity.an_pitch**2 + entity.an_yaw**2)


# -----------------------------------------------------------------------
# Aggregation across interceptors
# -----------------------------------------------------------------------

def aggregate_Z_i(Z_ij_list: List[float], beta: float) -> float:
    """Smooth-max (log-sum-exp) aggregation of Z_ij over interceptors.

    Z_tilde_i = (1/beta) * log( sum_j exp(beta * Z_ij) )
    """
    if not Z_ij_list:
        return 0.0
    Z_arr = np.array(Z_ij_list, dtype=np.float64)
    # Numerically stable log-sum-exp
    Z_max = np.max(Z_arr)
    return float(Z_max + (1.0 / beta) * np.log(np.sum(np.exp(beta * (Z_arr - Z_max)))))


# -----------------------------------------------------------------------
# Group cone cost
# -----------------------------------------------------------------------

def compute_group_cone_cost(
    offensives: list,
    defensives: list,
    config: dict,
    ap_config: dict,
    y_cache: YSystemCache,
    current_step: int,
    prev_q_matrix: Optional[np.ndarray] = None,
) -> Tuple[float, Dict, np.ndarray]:
    """Compute the group cone cost for all attackers.

    Args:
        offensives:  list of offensive Aircraft
        defensives:  list of defensive Aircraft
        config:      environment config
        ap_config:   analytic_priors config block
        y_cache:     YSystemCache instance
        current_step: current time step
        prev_q_matrix: (n_off, n_def) array of previous q values (None on first step)

    Returns:
        cone_cost:   scalar total cost
        info:        dict of logged quantities
        q_matrix:    (n_off, n_def) current q values (to cache for next step)
    """
    dt = config["dt"]
    max_steps = config["max_steps"]
    fov_half = config["fov_half_angle"]
    beta = ap_config.get("beta_cone_agg", 10.0)
    M_c = ap_config.get("M_c", 0.05)

    n_off = len(offensives)
    n_def = len(defensives)
    t_go = (max_steps - current_step) * dt

    # Current q matrix
    q_matrix = np.zeros((n_off, n_def))
    Z_matrix = np.zeros((n_off, n_def))
    Z_tilde = np.zeros(n_off)
    psi_agg = np.zeros(n_off)

    for i, off in enumerate(offensives):
        if not off.alive:
            continue
        Z_list = []
        a_A_i = compute_equivalent_normal_accel(off)

        for j, defn in enumerate(defensives):
            if not defn.alive:
                continue

            q_ij = compute_q_ij(off, defn, fov_half)
            q_matrix[i, j] = q_ij

            # q_dot
            q_prev = prev_q_matrix[i, j] if prev_q_matrix is not None else q_ij
            q_dot = compute_q_dot_ij(q_ij, q_prev, dt)

            # Equivalent interceptor normal accel
            a_I_j = compute_equivalent_normal_accel(defn)

            # Zero-effort cone margin
            Z_ij = compute_Z_ij(q_ij, q_dot, t_go, a_I_j, a_A_i, y_cache)
            Z_matrix[i, j] = Z_ij
            Z_list.append(Z_ij)

        if Z_list:
            Z_tilde[i] = aggregate_Z_i(Z_list, beta)
        else:
            Z_tilde[i] = 0.0

        psi_agg[i] = max(0.0, Z_tilde[i] + M_c)

    cone_cost = float(np.sum(psi_agg))

    # Info for logging
    alive_mask_off = np.array([off.alive for off in offensives])
    alive_mask_def = np.array([d.alive for d in defensives])
    Z_alive = Z_matrix[np.ix_(alive_mask_off, alive_mask_def)] if alive_mask_off.any() and alive_mask_def.any() else np.array([0.0])

    info = {
        "cone_cost": cone_cost,
        "avg_cone_cost_per_agent": cone_cost / max(alive_mask_off.sum(), 1),
        "Z_ij_mean": float(np.mean(Z_alive)) if Z_alive.size > 0 else 0.0,
        "Z_ij_max": float(np.max(Z_alive)) if Z_alive.size > 0 else 0.0,
        "Z_tilde_mean": float(np.mean(Z_tilde[alive_mask_off])) if alive_mask_off.any() else 0.0,
        "Z_tilde_max": float(np.max(Z_tilde[alive_mask_off])) if alive_mask_off.any() else 0.0,
        "psi_agg_per_agent": psi_agg.tolist(),
        "_Z_matrix": Z_matrix,   # internal: passed to mismatch module
        "_Z_tilde": Z_tilde,     # internal: per-agent Z_tilde for obs
    }
    return cone_cost, info, q_matrix

"""
Module 2: Assignment Mismatch Reward
======================================
Exploits the fact that the enemy interceptors fix their target
assignment at episode start and never reassign.

This module:
  1. At reset: records fixed assignment pi_0(j) for each interceptor j.
  2. Each step: computes pairwise intercept cost J_ij(t).
  3. Computes softmin optimal cost vs. fixed assignment cost.
  4. Outputs mismatch reward = lambda_M * delta(M_tilde).

Key quantities:
  J_ij          – pairwise intercept cost
  J_pi0_j       – cost of following fixed assignment
  J_min_j_tilde – soft minimum over all possible targets
  m_j_tilde     – mismatch:  J_pi0_j - J_min_j_tilde
  M_tilde       – sum of m_j_tilde over all interceptors
  r_mis         – lambda_M * (M_tilde(t) - M_tilde(t-1))

Approximations:
  - eta_ij (heading correction angle) is approximated from the
    interceptor's current heading and LOS to target.
  - Cone exposure E_ij_cone uses a smooth sigmoid membership function.
  - Z_ij from module 1 is reused if available, otherwise set to 0.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional

G = 9.81


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _closing_speed(off, defn) -> float:
    """Compute closing speed V_c between interceptor j and attacker i.

    V_c = -d(rho)/dt  ≈  -(r_hat . v_rel)
    Positive means closing.
    """
    dx = off.x - defn.x
    dy = off.y - defn.y
    dz = off.z - defn.z
    rho = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    if rho < 1e-6:
        return 0.0

    cos_ga = np.cos(off.gamma)
    vx_a = off.v * cos_ga * np.cos(off.heading)
    vy_a = off.v * cos_ga * np.sin(off.heading)
    vz_a = off.v * np.sin(off.gamma)

    cos_gd = np.cos(defn.gamma)
    vx_d = defn.v * cos_gd * np.cos(defn.heading)
    vy_d = defn.v * cos_gd * np.sin(defn.heading)
    vz_d = defn.v * np.sin(defn.gamma)

    # Relative velocity of attacker w.r.t. interceptor
    dvx = vx_a - vx_d
    dvy = vy_a - vy_d
    dvz = vz_a - vz_d

    # Unit range vector  r_hat = (p_i - p_j) / rho
    rx, ry, rz = dx / rho, dy / rho, dz / rho

    # Closing speed = negative range-rate
    return float(-(rx * dvx + ry * dvy + rz * dvz))


def _heading_correction_angle(defn, off) -> float:
    """Angle the interceptor j would need to turn to point at target i.

    eta_ij = angle between interceptor heading and LOS to target.
    """
    dx = off.x - defn.x
    dy = off.y - defn.y
    angle_to_target = np.arctan2(dy, dx)
    eta = angle_to_target - defn.heading
    return float(np.arctan2(np.sin(eta), np.cos(eta)))


def _smooth_cone_exposure(off, defn, fov_half: float,
                          k_smooth: float = 10.0) -> float:
    """Smooth cone-membership E_ij^cone ∈ [0, 1].

    Uses sigmoid: sigma( k * (cos(theta) - cos(alpha)) )
    where theta is off-axis angle.
    """
    dx = off.x - defn.x
    dy = off.y - defn.y
    dz = off.z - defn.z
    dist = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    if dist < 1e-6:
        return 1.0

    # Interceptor velocity direction
    cg = np.cos(defn.gamma)
    bx = cg * np.cos(defn.heading)
    by = cg * np.sin(defn.heading)
    bz = np.sin(defn.gamma)

    cos_theta = (bx * dx + by * dy + bz * dz) / dist
    cos_alpha = np.cos(fov_half)

    return float(1.0 / (1.0 + np.exp(-k_smooth * (cos_theta - cos_alpha))))


# -----------------------------------------------------------------------
# Pairwise intercept cost J_ij
# -----------------------------------------------------------------------

def compute_pairwise_intercept_cost(
    off, defn,
    fov_half: float,
    ap_config: dict,
    Z_ij: float = 0.0,
    M_c: float = 0.05,
) -> float:
    """Compute J_ij(t) – the cost for interceptor j to intercept target i.

    J_ij = w_T * T_ij + w_D * D_ij + w_F * F_ij

    Where:
      T_ij: geometric/time cost ( rho / V_c + lambda_eta * |eta| / dot_psi_max )
      D_ij: cone-escape risk  [-(Z_ij + M_c)]_+
      F_ij: info refresh cost  (1 - f_bar_ij)
    """
    eps = 1e-6
    w_T = ap_config.get("J_w_T", 1.0)
    w_D = ap_config.get("J_w_D", 0.5)
    w_F = ap_config.get("J_w_F", 0.3)
    lambda_eta = ap_config.get("lambda_eta", 1.0)
    f_low = ap_config.get("f_low", 1.0)
    f_high = ap_config.get("f_high", 10.0)

    # Interceptor max turn rate  dot_psi_max = ny_max * g / v
    ny_max = abs(defn.params.get("ny_max", 8.0))
    dot_psi_max = ny_max * G / max(defn.v, eps)

    # Distance
    rho = np.sqrt((off.x - defn.x) ** 2 + (off.y - defn.y) ** 2 +
                  (off.z - defn.z) ** 2)

    # Closing speed
    V_c = _closing_speed(off, defn)

    # Heading correction angle
    eta = _heading_correction_angle(defn, off)

    # T_ij: geometric / time cost
    T_ij = rho / (V_c + eps) + lambda_eta * abs(eta) / (dot_psi_max + eps)
    # Clamp to prevent extreme values when V_c is negative
    T_ij = np.clip(T_ij, 0.0, 200.0)

    # D_ij: cone escape cost
    D_ij = max(0.0, -(Z_ij + M_c))

    # F_ij: info refresh rate cost
    E_cone = _smooth_cone_exposure(off, defn, fov_half)
    f_ij = f_low + (f_high - f_low) * E_cone
    f_bar = (f_ij - f_low) / (f_high - f_low + eps)
    F_ij = 1.0 - f_bar

    return float(w_T * T_ij + w_D * D_ij + w_F * F_ij)


# -----------------------------------------------------------------------
# Initial fixed assignment  (done once at reset)
# -----------------------------------------------------------------------

def compute_initial_assignment(
    offensives: list,
    defensives: list,
    fov_half: float,
    ap_config: dict,
) -> Dict[int, int]:
    """Compute pi_0 : defender j -> attacker i at episode start.

    pi_0(j) = argmin_i J_ij(t_0)

    Returns:
        fixed_assignment: {def_idx: off_idx}
    """
    n_off = len(offensives)
    n_def = len(defensives)
    assignment = {}
    for j, defn in enumerate(defensives):
        if not defn.alive:
            continue
        best_i = 0
        best_cost = float('inf')
        for i, off in enumerate(offensives):
            if not off.alive:
                continue
            cost = compute_pairwise_intercept_cost(
                off, defn, fov_half, ap_config, Z_ij=0.0,
                M_c=ap_config.get("M_c", 0.05))
            if cost < best_cost:
                best_cost = cost
                best_i = i
        assignment[j] = best_i
    return assignment


# -----------------------------------------------------------------------
# Mismatch computation
# -----------------------------------------------------------------------

def compute_assignment_mismatch(
    offensives: list,
    defensives: list,
    config: dict,
    ap_config: dict,
    fixed_assignment: Dict[int, int],
    Z_matrix: Optional[np.ndarray] = None,
    prev_M_tilde: Optional[float] = None,
) -> Tuple[float, float, Dict]:
    """Compute assignment mismatch reward.

    Args:
        offensives:         list of offensive Aircraft
        defensives:         list of defensive Aircraft
        config:             env config
        ap_config:          analytic_priors config
        fixed_assignment:   {def_idx: off_idx} from reset
        Z_matrix:           (n_off, n_def) of Z_ij values from cone module (optional)
        prev_M_tilde:       previous step's M_tilde for delta reward

    Returns:
        mismatch_reward:    scalar  lambda_M * (M_tilde - prev_M_tilde)
        M_tilde:            current aggregated mismatch (to cache)
        info:               logging dict
    """
    fov_half = config["fov_half_angle"]
    M_c = ap_config.get("M_c", 0.05)
    beta_m = ap_config.get("beta_mismatch_softmin", 10.0)
    lambda_M = ap_config.get("mismatch_reward_weight", 0.3)
    eps = 1e-8

    n_off = len(offensives)
    n_def = len(defensives)

    # Build full J matrix
    J_matrix = np.zeros((n_off, n_def))
    for j, defn in enumerate(defensives):
        if not defn.alive:
            continue
        for i, off in enumerate(offensives):
            if not off.alive:
                continue
            Z_ij = float(Z_matrix[i, j]) if Z_matrix is not None else 0.0
            J_matrix[i, j] = compute_pairwise_intercept_cost(
                off, defn, fov_half, ap_config, Z_ij=Z_ij, M_c=M_c)

    # Per-interceptor mismatch
    m_j_list = []
    for j in range(n_def):
        if not defensives[j].alive:
            continue
        if j not in fixed_assignment:
            continue

        pi0_j = fixed_assignment[j]  # fixed target

        # J for fixed assignment
        if pi0_j < n_off and offensives[pi0_j].alive:
            J_fixed = J_matrix[pi0_j, j]
        else:
            # Fixed target is dead; use 0 mismatch
            m_j_list.append(0.0)
            continue

        # Softmin over all alive targets
        alive_idx = [i for i in range(n_off) if offensives[i].alive]
        if not alive_idx:
            m_j_list.append(0.0)
            continue

        J_vals = np.array([J_matrix[i, j] for i in alive_idx])

        # Softmin: -1/beta * log( sum exp(-beta * J) )
        J_shifted = -beta_m * J_vals
        J_max_shift = np.max(J_shifted)
        J_softmin = float(-(1.0 / beta_m) * (J_max_shift + np.log(
            np.sum(np.exp(J_shifted - J_max_shift)) + eps)))

        m_j = J_fixed - J_softmin
        m_j_list.append(max(m_j, 0.0))  # mismatch is non-negative

    M_tilde = float(np.sum(m_j_list)) if m_j_list else 0.0

    # Delta reward
    if prev_M_tilde is not None:
        delta_M = M_tilde - prev_M_tilde
    else:
        delta_M = 0.0

    mismatch_reward = lambda_M * delta_M

    # Per-attacker contribution to mismatch: how much each attacker i
    # "causes" the fixed assignments to become suboptimal.
    # Computed as: for each interceptor j whose fixed target is i,
    #   agent i's contribution = sum of m_j for those interceptors.
    # For agents that are NOT fixed targets: their contribution is
    #   how much they make themselves a "cheaper" target than the fixed one.
    per_agent_contribution = [0.0] * n_off
    j_idx = 0
    for j in range(n_def):
        if not defensives[j].alive or j not in fixed_assignment:
            continue
        pi0_j = fixed_assignment[j]
        m_val = m_j_list[j_idx] if j_idx < len(m_j_list) else 0.0
        # The fixed target gets credited for making itself "harder to catch"
        if pi0_j < n_off:
            per_agent_contribution[pi0_j] += m_val * 0.5
        # The agent who is now "cheapest" gets credited for luring
        alive_idx = [i for i in range(n_off) if offensives[i].alive]
        if alive_idx:
            cheapest = min(alive_idx, key=lambda i: J_matrix[i, j])
            if cheapest != pi0_j:
                per_agent_contribution[cheapest] += m_val * 0.5
        j_idx += 1

    # Info
    info = {
        "M_tilde": M_tilde,
        "mismatch_reward": mismatch_reward,
        "m_j_mean": float(np.mean(m_j_list)) if m_j_list else 0.0,
        "m_j_max": float(np.max(m_j_list)) if m_j_list else 0.0,
        "delta_M_tilde": delta_M,
        "M_tilde_value": M_tilde,
        "_per_agent_contribution": per_agent_contribution,
    }
    return mismatch_reward, M_tilde, info

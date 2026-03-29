import numpy as np


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + np.exp(-x))
    ex = np.exp(x)
    return ex / (1.0 + ex)


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

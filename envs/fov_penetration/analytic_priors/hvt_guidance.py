import numpy as np


def _velocity_3d(entity) -> np.ndarray:
    cg = np.cos(entity.gamma)
    return np.array([
        entity.v * cg * np.cos(entity.heading),
        entity.v * cg * np.sin(entity.heading),
        entity.v * np.sin(entity.gamma),
    ], dtype=np.float64)


def compute_hvt_guidance_features(
    off,
    hvt,
    dt: float,
    prev_omega_los: float,
    pn_nav_gain: float = 3.0,
):
    r = np.array([hvt.x - off.x, hvt.y - off.y, hvt.z - off.z], dtype=np.float64)
    rho = float(np.linalg.norm(r))
    if rho < 1e-6:
        return {
            "rho": 0.0,
            "closing_speed": 0.0,
            "omega_los": 0.0,
            "omega_los_dot": 0.0,
            "pn_hint": 0.0,
        }

    r_hat = r / rho
    v_off = _velocity_3d(off)
    v_rel = -v_off

    closing_speed = float(-np.dot(v_rel, r_hat))
    omega_los = float(np.linalg.norm(np.cross(r, v_rel)) / max(rho * rho, 1e-6))
    if dt > 0:
        omega_los_dot = (omega_los - float(prev_omega_los)) / dt
    else:
        omega_los_dot = 0.0

    pn_hint = float(pn_nav_gain * max(closing_speed, 0.0) * omega_los)

    return {
        "rho": rho,
        "closing_speed": closing_speed,
        "omega_los": omega_los,
        "omega_los_dot": float(omega_los_dot),
        "pn_hint": pn_hint,
    }

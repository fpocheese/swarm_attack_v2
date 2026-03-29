"""
Y-system Adjoint Solver
========================
Solves the adjoint ODE system for zero-effort cone-margin prediction.

ODE System:
  dY1/dt = (N_c / (tau_I * t^2)) * Y3,   Y1(0) = 1
  dY3/dt = -t * Y1 - (1/tau_I) * Y3,     Y3(0) = 0
  dY4/dt =  t * Y1 - (1/tau_A) * Y4,     Y4(0) = 0

where t is time-to-go (integrated from 0 to t_go).

The solver pre-computes a lookup table and uses linear interpolation
for fast runtime queries.
"""

import numpy as np
from typing import Tuple, Optional, Dict


def _rk4_step(y: np.ndarray, t: float, dt: float,
              N_c: float, tau_I: float, tau_A: float) -> np.ndarray:
    """Single RK4 step for the Y-system.

    State y = [Y1, Y3, Y4].
    Independent variable is time-to-go `t`.
    """
    def f(t_val, y_val):
        Y1, Y3, Y4 = y_val
        # Guard against t=0 (singularity at t_go=0)
        t_sq = max(t_val * t_val, 1e-12)
        dY1 = (N_c / (tau_I * t_sq)) * Y3
        dY3 = -t_val * Y1 - (1.0 / tau_I) * Y3
        dY4 = t_val * Y1 - (1.0 / tau_A) * Y4
        return np.array([dY1, dY3, dY4])

    k1 = f(t, y)
    k2 = f(t + 0.5 * dt, y + 0.5 * dt * k1)
    k3 = f(t + 0.5 * dt, y + 0.5 * dt * k2)
    k4 = f(t + dt, y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def solve_y_system(t_go: float, tau_I: float, tau_A: float,
                   N_c: float, n_steps: int = 200) -> Tuple[float, float, float]:
    """Numerically integrate the Y adjoint ODE from t=0 to t=t_go.

    Args:
        t_go:   time-to-go (seconds)
        tau_I:  interceptor first-order lag time constant
        tau_A:  attacker first-order lag time constant
        N_c:    navigation constant of the interceptor
        n_steps: integration steps

    Returns:
        (Y1, Y3, Y4) at t = t_go
    """
    if t_go <= 0:
        return 1.0, 0.0, 0.0

    dt = t_go / max(n_steps, 1)
    y = np.array([1.0, 0.0, 0.0], dtype=np.float64)  # [Y1, Y3, Y4]
    t = 1e-6  # start slightly > 0 to avoid singularity

    for _ in range(n_steps):
        y = _rk4_step(y, t, dt, N_c, tau_I, tau_A)
        t += dt

    return float(y[0]), float(y[1]), float(y[2])


class YSystemCache:
    """Pre-computed lookup table for Y-system values.

    Usage:
        cache = YSystemCache(tau_I=0.5, tau_A=0.3, N_c=3.0,
                             t_go_max=120.0, n_table=500)
        Y1, Y3, Y4 = cache.query(t_go)
    """

    def __init__(self, tau_I: float = 0.5, tau_A: float = 0.3,
                 N_c: float = 3.0, t_go_max: float = 150.0,
                 n_table: int = 500, n_integration_steps: int = 200):
        self.tau_I = tau_I
        self.tau_A = tau_A
        self.N_c = N_c
        self.t_go_max = t_go_max

        # Build lookup table
        self._t_go_table = np.linspace(0, t_go_max, n_table)
        self._Y1 = np.zeros(n_table)
        self._Y3 = np.zeros(n_table)
        self._Y4 = np.zeros(n_table)

        for k, tg in enumerate(self._t_go_table):
            Y1, Y3, Y4 = solve_y_system(tg, tau_I, tau_A, N_c,
                                         n_steps=n_integration_steps)
            self._Y1[k] = Y1
            self._Y3[k] = Y3
            self._Y4[k] = Y4

    def query(self, t_go: float) -> Tuple[float, float, float]:
        """Interpolated query.

        Args:
            t_go: time-to-go (clamped to [0, t_go_max])

        Returns:
            (Y1, Y3, Y4)
        """
        t_go_c = np.clip(t_go, 0.0, self.t_go_max)
        Y1 = float(np.interp(t_go_c, self._t_go_table, self._Y1))
        Y3 = float(np.interp(t_go_c, self._t_go_table, self._Y3))
        Y4 = float(np.interp(t_go_c, self._t_go_table, self._Y4))
        return Y1, Y3, Y4

    def query_batch(self, t_go_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Batch interpolated query.

        Args:
            t_go_array: array of t_go values

        Returns:
            (Y1_array, Y3_array, Y4_array)
        """
        t_clamped = np.clip(t_go_array, 0.0, self.t_go_max)
        Y1 = np.interp(t_clamped, self._t_go_table, self._Y1)
        Y3 = np.interp(t_clamped, self._t_go_table, self._Y3)
        Y4 = np.interp(t_clamped, self._t_go_table, self._Y4)
        return Y1, Y3, Y4

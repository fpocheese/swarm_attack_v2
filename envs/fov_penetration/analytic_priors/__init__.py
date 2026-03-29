"""
Analytic Priors for FOV Penetration Environment
=================================================
Module 1: Cone Margin Cost  (cone_margin.py)
Module 2: Assignment Mismatch Reward  (assignment_mismatch.py)
Module 3: LOS Escape Reward  (los_escape.py)
Shared : Y-system adjoint solver  (y_system.py)
"""

from .y_system import solve_y_system, YSystemCache
from .cone_margin import compute_group_cone_cost
from .assignment_mismatch import (
    compute_initial_assignment,
    compute_assignment_mismatch,
)
from .los_escape import compute_escape_reward
from .hvt_guidance import compute_hvt_guidance_features
from .penetration_phase import compute_penetration_success_score

__all__ = [
    "solve_y_system",
    "YSystemCache",
    "compute_group_cone_cost",
    "compute_initial_assignment",
    "compute_assignment_mismatch",
    "compute_escape_reward",
    "compute_hvt_guidance_features",
    "compute_penetration_success_score",
]

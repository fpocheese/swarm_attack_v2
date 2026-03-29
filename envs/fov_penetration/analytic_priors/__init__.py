"""
Analytic Priors for FOV Penetration Environment V22
=====================================================
Module 1 : Cone Margin Cost           (cone_margin.py)
Module 2a: Assignment Mismatch        (assignment_mismatch.py)  [deprecated V22]
Module 2b: Decoy Game                 (decoy_game.py)           [NEW V22]
Module 3a: LOS Escape Reward          (los_escape.py)
Module 3b: Effective Penetration      (penetration_phase.py)    [rewritten V22]
Module 4 : HVT Guidance               (hvt_guidance.py)
Shared   : Y-system adjoint solver    (y_system.py)
"""

from .y_system import solve_y_system, YSystemCache
from .cone_margin import compute_group_cone_cost
from .assignment_mismatch import (
    compute_initial_assignment,
    compute_assignment_mismatch,
)
from .decoy_game import compute_decoy_game
from .los_escape import compute_escape_reward
from .hvt_guidance import compute_hvt_guidance_features
from .penetration_phase import (
    compute_penetration_success_score,
    compute_effective_penetration,
)

__all__ = [
    "solve_y_system",
    "YSystemCache",
    "compute_group_cone_cost",
    "compute_initial_assignment",
    "compute_assignment_mismatch",
    "compute_decoy_game",
    "compute_escape_reward",
    "compute_hvt_guidance_features",
    "compute_penetration_success_score",
    "compute_effective_penetration",
]

#!/usr/bin/env python3
"""V22 smoke test — validates all refactored modules import and run."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

# --- 1. Import check ---
print("=== 1. Import check ===")
from envs.fov_penetration import FOVPenetrationEnv
from envs.fov_penetration.analytic_priors import (
    compute_decoy_game,
    compute_effective_penetration,
)
from envs.fov_penetration.analytic_priors.los_escape import compute_escape_reward
from envs.fov_penetration.analytic_priors.penetration_phase import compute_P_pen
from envs.fov_penetration.policies_interceptor import InterceptorPolicy
print("All imports OK")

# --- 2. Env creation ---
print("\n=== 2. Env creation ===")
env = FOVPenetrationEnv()
print(f"Env created: {env.n_agents} agents, obs_dim={env.obs_dim}")

# --- 3. Reset ---
print("\n=== 3. Reset ===")
obs, share_obs, avail = env.reset()
print(f"Reset OK: obs[0].shape={obs[0].shape}")

# --- 4. Step ---
print("\n=== 4. Step ===")
actions = [np.zeros(3) for _ in range(env.n_agents)]
obs2, share_obs2, rewards, costs, dones, infos, _ = env.step(actions)
info = infos[0]
total_r = sum(r[0] for r in rewards)
print(f"Step OK: total_reward={total_r:.4f}")

# Check V22 info fields
v22_fields = [
    "locked_target_by_defender", "locked_by_map",
    "n_locked_defenders", "terminal_miss_distance_min",
    "step_lock_events",
]
for f in v22_fields:
    val = info.get(f, "MISSING")
    print(f"  info[{f}] = {val}")

# --- 5. Multi-step run ---
print("\n=== 5. Multi-step run (50 steps) ===")
for s in range(50):
    obs2, share_obs2, rewards, costs, dones, infos, _ = env.step(actions)
    if any(dones):
        print(f"  Episode done at step {s+2}: {infos[0].get('done_reason', 'unknown')}")
        break
else:
    info = infos[0]
    print(f"  After 51 total steps:")
    print(f"    n_locked_defenders = {info.get('n_locked_defenders', 0)}")
    n_events = len(info.get('step_lock_events', []))
    print(f"    step_lock_events count = {n_events}")

# --- 6. Check interceptor policy state machine ---
print("\n=== 6. Interceptor policy V22 state machine ===")
for i, ip in enumerate(env.defensive_policies):
    print(f"  Defender {i}: lock_mode={ip.lock_mode}, "
          f"initial_target={ip.initial_assigned_target_idx}, "
          f"locked_target={ip.current_locked_target_idx}, "
          f"has_ever_locked={ip.has_ever_locked}")

# --- 7. Entity V22 attributes ---
print("\n=== 7. Entity V22 attributes ===")
for i, off in enumerate(env.offensives):
    print(f"  Offensive {i}: "
          f"locked_by_count={off.locked_by_count}, "
          f"min_miss_dist={off.min_miss_distance:.1f}, "
          f"hit_time={off.hit_time}")

print("\n=============================")
print("Smoke test V22 PASSED!")
print("=============================")

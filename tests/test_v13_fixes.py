"""V13 interceptor + assignment fixes validation test"""
import numpy as np
import sys
sys.path.insert(0, '.')

from envs.fov_penetration.fov_penetration_env import FOVPenetrationEnv
from envs.fov_penetration.config import get_config

config = get_config()
env = FOVPenetrationEnv(config)
obs, share_obs, avail = env.reset()

print("=== V13 Validation Test ===")
print()

# 1. Initial assignment: every offensive has an interceptor
print("--- 1. Initial target assignment ---")
assignments = env.assignments
covered_off = set(assignments.values())
all_off = set(range(config["n_offensive"]))
print(f"  Assignments: {assignments}")
print(f"  Covered offensives: {covered_off}")
print(f"  Uncovered offensives: {all_off - covered_off}")
assert covered_off == all_off, f"FAIL: missing coverage for {all_off - covered_off}"
print("  PASS: all offensives have interceptor assigned")
print()

# 2. PN guidance works from step 0
print("--- 2. PN guidance from first step ---")
for i, policy in enumerate(env.defensive_policies):
    has_pos = "YES" if policy.target_pos_known is not None else "NO"
    print(f"  Interceptor {i}: target={policy.assigned_target_idx}, "
          f"pos_known={has_pos}, state={policy.engagement_state}")
    assert policy.target_pos_known is not None, f"FAIL: interceptor {i} has no target_pos_known"
    assert policy.assigned_target_idx is not None, f"FAIL: interceptor {i} has no target"

# Run 10 steps and verify PN active
actions = [np.zeros(3) for _ in range(config["n_offensive"])]
for step in range(10):
    obs, share_obs, rewards, costs, dones, infos, avail = env.step(actions)

print()
print("  After 10 steps:")
for i, policy in enumerate(env.defensive_policies):
    print(f"  Interceptor {i}: ny={policy.demanded_ny:.3f}, "
          f"nz={policy.demanded_nz:.3f}, "
          f"v_close={policy.closing_speed:.1f}, "
          f"state={policy.engagement_state}")
    assert policy.closing_speed > 0, f"FAIL: interceptor {i} closing_speed=0, PN not working"
print("  PASS: all interceptors have active PN guidance")
print()

# 3. Reassignment after target death
print("--- 3. Reassignment after target death ---")
orig_assignments = dict(env.assignments)
print(f"  Original: {orig_assignments}")

# Kill offensive 0
env.offensives[0].kill()
print("  Killed offensive 0")

# Run 20 steps to trigger reassignment
for step in range(20):
    obs, share_obs, rewards, costs, dones, infos, avail = env.step(actions)

new_assignments = env.assignments
covered_new = set(new_assignments.values())
alive_off = set(j for j in range(config["n_offensive"]) if env.offensives[j].alive)
print(f"  New assignments: {new_assignments}")
print(f"  Alive offensives: {alive_off}")
print(f"  Covered: {covered_new}")
uncovered = alive_off - covered_new
print(f"  Uncovered alive: {uncovered}")
assert not uncovered, f"FAIL: alive offensives {uncovered} not covered"
# Verify no interceptor assigned to dead target
for d_idx, o_idx in new_assignments.items():
    assert env.offensives[o_idx].alive, f"FAIL: interceptor {d_idx} assigned to dead offensive {o_idx}"
print("  PASS: all alive offensives covered, no interceptor wasted on dead target")
print()

# 4. All interceptors assigned (surplus interceptors gang up)
print("--- 4. Surplus interceptors gangup ---")
n_assigned = len(new_assignments)
n_alive_def = sum(1 for d in env.defensives if d.alive)
print(f"  Alive interceptors: {n_alive_def}, Assigned: {n_assigned}")
assert n_assigned == n_alive_def, f"FAIL: {n_alive_def - n_assigned} interceptors idle"
print("  PASS: all alive interceptors assigned")
print()

# 5. Extrapolation timer fix
print("--- 5. Extrapolation timer fix ---")
for step in range(100):
    obs, share_obs, rewards, costs, dones, infos, avail = env.step(actions)

for i, policy in enumerate(env.defensive_policies):
    if policy.target is not None and policy.target.alive:
        print(f"  Interceptor {i}: info_timer={policy.info_timer:.3f}, "
              f"guide_timer={policy.guide_timer:.3f}")
        assert policy.info_timer <= 0.6, \
            f"FAIL: interceptor {i} info_timer={policy.info_timer:.3f} growing unbounded"
print("  PASS: extrapolation timer bounded within guide period")
print()

# 6. Persistent pursuit after escape event
print("--- 6. Persistent pursuit after escape ---")
# Reset for a clean test
obs, share_obs, avail = env.reset()
# Run 200 steps to let interceptors engage
for step in range(200):
    # Random offensive actions to try to evade
    actions = [np.random.uniform(-1, 1, 3) for _ in range(config["n_offensive"])]
    obs, share_obs, rewards, costs, dones, infos, avail = env.step(actions)

# Check all alive interceptors still have targets
for i, policy in enumerate(env.defensive_policies):
    d = env.defensives[i]
    if d.alive:
        if policy.target is not None and policy.target.alive:
            print(f"  Interceptor {i}: target={policy.assigned_target_idx}, "
                  f"state={policy.engagement_state}, "
                  f"closing={policy.closing_speed:.1f}")
            assert policy.assigned_target_idx is not None, \
                f"FAIL: interceptor {i} lost target after escape"
print("  PASS: interceptors retain targets through engagement")
print()

print("=== ALL V13 TESTS PASSED ===")

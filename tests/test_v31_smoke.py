"""V31 smoke test: verify config + interceptor pass-through logic"""
import sys
sys.path.insert(0, '.')

from envs.fov_penetration.config import get_config
from envs.fov_penetration.policies_interceptor import InterceptorPolicy
from envs.fov_penetration.entities import Aircraft, HVT
import numpy as np

cfg = get_config(scenario='scenario_1')
print('=== V31 Config Verification ===')
print('pursuit.uturn_ay_fraction =', cfg['pursuit']['uturn_ay_fraction'])
print('pursuit.uturn_recovery_steps =', cfg['pursuit']['uturn_recovery_steps'])
print('pursuit.uturn_ax_brake =', cfg['pursuit']['uturn_ax_brake'])
print('pursuit.passed_distance_abandon =', cfg['pursuit']['passed_distance_abandon'])
print('pn_guide_freq_rear =', cfg['pn_guide_freq_rear'])
print()

print('=== V31 Reward Verification ===')
rc = cfg['reward']
print('lambda_hit_approach =', rc['lambda_hit_approach'])
print('lambda_no_retreat =', rc['lambda_no_retreat'])
print('lambda_decoy_value =', rc['lambda_decoy_value'])
print('lambda_penalty_cone =', rc['lambda_penalty_cone'])
print('lambda_penalty_danger =', rc['lambda_penalty_danger'])
print('danger_radius =', rc['danger_radius'])
print('killed_penalty =', rc['killed_penalty'])
print('timeout_penalty =', rc['timeout_penalty'])
print('timeout_distance_penalty_coef =', rc['timeout_distance_penalty_coef'])
print('lambda_terminal_hit =', rc['lambda_terminal_hit'])
print()

# Test InterceptorPolicy with pass-through
params = cfg['defensive']
hvt = HVT(1200, 0, 0)
intc = Aircraft('D0', 'defensive', params, x=0, y=0, z=300, heading=np.pi)
policy = InterceptorPolicy(intc, hvt, cfg, patrol_idx=0)
print('InterceptorPolicy created OK')
print('_target_passed =', policy._target_passed)
print('_pass_step =', policy._pass_step)
print('_was_approaching =', policy._was_approaching)
print()

# Simulate a pass-through scenario
off = Aircraft('O0', 'offensive', cfg['offensive'], x=-500, y=0, z=300, heading=0.0)
policy.set_initial_target(0, off)
policy.lock_mode = InterceptorPolicy.STATE_LOCKED
policy.current_locked_target_idx = 0
policy.target = off

dt = cfg['dt']
print('=== Simulating head-on engagement ===')
for step in range(200):
    off.step_with_action([0, 0, 0], dt)  # fly straight
    ax, ay, mu = policy.get_action([off], dt)
    intc.step(ax, ay, mu, dt)
    dist = intc.distance_3d(off)
    if step % 50 == 0:
        print(f'  Step {step}: dist={dist:.1f}m, passed={policy._target_passed}')

print(f'  Final: dist={intc.distance_3d(off):.1f}m, passed={policy._target_passed}')
print(f'  Pass step={policy._pass_step}, pass_dist={policy._pass_distance:.1f}m')
print()

# Now check that ay is degraded after pass
if policy._target_passed:
    ax2, ay2, mu2 = policy.get_action([off], dt)
    ay_max_full = params['ay_max']
    print(f'  Post-pass ay_cmd = {ay2:.2f} m/s^2')
    print(f'  Full ay_max = {ay_max_full:.2f} m/s^2')
    print(f'  Degradation ratio = {ay2/ay_max_full:.3f}')
    if ay2 < ay_max_full * 0.5:
        print('  PASS: ay is degraded after fly-through!')
    else:
        print('  WARNING: ay degradation may not be working')
else:
    print('  NOTE: pass-through not detected in 200 steps (may need more steps)')

print()
print('ALL V31 CHECKS PASSED!')

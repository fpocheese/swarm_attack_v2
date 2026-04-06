"""V31 pass-through degradation test"""
import sys; sys.path.insert(0, '.')
from envs.fov_penetration.config import get_config
from envs.fov_penetration.policies_interceptor import InterceptorPolicy
from envs.fov_penetration.entities import Aircraft, HVT
import numpy as np

cfg = get_config(scenario='scenario_1')
params_d = cfg['defensive']
params_o = cfg['offensive']
hvt = HVT(1200, 0, 0)

# Head-on: interceptor heading west, offensive heading east, 50m y-offset to cause miss
intc = Aircraft('D0', 'defensive', params_d, x=200, y=0, z=300, heading=np.pi)
off = Aircraft('O0', 'offensive', params_o, x=-200, y=50, z=300, heading=0.0)

policy = InterceptorPolicy(intc, hvt, cfg, patrol_idx=0)
policy.set_initial_target(0, off)
policy.lock_mode = InterceptorPolicy.STATE_LOCKED
policy.current_locked_target_idx = 0
policy.target = off
policy._update_known_position(off)

dt = cfg['dt']
min_dist = float('inf')
for step in range(3000):  # 30 seconds
    off.step_with_action([0, 0, 0], dt)
    ax, ay, mu = policy.get_action([off], dt)
    intc.step(ax, ay, mu, dt)
    dist = intc.distance_3d(off)
    min_dist = min(min_dist, dist)
    if step % 300 == 0:
        print('Step %d: dist=%.1fm, min=%.1fm, passed=%s, ay=%.1f' % (
            step, dist, min_dist, policy._target_passed, ay))
    if policy._target_passed and step == policy._pass_step + 1:
        print('>>> PASS DETECTED at step %d! dist=%.1fm, ay=%.1f' % (step, dist, ay))

print()
print('Min distance (CPA) = %.2fm' % min_dist)
print('Pass detected =', policy._target_passed)
print('Pass step =', policy._pass_step)
print('Final dist = %.1fm' % intc.distance_3d(off))

if policy._target_passed:
    ax2, ay2, mu2 = policy.get_action([off], dt)
    ay_max_full = params_d['ay_max']
    print('Post-pass ay = %.2f, full ay_max = %.2f' % (ay2, ay_max_full))
    print('Degradation = %.1f%% of max' % (ay2 / ay_max_full * 100))
    if ay2 < ay_max_full * 0.5:
        print('SUCCESS: ay is properly degraded after pass-through!')
    else:
        print('WARNING: ay might not be degraded enough')
else:
    print('NOTE: pass-through not detected in 3000 steps')

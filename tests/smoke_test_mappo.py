#!/usr/bin/env python
"""Quick smoke test for MAPPO + dt=0.05 setup."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'third_party', 'MACPO', 'MACPO'))
import numpy as np

# Test 1: Import r_mappo
from macpo.algorithms.r_mappo.r_mappo import R_MAPPO
print('[OK] R_MAPPO imported')

# Test 2: Import rMAPPOPolicy
from macpo.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy
print('[OK] R_MAPPOPolicy imported')

# Test 3: Import base_runner (MAPPO)
from macpo.runner.separated.base_runner import Runner
print('[OK] MAPPO base_runner imported')

# Test 4: Import mujoco_runner (MAPPO)
from macpo.runner.separated.mujoco_runner import MujocoRunner
print('[OK] MAPPO MujocoRunner imported')

# Test 5: Import env
from envs.fov_penetration import FOVPenetrationEnv
env = FOVPenetrationEnv()
obs = env.reset()
obs0 = np.array(obs[0])
print('[OK] Env: obs shape = %s, n_agents = %d' % (obs0.shape, env.n_agents))

# Test 6: Check dt
from envs.fov_penetration.config import DEFAULT_CONFIG
dt = DEFAULT_CONFIG['dt']
max_steps = DEFAULT_CONFIG['max_steps']
hit_range = DEFAULT_CONFIG['point_target']['hit_threshold']
step_size = 50 * dt
status = 'OK' if step_size < hit_range else 'FAIL'
print('[OK] Config: dt=%.3f, max_steps=%d, total=%.1fs, hit_range=%.1fm' % (dt, max_steps, dt*max_steps, hit_range))
print('     step_size=%.2fm (%s: step < hit_range)' % (step_size, status))
assert step_size < hit_range, 'step_size must be < hit_range!'

# Test 7: Quick step
action = [np.zeros(env.action_space[0].shape) for _ in range(env.n_agents)]
result = env.step(action)
print('[OK] env.step() returns %d values' % len(result))
assert len(result) == 7, 'Expected 7 return values from step()'

print()
print('=== All smoke tests PASSED! ===')

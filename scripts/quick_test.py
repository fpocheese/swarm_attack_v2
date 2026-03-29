#!/usr/bin/env python
"""Quick environment verification test."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party", "MACPO", "MACPO"))

from envs.fov_penetration.fov_penetration_env import FOVPenetrationEnv
import numpy as np

env = FOVPenetrationEnv({'scenario': 'scenario_1'})
obs, share_obs, avail = env.reset()
print(f"obs_dim={env.obs_dim}, share_obs_dim={env.share_obs_dim}")
print(f"n_agents={env.n_agents}, action_space={env.action_space[0]}")
print(f"obs[0].shape={obs[0].shape}, share_obs[0].shape={share_obs[0].shape}")
print(f"max_steps={env.max_steps}, dt={env.dt}")
print(f"hit_hvt_range={env.config['hit_hvt_range']}")

for s in range(10):
    result = env.step(np.zeros((4, 3)))

print("Environment step test PASSED!")

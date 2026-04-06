#!/usr/bin/env python
"""V24 Smoke test"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.fov_penetration import FOVPenetrationEnv
import numpy as np

env = FOVPenetrationEnv(scenario='scenario_1')
env.seed(42)
obs, share_obs, avail = env.reset()
obs = np.array(obs)
share_obs = np.array(share_obs)
print(f'obs shape: {obs.shape}')
print(f'share_obs shape: {share_obs.shape}')
print(f'obs_dim: {env.obs_dim}')
print(f'n_agents: {env.n_agents}')

# Run 200 steps
for step in range(200):
    actions = [np.random.uniform(-1, 1, size=3) for _ in range(env.n_agents)]
    obs, share_obs, rewards, costs, dones, infos, avail = env.step(actions)

print(f'After 200 steps:')
print(f'  obs len={len(obs)}, per-agent shape={obs[0].shape}')
print(f'  rewards={rewards}')
info = infos[0]
print(f'  decoy_counts={info.get("decoy_counts_per_agent", "N/A")}')
print(f'  offensive_alive={info.get("offensive_alive")}')
print(f'  defensive_alive={info.get("defensive_alive")}')
print(f'  kill_events={len(info.get("kill_events", []))}')
print(f'  obs_dim expected={env.obs_dim}, actual={obs[0].shape[0]}')
assert obs[0].shape[0] == env.obs_dim, f"obs dim mismatch: {obs[0].shape[0]} vs {env.obs_dim}"
print('OK - V24 env works!')

"""V38 快速冒烟测试 — 验证环境+V5动力学能正常运行"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from envs.fov_penetration import FOVPenetrationEnv

env = FOVPenetrationEnv()
obs, share_obs, avail = env.reset()
print(f"n_agents: {env.n_agents}")
print(f"obs_dim: {np.array(obs).shape}")
print(f"share_obs_dim: {np.array(share_obs).shape}")
print(f"action_space: {env.action_space[0].shape}")
print(f"dt: {env.dt}")
print(f"max_steps: {env.max_steps}")

pt = env.config.get("point_target", {})
print(f"hit_threshold: {pt.get('hit_threshold', 'MISSING')}")
print(f"collision_kill_range: {env.config.get('collision_kill_range', 'MISSING')}")

for i in range(10):
    actions = [[0.0, 0.0, 0.0] for _ in range(env.n_agents)]
    obs, share_obs, rewards, costs, dones, infos, avail = env.step(actions)
    if i == 0:
        print(f"rewards shape: {np.array(rewards).shape}")
        print(f"costs shape: {np.array(costs).shape}")
        print(f"first step reward: {rewards}")
print("Smoke test PASSED!")

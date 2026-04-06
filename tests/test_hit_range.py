#!/usr/bin/env python3
"""快速验证: 在 50m hit range + CPA 下, random actions 能否产生 HVT hits"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from envs.fov_penetration import FOVPenetrationEnv

ap_override = {"analytic_priors": {
    "enable_cone_cost": True,
    "enable_assignment_mismatch_reward": False,
    "enable_escape_reward": True,
    "enable_decoy_game": True,
    "enable_effective_penetration": True,
}}

env = FOVPenetrationEnv(config=ap_override, scenario="scenario_1")

total_hits = 0
total_episodes = 20
for ep in range(total_episodes):
    env.seed(ep * 100)
    obs, _, _ = env.reset()
    ep_hits = 0
    min_dist = float('inf')
    for step in range(1200):
        # Random small actions
        actions = [np.random.uniform(-0.3, 0.3, 3) for _ in range(env.n_agents)]
        obs, _, rewards, costs, dones, infos, _ = env.step(actions)
        info = infos[0]
        md = info.get("terminal_miss_distance_min", float('inf'))
        if md < min_dist:
            min_dist = md
        if any(dones):
            ep_hits = info.get("num_hit_hvt", 0)
            break
    total_hits += ep_hits
    symbol = "HIT!" if ep_hits > 0 else "miss"
    print(f"  ep {ep+1:2d}: hits={ep_hits} min_dist={min_dist:.0f}m  {symbol}")

print(f"\nTotal: {total_hits}/{total_episodes} episodes had hits ({100*total_hits/total_episodes:.0f}%)")
print(f"avg cost: {np.mean([c[0] for c in costs]):.0f}")

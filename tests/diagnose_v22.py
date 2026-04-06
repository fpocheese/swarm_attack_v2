#!/usr/bin/env python3
"""V22 诊断脚本 — 逐步跑 episode, 分解 reward/cost 各模块贡献"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np

from envs.fov_penetration import FOVPenetrationEnv

# V22 config (same as training)
ap_override = {"analytic_priors": {
    "enable_cone_cost": True,
    "enable_assignment_mismatch_reward": False,
    "enable_escape_reward": True,
    "enable_decoy_game": True,
    "enable_effective_penetration": True,
}}

env = FOVPenetrationEnv(config=ap_override, scenario="scenario_1")
env.seed(42)

N_EPISODES = 5
for ep in range(N_EPISODES):
    obs, share_obs, avail = env.reset()
    
    total_rewards = np.zeros(env.n_agents)
    total_costs = np.zeros(env.n_agents)
    
    # Per-module reward accumulators
    module_rewards = {
        "ap_cone_cost": 0.0,
        "ap_escape_reward": 0.0,
        "ap_decoy_game": 0.0,
        "ap_effective_penetration": 0.0,
        "ap_hvt_guidance": 0.0,
        "ap_cooperative": 0.0,
        "base_reward": 0.0,
    }
    
    n_locked = 0
    n_lock_events = 0
    min_miss_dist = float('inf')
    hit_count = 0
    
    for step in range(1200):
        actions = [np.zeros(3) for _ in range(env.n_agents)]
        obs, share_obs, rewards, costs, dones, infos, _ = env.step(actions)
        
        info = infos[0]
        for i in range(env.n_agents):
            total_rewards[i] += rewards[i][0]
            total_costs[i] += costs[i][0]
        
        # Track lock events
        step_lock_events = info.get("step_lock_events", [])
        n_lock_events += len(step_lock_events)
        n_locked_now = info.get("n_locked_defenders", 0)
        if n_locked_now > n_locked:
            n_locked = n_locked_now
        
        # Track module-specific rewards from info
        for key in module_rewards:
            if key in info:
                module_rewards[key] += info[key]
        
        # Track AP reward components from info
        for k in info:
            if k.startswith("ap_") and k not in module_rewards:
                if isinstance(info[k], (int, float)):
                    module_rewards[k] = module_rewards.get(k, 0.0) + info[k]
        
        if any(dones):
            terminal_info = info
            hit_count = terminal_info.get("num_hit_hvt", 0)
            min_miss_dist = terminal_info.get("terminal_miss_distance_min", float('inf'))
            break
    
    print(f"\n=== Episode {ep+1} ===")
    print(f"  Steps: {step+1}")
    print(f"  Total reward per agent: {total_rewards}")
    print(f"  Mean reward: {np.mean(total_rewards):.1f}")
    print(f"  Total cost per agent: {total_costs}")
    print(f"  Mean cost: {np.mean(total_costs):.1f}")
    print(f"  Max locked defenders: {n_locked}")
    print(f"  Total lock events: {n_lock_events}")
    print(f"  Min miss distance: {min_miss_dist:.1f}")
    print(f"  HVT hits: {hit_count}")
    print(f"  Module reward breakdown:")
    for k, v in sorted(module_rewards.items()):
        if abs(v) > 0.01:
            print(f"    {k}: {v:.3f}")
    
    # Check info keys available
    if ep == 0:
        print(f"\n  Available info keys: {sorted(info.keys())}")

# Also run diagnostic: what does the base env reward look like WITHOUT AP?
print("\n\n=== BASELINE (no AP) comparison ===")
env_base = FOVPenetrationEnv(config={"analytic_priors": {
    "enable_cone_cost": False,
    "enable_assignment_mismatch_reward": False,
    "enable_escape_reward": False,
    "enable_decoy_game": False,
    "enable_effective_penetration": False,
}}, scenario="scenario_1")
env_base.seed(42)
obs, share_obs, avail = env_base.reset()

total_r_base = np.zeros(env_base.n_agents)
total_c_base = np.zeros(env_base.n_agents)
for step in range(1200):
    actions = [np.zeros(3) for _ in range(env_base.n_agents)]
    obs, share_obs, rewards, costs, dones, infos, _ = env_base.step(actions)
    for i in range(env_base.n_agents):
        total_r_base[i] += rewards[i][0]
        total_c_base[i] += costs[i][0]
    if any(dones):
        break

print(f"  Steps: {step+1}")
print(f"  Total reward per agent (no AP): {total_r_base}")
print(f"  Mean reward: {np.mean(total_r_base):.1f}")
print(f"  Total cost per agent (no AP): {total_c_base}")
print(f"  Mean cost: {np.mean(total_c_base):.1f}")

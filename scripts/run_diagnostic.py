#!/usr/bin/env python
"""
诊断episode — 分析飞行器行为是否合理
运行最新保存的模型或零动作，输出详细诊断信息
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party", "MACPO", "MACPO"))

from envs.fov_penetration.fov_penetration_env import FOVPenetrationEnv
import numpy as np

def run_diagnostic(action_mode="zero", n_episodes=5):
    print(f"\n{'='*60}")
    print(f"DIAGNOSTIC: mode={action_mode}, episodes={n_episodes}")
    print(f"{'='*60}")
    
    for ep in range(n_episodes):
        env = FOVPenetrationEnv({'scenario': 'scenario_1'})
        env.seed(ep * 42)
        obs, share_obs, avail = env.reset()
        
        ep_rew = np.zeros(4)
        ep_cost = np.zeros(4)
        
        # Track key metrics
        min_dists_hvt = [[] for _ in range(4)]
        altitudes = [[] for _ in range(4)]
        killed_step = [None] * 4
        hit_step = [None] * 4
        
        for step in range(1500):
            if action_mode == "zero":
                actions = np.zeros((4, 3))
            elif action_mode == "random":
                actions = np.random.uniform(-1, 1, (4, 3))
            else:
                actions = np.zeros((4, 3))
            
            result = env.step(actions)
            obs, share_obs, rewards, costs, dones, infos, avail = result
            ep_rew += np.array(rewards).flatten()
            ep_cost += np.array(costs).flatten()
            
            for i, off in enumerate(env.offensives):
                if off.alive and not off.hit_hvt:
                    d = off.distance_to(env.hvt.x, env.hvt.y, env.hvt.z)
                    min_dists_hvt[i].append(d)
                    altitudes[i].append(off.z)
                if not off.alive and killed_step[i] is None:
                    killed_step[i] = step
                if off.hit_hvt and hit_step[i] is None:
                    hit_step[i] = step
            
            if np.all(dones):
                break
        
        info = infos[0]
        print(f"\n--- Episode {ep} ---")
        print(f"  Done reason: {info['done_reason']}, steps: {env.current_step}")
        print(f"  Hits: {info['hit_count']}, Off alive: {info['offensive_alive']}, Def alive: {info['defensive_alive']}")
        print(f"  Mean reward: {np.mean(ep_rew):.1f}, Mean cost: {np.mean(ep_cost):.1f}")
        
        for i in range(4):
            dists = min_dists_hvt[i]
            alts = altitudes[i]
            if dists:
                print(f"  Agent {i}: dist_hvt {dists[0]:.0f}->{dists[-1]:.0f}m (min={min(dists):.0f}), "
                      f"alt {alts[0]:.0f}->{alts[-1]:.0f}m, "
                      f"killed@{killed_step[i]}, hit@{hit_step[i]}")
            else:
                print(f"  Agent {i}: no data (killed@step 0?)")

# Run diagnostics
run_diagnostic("zero", 5)
run_diagnostic("random", 5)

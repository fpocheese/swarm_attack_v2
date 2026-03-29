"""
快速分析: 跑1个episode, 打印每步各reward分量的值
目的: 理解为什么飞机不往HVT方向飞
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from envs.fov_penetration.fov_penetration_env import FOVPenetrationEnv

env = FOVPenetrationEnv({"scenario": "scenario_1"})
env.reset()
cfg = env.config
hvt = env.hvt

# 跑零动作（直飞）看reward分量
print("=== 零动作(直飞)下的reward分量分析 ===")
print(f"HVT位置: [{hvt.x}, {hvt.y}, {hvt.z}]")
for off in env.offensives:
    dist = off.distance_to(hvt.x, hvt.y, hvt.z)
    print(f"Off{off.uid}: pos=({off.x:.0f},{off.y:.0f},{off.z:.0f}), dist_hvt={dist:.0f}m, heading={np.degrees(off.heading):.1f}°")

print("\n--- Step-by-step reward breakdown (first 20 steps) ---")
for step in range(20):
    actions = [np.zeros(3) for _ in range(4)]
    
    # 手动计算各分量
    obs_range = cfg["obs_range"]
    rc = cfg["reward"]
    init_dist = 6000.0
    
    prev_dists = [off.distance_to(hvt.x, hvt.y, hvt.z) if off.alive else float('inf') 
                  for off in env.offensives]
    
    result = env.step(actions)
    rewards = [r[0] for r in result[2]]
    
    # 手动分解
    for i, off in enumerate(env.offensives):
        if not off.alive:
            continue
        dist = off.distance_to(hvt.x, hvt.y, hvt.z)
        delta = prev_dists[i] - dist
        
        approach_r = rc["approach_hvt_coef"] * delta / obs_range
        dist_ratio = max(1.0 - dist / init_dist, 0.0)
        approach_r *= (1.0 + 2.0 * dist_ratio)
        
        progress = max(1.0 - dist / init_dist, 0.0)
        progress_r = rc.get("progress_coef", 0.3) * progress
        
        retreat_pen = rc.get("retreat_penalty", -0.10) if delta < 0 else 0
        
        if step < 5 or step == 19:
            print(f"  Step{step} Off{i}: dist={dist:.0f} delta={delta:.2f}m | "
                  f"approach={approach_r:.4f} progress={progress_r:.4f} retreat={retreat_pen:.2f} | "
                  f"total_reward={rewards[i]:.4f} | z={off.z:.0f} gamma={np.degrees(off.gamma):.1f}°")

# 再跑到结束看最终距离
env.reset()
total_rewards = [0.0] * 4
for step in range(500):
    actions = [np.zeros(3) for _ in range(4)]
    result = env.step(actions)
    for i in range(4):
        total_rewards[i] += result[2][i][0]

print("\n=== 500步零动作(直飞)结果 ===")
for i, off in enumerate(env.offensives):
    dist = off.distance_to(hvt.x, hvt.y, hvt.z)
    print(f"Off{i}: dist_hvt={dist:.0f}m, z={off.z:.0f}m, alive={off.alive}, total_reward={total_rewards[i]:.1f}")

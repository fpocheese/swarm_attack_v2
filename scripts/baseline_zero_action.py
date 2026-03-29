#!/usr/bin/env python
"""
Zero-action baseline test — 所有飞机直飞不做任何机动
这是训练的最低基准：模型必须超过这个分数才算有效学习
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party", "MACPO", "MACPO"))

from envs.fov_penetration.fov_penetration_env import FOVPenetrationEnv
import numpy as np

N_EPISODES = 20
hits_total = 0
total_rews = []
done_reasons = {"success": 0, "all_killed": 0, "timeout": 0}

for seed in range(N_EPISODES):
    env = FOVPenetrationEnv({'scenario': 'scenario_1'})
    env.seed(seed * 100)
    obs, share_obs, avail = env.reset()
    ep_rew = np.zeros(4)
    for step in range(1500):
        result = env.step(np.zeros((4, 3)))
        obs, share_obs, rewards, costs, dones, infos, avail = result
        ep_rew += np.array(rewards).flatten()
        if np.all(dones):
            break
    
    n_hit = sum(o.hit_hvt for o in env.offensives)
    hits_total += n_hit
    reason = infos[0]["done_reason"]
    done_reasons[reason] = done_reasons.get(reason, 0) + 1
    off_alive = sum(1 for o in env.offensives if o.alive)
    def_alive = sum(1 for d in env.defensives if d.alive)
    total_rews.append(np.mean(ep_rew))
    print(f"  seed={seed:2d}: hit={n_hit}, off_alive={off_alive}, def_alive={def_alive}, "
          f"reason={reason:10s}, mean_rew={np.mean(ep_rew):.1f}, steps={env.current_step}")

print(f"\n{'='*60}")
print(f"Zero-Action Baseline Results ({N_EPISODES} episodes):")
print(f"  Hit rate: {hits_total}/{N_EPISODES*4} = {hits_total/(N_EPISODES*4)*100:.1f}%")
print(f"  Mean episode reward: {np.mean(total_rews):.1f} ± {np.std(total_rews):.1f}")
print(f"  Done reasons: {done_reasons}")
print(f"{'='*60}")

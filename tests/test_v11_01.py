#!/usr/bin/env python3
"""Quick test for V11_01 environment"""
import numpy as np
import sys
sys.path.insert(0, "/home/ps/2026/muti_uav_attack/muti_uav_attack_v11_01")

from envs.fov_penetration.fov_penetration_env import FOVPenetrationEnv

env = FOVPenetrationEnv(scenario='scenario_1')
obs, share_obs, avail = env.reset()

print('=== V11_01 Env Test ===')
print(f'Obs dim: {obs[0].shape}')
print(f'Share obs dim: {share_obs[0].shape}')
print(f'Action space: {env.action_space[0]}')
print(f'N agents: {env.n_agents}')
print(f'Kill range: {env.config["kill_range"]}')
print(f'Collision kill range: {env.config["collision_kill_range"]}')
print(f'FOV escape enabled: {env.config["fov_escape"]["enabled"]}')
print(f'Forward only pursuit: {env.config["pursuit"]["forward_only"]}')
print(f'Map size: {env.config["map_size"]}')
print(f'Episode length: {env.config["max_steps"]}')

# 跑10步测试
total_r = 0
for step in range(10):
    actions = [np.random.uniform(-1, 1, 3) for _ in range(env.n_agents)]
    obs, share_obs, rewards, costs, dones, infos, avail = env.step(actions)
    r = sum(r[0] for r in rewards)
    total_r += r
    
print(f'10 steps OK, total reward: {total_r:.2f}')

# 跑完整episode测试, 多次以看逃逸统计
total_escapes = 0
total_episodes = 5
for ep in range(total_episodes):
    env2 = FOVPenetrationEnv(scenario='scenario_1')
    env2.seed(ep)
    obs, share_obs, avail = env2.reset()
    for step in range(1200):
        actions = [np.random.uniform(-1, 1, 3) for _ in range(env2.n_agents)]
        obs, share_obs, rewards, costs, dones, infos, avail = env2.step(actions)
        if dones[0]:
            n_esc = infos[0].get('n_escapes_total', 0)
            total_escapes += n_esc
            print(f'Ep{ep}: step={step+1}, reason={infos[0]["done_reason"]}, '
                  f'hits={infos[0]["hit_count"]}, off_alive={infos[0]["offensive_alive"]}, '
                  f'def_alive={infos[0]["defensive_alive"]}, '
                  f'escapes={n_esc}, escaped_agents={infos[0]["n_escaped_agents"]}')
            break

print(f'\nTotal escapes across {total_episodes} episodes: {total_escapes}')
print('=== V11_01 Env Test PASSED ===')

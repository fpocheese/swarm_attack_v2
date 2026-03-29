#!/usr/bin/env python
"""
渲染随机策略 Episode
======================
使用随机动作运行一个 episode 并保存动画
"""
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from envs.fov_penetration import FOVPenetrationEnv
from envs.fov_penetration.render import render_episode, render_single_frame_to_file
import numpy as np


def main():
    env = FOVPenetrationEnv()
    env.seed(42)
    
    # 创建输出目录
    output_dir = os.path.join(PROJECT_ROOT, "outputs", "gifs")
    os.makedirs(output_dir, exist_ok=True)
    
    save_path = os.path.join(output_dir, "random_episode.gif")
    
    print("Running random policy episode...")
    print(f"Env info: {env.get_env_info()}")
    print(f"Obs dim: {env.obs_dim}, Share obs dim: {env.share_obs_dim}")
    
    # 先运行一个 episode 收集动作
    obs, share_obs, _ = env.reset()
    actions_list = []
    
    for step in range(env.max_steps):
        actions = [env.action_space[i].sample() for i in range(env.n_agents)]
        actions_list.append(actions)
        
        obs, share_obs, rewards, costs, dones, infos, _ = env.step(actions)
        
        if step % 50 == 0:
            info = infos[0]
            print(f"Step {step}: reward_avg={np.mean([r[0] for r in rewards]):.3f}, "
                  f"cost={sum(c[0] for c in costs):.3f}, "
                  f"atk_alive={env.attacker.alive}, "
                  f"escorts_alive={sum(1 for e in env.escorts if e.alive)}")
        
        if any(dones):
            print(f"\nEpisode ended at step {step+1}: {infos[0].get('done_reason', 'unknown')}")
            break
    
    # 重新运行并渲染
    print(f"\nRendering animation to {save_path}...")
    env.seed(42)
    
    def random_policy(obs):
        return [env.action_space[i].sample() for i in range(env.n_agents)]
    
    render_episode(env, policy_fn=random_policy, save_path=save_path, fps=10)
    
    # 同时保存初始帧
    env.seed(42)
    env.reset()
    frame_path = os.path.join(output_dir, "initial_frame.png")
    render_single_frame_to_file(env, frame_path)
    
    print("\nDone!")
    print(f"GIF: {save_path}")
    print(f"Initial frame: {frame_path}")


if __name__ == "__main__":
    main()

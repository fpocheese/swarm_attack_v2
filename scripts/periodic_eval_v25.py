#!/usr/bin/env python
"""V25 实时评估脚本 — 每隔N分钟测试一次，绘制GIF可视化"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from pathlib import Path
from envs.fov_penetration import FOVPenetrationEnv
from envs.fov_penetration.render import render_episode_to_gif

# 配置
EVAL_INTERVAL = 600  # 每600秒(10分钟)评估一次
NUM_EVAL_EPISODES = 5
OUTPUT_DIR = Path("outputs/v25_eval_gifs")
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR = Path("outputs/results/fov_penetration/mappo/v25_mappo_no_extrap_decoy/run1/models")

def load_policy(ckpt_path):
    """加载MAPPO策略"""
    try:
        state = torch.load(ckpt_path, map_location='cpu')
        # 假设模型入参: (obs_dim, action_dim)
        # 这里只是加载权重，实际推理需要完整的模型定义
        return state
    except:
        return None

def run_episode(env, render=False):
    """运行一个完整的episode，返回统计数据和轨迹"""
    obs, share_obs, avail = env.reset()
    episode_rewards = [0.0] * env.n_agents
    episode_length = 0
    trajectories = []  # [{step, positions, ...}]
    
    for step in range(env.max_steps):
        # 随机动作(测试阶段)
        actions = [np.random.uniform(-1, 1, size=3) for _ in range(env.n_agents)]
        obs, share_obs, rewards, costs, dones, infos, avail = env.step(actions)
        
        for i in range(env.n_agents):
            episode_rewards[i] += rewards[i][0] if isinstance(rewards[i], list) else rewards[i]
        
        episode_length += 1
        
        # 记录轨迹(渲染用)
        traj_data = {
            'step': step,
            'offensive_pos': [(off.x, off.y, off.z) for off in env.offensives],
            'defensive_pos': [(d.x, d.y, d.z) for d in env.defensives],
            'hvt_pos': (env.hvt.x, env.hvt.y, env.hvt.z),
            'offensive_alive': [off.alive for off in env.offensives],
            'defensive_alive': [d.alive for d in env.defensives],
        }
        trajectories.append(traj_data)
        
        if all(dones):
            break
    
    info = infos[0]
    return {
        'total_reward': np.mean(episode_rewards),
        'episode_length': episode_length,
        'penetration_success': any(off.hit_hvt for off in env.offensives),
        'offensive_alive': sum(1 for off in env.offensives if off.alive),
        'defensive_alive': sum(1 for d in env.defensives if d.alive),
        'trajectories': trajectories,
    }

def main():
    """主循环: 每隔EVAL_INTERVAL秒运行一次评估"""
    env = FOVPenetrationEnv(scenario='scenario_1')
    env.seed(42)
    
    eval_count = 0
    last_eval_time = time.time()
    
    print(f"[V25 Eval] 启动实时评估，间隔{EVAL_INTERVAL}秒，进度保存到 {OUTPUT_DIR}")
    
    while True:
        current_time = time.time()
        
        # 检查是否需要评估
        if current_time - last_eval_time >= EVAL_INTERVAL:
            eval_count += 1
            last_eval_time = current_time
            
            print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] ==== V25 评估 #{eval_count} ====")
            
            results = []
            for ep in range(NUM_EVAL_EPISODES):
                result = run_episode(env, render=True)
                results.append(result)
                print(f"  Ep{ep+1}: reward={result['total_reward']:.2f}, "
                      f"len={result['episode_length']}, "
                      f"hit_hvt={result['penetration_success']}, "
                      f"off_alive={result['offensive_alive']}, "
                      f"def_alive={result['defensive_alive']}")
            
            # 绘制一个代表性episode的GIF
            best_idx = np.argmax([r['total_reward'] for r in results])
            best_traj = results[best_idx]['trajectories']
            
            gif_path = OUTPUT_DIR / f"v25_eval_{eval_count:03d}_reward_{results[best_idx]['total_reward']:.0f}.gif"
            print(f"  绘制GIF: {gif_path}")
            
            try:
                render_episode_to_gif(
                    best_traj,
                    hvt_pos=(1200.0, 0.0, 0.0),
                    output_path=str(gif_path),
                    map_size=2000.0,
                )
                print(f"  ✓ GIF已保存")
            except Exception as e:
                print(f"  ✗ GIF绘制失败: {e}")
            
            # 统计
            avg_reward = np.mean([r['total_reward'] for r in results])
            success_rate = np.mean([r['penetration_success'] for r in results])
            avg_off_alive = np.mean([r['offensive_alive'] for r in results])
            avg_def_alive = np.mean([r['defensive_alive'] for r in results])
            
            print(f"  [统计] 平均奖励={avg_reward:.2f}, "
                  f"成功率={success_rate*100:.1f}%, "
                  f"平均进攻存活={avg_off_alive:.2f}, "
                  f"平均防守存活={avg_def_alive:.2f}")
            
            # 保存结果到文件
            result_file = OUTPUT_DIR / f"v25_eval_{eval_count:03d}_stats.txt"
            with open(result_file, 'w') as f:
                f.write(f"V25 Evaluation #{eval_count}\n")
                f.write(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Average Reward: {avg_reward:.2f}\n")
                f.write(f"Success Rate: {success_rate*100:.1f}%\n")
                f.write(f"Avg Offensive Alive: {avg_off_alive:.2f}\n")
                f.write(f"Avg Defensive Alive: {avg_def_alive:.2f}\n")
        
        time.sleep(5)  # 检查间隔5秒

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n[V25 Eval] 已停止")
        sys.exit(0)

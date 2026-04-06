#!/usr/bin/env python
"""V25 快速评估 — 10个episode，绘制轨迹可视化"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pathlib import Path
from envs.fov_penetration import FOVPenetrationEnv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.patches import FancyArrowPatch
import matplotlib.patches as mpatches

OUTPUT_DIR = Path("outputs/v25_eval_quick")
OUTPUT_DIR.mkdir(exist_ok=True)

def run_episode(env):
    """运行一个episode并记录轨迹"""
    obs, share_obs, avail = env.reset()
    
    trajectories = {
        'off': [[] for _ in range(env.n_agents)],
        'def': [[] for _ in range(len(env.defensives))],
        'hvt': [],
        'rewards': [],
        'episode_data': []
    }
    
    for step in range(env.max_steps):
        # 记录位置
        for i, off in enumerate(env.offensives):
            if off.alive:
                trajectories['off'][i].append((off.x, off.y))
        for i, d in enumerate(env.defensives):
            if d.alive:
                trajectories['def'][i].append((d.x, d.y))
        trajectories['hvt'].append((env.hvt.x, env.hvt.y))
        
        # 随机动作
        actions = [np.random.uniform(-1, 1, size=3) for _ in range(env.n_agents)]
        obs, share_obs, rewards, costs, dones, infos, avail = env.step(actions)
        
        for i in range(len(rewards)):
            if isinstance(rewards[i], list):
                trajectories['rewards'].append(rewards[i][0])
            else:
                trajectories['rewards'].append(rewards[i])
        
        episode_data = {
            'step': step,
            'off_alive': [off.alive for off in env.offensives],
            'def_alive': [d.alive for d in env.defensives],
        }
        trajectories['episode_data'].append(episode_data)
        
        if all(dones):
            break
    
    info = infos[0]
    return trajectories, {
        'avg_reward': np.mean(trajectories['rewards']) if trajectories['rewards'] else 0,
        'hit_hvt': any(off.hit_hvt for off in env.offensives),
        'steps': len(trajectories['episode_data']),
        'off_alive_end': sum(1 for off in env.offensives if off.alive),
        'def_alive_end': sum(1 for d in env.defensives if d.alive),
    }

def plot_trajectories(trajectories, stats, episode_num):
    """绘制一个episode的轨迹"""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 绘制战场边界
    ax.set_xlim(-2500, 2500)
    ax.set_ylim(-1500, 1500)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # 绘制HVT
    hvt_x, hvt_y = trajectories['hvt'][0]
    ax.plot(hvt_x, hvt_y, 'r*', markersize=25, label='HVT', zorder=10)
    
    # 绘制进攻方轨迹
    colors_off = ['blue', 'cyan', 'lime', 'magenta']
    for i, traj in enumerate(trajectories['off']):
        if traj:
            xs, ys = zip(*traj)
            ax.plot(xs, ys, color=colors_off[i], linewidth=2, alpha=0.7, label=f'Offensive {i}')
            ax.plot(xs[-1], ys[-1], 'o', color=colors_off[i], markersize=10)
    
    # 绘制防守方轨迹
    colors_def = ['red', 'orange', 'pink', 'brown']
    for i, traj in enumerate(trajectories['def']):
        if traj:
            xs, ys = zip(*traj)
            ax.plot(xs, ys, color=colors_def[i], linewidth=1.5, linestyle='--', alpha=0.6, label=f'Defensive {i}')
            ax.plot(xs[-1], ys[-1], 's', color=colors_def[i], markersize=8)
    
    # 标题和统计
    title = f"V25 Episode {episode_num}\n" \
            f"Reward={stats['avg_reward']:.1f}, Hit_HVT={stats['hit_hvt']}, " \
            f"Steps={stats['steps']}\n" \
            f"Off_Alive={stats['off_alive_end']}, Def_Alive={stats['def_alive_end']}"
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    return fig

def main():
    env = FOVPenetrationEnv(scenario='scenario_1')
    env.seed(42)
    
    print("[V25 Quick Eval] 运行10个episode...")
    
    all_stats = []
    
    for ep in range(10):
        print(f"  Episode {ep+1}/10...", end=' ', flush=True)
        traj, stats = run_episode(env)
        all_stats.append(stats)
        
        # 绘制前3个episode
        if ep < 3:
            fig = plot_trajectories(traj, stats, ep+1)
            gif_path = OUTPUT_DIR / f"v25_quick_ep{ep+1:02d}_reward_{stats['avg_reward']:.0f}.png"
            fig.savefig(gif_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            print(f"✓ reward={stats['avg_reward']:.1f}, hit={stats['hit_hvt']}")
        else:
            print(f"reward={stats['avg_reward']:.1f}, hit={stats['hit_hvt']}")
    
    # 统计
    print("\n" + "="*70)
    print("[V25 评估结果]")
    avg_reward = np.mean([s['avg_reward'] for s in all_stats])
    success_count = sum(1 for s in all_stats if s['hit_hvt'])
    success_rate = success_count / len(all_stats) * 100
    avg_off_alive = np.mean([s['off_alive_end'] for s in all_stats])
    avg_def_alive = np.mean([s['def_alive_end'] for s in all_stats])
    avg_steps = np.mean([s['steps'] for s in all_stats])
    
    print(f"平均奖励: {avg_reward:.2f}")
    print(f"成功率(击中HVT): {success_rate:.1f}% ({success_count}/{len(all_stats)})")
    print(f"平均进攻方存活: {avg_off_alive:.2f}/4")
    print(f"平均防守方存活: {avg_def_alive:.2f}/4")
    print(f"平均episode长度: {avg_steps:.1f}步")
    print(f"可视化已保存到: {OUTPUT_DIR}")
    print("="*70)

if __name__ == '__main__':
    main()

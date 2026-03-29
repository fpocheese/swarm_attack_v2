#!/usr/bin/env python
"""
飞行器集群博弈对抗诊断脚本
==========================
对第1个测试工况(episode_00)逐步回放，记录进攻方/防御方全量飞行状态，
并绘制以下常见分析曲线（保存为PNG图片）:

1. 三维轨迹总览
2. 高度-时间曲线（含z_min安全线）
3. 速度-时间曲线
4. 航向角-时间曲线
5. 俯仰角(gamma)-时间曲线
6. 纵向过载(nx)-时间曲线
7. 侧向过载(ny)-时间曲线
8. 法向过载(nz)-时间曲线
9. 各进攻飞行器到HVT距离-时间曲线
10. 各进攻飞行器到最近防御方距离-时间曲线
11. 动作输出(原始action[-1,1])-时间曲线
"""

import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['DejaVu Sans']
rcParams['axes.unicode_minus'] = False

# paths
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'third_party', 'MACPO', 'MACPO'))

from macpo.config import get_config as get_macpo_config
from macpo.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy
from envs.fov_penetration import FOVPenetrationEnv


def load_policies(env, model_dir):
    parser = get_macpo_config()
    args = parser.parse_known_args([
        '--algorithm_name', 'macpo',
        '--hidden_size', '256',
        '--layer_N', '3',
        '--lr', '5e-4',
        '--critic_lr', '5e-4',
    ])[0]

    obs_space = env.observation_space[0]
    share_obs_space = env.share_observation_space[0]
    act_space = env.action_space[0]

    policies = []
    for i in range(env.n_agents):
        policy = R_MAPPOPolicy(args, obs_space, share_obs_space, act_space, device=torch.device('cpu'))
        actor_state = torch.load(os.path.join(model_dir, f'actor_agent{i}.pt'), map_location='cpu')
        policy.actor.load_state_dict(actor_state)
        policy.actor.eval()
        policies.append(policy)
    return policies


def run_episode_collect_data(env, policies, seed=0, max_steps=500):
    """运行一个episode，收集全量飞行状态数据"""
    env.seed(seed)
    obs, share_obs, _ = env.reset()

    rnn_states = [np.zeros((1, 1, 256), dtype=np.float32) for _ in range(env.n_agents)]
    masks = [np.ones((1, 1), dtype=np.float32) for _ in range(env.n_agents)]

    n_off = env.n_offensive
    n_def = env.n_defensive

    # 数据容器 - 进攻方
    off_data = {
        'x': [[] for _ in range(n_off)],
        'y': [[] for _ in range(n_off)],
        'z': [[] for _ in range(n_off)],
        'v': [[] for _ in range(n_off)],
        'heading': [[] for _ in range(n_off)],
        'gamma': [[] for _ in range(n_off)],
        'nx': [[] for _ in range(n_off)],
        'ny': [[] for _ in range(n_off)],
        'nz': [[] for _ in range(n_off)],
        'alive': [[] for _ in range(n_off)],
        'dist_hvt': [[] for _ in range(n_off)],
        'dist_nearest_def': [[] for _ in range(n_off)],
        'action_raw': [[] for _ in range(n_off)],  # [-1,1]^3
    }

    # 数据容器 - 防御方
    def_data = {
        'x': [[] for _ in range(n_def)],
        'y': [[] for _ in range(n_def)],
        'z': [[] for _ in range(n_def)],
        'v': [[] for _ in range(n_def)],
        'heading': [[] for _ in range(n_def)],
        'gamma': [[] for _ in range(n_def)],
        'nx': [[] for _ in range(n_def)],
        'ny': [[] for _ in range(n_def)],
        'nz': [[] for _ in range(n_def)],
        'alive': [[] for _ in range(n_def)],
    }

    steps_list = []
    done_reason = 'unknown'

    # 记录初始状态
    def record_state(step):
        steps_list.append(step)
        for i, off in enumerate(env.offensives):
            off_data['x'][i].append(off.x)
            off_data['y'][i].append(off.y)
            off_data['z'][i].append(off.z)
            off_data['v'][i].append(off.v)
            off_data['heading'][i].append(np.degrees(off.heading))
            off_data['gamma'][i].append(np.degrees(off.gamma))
            off_data['nx'][i].append(off.nx)
            off_data['ny'][i].append(off.ny)
            off_data['nz'][i].append(off.nz)
            off_data['alive'][i].append(1.0 if off.alive else 0.0)
            off_data['dist_hvt'][i].append(off.distance_to(env.hvt.x, env.hvt.y, env.hvt.z))
            # 到最近防御方距离
            min_d = float('inf')
            for d in env.defensives:
                if d.alive:
                    dd = off.distance_3d(d)
                    if dd < min_d:
                        min_d = dd
            off_data['dist_nearest_def'][i].append(min_d if min_d < float('inf') else 0)

        for i, d in enumerate(env.defensives):
            def_data['x'][i].append(d.x)
            def_data['y'][i].append(d.y)
            def_data['z'][i].append(d.z)
            def_data['v'][i].append(d.v)
            def_data['heading'][i].append(np.degrees(d.heading))
            def_data['gamma'][i].append(np.degrees(d.gamma))
            def_data['nx'][i].append(d.nx)
            def_data['ny'][i].append(d.ny)
            def_data['nz'][i].append(d.nz)
            def_data['alive'][i].append(1.0 if d.alive else 0.0)

    record_state(0)

    for step in range(max_steps):
        actions = []
        actions_raw = []
        for i in range(env.n_agents):
            with torch.no_grad():
                obs_t = np.array(obs[i]).reshape(1, 1, -1)
                action, _, rnn_out = policies[i].actor(obs_t, rnn_states[i], masks[i], deterministic=True)
                act_np = action.squeeze().numpy()
                actions.append(act_np)
                actions_raw.append(act_np.copy())
                rnn_states[i] = rnn_out.numpy()

        # 记录原始动作
        for i in range(n_off):
            off_data['action_raw'][i].append(actions_raw[i].tolist())

        obs, share_obs, rewards, costs, dones, infos, _ = env.step(actions)

        record_state(step + 1)

        if dones[0]:
            done_reason = infos[0].get('done_reason', 'unknown')
            break

    # 把action_raw补齐最后一步(占位)
    for i in range(n_off):
        while len(off_data['action_raw'][i]) < len(steps_list):
            off_data['action_raw'][i].append([0, 0, 0])

    return steps_list, off_data, def_data, done_reason


def plot_diagnostics(steps, off_data, def_data, done_reason, out_dir, config):
    os.makedirs(out_dir, exist_ok=True)
    n_off = len(off_data['x'])
    n_def = len(def_data['x'])
    t = np.array(steps) * config['dt']  # 转换为秒

    off_colors = ['blue', 'dodgerblue', 'cyan', 'navy']
    def_colors = ['red', 'orangered', 'salmon', 'darkred']

    z_min = config.get('z_min', 100.0)

    # ============ 1. 三维轨迹总览 ============
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    for i in range(n_off):
        ax.plot3D(off_data['x'][i], off_data['y'][i], off_data['z'][i],
                  color=off_colors[i % len(off_colors)], linewidth=1.5, label=f'Off{i}')
        # 标记起点和终点
        ax.scatter3D(off_data['x'][i][0], off_data['y'][i][0], off_data['z'][i][0],
                     marker='o', color=off_colors[i % len(off_colors)], s=60)
        ax.scatter3D(off_data['x'][i][-1], off_data['y'][i][-1], off_data['z'][i][-1],
                     marker='x', color=off_colors[i % len(off_colors)], s=100)
    for i in range(n_def):
        ax.plot3D(def_data['x'][i], def_data['y'][i], def_data['z'][i],
                  color=def_colors[i % len(def_colors)], linewidth=1.5, linestyle='--', label=f'Def{i}')
    # HVT
    hvt_pos = config['hvt_position']
    ax.scatter3D(hvt_pos[0], hvt_pos[1], hvt_pos[2] if len(hvt_pos) > 2 else 0,
                 marker='*', color='gold', s=300, edgecolor='black', linewidth=1.5, label='HVT')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'3D Trajectory Overview (done: {done_reason})')
    ax.legend(fontsize=8)
    fig.savefig(os.path.join(out_dir, '01_3d_trajectory.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  [1/11] 3D trajectory saved")

    # ============ 2. 高度-时间曲线 ============
    fig, ax = plt.subplots(figsize=(14, 6))
    for i in range(n_off):
        ax.plot(t, off_data['z'][i], color=off_colors[i % len(off_colors)], linewidth=1.5, label=f'Off{i}')
    for i in range(n_def):
        ax.plot(t, def_data['z'][i], color=def_colors[i % len(def_colors)], linewidth=1.2, linestyle='--', label=f'Def{i}')
    ax.axhline(y=z_min, color='red', linestyle=':', linewidth=2, label=f'z_min={z_min}m (CRASH)')
    ax.axhline(y=z_min * 1.5, color='orange', linestyle=':', linewidth=1.5, label=f'z_safe={z_min*1.5:.0f}m')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Altitude Z (m)')
    ax.set_title('Altitude vs Time')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(out_dir, '02_altitude.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  [2/11] Altitude saved")

    # ============ 3. 速度-时间曲线 ============
    fig, ax = plt.subplots(figsize=(14, 6))
    for i in range(n_off):
        ax.plot(t, off_data['v'][i], color=off_colors[i % len(off_colors)], linewidth=1.5, label=f'Off{i}')
    for i in range(n_def):
        ax.plot(t, def_data['v'][i], color=def_colors[i % len(def_colors)], linewidth=1.2, linestyle='--', label=f'Def{i}')
    ax.axhline(y=config['offensive']['v_min'], color='blue', linestyle=':', alpha=0.5, label=f"Off v_min={config['offensive']['v_min']}")
    ax.axhline(y=config['offensive']['v_max'], color='blue', linestyle=':', alpha=0.5, label=f"Off v_max={config['offensive']['v_max']}")
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Speed (m/s)')
    ax.set_title('Speed vs Time')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(out_dir, '03_speed.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  [3/11] Speed saved")

    # ============ 4. 航向角-时间曲线 ============
    fig, ax = plt.subplots(figsize=(14, 6))
    for i in range(n_off):
        ax.plot(t, off_data['heading'][i], color=off_colors[i % len(off_colors)], linewidth=1.5, label=f'Off{i}')
    for i in range(n_def):
        ax.plot(t, def_data['heading'][i], color=def_colors[i % len(def_colors)], linewidth=1.2, linestyle='--', label=f'Def{i}')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Heading (deg)')
    ax.set_title('Heading Angle vs Time')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(out_dir, '04_heading.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  [4/11] Heading saved")

    # ============ 5. 俯仰角(gamma)-时间曲线 ============
    fig, ax = plt.subplots(figsize=(14, 6))
    for i in range(n_off):
        ax.plot(t, off_data['gamma'][i], color=off_colors[i % len(off_colors)], linewidth=1.5, label=f'Off{i}')
    for i in range(n_def):
        ax.plot(t, def_data['gamma'][i], color=def_colors[i % len(def_colors)], linewidth=1.2, linestyle='--', label=f'Def{i}')
    gamma_min_deg = np.degrees(config['offensive']['gamma_min'])
    gamma_max_deg = np.degrees(config['offensive']['gamma_max'])
    ax.axhline(y=gamma_min_deg, color='blue', linestyle=':', alpha=0.5, label=f'Off gamma_min={gamma_min_deg:.0f} deg')
    ax.axhline(y=gamma_max_deg, color='blue', linestyle=':', alpha=0.5, label=f'Off gamma_max={gamma_max_deg:.0f} deg')
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Flight Path Angle gamma (deg)')
    ax.set_title('Flight Path Angle (Gamma) vs Time')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(out_dir, '05_gamma.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  [5/11] Gamma saved")

    # ============ 6. 纵向过载(nx)-时间曲线 ============
    fig, ax = plt.subplots(figsize=(14, 6))
    for i in range(n_off):
        ax.plot(t, off_data['nx'][i], color=off_colors[i % len(off_colors)], linewidth=1.5, label=f'Off{i}')
    for i in range(n_def):
        ax.plot(t, def_data['nx'][i], color=def_colors[i % len(def_colors)], linewidth=1.2, linestyle='--', label=f'Def{i}')
    ax.axhline(y=config['offensive']['nx_min'], color='blue', linestyle=':', alpha=0.5)
    ax.axhline(y=config['offensive']['nx_max'], color='blue', linestyle=':', alpha=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('nx (longitudinal overload)')
    ax.set_title('Longitudinal Overload (nx) vs Time')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(out_dir, '06_nx.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  [6/11] nx saved")

    # ============ 7. 侧向过载(ny)-时间曲线 ============
    fig, ax = plt.subplots(figsize=(14, 6))
    for i in range(n_off):
        ax.plot(t, off_data['ny'][i], color=off_colors[i % len(off_colors)], linewidth=1.5, label=f'Off{i}')
    for i in range(n_def):
        ax.plot(t, def_data['ny'][i], color=def_colors[i % len(def_colors)], linewidth=1.2, linestyle='--', label=f'Def{i}')
    ax.axhline(y=config['offensive']['ny_min'], color='blue', linestyle=':', alpha=0.5)
    ax.axhline(y=config['offensive']['ny_max'], color='blue', linestyle=':', alpha=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('ny (lateral overload)')
    ax.set_title('Lateral Overload (ny) vs Time')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(out_dir, '07_ny.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  [7/11] ny saved")

    # ============ 8. 法向过载(nz)-时间曲线 ============
    fig, ax = plt.subplots(figsize=(14, 6))
    for i in range(n_off):
        ax.plot(t, off_data['nz'][i], color=off_colors[i % len(off_colors)], linewidth=1.5, label=f'Off{i}')
    for i in range(n_def):
        ax.plot(t, def_data['nz'][i], color=def_colors[i % len(def_colors)], linewidth=1.2, linestyle='--', label=f'Def{i}')
    ax.axhline(y=config['offensive']['nz_min'], color='blue', linestyle=':', alpha=0.5)
    ax.axhline(y=config['offensive']['nz_max'], color='blue', linestyle=':', alpha=0.5)
    ax.axhline(y=1.0, color='green', linestyle='-', alpha=0.5, linewidth=2, label='nz=1 (level flight)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('nz (normal overload)')
    ax.set_title('Normal Overload (nz) vs Time  [nz=1 => level flight, nz<1 => descending]')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(out_dir, '08_nz.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  [8/11] nz saved")

    # ============ 9. 到HVT距离-时间曲线 ============
    fig, ax = plt.subplots(figsize=(14, 6))
    for i in range(n_off):
        ax.plot(t, off_data['dist_hvt'][i], color=off_colors[i % len(off_colors)], linewidth=1.5, label=f'Off{i}')
    ax.axhline(y=config['kill_range'], color='green', linestyle=':', linewidth=2, label=f"kill_range={config['kill_range']}m")
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Distance to HVT (m)')
    ax.set_title('Distance to HVT vs Time')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(out_dir, '09_dist_hvt.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  [9/11] dist_hvt saved")

    # ============ 10. 到最近防御方距离-时间曲线 ============
    fig, ax = plt.subplots(figsize=(14, 6))
    for i in range(n_off):
        ax.plot(t, off_data['dist_nearest_def'][i], color=off_colors[i % len(off_colors)], linewidth=1.5, label=f'Off{i}')
    ax.axhline(y=config['collision_range'], color='red', linestyle=':', linewidth=2, label=f"kill_range={config['collision_range']}m")
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Distance to Nearest Defender (m)')
    ax.set_title('Distance to Nearest Defender vs Time')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(out_dir, '10_dist_def.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  [10/11] dist_def saved")

    # ============ 11. 原始动作输出[-1,1] ============
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    action_labels = ['action[0] -> nx_cmd', 'action[1] -> ny_cmd', 'action[2] -> nz_cmd']
    for dim in range(3):
        ax = axes[dim]
        for i in range(n_off):
            vals = [a[dim] if isinstance(a, (list, np.ndarray)) else 0 for a in off_data['action_raw'][i]]
            ax.plot(t, vals, color=off_colors[i % len(off_colors)], linewidth=1.2, label=f'Off{i}')
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        ax.axhline(y=-1, color='gray', linestyle=':', alpha=0.3)
        ax.axhline(y=1, color='gray', linestyle=':', alpha=0.3)
        ax.set_ylabel(action_labels[dim])
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-1.3, 1.3)
    axes[2].set_xlabel('Time (s)')
    axes[0].set_title('Raw Action Output [-1, 1] vs Time')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, '11_actions_raw.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  [11/11] actions saved")

    # ============ 综合摘要文本 ============
    summary = []
    summary.append(f"Episode Done Reason: {done_reason}")
    summary.append(f"Total Steps: {len(steps)}")
    summary.append(f"Total Time: {t[-1]:.1f} s")
    summary.append("")
    for i in range(n_off):
        z_arr = np.array(off_data['z'][i])
        gamma_arr = np.array(off_data['gamma'][i])
        nz_arr = np.array(off_data['nz'][i])
        alive_arr = np.array(off_data['alive'][i])
        death_step = np.argmax(alive_arr < 1.0) if np.any(alive_arr < 1.0) else -1
        summary.append(f"=== Offensive {i} ===")
        summary.append(f"  z_init={z_arr[0]:.1f}m, z_min_reached={z_arr.min():.1f}m, z_final={z_arr[-1]:.1f}m")
        summary.append(f"  gamma_mean={gamma_arr.mean():.2f} deg, gamma_min={gamma_arr.min():.2f}, gamma_max={gamma_arr.max():.2f}")
        summary.append(f"  nz_mean={nz_arr.mean():.3f}, nz_min={nz_arr.min():.3f}, nz_max={nz_arr.max():.3f}")
        summary.append(f"  nz<1 ratio: {(nz_arr < 1.0).sum()}/{len(nz_arr)} = {(nz_arr < 1.0).mean()*100:.1f}%")
        summary.append(f"  dist_hvt_final={off_data['dist_hvt'][i][-1]:.1f}m")
        if death_step > 0:
            summary.append(f"  KILLED at step {death_step}, z={z_arr[death_step]:.1f}m, t={t[death_step]:.1f}s")
        summary.append("")

    summary_text = "\n".join(summary)
    with open(os.path.join(out_dir, 'summary.txt'), 'w') as f:
        f.write(summary_text)
    print("\n" + summary_text)


def main():
    model_dir = os.path.join(project_root, 'outputs', 'results', 'fov_penetration',
                             'macpo', 'v9_coop_penetration', 'run1', 'models')
    out_dir = os.path.join(project_root, 'outputs', 'diagnostics', 'v9_ep00')

    env = FOVPenetrationEnv(scenario='scenario_1')
    policies = load_policies(env, model_dir)

    print(f"Model: {model_dir}")
    print(f"Output: {out_dir}")
    print("Running episode 0 with seed=0 ...")

    steps, off_data, def_data, done_reason = run_episode_collect_data(env, policies, seed=0, max_steps=500)

    print(f"Episode done: {done_reason}, total steps: {len(steps)}")
    print("Generating diagnostic plots ...")

    plot_diagnostics(steps, off_data, def_data, done_reason, out_dir, env.config)

    print(f"\nAll diagnostic plots saved to: {out_dir}")


if __name__ == '__main__':
    main()

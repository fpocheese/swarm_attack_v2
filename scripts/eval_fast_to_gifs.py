#!/usr/bin/env python
"""
评估 MAPPO 模型并导出 GIF 与关键曲线图（过载/速度/距离）。
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "MACPO", "MACPO"))

from macpo.config import get_config as get_macpo_config
from macpo.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy
from envs.fov_penetration import FOVPenetrationEnv
from envs.fov_penetration.render import render_frame


def load_policies(env, model_dir, hidden_size=256, layer_N=3):
    parser = get_macpo_config()

    args = parser.parse_known_args([
        '--algorithm_name', 'mappo',
        '--hidden_size', str(hidden_size),
        '--layer_N', str(layer_N),
        '--lr', '5e-4',
        '--critic_lr', '5e-4',
        '--use_feature_normalization',
        '--use_recurrent_policy',
    ])[0]

    obs_space = env.observation_space[0]
    share_obs_space = env.share_observation_space[0]
    act_space = env.action_space[0]

    policies = []
    for i in range(env.n_agents):
        policy = R_MAPPOPolicy(args, obs_space, share_obs_space, act_space, device=torch.device('cpu'))
        actor_state = torch.load(os.path.join(model_dir, f'actor_agent{i}.pt'), map_location='cpu')
        policy.actor.load_state_dict(actor_state, strict=False)
        policy.actor.eval()
        policies.append(policy)
    return policies


def _record_telemetry(env, telemetry):
    telemetry['step'].append(env.current_step)
    for i, off in enumerate(env.offensives):
        telemetry['off_speed'][i].append(float(off.v))
        telemetry['off_z'][i].append(float(off.z))
        telemetry['off_alive'][i].append(1.0 if off.alive else 0.0)
        g0 = 9.81
        n_pitch = float(off.an_pitch / g0)
        n_yaw = float(off.an_yaw / g0)
        n_total = float(np.sqrt(off.an_pitch ** 2 + off.an_yaw ** 2) / g0)

        telemetry['off_ax'][i].append(float(off.ax))
        telemetry['off_n_pitch'][i].append(n_pitch)
        telemetry['off_n_yaw'][i].append(n_yaw)
        telemetry['off_n_total'][i].append(n_total)

        dist_hvt = float(off.distance_to(env.hvt.x, env.hvt.y, env.hvt.z))
        telemetry['off_dist_hvt'][i].append(dist_hvt)

        min_d = float('inf')
        for d in env.defensives:
            if d.alive:
                min_d = min(min_d, float(off.distance_3d(d)))
        telemetry['off_dist_nearest_def'][i].append(min_d if min_d < float('inf') else 0.0)


def save_telemetry_plot(telemetry, dt, out_png):
    t = np.array(telemetry['step'], dtype=np.float32) * dt
    n_off = len(telemetry['off_speed'])
    colors = ['blue', 'dodgerblue', 'cyan', 'navy']

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    ax = axes[0, 0]
    for i in range(n_off):
        ax.plot(t, telemetry['off_speed'][i], color=colors[i % len(colors)], linewidth=1.5, label=f'Off{i}')
    ax.set_title('Speed vs Time')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Speed (m/s)')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    for i in range(n_off):
        ax.plot(t, telemetry['off_n_total'][i], color=colors[i % len(colors)], linewidth=1.5, label=f'Off{i}')
    ax.set_title('Total Overload |n| vs Time')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('|n| (g-like unit)')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    for i in range(n_off):
        ax.plot(t, telemetry['off_dist_hvt'][i], color=colors[i % len(colors)], linewidth=1.5, label=f'Off{i}')
    ax.set_title('Distance to HVT vs Time')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Distance (m)')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    for i in range(n_off):
        ax.plot(t, telemetry['off_dist_nearest_def'][i], color=colors[i % len(colors)], linewidth=1.5, label=f'Off{i}')
    ax.set_title('Distance to Nearest Defender vs Time')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Distance (m)')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_png, dpi=160, bbox_inches='tight')
    plt.close(fig)


def save_telemetry_csv(telemetry, dt, out_csv):
    n_off = len(telemetry['off_speed'])
    header = ['step', 'time_s']
    for i in range(n_off):
        header.extend([
            f'off{i}_speed', f'off{i}_z', f'off{i}_alive', f'off{i}_ax', f'off{i}_n_pitch', f'off{i}_n_yaw', f'off{i}_n_total',
            f'off{i}_dist_hvt', f'off{i}_dist_nearest_def'
        ])

    rows = []
    for idx, step in enumerate(telemetry['step']):
        row = [step, step * dt]
        for i in range(n_off):
            row.extend([
                telemetry['off_speed'][i][idx],
                telemetry['off_z'][i][idx],
                telemetry['off_alive'][i][idx],
                telemetry['off_ax'][i][idx],
                telemetry['off_n_pitch'][i][idx],
                telemetry['off_n_yaw'][i][idx],
                telemetry['off_n_total'][i][idx],
                telemetry['off_dist_hvt'][i][idx],
                telemetry['off_dist_nearest_def'][i][idx],
            ])
        rows.append(row)

    np.savetxt(out_csv, np.array(rows, dtype=np.float64), delimiter=',', header=','.join(header), comments='')


def run_episode_and_collect_outputs(
    env,
    policies,
    seed,
    gif_path,
    hidden_size=256,
    max_steps=8000,
    figsize=(10, 8),
    fps=12,
    speedup=2.0,
    frame_stride=1,
):
    env.seed(seed)
    obs, share_obs, _ = env.reset()

    rnn_states = [np.zeros((1, 1, hidden_size), dtype=np.float32) for _ in range(env.n_agents)]
    masks = [np.ones((1, 1), dtype=np.float32) for _ in range(env.n_agents)]

    n_off = env.n_offensive
    telemetry = {
        'step': [],
        'off_speed': [[] for _ in range(n_off)],
        'off_z': [[] for _ in range(n_off)],
        'off_alive': [[] for _ in range(n_off)],
        'off_ax': [[] for _ in range(n_off)],
        'off_n_pitch': [[] for _ in range(n_off)],
        'off_n_yaw': [[] for _ in range(n_off)],
        'off_n_total': [[] for _ in range(n_off)],
        'off_dist_hvt': [[] for _ in range(n_off)],
        'off_dist_nearest_def': [[] for _ in range(n_off)],
    }
    death_events = []
    prev_alive = [True for _ in range(n_off)]

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    ep_reward = 0.0
    gif_fps = max(1, int(round(fps * speedup)))
    writer = imageio.get_writer(gif_path, mode='I', fps=gif_fps)
    frame_count = 0

    # 初始状态与初始帧
    _record_telemetry(env, telemetry)
    render_frame(ax, env, step_num=0, show_fov=True)
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    writer.append_data(image)
    frame_count += 1

    done = False
    info = None

    for step in range(max_steps):
        actions = []
        for i in range(env.n_agents):
            with torch.no_grad():
                obs_t = np.array(obs[i]).reshape(1, -1)
                action, _, rnn_out = policies[i].actor(obs_t, rnn_states[i], masks[i], deterministic=True)
                actions.append(action.squeeze().numpy())
                rnn_states[i] = rnn_out.numpy()

        obs, share_obs, rewards, costs, dones, infos, _ = env.step(actions)
        ep_reward += sum(r[0] for r in rewards)
        info = infos[0]

        _record_telemetry(env, telemetry)

        for i, off in enumerate(env.offensives):
            now_alive = bool(off.alive)
            if prev_alive[i] and not now_alive:
                death_events.append({
                    'agent_id': i,
                    'step': int(env.current_step),
                    'time_s': float(env.current_step * env.config.get('dt', 0.01)),
                    'z_m': float(off.z),
                    'dist_hvt_m': float(off.distance_to(env.hvt.x, env.hvt.y, env.hvt.z)),
                })
            prev_alive[i] = now_alive

        if ((step + 1) % frame_stride) == 0 or dones[0]:
            render_frame(ax, env, step_num=step + 1, show_fov=True)
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            writer.append_data(image)
            frame_count += 1

        if dones[0]:
            done = True
            break

    writer.close()
    plt.close(fig)

    if info is None:
        info = {
            'success': False,
            'done_reason': 'unknown',
            'hit_count': 0,
            'offensive_alive': 0,
        }

    result = {
        'success': info.get('success', False),
        'done_reason': info.get('done_reason', 'unknown'),
        'hit_count': info.get('hit_count', 0),
        'offensive_alive': info.get('offensive_alive', 0),
        'steps': env.current_step,
        'reward': ep_reward,
        'frames': frame_count,
        'terminated': done,
        'kill_events': info.get('kill_events', []),
        'death_events': death_events,
        'telemetry': telemetry,
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str,
                        default=os.path.join(PROJECT_ROOT, 'outputs', 'results', 'fov_penetration', 'mappo', 'v38_inertial_accel', 'run3', 'models'))
    parser.add_argument('--out_dir', type=str,
                        default=os.path.join(PROJECT_ROOT, 'outputs', 'gifs', 'v38_run3_eval_fast'))
    parser.add_argument('--n_episodes', type=int, default=1)
    parser.add_argument('--seed_base', type=int, default=1000)
    parser.add_argument('--max_steps', type=int, default=8000)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--layer_N', type=int, default=3)
    parser.add_argument('--fps', type=int, default=12)
    parser.add_argument('--speedup', type=float, default=2.0, help='GIF播放加速倍率（2.0=2倍速）')
    parser.add_argument('--frame_stride', type=int, default=1, help='每隔多少仿真步渲染一帧')
    args = parser.parse_args()

    model_dir = args.model_dir
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    env = FOVPenetrationEnv(scenario='scenario_1')
    policies = load_policies(env, model_dir, hidden_size=args.hidden_size, layer_N=args.layer_N)
    dt = env.config.get('dt', 0.1)

    print(f'Loaded model from: {model_dir}')
    print(f'Output dir: {out_dir}')
    print(f'Algorithm: mappo')
    print(f'Max steps per episode: {args.max_steps}')
    print(f'GIF speedup: {args.speedup}x, fps={args.fps}, frame_stride={args.frame_stride}')

    all_results = []
    summary_lines = []

    for ep in range(args.n_episodes):
        seed = args.seed_base + ep
        gif_path = os.path.join(out_dir, f'episode_{ep:02d}.gif')
        result = run_episode_and_collect_outputs(
            env, policies, seed=seed, gif_path=gif_path,
            hidden_size=args.hidden_size,
            max_steps=args.max_steps,
            fps=args.fps,
            speedup=args.speedup,
            frame_stride=max(1, args.frame_stride),
        )

        telemetry_csv = os.path.join(out_dir, f'episode_{ep:02d}_telemetry.csv')
        telemetry_png = os.path.join(out_dir, f'episode_{ep:02d}_telemetry.png')
        events_json = os.path.join(out_dir, f'episode_{ep:02d}_events.json')
        save_telemetry_csv(result['telemetry'], dt, telemetry_csv)
        save_telemetry_plot(result['telemetry'], dt, telemetry_png)

        diag_payload = {
            'done_reason': result['done_reason'],
            'offensive_alive': result['offensive_alive'],
            'kill_events': result.get('kill_events', []),
            'death_events': result.get('death_events', []),
            'min_altitude_m': [float(np.min(result['telemetry']['off_z'][i])) for i in range(env.n_offensive)],
            'final_altitude_m': [float(result['telemetry']['off_z'][i][-1]) for i in range(env.n_offensive)],
            'final_dist_hvt_m': [float(result['telemetry']['off_dist_hvt'][i][-1]) for i in range(env.n_offensive)],
        }
        with open(events_json, 'w', encoding='utf-8') as f:
            json.dump(diag_payload, f, ensure_ascii=False, indent=2)

        result.pop('telemetry', None)
        result.pop('kill_events', None)
        result.pop('death_events', None)

        status = 'SUCCESS' if result['success'] else result['done_reason']
        line = (f"Ep{ep:02d}: {status}, steps={result['steps']}, "
                f"hits={result['hit_count']}, off_alive={result['offensive_alive']}, "
                f"reward={result['reward']:.1f}, frames={result['frames']}, gif={os.path.basename(gif_path)}, "
                f"plot={os.path.basename(telemetry_png)}, events={os.path.basename(events_json)}")
        print(line)
        summary_lines.append(line)
        all_results.append(result)

    successes = sum(1 for r in all_results if r['success'])
    avg_steps = float(np.mean([r['steps'] for r in all_results]))
    avg_reward = float(np.mean([r['reward'] for r in all_results]))

    summary_txt = os.path.join(out_dir, 'summary.txt')
    with open(summary_txt, 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary_lines))
        f.write('\n\n')
        f.write(f"SUMMARY: success_rate={successes}/{args.n_episodes}={successes*100.0/max(1,args.n_episodes):.1f}%, avg_steps={avg_steps:.1f}, avg_reward={avg_reward:.1f}\n")
        f.write(f"max_steps={args.max_steps}, speedup={args.speedup}x, fps={args.fps}, frame_stride={args.frame_stride}\n")

    print('\n=== SUMMARY ===')
    print(f"success_rate={successes}/{args.n_episodes}={successes*100.0/max(1,args.n_episodes):.1f}%")
    print(f"avg_steps={avg_steps:.1f}, avg_reward={avg_reward:.1f}")
    print(f"summary txt: {summary_txt}")


if __name__ == '__main__':
    main()

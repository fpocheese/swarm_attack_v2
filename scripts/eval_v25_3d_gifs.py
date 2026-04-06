#!/usr/bin/env python
"""生成V25模型的3个三维评估GIF。"""

import os
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio
import sys

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
        '--lr', '3e-4',
        '--critic_lr', '3e-4',
        '--use_recurrent_policy',
        '--use_feature_normalization',
        '--use_orthogonal',
        '--use_ReLU',
    ])[0]

    obs_space = env.observation_space[0]
    share_obs_space = env.share_observation_space[0]
    act_space = env.action_space[0]

    policies = []
    for i in range(env.n_agents):
        policy = R_MAPPOPolicy(args, obs_space, share_obs_space, act_space, device=torch.device('cpu'))
        actor_path = os.path.join(model_dir, f'actor_agent{i}.pt')
        if not os.path.exists(actor_path):
            raise FileNotFoundError(f'Missing model file: {actor_path}')
        actor_state = torch.load(actor_path, map_location='cpu')
        policy.actor.load_state_dict(actor_state)
        policy.actor.eval()
        policies.append(policy)
    return policies


def run_episode_and_collect_frames(env, policies, seed, hidden_size=256, frame_stride=5, max_steps=None):
    env.seed(seed)
    obs, share_obs, _ = env.reset()

    if max_steps is None:
        max_steps = env.max_steps

    rnn_states = [np.zeros((1, 1, hidden_size), dtype=np.float32) for _ in range(env.n_agents)]
    masks = [1.0 for _ in range(env.n_agents)]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    frames = []
    ep_reward = 0.0
    info = None

    # 初始帧
    render_frame(ax, env, step_num=0, show_fov=True)
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    frames.append(image)

    for step in range(max_steps):
        actions = []
        for i in range(env.n_agents):
            with torch.no_grad():
                obs_i = np.array(obs[i], dtype=np.float32).flatten()
                obs_t = torch.FloatTensor(obs_i).unsqueeze(0)

                rnn_t = torch.FloatTensor(rnn_states[i])
                if rnn_t.dim() == 2:
                    rnn_t = rnn_t.unsqueeze(0)

                mask_t = torch.FloatTensor([[masks[i]]])
                action, _, rnn_out = policies[i].actor(obs_t, rnn_t, mask_t, deterministic=True)
                actions.append(action.squeeze().numpy())
                rnn_states[i] = rnn_out.numpy().squeeze(0)

        obs, share_obs, rewards, costs, dones, infos, _ = env.step(actions)
        ep_reward += sum(r[0] for r in rewards)
        info = infos[0]

        if step % max(frame_stride, 1) == 0:
            render_frame(ax, env, step_num=step + 1, show_fov=True)
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frames.append(image)

        if dones[0]:
            break

    plt.close(fig)

    if info is None:
        info = {}

    result = {
        'success': info.get('success', False),
        'done_reason': info.get('done_reason', 'unknown'),
        'hit_count': info.get('hit_count', 0),
        'offensive_alive': info.get('offensive_alive', 0),
        'defensive_alive': info.get('defensive_alive', 0),
        'steps': env.current_step,
        'reward': float(ep_reward),
        'frames': len(frames),
    }
    return frames, result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', type=str, default='scenario_1')
    parser.add_argument('--num_episodes', type=int, default=3)
    parser.add_argument('--base_seed', type=int, default=20260)
    parser.add_argument('--fps', type=int, default=12)
    parser.add_argument('--frame_stride', type=int, default=5)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--layer_N', type=int, default=3)
    parser.add_argument('--max_steps', type=int, default=6000)
    parser.add_argument('--model_dir', type=str,
                        default='outputs/results/fov_penetration/mappo/v25_mappo_no_extrap_decoy/run1/models')
    parser.add_argument('--out_dir', type=str,
                        default='outputs/gifs/v25_eval_3d')
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = args.model_dir if os.path.isabs(args.model_dir) else os.path.join(project_root, args.model_dir)
    out_dir = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(project_root, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    env = FOVPenetrationEnv(scenario=args.scenario)
    policies = load_policies(env, model_dir, hidden_size=args.hidden_size, layer_N=args.layer_N)

    print(f'Loaded model: {model_dir}')
    print(f'Output dir: {out_dir}')

    summary_lines = []
    for ep in range(args.num_episodes):
        seed = args.base_seed + ep
        frames, result = run_episode_and_collect_frames(
            env, policies, seed=seed,
            hidden_size=args.hidden_size,
            frame_stride=args.frame_stride,
            max_steps=args.max_steps
        )

        gif_path = os.path.join(out_dir, f'v25_3d_episode_{ep+1:02d}.gif')
        imageio.mimsave(gif_path, frames, fps=args.fps)

        status = 'SUCCESS' if result['success'] else result['done_reason']
        line = (f"Ep{ep+1:02d} seed={seed} | {status} | steps={result['steps']} | "
                f"hits={result['hit_count']} | off_alive={result['offensive_alive']} | "
                f"def_alive={result['defensive_alive']} | reward={result['reward']:.1f} | "
                f"frames={result['frames']} | file={os.path.basename(gif_path)}")
        print(line)
        summary_lines.append(line)

    summary_path = os.path.join(out_dir, 'summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary_lines) + '\n')

    print(f'\nSaved summary: {summary_path}')


if __name__ == '__main__':
    main()

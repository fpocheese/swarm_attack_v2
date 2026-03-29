#!/usr/bin/env python
"""
将 run4 模型在 scenario_1 上评估 10 次，并保存为 GIF。
"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio

from macpo.config import get_config as get_macpo_config
from macpo.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy
from envs.fov_penetration import FOVPenetrationEnv
from envs.fov_penetration.render import render_frame


def load_policies(env, model_dir):
    parser = get_macpo_config()
    args = parser.parse_known_args([
        '--algorithm_name', 'macpo',
        '--hidden_size', '128',
        '--layer_N', '2',
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


def run_episode_and_collect_frames(env, policies, seed, max_steps=500, figsize=(10, 8)):
    env.seed(seed)
    obs, share_obs, _ = env.reset()

    rnn_states = [np.zeros((1, 1, 128), dtype=np.float32) for _ in range(env.n_agents)]
    masks = [np.ones((1, 1), dtype=np.float32) for _ in range(env.n_agents)]

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    frames = []
    ep_reward = 0.0

    # 初始帧
    render_frame(ax, env, step_num=0, show_fov=True)
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    frames.append(image)

    done = False
    info = None

    for step in range(max_steps):
        actions = []
        for i in range(env.n_agents):
            with torch.no_grad():
                obs_t = np.array(obs[i]).reshape(1, 1, -1)
                action, _, rnn_out = policies[i].actor(obs_t, rnn_states[i], masks[i], deterministic=True)
                actions.append(action.squeeze().numpy())
                rnn_states[i] = rnn_out.numpy()

        obs, share_obs, rewards, costs, dones, infos, _ = env.step(actions)
        ep_reward += sum(r[0] for r in rewards)
        info = infos[0]

        render_frame(ax, env, step_num=step + 1, show_fov=True)
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(image)

        if dones[0]:
            done = True
            break

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
        'frames': len(frames),
        'terminated': done,
    }
    return frames, result


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(project_root, 'outputs', 'results', 'fov_penetration', 'macpo', 'v3_scenario1_5M', 'run4', 'models')
    out_dir = os.path.join(project_root, 'outputs', 'gifs', 'run4_eval_10eps')
    os.makedirs(out_dir, exist_ok=True)

    env = FOVPenetrationEnv(scenario='scenario_1')
    policies = load_policies(env, model_dir)

    print(f'Loaded model from: {model_dir}')
    print(f'GIF output dir: {out_dir}')

    all_results = []
    summary_lines = []

    for ep in range(10):
        seed = 1000 + ep
        frames, result = run_episode_and_collect_frames(env, policies, seed=seed, max_steps=env.max_steps)

        gif_path = os.path.join(out_dir, f'episode_{ep:02d}.gif')
        imageio.mimsave(gif_path, frames, fps=12)

        status = 'SUCCESS' if result['success'] else result['done_reason']
        line = (f"Ep{ep:02d}: {status}, steps={result['steps']}, "
                f"hits={result['hit_count']}, off_alive={result['offensive_alive']}, "
                f"reward={result['reward']:.1f}, frames={result['frames']}, file={os.path.basename(gif_path)}")
        print(line)
        summary_lines.append(line)
        all_results.append(result)

    # 不再做大拼接 GIF（内存占用过高）
    montage_path = None

    successes = sum(1 for r in all_results if r['success'])
    avg_steps = float(np.mean([r['steps'] for r in all_results]))
    avg_reward = float(np.mean([r['reward'] for r in all_results]))

    summary_txt = os.path.join(out_dir, 'summary.txt')
    with open(summary_txt, 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary_lines))
        f.write('\n\n')
        f.write(f"SUMMARY: success_rate={successes}/10={successes*10}%, avg_steps={avg_steps:.1f}, avg_reward={avg_reward:.1f}\n")
        f.write("concat_gif=skipped_for_memory\n")

    print('\n=== SUMMARY ===')
    print(f"success_rate={successes}/10={successes*10}%")
    print(f"avg_steps={avg_steps:.1f}, avg_reward={avg_reward:.1f}")
    print("concat gif: skipped_for_memory")
    print(f"summary txt: {summary_txt}")


if __name__ == '__main__':
    main()

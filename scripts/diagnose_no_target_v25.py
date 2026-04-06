#!/usr/bin/env python
"""诊断为什么某些防守机后期没有打击目标。"""
import os
import sys
import argparse
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "MACPO", "MACPO"))

from macpo.config import get_config as get_macpo_config
from macpo.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy
from envs.fov_penetration import FOVPenetrationEnv


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
        actor_state = torch.load(actor_path, map_location='cpu')
        policy.actor.load_state_dict(actor_state)
        policy.actor.eval()
        policies.append(policy)
    return policies


def step_actions(policies, obs, rnn_states, masks):
    actions = []
    for i in range(len(policies)):
        obs_i = np.array(obs[i], dtype=np.float32).flatten()
        obs_t = torch.FloatTensor(obs_i).unsqueeze(0)

        rnn_t = torch.FloatTensor(rnn_states[i])
        if rnn_t.dim() == 2:
            rnn_t = rnn_t.unsqueeze(0)

        mask_t = torch.FloatTensor([[masks[i]]])
        with torch.no_grad():
            action, _, rnn_out = policies[i].actor(obs_t, rnn_t, mask_t, deterministic=True)

        actions.append(action.squeeze().numpy())
        rnn_states[i] = rnn_out.numpy().squeeze(0)
    return actions, rnn_states


def run_diagnosis(seed, model_dir, max_steps=6000):
    env = FOVPenetrationEnv(scenario='scenario_1')
    env.seed(seed)
    policies = load_policies(env, model_dir)

    obs, share_obs, _ = env.reset()
    rnn_states = [np.zeros((1, 1, 256), dtype=np.float32) for _ in range(env.n_agents)]
    masks = [1.0 for _ in range(env.n_agents)]

    records = []

    for step in range(max_steps):
        actions, rnn_states = step_actions(policies, obs, rnn_states, masks)
        obs, share_obs, rewards, costs, dones, infos, _ = env.step(actions)

        row = {'step': step + 1, 'off_alive': [], 'def_alive': []}
        for oi, off in enumerate(env.offensives):
            row['off_alive'].append((oi, off.alive, getattr(off, 'hit_hvt', False)))
        for di, d in enumerate(env.defensives):
            pol = env.defensive_policies[di]
            tgt = getattr(pol, 'target', None)
            tgt_idx = None
            if tgt is not None:
                for oi, off in enumerate(env.offensives):
                    if tgt is off:
                        tgt_idx = oi
                        break
            row['def_alive'].append({
                'di': di,
                'alive': d.alive,
                'lock_mode': int(getattr(pol, 'lock_mode', -1)),
                'target_idx': tgt_idx,
                'target_alive': (tgt.alive if tgt is not None else None),
                'target_hit_hvt': (getattr(tgt, 'hit_hvt', False) if tgt is not None else None),
            })
        records.append(row)

        if all(dones):
            break

    return records, env


def summarize(records, focus_defs):
    lines = []
    lines.append(f"total_steps={len(records)}")
    for d in focus_defs:
        lines.append(f"\n[D{d}] timeline checkpoints:")
        checkpoints = [1, 500, 1000, 2000, 3000, 4000, 5000, len(records)]
        seen = set()
        for cp in checkpoints:
            if cp < 1 or cp > len(records) or cp in seen:
                continue
            seen.add(cp)
            r = records[cp - 1]
            info = r['def_alive'][d]
            lines.append(
                f" step={cp:4d} alive={info['alive']} lock_mode={info['lock_mode']} "
                f"target={info['target_idx']} target_alive={info['target_alive']} target_hit_hvt={info['target_hit_hvt']}"
            )

        # 找首次 target=None 时刻
        first_none = None
        for r in records:
            info = r['def_alive'][d]
            if info['alive'] and info['target_idx'] is None:
                first_none = (r['step'], info)
                break
        if first_none is not None:
            lines.append(
                f" first_target_none_at_step={first_none[0]} (alive={first_none[1]['alive']}, lock_mode={first_none[1]['lock_mode']})"
            )
        else:
            lines.append(" first_target_none_at_step=None")
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--focus_defs', type=str, default='0,1,3')
    parser.add_argument('--max_steps', type=int, default=6000)
    parser.add_argument('--model_dir', type=str,
                        default='outputs/results/fov_penetration/mappo/v25_mappo_no_extrap_decoy/run1/models')
    parser.add_argument('--out', type=str, default=None)
    args = parser.parse_args()

    model_dir = args.model_dir if os.path.isabs(args.model_dir) else os.path.join(PROJECT_ROOT, args.model_dir)
    focus_defs = [int(x) for x in args.focus_defs.split(',') if x.strip()]

    records, env = run_diagnosis(args.seed, model_dir, max_steps=args.max_steps)
    report = summarize(records, focus_defs)

    out_path = args.out
    if out_path is None:
        out_path = os.path.join(PROJECT_ROOT, 'outputs', 'gifs', 'v25_eval_3d', f'diagnose_seed_{args.seed}.txt')
    else:
        out_path = out_path if os.path.isabs(out_path) else os.path.join(PROJECT_ROOT, out_path)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(report + '\n')

    print(report)
    print(f"saved: {out_path}")


if __name__ == '__main__':
    main()

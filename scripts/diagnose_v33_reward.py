#!/usr/bin/env python3
"""V33 reward diagnostic: run 1 episode, log per-step reward breakdown per agent.

Prints component-level breakdowns at key intervals, plus a full summary.
"""
import sys, os, json
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "third_party" / "MACPO" / "MACPO"))

import torch
from envs.fov_penetration import FOVPenetrationEnv
from envs.fov_penetration.reward_cost_v28 import compute_rewards, compute_terminal_rewards


def load_policy(model_dir, env, device, hidden_size=256, layer_N=3):
    from macpo.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy
    from macpo.config import get_config
    parser = get_config()
    all_args = parser.parse_known_args([])[0]
    all_args.algorithm_name = "mappo"
    all_args.hidden_size = hidden_size
    all_args.layer_N = layer_N
    all_args.use_recurrent_policy = True
    all_args.use_naive_recurrent_policy = False
    all_args.use_feature_normalization = False
    all_args.use_orthogonal = True
    all_args.use_ReLU = True
    all_args.gain = 0.01
    policies = []
    for aid in range(env.n_agents):
        po = R_MAPPOPolicy(all_args, env.observation_space[aid],
                           env.share_observation_space[aid],
                           env.action_space[aid], device=device)
        fpath = os.path.join(model_dir, f"actor_agent{aid}.pt")
        if os.path.exists(fpath):
            po.actor.load_state_dict(torch.load(fpath, map_location=device))
        po.actor.eval()
        policies.append(po)
    return policies


def get_actions(policies, obs, device, hidden_size=256):
    actions = []
    for aid, po in enumerate(policies):
        obs_t = torch.FloatTensor(obs[aid]).unsqueeze(0).to(device)
        rnn = torch.zeros(1, 1, hidden_size).to(device)
        masks = torch.ones(1, 1).to(device)
        with torch.no_grad():
            action, _, _ = po.actor(obs_t, rnn, masks, deterministic=True)
        actions.append(action.cpu().numpy().flatten())
    return actions


def run_diagnostic(model_dir, seed=42):
    device = torch.device("cpu")
    env = FOVPenetrationEnv(scenario="scenario_1")
    env.seed(seed)
    policies = load_policy(model_dir, env, device)

    obs, _, _ = env.reset()
    cfg = env.config
    hvt = env.hvt

    # accumulators per reward component
    comp_keys = [
        "reward_penetration", "reward_hit_geometry", "reward_no_retreat",
        "reward_decoy_value", "reward_decoy_potential", "reward_attention_redirect",
        "reward_escape", "reward_escape_progress",
        "penalty_cone", "penalty_fov", "penalty_danger",
        "penalty_boundary", "penalty_ground", "penalty_collision",
        "hit_hvt",
    ]
    comp_sums = {k: 0.0 for k in comp_keys}
    total_reward = 0.0
    n_steps = 0

    # per-agent tracking
    agent_dists = {i: [] for i in range(env.n_agents)}  # dist to HVT
    agent_alive_steps = {i: 0 for i in range(env.n_agents)}
    agent_closing_speeds = {i: [] for i in range(env.n_agents)}

    max_steps = env.max_steps
    print_interval = max_steps // 10  # print snapshot every 10%

    for step in range(max_steps):
        # record distances
        for i, off in enumerate(env.offensives):
            if off.alive:
                d = off.distance_to(hvt.x, hvt.y, hvt.z)
                agent_dists[i].append(d)
                agent_alive_steps[i] += 1
                # approx closing speed
                dx = hvt.x - off.x
                dy = hvt.y - off.y
                dz = hvt.z - off.z
                R = np.sqrt(dx**2 + dy**2 + dz**2)
                cg = np.cos(off.gamma)
                vx = off.v * cg * np.cos(off.heading)
                vy = off.v * cg * np.sin(off.heading)
                vz = off.v * np.sin(off.gamma)
                if R > 1e-6:
                    Vc = (dx*vx + dy*vy + dz*vz) / R
                    agent_closing_speeds[i].append(Vc)

        actions = get_actions(policies, obs, device)
        obs, _, rewards, costs, dones, infos, _ = env.step(actions)
        step_r = sum(r[0] if isinstance(r, (list, tuple, np.ndarray)) else r for r in rewards) / env.n_agents
        total_reward += step_r
        n_steps += 1

        # extract reward_info from env if available
        ri = getattr(env, '_last_reward_info', {}) if hasattr(env, '_last_reward_info') else {}
        for k in comp_keys:
            if k in ri:
                comp_sums[k] += ri[k]

        if (step + 1) % print_interval == 0 or any(dones):
            pct = (step + 1) / max_steps * 100
            alive_off = sum(1 for o in env.offensives if o.alive)
            alive_def = sum(1 for d in env.defensives if d.alive)
            min_d = min((off.distance_to(hvt.x, hvt.y, hvt.z) for off in env.offensives if off.alive), default=9999)
            print(f"Step {step+1:5d}/{max_steps} ({pct:5.1f}%) | "
                  f"reward={step_r:+.3f} cumR={total_reward:.1f} | "
                  f"alive: off={alive_off} def={alive_def} | "
                  f"min_d_hvt={min_d:.1f}m")

        if any(dones):
            break

    info = infos[0] if infos else {}
    print("\n" + "=" * 72)
    print(f"EPISODE COMPLETE: {n_steps} steps, reason={info.get('done_reason', 'unknown')}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Hits: {sum(1 for o in env.offensives if o.hit_hvt)}")
    print(f"Alive offensives: {sum(1 for o in env.offensives if o.alive)}")
    print(f"Alive defensives: {sum(1 for d in env.defensives if d.alive)}")

    print("\n--- Per-Agent Distance Profile ---")
    for i in range(env.n_agents):
        if agent_dists[i]:
            darr = np.array(agent_dists[i])
            cs = np.array(agent_closing_speeds[i]) if agent_closing_speeds[i] else np.array([0.0])
            print(f"  Agent {i}: alive_steps={agent_alive_steps[i]}, "
                  f"init_d={darr[0]:.0f}m, min_d={darr.min():.0f}m, "
                  f"final_d={darr[-1]:.0f}m, "
                  f"avg_closing={cs.mean():.1f} m/s, "
                  f"hit={env.offensives[i].hit_hvt}")
        else:
            print(f"  Agent {i}: never alive")

    print("\n--- Reward Component Breakdown ---")
    if any(v != 0 for v in comp_sums.values()):
        for k in comp_keys:
            if comp_sums[k] != 0:
                print(f"  {k:35s}: {comp_sums[k]:+10.2f}")
    else:
        print("  (reward_info not captured; using env-level metrics instead)")
        print(f"  Total episode reward:    {total_reward:.2f}")

    # Also compute what the terminal reward would be
    try:
        ap_data = getattr(env, '_last_ap_data', {}) if hasattr(env, '_last_ap_data') else {}
        term_r, term_info = compute_terminal_rewards(env.offensives, hvt, cfg, ap_data)
        print("\n--- Terminal Reward Diagnostics ---")
        for k, v in term_info.items():
            print(f"  {k:30s}: {v}")
    except Exception as e:
        print(f"  Could not compute terminal: {e}")

    env.close()
    return total_reward, n_steps


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_diagnostic(args.model_dir, args.seed)

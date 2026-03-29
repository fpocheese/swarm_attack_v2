#!/usr/bin/env python
"""FOV Penetration MACPO 评估脚本 V2"""
import sys, os, argparse
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "MACPO", "MACPO"))

from envs.fov_penetration import FOVPenetrationEnv


def load_policies(model_dir, env, device, hidden_size=64, layer_N=1):
    from macpo.algorithms.r_mappo.algorithm.MACPPOPolicy import MACPPOPolicy
    from macpo.config import get_config

    parser = get_config()
    all_args = parser.parse_known_args([])[0]
    all_args.algorithm_name = "macpo"
    all_args.hidden_size = hidden_size
    all_args.layer_N = layer_N

    policies = []
    for agent_id in range(env.n_agents):
        po = MACPPOPolicy(all_args,
                          env.observation_space[agent_id],
                          env.share_observation_space[agent_id],
                          env.action_space[agent_id],
                          device=device)

        actor_path = os.path.join(model_dir, f"actor_agent{agent_id}.pt")
        if os.path.exists(actor_path):
            state_dict = torch.load(actor_path, map_location=device)
            po.actor.load_state_dict(state_dict)
            print(f"Loaded actor for agent {agent_id}")
        else:
            print(f"Warning: {actor_path} not found, random policy")

        po.actor.eval()
        policies.append(po)
    return policies


def get_actions(policies, obs, device, hidden_size=64):
    actions = []
    for agent_id, po in enumerate(policies):
        obs_t = torch.FloatTensor(obs[agent_id]).unsqueeze(0).to(device)
        rnn = torch.zeros(1, 1, hidden_size).to(device)
        masks = torch.ones(1, 1).to(device)
        with torch.no_grad():
            action, _, _ = po.actor(obs_t, rnn, masks, deterministic=True)
        actions.append(action.cpu().numpy().flatten())
    return actions


def evaluate(model_dir=None, n_episodes=20, save_gif=False,
             hidden_size=64, layer_N=1):
    device = torch.device("cpu")
    env = FOVPenetrationEnv()
    env.seed(42)

    if model_dir and os.path.exists(model_dir):
        policies = load_policies(model_dir, env, device, hidden_size, layer_N)
        policy_name = "trained"
    else:
        policies = None
        policy_name = "random"
        print("No model dir, using random policy")

    results = {"success": [], "attacker_killed": [], "timeout": [],
               "escorts_alive": [], "intc_alive": [],
               "episode_reward": [], "episode_cost": [],
               "episode_length": [], "escort_kills": [],
               # V22: 新增点目标/锁定统计
               "hit_count": [], "first_hit_time": [],
               "terminal_miss_dist_min": [],
               "n_locked_defenders": [],
               "n_escapes_total": [],
               "N_eff": [],
               }

    for ep in range(n_episodes):
        obs, _, _ = env.reset()
        ep_reward = 0.0
        ep_cost = 0.0

        for step in range(env.max_steps):
            if policies:
                actions = get_actions(policies, obs, device, hidden_size)
            else:
                actions = [env.action_space[i].sample() for i in range(env.n_agents)]

            obs, _, rewards, costs, dones, infos, _ = env.step(actions)
            ep_reward += sum(r[0] for r in rewards) / env.n_agents
            ep_cost += sum(c[0] for c in costs) / env.n_agents

            if any(dones):
                break

        info = infos[0]
        results["success"].append(info.get("success", False))
        results["attacker_killed"].append(info.get("attacker_killed", False))
        results["timeout"].append(info.get("done_reason") == "timeout")
        results["escorts_alive"].append(info.get("escorts_alive_count", 0))
        results["intc_alive"].append(info.get("interceptors_alive_count", 0))
        results["episode_reward"].append(ep_reward)
        results["episode_cost"].append(ep_cost)
        results["episode_length"].append(step + 1)
        results["escort_kills"].append(len(info.get("escort_kill_events", [])))
        # V22 metrics
        results["hit_count"].append(info.get("hit_count", 0))
        results["first_hit_time"].append(info.get("first_hit_time", -1))
        results["terminal_miss_dist_min"].append(
            info.get("terminal_miss_distance_min", float('inf')))
        results["n_locked_defenders"].append(
            info.get("n_locked_defenders", 0))
        results["n_escapes_total"].append(
            info.get("n_escapes_total", 0))
        results["N_eff"].append(info.get("N_eff", 0.0))

        print(f"Ep {ep+1}/{n_episodes}: reward={ep_reward:.1f}, cost={ep_cost:.1f}, "
              f"steps={step+1}, reason={info.get('done_reason')}, "
              f"hits={info.get('hit_count', 0)}, "
              f"miss_dist_min={info.get('terminal_miss_distance_min', float('inf')):.1f}m")

    print("\n" + "=" * 60)
    print(f"Results ({policy_name}, {n_episodes} eps)")
    print("=" * 60)
    print(f"Success rate:       {np.mean(results['success']):.1%}")
    print(f"Attacker killed:    {np.mean(results['attacker_killed']):.1%}")
    print(f"Timeout:            {np.mean(results['timeout']):.1%}")
    print(f"Avg reward:         {np.mean(results['episode_reward']):.2f}")
    print(f"Avg cost:           {np.mean(results['episode_cost']):.2f}")
    print(f"Avg length:         {np.mean(results['episode_length']):.1f}")
    print(f"Avg escorts alive:  {np.mean(results['escorts_alive']):.2f}")
    print(f"Avg intc alive:     {np.mean(results['intc_alive']):.2f}")
    print(f"Avg escort kills:   {np.mean(results['escort_kills']):.2f}")
    # V22 stats
    print(f"Avg hit count:      {np.mean(results['hit_count']):.2f}")
    valid_hits = [t for t in results['first_hit_time'] if t >= 0]
    print(f"Avg first hit time: {np.mean(valid_hits):.1f}" if valid_hits else "First hit time: N/A")
    valid_miss = [d for d in results['terminal_miss_dist_min'] if d < float('inf')]
    print(f"Avg min miss dist:  {np.mean(valid_miss):.1f}m" if valid_miss else "Min miss dist: N/A")

    if save_gif:
        from envs.fov_penetration.render import render_episode
        os.makedirs(os.path.join(PROJECT_ROOT, "outputs", "gifs"), exist_ok=True)
        save_path = os.path.join(PROJECT_ROOT, "outputs", "gifs", f"eval_{policy_name}_v2.gif")

        def policy_fn(obs_list):
            if policies:
                return get_actions(policies, obs_list, device, hidden_size)
            return [env.action_space[i].sample() for i in range(env.n_agents)]

        env.seed(42)
        render_episode(env, policy_fn=policy_fn, save_path=save_path, fps=10)

    env.close()
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--n_episodes", type=int, default=20)
    parser.add_argument("--save_gif", action="store_true", default=False)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--layer_N", type=int, default=2)
    args = parser.parse_args()

    evaluate(model_dir=args.model_dir,
             n_episodes=args.n_episodes,
             save_gif=args.save_gif,
             hidden_size=args.hidden_size,
             layer_N=args.layer_N)

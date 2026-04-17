#!/usr/bin/env python3
"""Latest-model evaluation for FOV penetration.

Loads the newest checkpoint if no model directory is provided, runs 10
deterministic evaluation episodes, records metrics, saves PNG summaries,
and optionally exports a GIF for one rollout.
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "third_party" / "MACPO" / "MACPO"))

from envs.fov_penetration import FOVPenetrationEnv
from envs.fov_penetration.render import render_frame


DEFAULT_MODEL_DIR = PROJECT_ROOT / "outputs" / "debug_snapshot_20260324_155939" / "checkpoints"


def find_latest_model_dir() -> Optional[str]:
    """Find the newest checkpoint folder that contains actor_agent0.pt."""
    candidates = []
    for actor_file in (PROJECT_ROOT / "outputs").rglob("actor_agent0.pt"):
        candidates.append(actor_file.parent)
    if not candidates:
        if DEFAULT_MODEL_DIR.exists():
            return str(DEFAULT_MODEL_DIR)
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(candidates[0])


def load_policies(model_dir, env, device, hidden_size=64, layer_N=1):
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
    for agent_id in range(env.n_agents):
        po = R_MAPPOPolicy(
            all_args,
            env.observation_space[agent_id],
            env.share_observation_space[agent_id],
            env.action_space[agent_id],
            device=device,
        )

        actor_path = os.path.join(model_dir, f"actor_agent{agent_id}.pt")
        if os.path.exists(actor_path):
            state_dict = torch.load(actor_path, map_location=device)
            po.actor.load_state_dict(state_dict)
            print(f"Loaded actor for agent {agent_id}: {actor_path}")
        else:
            print(f"Warning: missing {actor_path}; using random policy for this agent")

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


def _metric_dict():
    return {
        "success": [],
        "attacker_killed": [],
        "timeout": [],
        "escorts_alive": [],
        "intc_alive": [],
        "episode_reward": [],
        "episode_cost": [],
        "episode_length": [],
        "escort_kills": [],
        "hit_count": [],
        "first_hit_time": [],
        "terminal_miss_dist_min": [],
        "n_locked_defenders": [],
        "n_escapes_total": [],
        "N_eff": [],
        "min_distance_to_hvt": [],
        "final_distance_to_hvt": [],
    }


def save_plots(results, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    ep = np.arange(1, len(results["episode_reward"]) + 1)

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Latest Model Evaluation Summary (10 Episodes)", fontsize=16)

    axs[0, 0].plot(ep, results["episode_reward"], marker="o", linewidth=2)
    axs[0, 0].set_title("Episode Reward")
    axs[0, 0].set_xlabel("Episode")
    axs[0, 0].set_ylabel("Reward")
    axs[0, 0].grid(True, alpha=0.3)

    axs[0, 1].plot(ep, results["episode_length"], marker="o", color="tab:orange", linewidth=2)
    axs[0, 1].set_title("Episode Length")
    axs[0, 1].set_xlabel("Episode")
    axs[0, 1].set_ylabel("Steps")
    axs[0, 1].grid(True, alpha=0.3)

    axs[1, 0].plot(ep, results["hit_count"], marker="o", color="tab:red", linewidth=2)
    axs[1, 0].set_title("Hit Count")
    axs[1, 0].set_xlabel("Episode")
    axs[1, 0].set_ylabel("Hits")
    axs[1, 0].grid(True, alpha=0.3)

    axs[1, 1].plot(ep, results["min_distance_to_hvt"], marker="o", color="tab:green", linewidth=2, label="Min Dist")
    axs[1, 1].plot(ep, results["final_distance_to_hvt"], marker="s", color="tab:purple", linewidth=2, label="Final Dist")
    axs[1, 1].set_title("Distance to HVT")
    axs[1, 1].set_xlabel("Episode")
    axs[1, 1].set_ylabel("Meters")
    axs[1, 1].legend()
    axs[1, 1].grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(out_dir / "episode_metrics.png", dpi=180)
    plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    axs[0].boxplot(results["episode_reward"], vert=True)
    axs[0].set_title("Reward Distribution")
    axs[0].set_xticks([1])
    axs[0].set_xticklabels(["reward"])

    axs[1].bar(["success", "timeout", "fail"], [
        sum(results["success"]),
        sum(results["timeout"]),
        len(results["success"]) - sum(results["success"]) - sum(results["timeout"]),
    ], color=["tab:green", "tab:orange", "tab:red"])
    axs[1].set_title("Outcome Counts")

    axs[2].plot(ep, np.cumsum(results["episode_reward"]), color="tab:blue", linewidth=2)
    axs[2].set_title("Cumulative Reward")
    axs[2].set_xlabel("Episode")
    axs[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "reward_distribution.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    success_rate = np.mean(results["success"]) if results["success"] else 0.0
    timeout_rate = np.mean(results["timeout"]) if results["timeout"] else 0.0
    ax.bar(["success_rate", "timeout_rate"], [success_rate, timeout_rate], color=["tab:green", "tab:orange"])
    ax.set_ylim(0, 1)
    ax.set_title("Episode-Level Rates")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "summary_rates.png", dpi=180)
    plt.close(fig)


def render_policy_gif(env, policies, out_path: Path, device, hidden_size=64, fps=10, stride=20, max_frames=500):
    try:
        import imageio.v2 as imageio
        use_imageio = True
    except Exception:
        from PIL import Image
        use_imageio = False

    obs, _, _ = env.reset()
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    frames = []
    max_steps = getattr(env, "max_steps", env.config["max_steps"])

    for step_i in range(max_steps):
        actions = get_actions(policies, obs, device, hidden_size)
        obs, _, _, _, dones, _, _ = env.step(actions)

        if step_i % stride == 0 or any(dones):
            render_frame(ax, env, step_num=step_i + 1)
            fig.canvas.draw()
            frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
            frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frames.append(frame)
            if len(frames) >= max_frames:
                break

        if any(dones):
            break

    if frames:
        if use_imageio:
            imageio.mimsave(out_path, frames, fps=fps)
        else:
            pil_frames = [Image.fromarray(frame) for frame in frames]
            pil_frames[0].save(
                out_path,
                save_all=True,
                append_images=pil_frames[1:],
                duration=int(1000 / max(fps, 1)),
                loop=0,
            )
        print(f"Saved GIF: {out_path}")
    plt.close(fig)


def evaluate(model_dir=None, n_episodes=10, save_gif=True, hidden_size=128, layer_N=2):
    device = torch.device("cpu")
    env = FOVPenetrationEnv(scenario="scenario_1")
    env.seed(42)

    if model_dir is None:
        model_dir = find_latest_model_dir()
    if model_dir and os.path.exists(model_dir):
        policies = load_policies(model_dir, env, device, hidden_size, layer_N)
        policy_name = Path(model_dir).parent.name if Path(model_dir).name == "checkpoints" else Path(model_dir).name
    else:
        policies = None
        policy_name = "random"
        print("No valid model dir found; using random policy")

    results = _metric_dict()
    results["n_episodes"] = n_episodes
    results["model_dir"] = str(model_dir) if model_dir else None
    results["policy_name"] = policy_name

    for ep in range(n_episodes):
        obs, _, _ = env.reset()
        ep_reward = 0.0
        ep_cost = 0.0
        max_dist = 0.0
        min_dist = float("inf")
        done = False

        for step in range(getattr(env, "max_steps", env.config["max_steps"])):
            if policies:
                actions = get_actions(policies, obs, device, hidden_size)
            else:
                actions = [env.action_space[i].sample() for i in range(env.n_agents)]

            obs, _, rewards, costs, dones, infos, _ = env.step(actions)
            ep_reward += sum(r[0] if isinstance(r, (list, tuple, np.ndarray)) else r for r in rewards) / env.n_agents
            ep_cost += sum(c[0] if isinstance(c, (list, tuple, np.ndarray)) else c for c in costs) / env.n_agents

            for off in env.offensives:
                if off.alive:
                    d = off.distance_to(env.hvt.x, env.hvt.y, env.hvt.z)
                    max_dist = max(max_dist, d)
                    min_dist = min(min_dist, d)

            if any(dones):
                done = True
                break

        info = infos[0] if infos else {}
        hits = sum(1 for off in env.offensives if off.hit_hvt)

        results["success"].append(bool(info.get("success", False)))
        results["attacker_killed"].append(bool(info.get("attacker_killed", False)))
        results["timeout"].append(bool(info.get("done_reason") == "timeout" or (not done and step + 1 >= getattr(env, "max_steps", env.config["max_steps"]))))
        results["escorts_alive"].append(info.get("escorts_alive_count", 0))
        results["intc_alive"].append(info.get("interceptors_alive_count", 0))
        results["episode_reward"].append(float(ep_reward))
        results["episode_cost"].append(float(ep_cost))
        results["episode_length"].append(int(step + 1))
        results["escort_kills"].append(len(info.get("escort_kill_events", [])))
        results["hit_count"].append(int(info.get("hit_count", hits)))
        results["first_hit_time"].append(int(info.get("first_hit_time", -1)))
        results["terminal_miss_dist_min"].append(float(info.get("terminal_miss_distance_min", float("inf"))))
        results["n_locked_defenders"].append(int(info.get("n_locked_defenders", 0)))
        results["n_escapes_total"].append(int(info.get("n_escapes_total", 0)))
        results["N_eff"].append(float(info.get("N_eff", 0.0)))
        results["min_distance_to_hvt"].append(float(min_dist if np.isfinite(min_dist) else 9999.0))
        alive_dists = [off.distance_to(env.hvt.x, env.hvt.y, env.hvt.z) for off in env.offensives if off.alive]
        results["final_distance_to_hvt"].append(float(min(alive_dists) if alive_dists else 9999.0))

        print(
            f"Ep {ep+1:02d}/{n_episodes}: "
            f"reward={ep_reward:.1f}, cost={ep_cost:.1f}, steps={step+1}, "
            f"hits={hits}, success={results['success'][-1]}, reason={info.get('done_reason')}, "
            f"min_dist={results['min_distance_to_hvt'][-1]:.1f}m"
        )

    results["hit_success_rate"] = results["hit_count"].count(1) / max(n_episodes, 1)
    results["episode_success_rate"] = float(np.mean(results["success"])) if results["success"] else 0.0
    results["timeout_rate"] = float(np.mean(results["timeout"])) if results["timeout"] else 0.0
    results["avg_episode_reward"] = float(np.mean(results["episode_reward"])) if results["episode_reward"] else 0.0
    results["avg_episode_cost"] = float(np.mean(results["episode_cost"])) if results["episode_cost"] else 0.0
    results["avg_episode_length"] = float(np.mean(results["episode_length"])) if results["episode_length"] else 0.0
    results["avg_min_distance"] = float(np.mean(results["min_distance_to_hvt"])) if results["min_distance_to_hvt"] else 0.0
    results["avg_final_distance"] = float(np.mean(results["final_distance_to_hvt"])) if results["final_distance_to_hvt"] else 0.0

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = PROJECT_ROOT / "outputs" / "results" / "latest_model_eval" / f"manual_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "eval_10eps.json"
    csv_path = out_dir / "eval_10eps.csv"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward", "cost", "length", "hit_count", "min_dist", "final_dist", "success", "timeout"])
        for i in range(n_episodes):
            writer.writerow([
                i + 1,
                results["episode_reward"][i],
                results["episode_cost"][i],
                results["episode_length"][i],
                results["hit_count"][i],
                results["min_distance_to_hvt"][i],
                results["final_distance_to_hvt"][i],
                int(results["success"][i]),
                int(results["timeout"][i]),
            ])

    save_plots(results, out_dir)

    if save_gif and policies is not None:
        gif_dir = PROJECT_ROOT / "outputs" / "gifs"
        gif_dir.mkdir(parents=True, exist_ok=True)
        gif_path = gif_dir / f"eval_latest_{ts}.gif"
        env.seed(42)
        render_policy_gif(env, policies, gif_path, device, hidden_size=hidden_size)

    print("\n" + "=" * 64)
    print("EVALUATION SUMMARY (10 EPISODES)")
    print("=" * 64)
    print(f"Model dir:           {model_dir}")
    print(f"Success rate:         {results['episode_success_rate']*100:.1f}%")
    print(f"Timeout rate:         {results['timeout_rate']*100:.1f}%")
    print(f"Hit success rate:     {results['hit_success_rate']*100:.1f}%")
    print(f"Avg episode reward:    {results['avg_episode_reward']:.2f}")
    print(f"Avg episode cost:      {results['avg_episode_cost']:.2f}")
    print(f"Avg episode length:    {results['avg_episode_length']:.1f} steps")
    print(f"Avg min distance HVT:  {results['avg_min_distance']:.1f} m")
    print(f"Avg final distance:    {results['avg_final_distance']:.1f} m")
    print(f"Saved JSON:            {json_path}")
    print(f"Saved CSV:             {csv_path}")
    print(f"Saved plots in:        {out_dir}")
    print("=" * 64)

    env.close()
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default=None, help="Checkpoint directory containing actor_agent*.pt")
    parser.add_argument("--n_episodes", type=int, default=10)
    parser.add_argument("--save_gif", action="store_true", default=True)
    parser.add_argument("--no_gif", action="store_true", default=False)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--layer_N", type=int, default=3)
    args = parser.parse_args()

    model_dir = args.model_dir or find_latest_model_dir()
    save_gif = False if args.no_gif else args.save_gif
    evaluate(
        model_dir=model_dir,
        n_episodes=args.n_episodes,
        save_gif=save_gif,
        hidden_size=args.hidden_size,
        layer_N=args.layer_N,
    )

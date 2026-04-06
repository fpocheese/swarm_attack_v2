#!/usr/bin/env python
"""
自动评估最新模型：持续测试直到出现成功episode，然后输出数据分析图和3D动图。
"""

import os
import sys
import json
import argparse
from datetime import datetime

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "MACPO", "MACPO"))

from envs.fov_penetration import FOVPenetrationEnv
from envs.fov_penetration.render import render_frame
from macpo.algorithms.r_mappo.algorithm.MACPPOPolicy import MACPPOPolicy
from macpo.config import get_config as macpo_get_config


def find_latest_model_dir(root_dir):
    candidates = []
    for exp_name in os.listdir(root_dir):
        exp_dir = os.path.join(root_dir, exp_name)
        if not os.path.isdir(exp_dir):
            continue
        for run_name in os.listdir(exp_dir):
            run_dir = os.path.join(exp_dir, run_name)
            model_dir = os.path.join(run_dir, "models")
            actor0 = os.path.join(model_dir, "actor_agent0.pt")
            if os.path.exists(actor0):
                mtime = os.path.getmtime(actor0)
                candidates.append((mtime, exp_name, run_name, model_dir))

    if not candidates:
        raise FileNotFoundError(f"No model found under {root_dir}")

    candidates.sort(key=lambda x: x[0], reverse=True)
    _, exp_name, run_name, model_dir = candidates[0]
    return exp_name, run_name, model_dir


def load_policies(env, model_dir, hidden_size=256, layer_N=3, device=torch.device("cpu")):
    parser = macpo_get_config()
    all_args = parser.parse_known_args([])[0]
    all_args.algorithm_name = "macpo"
    all_args.hidden_size = hidden_size
    all_args.layer_N = layer_N
    all_args.use_recurrent_policy = True
    all_args.use_feature_normalization = False

    policies = []
    for agent_id in range(env.n_agents):
        po = MACPPOPolicy(all_args,
                          env.observation_space[agent_id],
                          env.share_observation_space[agent_id],
                          env.action_space[agent_id],
                          device=device)
        actor_path = os.path.join(model_dir, f"actor_agent{agent_id}.pt")
        if not os.path.exists(actor_path):
            raise FileNotFoundError(f"Missing model file: {actor_path}")
        state_dict = torch.load(actor_path, map_location=device)
        po.actor.load_state_dict(state_dict)
        po.actor.eval()
        policies.append(po)
    return policies


def get_actions(policies, obs, device, hidden_size):
    actions = []
    for agent_id, po in enumerate(policies):
        obs_t = torch.FloatTensor(obs[agent_id]).unsqueeze(0).to(device)
        rnn = torch.zeros(1, 1, hidden_size).to(device)
        masks = torch.ones(1, 1).to(device)
        with torch.no_grad():
            action, _, _ = po.actor(obs_t, rnn, masks, deterministic=True)
        actions.append(action.cpu().numpy().flatten())
    return actions


def run_one_episode(env, policies, seed, hidden_size, device, capture_frames=False):
    env.seed(seed)
    obs, _, _ = env.reset()

    n_agents = env.n_agents
    ep_reward = 0.0
    ep_cost = 0.0

    step_rewards = []
    step_costs = []
    dists = [[] for _ in range(n_agents)]
    z_pos = [[] for _ in range(n_agents)]
    speeds = [[] for _ in range(n_agents)]
    headings = [[] for _ in range(n_agents)]
    gammas = [[] for _ in range(n_agents)]
    accels = [[] for _ in range(n_agents)]
    xy_traj = [[] for _ in range(n_agents)]

    dt = float(getattr(env, 'dt', 0.1))

    frames = []
    fig = None
    ax = None
    if capture_frames:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

    done_reason = "unknown"
    info = {}

    for step in range(env.max_steps):
        for i, off in enumerate(env.offensives):
            dist = off.distance_to(env.hvt.x, env.hvt.y, env.hvt.z) if off.alive else np.nan
            dists[i].append(float(dist))
            z_pos[i].append(float(off.z) if off.alive else np.nan)
            speeds[i].append(float(off.v) if off.alive else np.nan)
            headings[i].append(float(np.degrees(off.heading)) if off.alive else np.nan)
            gammas[i].append(float(np.degrees(off.gamma)) if off.alive else np.nan)
            xy_traj[i].append((float(off.x), float(off.y)))

            if off.alive and len(speeds[i]) >= 2 and np.isfinite(speeds[i][-1]) and np.isfinite(speeds[i][-2]):
                accels[i].append((speeds[i][-1] - speeds[i][-2]) / dt)
            else:
                accels[i].append(np.nan)

        actions = get_actions(policies, obs, device, hidden_size)
        obs, _, rewards, costs, dones, infos, _ = env.step(actions)

        r_vals = [rewards[i][0] if isinstance(rewards[i], (list, np.ndarray)) else rewards[i] for i in range(n_agents)]
        c_vals = [costs[i][0] if isinstance(costs[i], (list, np.ndarray)) else costs[i] for i in range(n_agents)]

        mean_r = float(np.mean(r_vals))
        mean_c = float(np.mean(c_vals))
        step_rewards.append(mean_r)
        step_costs.append(mean_c)
        ep_reward += mean_r
        ep_cost += mean_c

        if capture_frames:
            render_frame(ax, env, step_num=step + 1, show_fov=True)
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frames.append(image)

        if any(dones):
            info = infos[0] if infos else {}
            done_reason = info.get("done_reason", "unknown")
            break

    if fig is not None:
        plt.close(fig)

    success = bool(info.get("success", False))
    hit_count = int(info.get("hit_count", 0))

    return {
        "seed": int(seed),
        "success": success,
        "done_reason": done_reason,
        "hit_count": hit_count,
        "episode_reward": float(ep_reward),
        "episode_cost": float(ep_cost),
        "episode_length": int(len(step_rewards)),
        "step_rewards": step_rewards,
        "step_costs": step_costs,
        "dists": dists,
        "z_pos": z_pos,
        "speeds": speeds,
        "headings": headings,
        "gammas": gammas,
        "accels": accels,
        "xy_traj": xy_traj,
        "frames": frames,
    }


def plot_attempt_trend(attempts, out_path):
    x = np.arange(1, len(attempts) + 1)
    rewards = np.array([a["episode_reward"] for a in attempts], dtype=np.float32)
    costs = np.array([a["episode_cost"] for a in attempts], dtype=np.float32)
    lengths = np.array([a["episode_length"] for a in attempts], dtype=np.float32)
    succ = np.array([1.0 if a["success"] else 0.0 for a in attempts], dtype=np.float32)
    cum_succ_rate = np.cumsum(succ) / x

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    axes[0, 0].plot(x, rewards, color='tab:blue')
    axes[0, 0].set_title('Episode Reward by Attempt')
    axes[0, 0].set_xlabel('Attempt')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(x, costs, color='tab:red')
    axes[0, 1].set_title('Episode Cost by Attempt')
    axes[0, 1].set_xlabel('Attempt')
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(x, lengths, color='tab:green')
    axes[1, 0].set_title('Episode Length by Attempt')
    axes[1, 0].set_xlabel('Attempt')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(x, cum_succ_rate, color='tab:purple', label='Cumulative success rate')
    axes[1, 1].scatter(x[succ > 0], cum_succ_rate[succ > 0], color='gold', s=40, label='Successful attempt')
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_title('Cumulative Success Rate')
    axes[1, 1].set_xlabel('Attempt')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_success_diagnostics(success_ep, out_path):
    n_agents = len(success_ep["dists"])
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']

    fig, axes = plt.subplots(4, 2, figsize=(16, 17))
    fig.suptitle(
        f"Successful Episode Diagnostics | seed={success_ep['seed']} | "
        f"len={success_ep['episode_length']} | reward={success_ep['episode_reward']:.1f} | "
        f"cost={success_ep['episode_cost']:.1f}",
        fontsize=12
    )

    ax = axes[0, 0]
    for i in range(n_agents):
        ax.plot(success_ep["dists"][i], color=colors[i % len(colors)], label=f"Agent{i}")
    ax.set_title('Distance to HVT')
    ax.set_ylabel('m')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    for i in range(n_agents):
        ax.plot(success_ep["z_pos"][i], color=colors[i % len(colors)], label=f"Agent{i}")
    ax.set_title('Altitude')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.plot(np.cumsum(success_ep["step_rewards"]), color='tab:blue', label='Cum reward')
    ax.plot(np.cumsum(success_ep["step_costs"]), color='tab:red', label='Cum cost')
    ax.set_title('Cumulative Reward vs Cost')
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 1]
    for i in range(n_agents):
        ax.plot(success_ep["speeds"][i], color=colors[i % len(colors)], label=f"Agent{i}")
    ax.set_title('Speed')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[2, 0]
    for i in range(n_agents):
        ax.plot(success_ep["headings"][i], color=colors[i % len(colors)], label=f"Agent{i}")
    ax.set_title('Heading (deg)')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[2, 1]
    for i in range(n_agents):
        ax.plot(success_ep["gammas"][i], color=colors[i % len(colors)], label=f"Agent{i}")
    ax.set_title('Flight-path angle gamma (deg)')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[3, 0]
    for i in range(n_agents):
        ax.plot(success_ep["accels"][i], color=colors[i % len(colors)], label=f"Agent{i}")
    ax.set_title('Approx acceleration dv/dt (m/s^2)')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[3, 1]
    for i in range(n_agents):
        traj = success_ep["xy_traj"][i]
        xs = [p[0] for p in traj]
        ys = [p[1] for p in traj]
        ax.plot(xs, ys, color=colors[i % len(colors)], label=f"Agent{i}")
        if xs and ys:
            ax.scatter(xs[0], ys[0], color=colors[i % len(colors)], marker='o', s=35)
            ax.scatter(xs[-1], ys[-1], color=colors[i % len(colors)], marker='x', s=45)
    ax.set_title('XY Trajectory')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close(fig)


def save_gif(frames, out_path, fps=12):
    if not frames:
        raise RuntimeError("No frames collected for GIF")
    imageio.mimsave(out_path, frames, fps=fps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default=None, help="模型目录，默认自动找最新")
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--layer_N", type=int, default=3)
    parser.add_argument("--start_seed", type=int, default=42)
    parser.add_argument("--max_attempts", type=int, default=0, help="0表示不限，直到成功")
    parser.add_argument("--gif_fps", type=int, default=12)
    args = parser.parse_args()

    model_root = os.path.join(PROJECT_ROOT, "outputs", "results", "fov_penetration", "macpo")

    if args.model_dir:
        model_dir = args.model_dir
        exp_name = "manual"
        run_name = "manual"
    else:
        exp_name, run_name, model_dir = find_latest_model_dir(model_root)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(PROJECT_ROOT, "outputs", "results", "latest_model_eval", f"{exp_name}_{run_name}_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print("Evaluate latest model until first success")
    print("=" * 70)
    print(f"Model dir: {model_dir}")
    print(f"Output dir: {out_dir}")

    device = torch.device("cpu")
    env = FOVPenetrationEnv()
    policies = load_policies(env, model_dir, args.hidden_size, args.layer_N, device)

    attempts = []
    success_ep = None
    attempt = 0

    while True:
        if args.max_attempts > 0 and attempt >= args.max_attempts:
            break

        seed = args.start_seed + attempt
        capture = False
        result = run_one_episode(env, policies, seed, args.hidden_size, device, capture_frames=capture)
        attempts.append({k: result[k] for k in [
            "seed", "success", "done_reason", "hit_count", "episode_reward", "episode_cost", "episode_length"
        ]})

        print(
            f"Attempt {attempt + 1:03d} | seed={seed} | success={result['success']} | "
            f"reason={result['done_reason']} | reward={result['episode_reward']:.1f} | "
            f"cost={result['episode_cost']:.1f} | len={result['episode_length']}"
        )

        if result["success"]:
            print("Success found. Re-running same seed with frame capture for plotting/GIF...")
            success_ep = run_one_episode(env, policies, seed, args.hidden_size, device, capture_frames=True)
            break

        attempt += 1

    env.close()

    if success_ep is None:
        summary = {
            "status": "no_success",
            "attempts": len(attempts),
            "model_dir": model_dir,
            "exp_name": exp_name,
            "run_name": run_name,
            "records": attempts,
        }
        with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print("No successful episode found in allowed attempts.")
        print(f"Summary saved: {os.path.join(out_dir, 'summary.json')}")
        return

    trend_png = os.path.join(out_dir, "attempt_trends.png")
    diag_png = os.path.join(out_dir, "success_diagnostics.png")
    gif_path = os.path.join(out_dir, "success_episode_3d.gif")

    plot_attempt_trend(attempts, trend_png)
    plot_success_diagnostics(success_ep, diag_png)
    save_gif(success_ep["frames"], gif_path, fps=args.gif_fps)

    success_index = len(attempts)
    success_rate = 1.0 / success_index

    summary = {
        "status": "success",
        "model_dir": model_dir,
        "exp_name": exp_name,
        "run_name": run_name,
        "attempts_until_success": success_index,
        "empirical_success_rate_until_first": success_rate,
        "successful_episode": {
            "seed": success_ep["seed"],
            "done_reason": success_ep["done_reason"],
            "hit_count": success_ep["hit_count"],
            "episode_reward": success_ep["episode_reward"],
            "episode_cost": success_ep["episode_cost"],
            "episode_length": success_ep["episode_length"],
            "frames": len(success_ep["frames"]),
        },
        "records": attempts,
        "artifacts": {
            "attempt_trends": trend_png,
            "success_diagnostics": diag_png,
            "success_gif_3d": gif_path,
        }
    }

    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("SUCCESS SUMMARY")
    print("=" * 70)
    print(f"Attempts until success: {success_index}")
    print(f"Success seed: {success_ep['seed']}")
    print(f"Successful episode reward: {success_ep['episode_reward']:.1f}")
    print(f"Successful episode cost: {success_ep['episode_cost']:.1f}")
    print(f"Successful episode len: {success_ep['episode_length']}")
    print(f"Saved: {trend_png}")
    print(f"Saved: {diag_png}")
    print(f"Saved: {gif_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()

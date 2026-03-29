#!/usr/bin/env python
import argparse
import json
import os
import sys

import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "MACPO", "MACPO"))

from envs.fov_penetration import FOVPenetrationEnv
from envs.fov_penetration.render import render_frame
from macpo.algorithms.r_mappo.algorithm.MACPPOPolicy import MACPPOPolicy
from macpo.config import get_config


def load_policies(env, model_dir, device, hidden_size, layer_n):
    parser = get_config()
    all_args = parser.parse_known_args([])[0]
    all_args.algorithm_name = "macpo"
    all_args.hidden_size = hidden_size
    all_args.layer_N = layer_n

    policies = []
    for agent_id in range(env.n_agents):
        policy = MACPPOPolicy(
            all_args,
            env.observation_space[agent_id],
            env.share_observation_space[agent_id],
            env.action_space[agent_id],
            device=device,
        )
        actor_path = os.path.join(model_dir, f"actor_agent{agent_id}.pt")
        if not os.path.exists(actor_path):
            raise FileNotFoundError(f"Missing checkpoint: {actor_path}")
        state_dict = torch.load(actor_path, map_location=device)
        policy.actor.load_state_dict(state_dict)
        policy.actor.eval()
        policies.append(policy)
    return policies


def get_actions(policies, obs, device, hidden_size):
    actions = []
    for agent_id, policy in enumerate(policies):
        obs_t = torch.FloatTensor(obs[agent_id]).unsqueeze(0).to(device)
        rnn = torch.zeros(1, 1, hidden_size).to(device)
        masks = torch.ones(1, 1).to(device)
        with torch.no_grad():
            action, _, _ = policy.actor(obs_t, rnn, masks, deterministic=True)
        actions.append(action.cpu().numpy().flatten())
    return actions


def run_episode_and_collect(env, policies, device, hidden_size, seed, max_steps, frame_skip):
    env.seed(seed)
    obs, _, _ = env.reset()

    telemetry = {
        "steps": [],
        "offensive": [],
        "defensive": [],
        "done_reason": "unknown",
    }

    for _ in range(env.n_offensive):
        telemetry["offensive"].append({
            "x": [], "y": [], "z": [],
            "v": [], "heading": [],
            "nx": [], "ny": [], "nz": [],
            "alive": [],
        })
    for _ in range(env.n_defensive):
        telemetry["defensive"].append({
            "x": [], "y": [], "z": [],
            "v": [], "heading": [],
            "nx": [], "ny": [], "nz": [],
            "alive": [],
        })

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    frames = []

    def record(step_idx):
        telemetry["steps"].append(step_idx)
        for i, off in enumerate(env.offensives):
            telemetry["offensive"][i]["x"].append(float(off.x))
            telemetry["offensive"][i]["y"].append(float(off.y))
            telemetry["offensive"][i]["z"].append(float(off.z))
            telemetry["offensive"][i]["v"].append(float(off.v))
            telemetry["offensive"][i]["heading"].append(float(np.degrees(off.heading)))
            telemetry["offensive"][i]["nx"].append(float(off.nx))
            telemetry["offensive"][i]["ny"].append(float(off.ny))
            telemetry["offensive"][i]["nz"].append(float(off.nz))
            telemetry["offensive"][i]["alive"].append(1 if off.alive else 0)
        for i, d in enumerate(env.defensives):
            telemetry["defensive"][i]["x"].append(float(d.x))
            telemetry["defensive"][i]["y"].append(float(d.y))
            telemetry["defensive"][i]["z"].append(float(d.z))
            telemetry["defensive"][i]["v"].append(float(d.v))
            telemetry["defensive"][i]["heading"].append(float(np.degrees(d.heading)))
            telemetry["defensive"][i]["nx"].append(float(d.nx))
            telemetry["defensive"][i]["ny"].append(float(d.ny))
            telemetry["defensive"][i]["nz"].append(float(d.nz))
            telemetry["defensive"][i]["alive"].append(1 if d.alive else 0)

    record(0)
    render_frame(ax, env, step_num=0, show_fov=True)
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    frames.append(img)

    done = False
    info = {}
    reward_total = 0.0
    cost_total = 0.0

    for step in range(max_steps):
        actions = get_actions(policies, obs, device, hidden_size)
        obs, _, rewards, costs, dones, infos, _ = env.step(actions)
        reward_total += float(sum(r[0] for r in rewards) / env.n_agents)
        cost_total += float(sum(c[0] for c in costs) / env.n_agents)

        record(step + 1)
        if (step + 1) % frame_skip == 0 or any(dones):
            render_frame(ax, env, step_num=step + 1, show_fov=True)
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frames.append(img)

        if any(dones):
            done = True
            info = infos[0]
            telemetry["done_reason"] = info.get("done_reason", "unknown")
            break

    plt.close(fig)
    return telemetry, frames, {
        "done": done,
        "done_reason": telemetry["done_reason"],
        "steps": int(telemetry["steps"][-1]),
        "avg_reward": reward_total,
        "avg_cost": cost_total,
        "success": bool(info.get("success", False)),
        "attacker_killed": bool(info.get("attacker_killed", False)),
        "timeout": telemetry["done_reason"] == "timeout",
    }


def plot_trajectory_3d(telemetry, out_path):
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")

    for i, off in enumerate(telemetry["offensive"]):
        ax.plot(off["x"], off["y"], off["z"], linewidth=1.8, label=f"Off{i}")
        ax.scatter(off["x"][0], off["y"][0], off["z"][0], s=40, marker="o")
        ax.scatter(off["x"][-1], off["y"][-1], off["z"][-1], s=60, marker="x")

    for i, d in enumerate(telemetry["defensive"]):
        ax.plot(d["x"], d["y"], d["z"], "--", linewidth=1.4, label=f"Def{i}")

    ax.set_title("V18 Episode 3D Trajectory")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def plot_trajectory_xy(telemetry, out_path):
    fig, ax = plt.subplots(figsize=(10, 8))
    for i, off in enumerate(telemetry["offensive"]):
        ax.plot(off["x"], off["y"], linewidth=1.8, label=f"Off{i}")
        ax.scatter(off["x"][0], off["y"][0], s=30, marker="o")
        ax.scatter(off["x"][-1], off["y"][-1], s=45, marker="x")
    for i, d in enumerate(telemetry["defensive"]):
        ax.plot(d["x"], d["y"], "--", linewidth=1.4, label=f"Def{i}")
    ax.set_title("V18 Episode 2D Trajectory (XY)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axis("equal")
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def plot_speed_heading(telemetry, dt, speed_path, heading_path):
    t = np.array(telemetry["steps"], dtype=np.float32) * dt

    fig1, ax1 = plt.subplots(figsize=(12, 5))
    for i, off in enumerate(telemetry["offensive"]):
        ax1.plot(t, off["v"], linewidth=1.6, label=f"Off{i}")
    for i, d in enumerate(telemetry["defensive"]):
        ax1.plot(t, d["v"], "--", linewidth=1.2, label=f"Def{i}")
    ax1.set_title("Speed Curve")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Speed (m/s)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)
    fig1.tight_layout()
    fig1.savefig(speed_path, dpi=170)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(12, 5))
    for i, off in enumerate(telemetry["offensive"]):
        ax2.plot(t, off["heading"], linewidth=1.6, label=f"Off{i}")
    for i, d in enumerate(telemetry["defensive"]):
        ax2.plot(t, d["heading"], "--", linewidth=1.2, label=f"Def{i}")
    ax2.set_title("Heading Curve")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Heading (deg)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)
    fig2.tight_layout()
    fig2.savefig(heading_path, dpi=170)
    plt.close(fig2)


def plot_overload(telemetry, dt, out_path):
    t = np.array(telemetry["steps"], dtype=np.float32) * dt
    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
    dims = ["nx", "ny", "nz"]
    titles = ["Longitudinal Overload nx", "Lateral Overload ny", "Normal Overload nz"]

    for idx, (dim, title) in enumerate(zip(dims, titles)):
        ax = axes[idx]
        for i, off in enumerate(telemetry["offensive"]):
            ax.plot(t, off[dim], linewidth=1.4, label=f"Off{i}")
        for i, d in enumerate(telemetry["defensive"]):
            ax.plot(t, d[dim], "--", linewidth=1.1, label=f"Def{i}")
        if dim == "nz":
            ax.axhline(y=1.0, color="green", linestyle=":", alpha=0.7)
        ax.set_ylabel(dim)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="V18 latest model visual simulation test")
    parser.add_argument(
        "--model_dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "outputs", "results", "fov_penetration", "macpo", "v18_attack_gate_long", "run1", "models"),
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "outputs", "results", "v18_visual_test"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_steps", type=int, default=1200)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--frame_skip", type=int, default=2)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--layer_N", type=int, default=3)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cpu")
    env = FOVPenetrationEnv(scenario="scenario_1")
    policies = load_policies(env, args.model_dir, device, args.hidden_size, args.layer_N)

    telemetry, frames, episode_metrics = run_episode_and_collect(
        env=env,
        policies=policies,
        device=device,
        hidden_size=args.hidden_size,
        seed=args.seed,
        max_steps=args.max_steps,
        frame_skip=max(1, args.frame_skip),
    )
    env.close()

    gif_path = os.path.join(args.out_dir, "v18_episode_trajectory.gif")
    imageio.mimsave(gif_path, frames, fps=args.fps)

    plot_trajectory_3d(telemetry, os.path.join(args.out_dir, "v18_trajectory_3d.png"))
    plot_trajectory_xy(telemetry, os.path.join(args.out_dir, "v18_trajectory_xy.png"))

    dt = 0.1
    plot_speed_heading(
        telemetry,
        dt=dt,
        speed_path=os.path.join(args.out_dir, "v18_speed_curve.png"),
        heading_path=os.path.join(args.out_dir, "v18_heading_curve.png"),
    )
    plot_overload(telemetry, dt=dt, out_path=os.path.join(args.out_dir, "v18_overload_curves.png"))

    with open(os.path.join(args.out_dir, "v18_episode_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(episode_metrics, f, ensure_ascii=False, indent=2)

    print("Visual simulation test completed.")
    print(f"Output dir: {args.out_dir}")
    print(json.dumps(episode_metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

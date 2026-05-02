#!/usr/bin/env python
"""Batched Monte Carlo hit-rate evaluation for v69 policies."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from eval_v28_10episodes import load_policies
from scripts.eval_v69_collect import DEFAULT_MODEL_DIR, HIDDEN, LAYER_N, make_env, model_complete, scenario_spec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=os.environ.get("MODEL_DIR", DEFAULT_MODEL_DIR))
    parser.add_argument("--out-root", default="outputs/v69_monte_carlo")
    parser.add_argument("--tag", default="mc1000_batch")
    parser.add_argument("--seed-start", type=int, default=20000)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--parallel-envs", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=8000)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--obs-mask", default=os.environ.get("FOV_OBS_PHASE_MASK", "v65_strict_los"))
    parser.add_argument("--terminal-guidance", default=os.environ.get("FOV_TERMINAL_GUIDANCE", "pn_los"))
    parser.add_argument("--pn-gain", type=float, default=float(os.environ.get("FOV_TERMINAL_PN_GAIN", "3.0")))
    parser.add_argument("--pn-max-action", type=float, default=float(os.environ.get("FOV_TERMINAL_PN_MAX_ACTION", "0.8")))
    parser.add_argument("--scenario-case", default=os.environ.get("FOV_SCENARIO_CASE", "baseline"),
                        choices=["baseline", "strong_defense", "six_defenders"])
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    return parser.parse_args()


def summarize(rows: list[dict], args: argparse.Namespace, out_dir: Path) -> dict:
    n = len(rows)
    success_count = sum(int(row["success"]) for row in rows)
    hit_count = sum(int(row["hit_count"]) for row in rows)
    hit_rate = success_count / max(n, 1)
    se = float(np.sqrt(hit_rate * (1.0 - hit_rate) / max(n, 1))) if n else 0.0
    best_values = [float(row["best_min_dist_m"]) for row in rows]
    return {
        "timestamp": "_".join(out_dir.name.split("_")[:2]),
        "tag": args.tag,
        "model_dir": args.model_dir,
        "scenario_case": args.scenario_case,
        "scenario_info": scenario_spec(args.scenario_case)[2],
        "episodes_requested": args.episodes,
        "episodes_completed": n,
        "parallel_envs": args.parallel_envs,
        "seed_start": args.seed_start,
        "success_count": success_count,
        "total_hits": hit_count,
        "success_rate": hit_rate,
        "success_rate_ci95_normal": [max(0.0, hit_rate - 1.96 * se), min(1.0, hit_rate + 1.96 * se)],
        "best_min_dist_m": min(best_values) if best_values else None,
        "mean_best_min_dist_m": float(np.mean(best_values)) if best_values else None,
        "median_best_min_dist_m": float(np.median(best_values)) if best_values else None,
    }


def write_outputs(out_dir: Path, rows: list[dict], args: argparse.Namespace) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    if rows:
        fieldnames = list(rows[0].keys())
        with (out_dir / "episodes.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    summary = summarize(rows, args, out_dir)
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def init_slot(slot_id: int, state: dict, env, seed: int, args: argparse.Namespace, device: torch.device):
    env.seed(seed)
    obs, _, _ = env.reset()
    hvt = env.hvt
    state["active"][slot_id] = True
    state["seed"][slot_id] = seed
    state["episode_idx"][slot_id] = seed - args.seed_start
    state["obs"][slot_id] = np.asarray(obs, dtype=np.float32)
    state["step"][slot_id] = 0
    state["final_info"][slot_id] = {}
    state["min_d"][slot_id] = np.asarray(
        [off.distance_to(hvt.x, hvt.y, hvt.z) for off in env.offensives], dtype=np.float64
    )
    state["min_step"][slot_id] = np.zeros(env.n_agents, dtype=np.int32)
    for agent_id in range(env.n_agents):
        state["rnn"][agent_id][slot_id] = torch.zeros(1, HIDDEN, device=device)
        state["mask"][agent_id][slot_id] = torch.ones(1, device=device)


def finish_slot(slot_id: int, state: dict, env) -> dict:
    min_d = state["min_d"][slot_id]
    min_step = state["min_step"][slot_id]
    best_agent = int(np.argmin(min_d))
    final_info = state["final_info"][slot_id]
    row = {
        "seed": int(state["seed"][slot_id]),
        "success": int(env.hit_count > 0),
        "hit_count": int(env.hit_count),
        "hit_indices": ";".join(str(i) for i in env.hit_indices),
        "final_step": int(state["step"][slot_id]),
        "done_reason": final_info.get("done_reason", "unknown") if isinstance(final_info, dict) else "unknown",
        "best_min_dist_m": float(min_d[best_agent]),
        "min_step_best": int(min_step[best_agent]),
        "min_d_agent0_m": float(min_d[0]),
        "min_d_agent1_m": float(min_d[1]),
        "min_d_agent2_m": float(min_d[2]),
        "min_d_agent3_m": float(min_d[3]),
        "episode_idx": int(state["episode_idx"][slot_id]),
    }
    state["active"][slot_id] = False
    return row


def batched_actions(policies, obs_batch: np.ndarray, state: dict, active_slots: list[int], device: torch.device):
    n_active, n_agents, _ = obs_batch.shape
    actions_by_slot = np.zeros((n_active, n_agents, 3), dtype=np.float32)
    slot_tensor = torch.tensor(active_slots, dtype=torch.long, device=device)
    with torch.no_grad():
        for agent_id, policy in enumerate(policies):
            obs_tensor = torch.as_tensor(obs_batch[:, agent_id, :], dtype=torch.float32, device=device)
            rnn_states = state["rnn"][agent_id].index_select(0, slot_tensor)
            masks = state["mask"][agent_id].index_select(0, slot_tensor)
            action, _, hidden = policy.actor(obs_tensor, rnn_states, masks, deterministic=True)
            actions_by_slot[:, agent_id, :] = action.detach().cpu().numpy()
            state["rnn"][agent_id][slot_tensor] = hidden.detach()
    return actions_by_slot


def run(args: argparse.Namespace, policies, device: torch.device, out_dir: Path) -> list[dict]:
    n_slots = max(1, int(args.parallel_envs))
    envs = [make_env(args) for _ in range(n_slots)]
    n_agents = envs[0].n_agents
    obs_dim = int(np.asarray(envs[0].reset()[0]).shape[-1])
    state = {
        "active": [False] * n_slots,
        "seed": [-1] * n_slots,
        "episode_idx": [-1] * n_slots,
        "obs": np.zeros((n_slots, n_agents, obs_dim), dtype=np.float32),
        "step": np.zeros(n_slots, dtype=np.int32),
        "final_info": [{} for _ in range(n_slots)],
        "min_d": np.zeros((n_slots, n_agents), dtype=np.float64),
        "min_step": np.zeros((n_slots, n_agents), dtype=np.int32),
        "rnn": [torch.zeros(n_slots, 1, HIDDEN, device=device) for _ in range(n_agents)],
        "mask": [torch.ones(n_slots, 1, device=device) for _ in range(n_agents)],
    }

    next_seed = args.seed_start
    last_seed = args.seed_start + args.episodes
    rows: list[dict] = []

    for slot_id, env in enumerate(envs):
        if next_seed < last_seed:
            init_slot(slot_id, state, env, next_seed, args, device)
            next_seed += 1

    try:
        while any(state["active"]):
            active_slots = [idx for idx, is_active in enumerate(state["active"]) if is_active]
            obs_batch = state["obs"][active_slots]
            action_batch = batched_actions(policies, obs_batch, state, active_slots, device)

            for batch_idx, slot_id in enumerate(active_slots):
                env = envs[slot_id]
                obs, _, _, _, dones, infos, _ = env.step([action_batch[batch_idx, aid] for aid in range(n_agents)])
                state["obs"][slot_id] = np.asarray(obs, dtype=np.float32)
                state["step"][slot_id] += 1
                state["final_info"][slot_id] = infos[0] if infos else {}

                hvt = env.hvt
                for agent_id, off in enumerate(env.offensives):
                    dist = off.distance_to(hvt.x, hvt.y, hvt.z)
                    if (off.alive or off.hit_hvt) and dist < state["min_d"][slot_id, agent_id]:
                        state["min_d"][slot_id, agent_id] = dist
                        state["min_step"][slot_id, agent_id] = state["step"][slot_id]

                if all(dones) or state["step"][slot_id] >= args.max_steps:
                    rows.append(finish_slot(slot_id, state, env))
                    if len(rows) % max(args.save_every, 1) == 0 or len(rows) == args.episodes:
                        summary = write_outputs(out_dir, rows, args)
                        print(json.dumps({
                            "completed": len(rows),
                            "episodes": args.episodes,
                            "success_rate": summary["success_rate"],
                            "success_count": summary["success_count"],
                            "best_min_dist_m": summary["best_min_dist_m"],
                            "out_dir": str(out_dir),
                        }, ensure_ascii=False), flush=True)
                    if next_seed < last_seed:
                        init_slot(slot_id, state, env, next_seed, args, device)
                        next_seed += 1
    finally:
        for env in envs:
            if hasattr(env, "close"):
                env.close()
    return rows


def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = PROJECT_ROOT / args.out_root / f"{timestamp}_{args.tag}_{args.scenario_case}"
    out_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("FOV_REWARD_PROFILE", "v68strictpnfix")
    os.environ.setdefault("FOV_OBS_PHASE_MASK", args.obs_mask)
    os.environ.setdefault("FOV_TERMINAL_GUIDANCE", args.terminal_guidance)
    os.environ.setdefault("FOV_TERMINAL_PN_GAIN", str(args.pn_gain))
    os.environ.setdefault("FOV_TERMINAL_PN_MAX_ACTION", str(args.pn_max_action))

    model_dir = (PROJECT_ROOT / args.model_dir).resolve() if not os.path.isabs(args.model_dir) else Path(args.model_dir)
    if not model_complete(model_dir):
        raise SystemExit(f"Incomplete model directory: {model_dir}")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")

    base_env = make_env(args)
    policies = load_policies(str(model_dir), base_env, device, hidden_size=HIDDEN, layer_N=LAYER_N)
    if hasattr(base_env, "close"):
        base_env.close()

    rows = run(args, policies, device, out_dir)
    summary = write_outputs(out_dir, rows, args)

    latest_link = PROJECT_ROOT / args.out_root / f"latest_{args.scenario_case}"
    if latest_link.exists() or latest_link.is_symlink():
        latest_link.unlink()
    latest_link.symlink_to(out_dir.resolve())
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
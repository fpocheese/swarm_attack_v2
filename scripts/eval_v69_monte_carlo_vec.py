#!/usr/bin/env python
"""Parallel environment-pool Monte Carlo evaluation for v69 policies."""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

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
    parser.add_argument("--tag", default="mc1000_vec")
    parser.add_argument("--seed-start", type=int, default=20000)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--parallel-envs", type=int, default=48)
    parser.add_argument("--mp-start-method", default="fork", choices=["fork", "spawn", "forkserver"])
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


def args_for_worker(args: argparse.Namespace) -> dict:
    return {
        "obs_mask": args.obs_mask,
        "terminal_guidance": args.terminal_guidance,
        "pn_gain": args.pn_gain,
        "pn_max_action": args.pn_max_action,
        "scenario_case": args.scenario_case,
        "max_steps": args.max_steps,
    }


def init_episode(env, seed: int, max_steps: int) -> dict:
    env.seed(seed)
    obs, _, _ = env.reset()
    hvt = env.hvt
    return {
        "seed": seed,
        "obs": np.asarray(obs, dtype=np.float32),
        "step": 0,
        "max_steps": max_steps,
        "final_info": {},
        "min_d": np.asarray([off.distance_to(hvt.x, hvt.y, hvt.z) for off in env.offensives], dtype=np.float64),
        "min_step": np.zeros(env.n_agents, dtype=np.int32),
    }


def finish_episode(env, state: dict) -> dict:
    min_d = state["min_d"]
    min_step = state["min_step"]
    best_agent = int(np.argmin(min_d))
    final_info = state["final_info"]
    return {
        "seed": int(state["seed"]),
        "success": int(env.hit_count > 0),
        "hit_count": int(env.hit_count),
        "hit_indices": ";".join(str(i) for i in env.hit_indices),
        "final_step": int(state["step"]),
        "done_reason": final_info.get("done_reason", "unknown") if isinstance(final_info, dict) else "unknown",
        "best_min_dist_m": float(min_d[best_agent]),
        "min_step_best": int(min_step[best_agent]),
        "min_d_agent0_m": float(min_d[0]),
        "min_d_agent1_m": float(min_d[1]),
        "min_d_agent2_m": float(min_d[2]),
        "min_d_agent3_m": float(min_d[3]),
    }


def env_worker(remote, worker_args: dict):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    args = SimpleNamespace(**worker_args)
    env = make_env(args)
    state = None
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "reset":
                state = init_episode(env, int(data), int(worker_args["max_steps"]))
                remote.send({"obs": state["obs"]})
            elif cmd == "step":
                if state is None:
                    raise RuntimeError("Worker step called before reset")
                obs, _, _, _, dones, infos, _ = env.step(data)
                state["step"] += 1
                state["final_info"] = infos[0] if infos else {}
                hvt = env.hvt
                for agent_id, off in enumerate(env.offensives):
                    dist = off.distance_to(hvt.x, hvt.y, hvt.z)
                    if (off.alive or off.hit_hvt) and dist < state["min_d"][agent_id]:
                        state["min_d"][agent_id] = dist
                        state["min_step"][agent_id] = state["step"]
                if all(dones) or state["step"] >= state["max_steps"]:
                    row = finish_episode(env, state)
                    state = None
                    remote.send({"done": True, "row": row})
                else:
                    state["obs"] = np.asarray(obs, dtype=np.float32)
                    remote.send({"done": False, "obs": state["obs"]})
            elif cmd == "close":
                break
            else:
                raise RuntimeError(f"Unknown command: {cmd}")
    finally:
        if hasattr(env, "close"):
            env.close()
        remote.close()


def summarize(rows: list[dict], args: argparse.Namespace, out_dir: Path) -> dict:
    n = len(rows)
    success_count = sum(int(row["success"]) for row in rows)
    hit_count = sum(int(row["hit_count"]) for row in rows)
    hit_rate = success_count / max(n, 1)
    se = float(np.sqrt(hit_rate * (1.0 - hit_rate) / max(n, 1))) if n else 0.0
    best_values = [float(row["best_min_dist_m"]) for row in rows]
    z = 1.96
    denom = 1.0 + z * z / max(n, 1)
    center = (hit_rate + z * z / (2.0 * max(n, 1))) / denom
    margin = z * np.sqrt((hit_rate * (1.0 - hit_rate) + z * z / (4.0 * max(n, 1))) / max(n, 1)) / denom
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
        "seed_end": args.seed_start + args.episodes - 1,
        "success_count": success_count,
        "total_hits": hit_count,
        "success_rate": hit_rate,
        "success_rate_ci95_normal": [max(0.0, hit_rate - 1.96 * se), min(1.0, hit_rate + 1.96 * se)],
        "success_rate_ci95_wilson": [float(max(0.0, center - margin)), float(min(1.0, center + margin))],
        "best_min_dist_m": min(best_values) if best_values else None,
        "mean_best_min_dist_m": float(np.mean(best_values)) if best_values else None,
        "median_best_min_dist_m": float(np.median(best_values)) if best_values else None,
    }


def write_outputs(out_dir: Path, rows: list[dict], args: argparse.Namespace) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    if rows:
        fieldnames = list(rows[0].keys()) + ["episode_idx", "scenario_case"]
        with (out_dir / "episodes.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for idx, row in enumerate(sorted(rows, key=lambda item: int(item["seed"]))):
                out_row = dict(row)
                out_row["episode_idx"] = idx
                out_row["scenario_case"] = args.scenario_case
                writer.writerow(out_row)
    summary = summarize(rows, args, out_dir)
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def batched_actions(policies, obs_by_slot: dict[int, np.ndarray], rnn_states, masks, device: torch.device):
    slots = sorted(obs_by_slot)
    obs_batch = np.stack([obs_by_slot[slot] for slot in slots], axis=0)
    _, n_agents, _ = obs_batch.shape
    slot_tensor = torch.tensor(slots, dtype=torch.long, device=device)
    actions_by_slot = {slot: [] for slot in slots}
    with torch.no_grad():
        for agent_id, policy in enumerate(policies):
            obs_tensor = torch.as_tensor(obs_batch[:, agent_id, :], dtype=torch.float32, device=device)
            hidden = rnn_states[agent_id].index_select(0, slot_tensor)
            mask = masks[agent_id].index_select(0, slot_tensor)
            action, _, next_hidden = policy.actor(obs_tensor, hidden, mask, deterministic=True)
            rnn_states[agent_id][slot_tensor] = next_hidden.detach()
            action_np = action.detach().cpu().numpy()
            for row_idx, slot in enumerate(slots):
                actions_by_slot[slot].append(action_np[row_idx].astype(np.float32))
    return actions_by_slot


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
    torch.set_num_threads(1)

    model_dir = (PROJECT_ROOT / args.model_dir).resolve() if not os.path.isabs(args.model_dir) else Path(args.model_dir)
    if not model_complete(model_dir):
        raise SystemExit(f"Incomplete model directory: {model_dir}")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")

    base_env = make_env(args)
    n_agents = base_env.n_agents

    n_workers = min(max(1, args.parallel_envs), args.episodes)
    ctx = mp.get_context(args.mp_start_method)
    remotes = []
    processes = []
    for _ in range(n_workers):
        parent_remote, child_remote = ctx.Pipe()
        process = ctx.Process(target=env_worker, args=(child_remote, args_for_worker(args)))
        process.daemon = True
        process.start()
        child_remote.close()
        remotes.append(parent_remote)
        processes.append(process)

    policies = load_policies(str(model_dir), base_env, device, hidden_size=HIDDEN, layer_N=LAYER_N)
    if hasattr(base_env, "close"):
        base_env.close()

    rnn_states = [torch.zeros(n_workers, 1, HIDDEN, device=device) for _ in range(n_agents)]
    masks = [torch.ones(n_workers, 1, device=device) for _ in range(n_agents)]
    obs_by_slot: dict[int, np.ndarray] = {}
    rows: list[dict] = []
    next_seed = args.seed_start
    last_seed = args.seed_start + args.episodes

    def assign(slot: int, seed: int):
        for agent_id in range(n_agents):
            rnn_states[agent_id][slot].zero_()
            masks[agent_id][slot].fill_(1.0)
        remotes[slot].send(("reset", seed))
        obs_by_slot[slot] = remotes[slot].recv()["obs"]

    try:
        for slot in range(n_workers):
            if next_seed < last_seed:
                assign(slot, next_seed)
                next_seed += 1

        while obs_by_slot:
            actions = batched_actions(policies, obs_by_slot, rnn_states, masks, device)
            active_slots = sorted(obs_by_slot)
            for slot in active_slots:
                remotes[slot].send(("step", actions[slot]))
            for slot in active_slots:
                result = remotes[slot].recv()
                if result["done"]:
                    rows.append(result["row"])
                    obs_by_slot.pop(slot, None)
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
                        assign(slot, next_seed)
                        next_seed += 1
                else:
                    obs_by_slot[slot] = result["obs"]
    finally:
        for remote in remotes:
            try:
                remote.send(("close", None))
            except Exception:
                pass
        for process in processes:
            process.join(timeout=2)
            if process.is_alive():
                process.terminate()

    summary = write_outputs(out_dir, rows, args)
    latest_link = PROJECT_ROOT / args.out_root / f"latest_{args.scenario_case}"
    if latest_link.exists() or latest_link.is_symlink():
        latest_link.unlink()
    latest_link.symlink_to(out_dir.resolve())
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
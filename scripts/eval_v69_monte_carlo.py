#!/usr/bin/env python
"""Lightweight Monte Carlo hit-rate evaluation for v69 policies."""

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
from scripts.eval_v69_collect import (
    DEFAULT_MODEL_DIR,
    HIDDEN,
    LAYER_N,
    get_actions,
    make_env,
    model_complete,
    scenario_spec,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=os.environ.get("MODEL_DIR", DEFAULT_MODEL_DIR))
    parser.add_argument("--out-root", default="outputs/v69_monte_carlo")
    parser.add_argument("--tag", default="mc1000")
    parser.add_argument("--seed-start", type=int, default=20000)
    parser.add_argument("--episodes", type=int, default=1000)
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


def summarize(rows, args: argparse.Namespace, out_dir: Path) -> dict:
    n = len(rows)
    success_count = sum(int(row["success"]) for row in rows)
    hit_count = sum(int(row["hit_count"]) for row in rows)
    hit_rate = success_count / max(n, 1)
    se = float(np.sqrt(hit_rate * (1.0 - hit_rate) / max(n, 1))) if n else 0.0
    best_values = [float(row["best_min_dist_m"]) for row in rows]
    summary = {
        "timestamp": "_".join(out_dir.name.split("_")[:2]),
        "tag": args.tag,
        "model_dir": args.model_dir,
        "scenario_case": args.scenario_case,
        "scenario_info": scenario_spec(args.scenario_case)[2],
        "episodes_requested": args.episodes,
        "episodes_completed": n,
        "seed_start": args.seed_start,
        "success_count": success_count,
        "total_hits": hit_count,
        "success_rate": hit_rate,
        "success_rate_ci95_normal": [max(0.0, hit_rate - 1.96 * se), min(1.0, hit_rate + 1.96 * se)],
        "best_min_dist_m": min(best_values) if best_values else None,
        "mean_best_min_dist_m": float(np.mean(best_values)) if best_values else None,
        "median_best_min_dist_m": float(np.median(best_values)) if best_values else None,
    }
    return summary


def write_outputs(out_dir: Path, rows: list[dict], args: argparse.Namespace):
    out_dir.mkdir(parents=True, exist_ok=True)
    if rows:
        with (out_dir / "episodes.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    summary = summarize(rows, args, out_dir)
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def run_episode(args: argparse.Namespace, policies, device, seed: int) -> dict:
    env = make_env(args)
    env.seed(seed)
    obs, _, _ = env.reset()
    rnn_states = [torch.zeros(1, 1, HIDDEN).to(device) for _ in range(env.n_agents)]
    masks = [torch.ones(1, 1).to(device) for _ in range(env.n_agents)]
    hvt = env.hvt
    min_d = [off.distance_to(hvt.x, hvt.y, hvt.z) for off in env.offensives]
    min_step = [0 for _ in env.offensives]
    final_info = {}
    final_step = 0

    for step in range(args.max_steps):
        actions, rnn_states = get_actions(policies, obs, device, rnn_states, masks)
        obs, _, _, _, dones, infos, _ = env.step(actions)
        final_info = infos[0] if infos else {}
        final_step = step + 1
        for agent_id, off in enumerate(env.offensives):
            dist = off.distance_to(hvt.x, hvt.y, hvt.z)
            if off.alive or off.hit_hvt:
                if dist < min_d[agent_id]:
                    min_d[agent_id] = dist
                    min_step[agent_id] = final_step
        if all(dones):
            break

    return {
        "seed": seed,
        "success": int(env.hit_count > 0),
        "hit_count": int(env.hit_count),
        "hit_indices": ";".join(str(i) for i in env.hit_indices),
        "final_step": final_step,
        "done_reason": final_info.get("done_reason", "unknown") if isinstance(final_info, dict) else "unknown",
        "best_min_dist_m": float(min(min_d)),
        "min_step_best": int(min_step[int(np.argmin(min_d))]),
        "min_d_agent0_m": float(min_d[0]),
        "min_d_agent1_m": float(min_d[1]),
        "min_d_agent2_m": float(min_d[2]),
        "min_d_agent3_m": float(min_d[3]),
    }


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

    if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()):
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    base_env = make_env(args)
    policies = load_policies(str(model_dir), base_env, device, hidden_size=HIDDEN, layer_N=LAYER_N)

    rows = []
    for idx in range(args.episodes):
        seed = args.seed_start + idx
        row = run_episode(args, policies, device, seed)
        row["episode_idx"] = idx
        row["scenario_case"] = args.scenario_case
        rows.append(row)
        if (idx + 1) % max(args.save_every, 1) == 0 or idx + 1 == args.episodes:
            summary = write_outputs(out_dir, rows, args)
            print(json.dumps({
                "completed": idx + 1,
                "episodes": args.episodes,
                "success_rate": summary["success_rate"],
                "success_count": summary["success_count"],
                "best_min_dist_m": summary["best_min_dist_m"],
                "out_dir": str(out_dir),
            }, ensure_ascii=False), flush=True)

    latest_link = PROJECT_ROOT / args.out_root / f"latest_{args.scenario_case}"
    if latest_link.exists() or latest_link.is_symlink():
        latest_link.unlink()
    latest_link.symlink_to(out_dir.resolve())
    print(json.dumps(summarize(rows, args, out_dir), ensure_ascii=False))


if __name__ == "__main__":
    main()

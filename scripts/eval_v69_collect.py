#!/usr/bin/env python
"""Hourly deterministic evaluation and plotting for v69 terminal-PN runs.

The script evaluates the latest remote MAPPO checkpoint under the strict
two-stage observation mask and optional terminal LOS-rate PN action wrapper. It
saves raw telemetry, aggregate summaries, and publication-style figures.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from envs.fov_penetration import FOVPenetrationEnv
from eval_v28_10episodes import load_policies
from scripts.phase_obs_wrapper import PhaseMaskedFOVWrapper
from scripts.terminal_pn_action_wrapper import TerminalPNActionWrapper

HIDDEN = 256
LAYER_N = 3
G0 = 9.80665
DEFAULT_MODEL_DIR = "outputs/results/fov_penetration/mappo/v69_hybrid_terminal_pn/run1/models"


SCENARIO_CASES = {
    "baseline": "Nominal 4v4 defense",
    "strong_defense": "4v4 with strengthened defender maneuver envelope",
    "six_defenders": "4v6 defense with nominal maneuver envelope",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=os.environ.get("MODEL_DIR", DEFAULT_MODEL_DIR))
    parser.add_argument("--out-root", default="outputs/v69_hourly_eval")
    parser.add_argument("--tag", default="")
    parser.add_argument("--seeds", default=os.environ.get("SEEDS", "1000,1001,1002"))
    parser.add_argument("--max-steps", type=int, default=8000)
    parser.add_argument("--sample-stride", type=int, default=10)
    parser.add_argument("--obs-mask", default=os.environ.get("FOV_OBS_PHASE_MASK", "v65_strict_los"))
    parser.add_argument("--terminal-guidance", default=os.environ.get("FOV_TERMINAL_GUIDANCE", "pn_los"))
    parser.add_argument("--pn-gain", type=float, default=float(os.environ.get("FOV_TERMINAL_PN_GAIN", "3.0")))
    parser.add_argument("--pn-max-action", type=float, default=float(os.environ.get("FOV_TERMINAL_PN_MAX_ACTION", "0.8")))
    parser.add_argument("--scenario-case", default=os.environ.get("FOV_SCENARIO_CASE", "baseline"),
                        choices=sorted(SCENARIO_CASES.keys()))
    parser.add_argument("--case-label", default="")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def scenario_spec(case_name: str) -> tuple[str, dict, dict]:
    """Return a scenario name plus eval-only overrides for adversary studies.

    The stable scenario_1 definition is not edited on disk. These overrides are
    applied only to the transient evaluation environment.
    """
    case_name = case_name.strip().lower()
    if case_name == "baseline":
        return "scenario_1", {
            "offensive_init": {
                "center_x": -1200.0, "center_y": 0.0, "center_z": 300.0,
                "spread_xy": 150.0, "spread_z": 30.0,
                "heading_to_hvt": True, "heading_noise": 0.2, "gamma_noise": 0.02,
                "pos_noise_xy": 100.0, "pos_noise_z": 20.0,
            },
            "defensive_init": {
                "center_x": 600.0, "center_y": 0.0, "center_z": 350.0,
                "spread_xy": 200.0, "spread_z": 50.0,
                "heading_to_offense": True, "heading_noise": 0.3, "gamma_noise": 0.1,
                "pos_noise_xy": 100.0, "pos_noise_z": 30.0,
            },
        }, {
            "case": case_name,
            "label": SCENARIO_CASES[case_name],
            "n_offensive": 4,
            "n_defensive": 4,
            "defender_maneuver": "nominal 5g, v=55 m/s",
            "geometry": "head-on, defenders forward at +600m on x-axis",
        }
    if case_name == "strong_defense":
        return "scenario_1", {
            # geometry: defenders pushed forward and laterally offset to the
            # north so attackers must engage at a 30-deg crossing angle; this
            # makes the visual distinct from baseline head-on.
            "offensive_init": {
                "center_x": -1200.0, "center_y": -300.0, "center_z": 350.0,
                "spread_xy": 150.0, "spread_z": 30.0,
                "heading_to_hvt": True, "heading_noise": 0.2, "gamma_noise": 0.02,
                "pos_noise_xy": 100.0, "pos_noise_z": 20.0,
            },
            "defensive_init": {
                "center_x": 300.0, "center_y": 250.0, "center_z": 450.0,
                "spread_xy": 220.0, "spread_z": 60.0,
                "heading_to_offense": True, "heading_noise": 0.3, "gamma_noise": 0.1,
                "pos_noise_xy": 100.0, "pos_noise_z": 30.0,
            },
            "defensive": {
                "v_min": 50.0, "v_nominal": 60.0, "v_max": 65.0,
                "ax_min": -12.0, "ax_max": 40.0,
                "an_pitch_max": 7.0 * G0, "an_yaw_max": 7.0 * G0,
                "dax_max": 110.0, "dan_pitch_max": 220.0, "dan_yaw_max": 220.0,
            },
        }, {
            "case": case_name,
            "label": SCENARIO_CASES[case_name],
            "n_offensive": 4,
            "n_defensive": 4,
            "defender_maneuver": "strengthened 7g, v=60 m/s",
            "geometry": "30-deg crossing, defenders forward-pushed and laterally offset (+250 m y)",
        }
    if case_name == "six_defenders":
        return "scenario_1", {
            "n_defensive": 6,
            # geometry: defenders deployed in an arc; attackers approach from
            # the southeast so the 6 interceptors are visually distinguishable.
            "offensive_init": {
                "center_x": -1100.0, "center_y": 400.0, "center_z": 320.0,
                "spread_xy": 200.0, "spread_z": 40.0,
                "heading_to_hvt": True, "heading_noise": 0.25, "gamma_noise": 0.02,
                "pos_noise_xy": 120.0, "pos_noise_z": 20.0,
            },
            "defensive_init": {
                "center_x": 500.0, "center_y": 0.0, "center_z": 400.0,
                "spread_xy": 450.0, "spread_z": 80.0,
                "heading_to_offense": True, "heading_noise": 0.4, "gamma_noise": 0.1,
                "pos_noise_xy": 150.0, "pos_noise_z": 40.0,
            },
        }, {
            "case": case_name,
            "label": SCENARIO_CASES[case_name],
            "n_offensive": 4,
            "n_defensive": 6,
            "defender_maneuver": "nominal 5g, v=55 m/s",
            "geometry": "attackers approach from southeast; 6 interceptors deployed in a 450m-wide arc",
            "policy_observation": "fixed actor observation; highest-threat defenders are selected without expanding input dimension",
        }
    raise ValueError(f"Unknown scenario case: {case_name}")


def wrap_angle(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def vel3d(aircraft) -> np.ndarray:
    return np.array([
        aircraft.v * np.cos(aircraft.gamma) * np.cos(aircraft.heading),
        aircraft.v * np.cos(aircraft.gamma) * np.sin(aircraft.heading),
        aircraft.v * np.sin(aircraft.gamma),
    ], dtype=np.float64)


def nearest_def_hvt(env) -> float:
    hvt = env.hvt
    vals = [d.distance_to(hvt.x, hvt.y, hvt.z) for d in env.defensives if d.alive]
    return float(min(vals)) if vals else float("inf")


def get_actions(policies, obs, device, rnn_states, masks):
    actions = []
    next_rnn = []
    obs_all = obs[0] if isinstance(obs, tuple) else obs
    for agent_id, policy in enumerate(policies):
        obs_tensor = torch.FloatTensor(np.asarray(obs_all[agent_id]).flatten()).unsqueeze(0).to(device)
        with torch.no_grad():
            action, _, hidden = policy.actor(obs_tensor, rnn_states[agent_id], masks[agent_id], deterministic=True)
        actions.append(action.cpu().numpy().flatten())
        next_rnn.append(hidden)
    return actions, next_rnn


def make_env(args: argparse.Namespace):
    scenario_name, overrides, _ = scenario_spec(getattr(args, "scenario_case", "baseline"))
    env = FOVPenetrationEnv(config=overrides or None, scenario=scenario_name)
    if args.obs_mask and args.obs_mask != "none":
        env = PhaseMaskedFOVWrapper(env, mode=args.obs_mask)
    if args.terminal_guidance == "pn_los":
        env = TerminalPNActionWrapper(env, gain=args.pn_gain, max_action=args.pn_max_action)
    return env


def model_complete(model_dir: Path) -> bool:
    return all((model_dir / f"actor_agent{i}.pt").exists() for i in range(4))


def list_value(mapping, key: str, agent_id: int, default: float = 0.0) -> float:
    vals = mapping.get(key, []) if isinstance(mapping, dict) else []
    try:
        return float(vals[agent_id])
    except Exception:
        return float(default)


def matrix_mean(mapping, key: str) -> float:
    vals = mapping.get(key, None) if isinstance(mapping, dict) else None
    if vals is None:
        return 0.0
    arr = np.asarray(vals, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    return float(np.nanmean(arr))


def collect_episode(args, policies, device, seed: int, out_dir: Path):
    env = make_env(args)
    env.seed(seed)
    obs, _, _ = env.reset()
    n_agents = env.n_agents
    hvt = env.hvt
    rnn_states = [torch.zeros(1, 1, HIDDEN).to(device) for _ in range(n_agents)]
    masks = [torch.ones(1, 1).to(device) for _ in range(n_agents)]
    telemetry_rows = []
    metric_rows = []
    min_d = [off.distance_to(hvt.x, hvt.y, hvt.z) for off in env.offensives]
    min_step = [0 for _ in env.offensives]
    final_info = {}
    final_step = 0

    for step in range(args.max_steps):
        actor_actions, rnn_states = get_actions(policies, obs, device, rnn_states, masks)
        executed_actions = actor_actions
        guided_count = 0
        terminal_flags = env._terminal_flags() if hasattr(env, "_terminal_flags") else [False] * n_agents
        if hasattr(env, "guide_actions"):
            executed_actions, guided_count = env.guide_actions(actor_actions)

        obs_before = np.asarray(obs, dtype=np.float32)
        obs, _, rewards, _, dones, infos, _ = env.step(actor_actions)
        info = infos[0] if infos else {}
        final_info = info if isinstance(info, dict) else {}
        final_step = step + 1

        nearest_def = nearest_def_hvt(env)
        alive_dists = []
        for agent_id, off in enumerate(env.offensives):
            dist_hvt = float(off.distance_to(hvt.x, hvt.y, hvt.z))
            if off.alive or off.hit_hvt:
                if dist_hvt < min_d[agent_id]:
                    min_d[agent_id] = dist_hvt
                    min_step[agent_id] = step + 1
            if off.alive and not off.hit_hvt:
                alive_dists.append(dist_hvt)

        should_sample = (step % args.sample_stride == 0) or bool(env.hit_count > 0) or all(dones)
        if should_sample:
            decoy = getattr(env, "_ap_decoy_info", {})
            pen = getattr(env, "_ap_pen_info", {})
            esc = getattr(env, "_ap_esc_info", {})
            hvt_info = getattr(env, "_ap_hvt_info", {})
            gamma_mean = float(final_info.get("Gamma_mean", matrix_mean(esc, "_Gamma_matrix")))
            xi_mean = float(final_info.get("Xi_mean", matrix_mean(esc, "_Xi_matrix")))
            metric_rows.append({
                "seed": seed,
                "step": step + 1,
                "time_s": (step + 1) * env.dt,
                "hit_count": int(env.hit_count),
                "min_off_hvt_dist_m": float(min(alive_dists) if alive_dists else min(min_d)),
                "nearest_def_hvt_dist_m": nearest_def,
                "phase_terminal_count": int(final_info.get("phase_terminal_count", sum(terminal_flags))),
                "terminal_pn_guided_agents": int(final_info.get("terminal_pn_guided_agents", guided_count)),
                "Phi_decoy": float(final_info.get("Phi_decoy", decoy.get("Phi_decoy", 0.0))),
                "N_eff": float(final_info.get("N_eff", pen.get("N_eff", 0.0))),
                "Gamma_mean": gamma_mean,
                "Gamma_max": float(final_info.get("Gamma_max", 0.0)),
                "Xi_mean": xi_mean,
                "Xi_max": float(final_info.get("Xi_max", 0.0)),
                "hvt_closing_speed_mean": float(final_info.get("hvt_closing_speed_mean", 0.0)),
                "n_locked_defenders": int(final_info.get("n_locked_defenders", 0)),
                "reward_mean": float(np.mean(rewards)),
            })
            for agent_id, off in enumerate(env.offensives):
                obs_i = obs_before[agent_id] if agent_id < len(obs_before) else np.zeros(23, dtype=np.float32)
                act = np.asarray(executed_actions[agent_id], dtype=np.float32)
                raw_act = np.asarray(actor_actions[agent_id], dtype=np.float32)
                def_geom = nearest_defender_geometry(env, off)
                r_vec = np.array([hvt.x - off.x, hvt.y - off.y, hvt.z - off.z], dtype=np.float64)
                rho = max(float(np.linalg.norm(r_vec)), 1e-6)
                v_vec = vel3d(off)
                closing = float(np.dot(r_vec, v_vec) / rho)
                bearing = math.atan2(hvt.y - off.y, hvt.x - off.x)
                heading_error = wrap_angle(bearing - off.heading)
                telemetry_rows.append({
                    "seed": seed,
                    "step": step + 1,
                    "time_s": (step + 1) * env.dt,
                    "team": "offense",
                    "agent": agent_id,
                    "x_m": float(off.x), "y_m": float(off.y), "z_m": float(off.z),
                    "speed_mps": float(off.v),
                    "heading_deg": float(np.degrees(off.heading)),
                    "gamma_deg": float(np.degrees(off.gamma)),
                    "heading_error_deg": float(np.degrees(heading_error)),
                    "dist_hvt_m": float(off.distance_to(hvt.x, hvt.y, hvt.z)),
                    "closing_hvt_mps": closing,
                    "alive": int(off.alive),
                    "hit_hvt": int(off.hit_hvt),
                    "terminal_phase": int(bool(terminal_flags[agent_id])),
                    "detected": int(getattr(off, "detected", False)),
                    "locked_by_count": int(getattr(off, "locked_by_count", 0)),
                    "nearest_def_id": int(def_geom["nearest_def_id"]),
                    "nearest_def_dist_m": float(def_geom["nearest_def_dist_m"]),
                    "nearest_def_hvt_dist_m": float(def_geom["nearest_def_hvt_dist_m"]),
                    "off_hvt_def_angle_deg": float(def_geom["off_hvt_def_angle_deg"]),
                    "hvt_angular_separation_deg": float(def_geom["hvt_angular_separation_deg"]),
                    "nearest_def_lateral_sep_m": float(def_geom["nearest_def_lateral_sep_m"]),
                    "nearest_def_along_los_m": float(def_geom["nearest_def_along_los_m"]),
                    "terminal_dominance_margin_m": float(def_geom["terminal_dominance_margin_m"]),
                    "action_ax": float(act[0]),
                    "action_pitch": float(act[1]),
                    "action_yaw": float(act[2]),
                    "actor_action_ax": float(raw_act[0]),
                    "actor_action_pitch": float(raw_act[1]),
                    "actor_action_yaw": float(raw_act[2]),
                    "ax_mps2": float(getattr(off, "ax", 0.0)),
                    "an_pitch_g": float(getattr(off, "an_pitch", 0.0) / G0),
                    "an_yaw_g": float(getattr(off, "an_yaw", 0.0) / G0),
                    "obs_los_az": float(obs_i[5]) if len(obs_i) > 6 else 0.0,
                    "obs_los_el": float(obs_i[6]) if len(obs_i) > 6 else 0.0,
                    "pn_cmd_pitch": float(np.clip(-args.pn_gain * float(obs_i[6]), -args.pn_max_action, args.pn_max_action)),
                    "pn_cmd_yaw": float(np.clip(-args.pn_gain * float(obs_i[5]), -args.pn_max_action, args.pn_max_action)),
                    "role_decoy": list_value(decoy, "role_decoy_per_agent", agent_id),
                    "role_penetrate": list_value(decoy, "role_penetrate_per_agent", agent_id),
                    "role_stealth": list_value(decoy, "role_stealth_per_agent", agent_id),
                    "lock_pressure": list_value(decoy, "lock_pressure_per_agent", agent_id),
                    "P_pen": list_value(pen, "P_pen_per_agent", agent_id),
                    "P_hit": list_value(hvt_info, "P_hit_per_agent", agent_id),
                    "E_esc": list_value(esc, "E_i_esc", agent_id),
                })
            for defender_id, defender in enumerate(env.defensives):
                telemetry_rows.append({
                    "seed": seed,
                    "step": step + 1,
                    "time_s": (step + 1) * env.dt,
                    "team": "defense",
                    "agent": defender_id,
                    "x_m": float(defender.x), "y_m": float(defender.y), "z_m": float(defender.z),
                    "speed_mps": float(defender.v),
                    "heading_deg": float(np.degrees(defender.heading)),
                    "gamma_deg": float(np.degrees(defender.gamma)),
                    "heading_error_deg": 0.0,
                    "dist_hvt_m": float(defender.distance_to(hvt.x, hvt.y, hvt.z)),
                    "closing_hvt_mps": 0.0,
                    "alive": int(defender.alive),
                    "hit_hvt": 0,
                    "terminal_phase": 0,
                    "detected": 0,
                    "locked_by_count": 0,
                    "nearest_def_id": -1,
                    "nearest_def_dist_m": 0.0,
                    "nearest_def_hvt_dist_m": float(defender.distance_to(hvt.x, hvt.y, hvt.z)),
                    "off_hvt_def_angle_deg": 0.0,
                    "hvt_angular_separation_deg": 0.0,
                    "nearest_def_lateral_sep_m": 0.0,
                    "nearest_def_along_los_m": 0.0,
                    "terminal_dominance_margin_m": 0.0,
                    "action_ax": 0.0, "action_pitch": 0.0, "action_yaw": 0.0,
                    "actor_action_ax": 0.0, "actor_action_pitch": 0.0, "actor_action_yaw": 0.0,
                    "ax_mps2": float(getattr(defender, "ax", 0.0)),
                    "an_pitch_g": float(getattr(defender, "an_pitch", 0.0) / G0),
                    "an_yaw_g": float(getattr(defender, "an_yaw", 0.0) / G0),
                    "obs_los_az": 0.0, "obs_los_el": 0.0,
                    "pn_cmd_pitch": 0.0, "pn_cmd_yaw": 0.0,
                    "role_decoy": 0.0, "role_penetrate": 0.0, "role_stealth": 0.0,
                    "lock_pressure": 0.0, "P_pen": 0.0, "P_hit": 0.0, "E_esc": 0.0,
                })

        if all(dones):
            break

    episode_summary = {
        "seed": seed,
        "hit_count": int(env.hit_count),
        "hit_indices": list(env.hit_indices),
        "final_step": final_step,
        "done_reason": final_info.get("done_reason", "unknown"),
        "min_dist_per_agent_m": [float(x) for x in min_d],
        "min_step_per_agent": [int(x) for x in min_step],
        "best_min_dist_m": float(min(min_d)),
        "success": bool(env.hit_count > 0),
    }
    return episode_summary, telemetry_rows, metric_rows


def write_csv(path: Path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_rows(path: Path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def farr(rows, key):
    return np.asarray([float(r[key]) for r in rows], dtype=np.float64)


def angle_between_deg(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    denom = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if denom <= 1e-9:
        return float("nan")
    value = float(np.clip(np.dot(vec_a, vec_b) / denom, -1.0, 1.0))
    return float(np.degrees(np.arccos(value)))


def nearest_defender_geometry(env, off) -> dict:
    hvt = env.hvt
    off_pos = np.array([off.x, off.y, off.z], dtype=np.float64)
    hvt_pos = np.array([hvt.x, hvt.y, hvt.z], dtype=np.float64)
    nearest_id = -1
    nearest_dist = float("inf")
    nearest_pos = None
    for defender_id, defender in enumerate(env.defensives):
        if not defender.alive:
            continue
        def_pos = np.array([defender.x, defender.y, defender.z], dtype=np.float64)
        dist = float(np.linalg.norm(def_pos - off_pos))
        if dist < nearest_dist:
            nearest_dist = dist
            nearest_id = defender_id
            nearest_pos = def_pos
    dist_hvt = float(np.linalg.norm(hvt_pos - off_pos))
    if nearest_pos is None:
        return {
            "nearest_def_id": -1,
            "nearest_def_dist_m": float("nan"),
            "nearest_def_hvt_dist_m": float("nan"),
            "off_hvt_def_angle_deg": float("nan"),
            "hvt_angular_separation_deg": float("nan"),
            "nearest_def_lateral_sep_m": float("nan"),
            "nearest_def_along_los_m": float("nan"),
            "terminal_dominance_margin_m": float("nan"),
        }
    to_hvt = hvt_pos - off_pos
    to_def = nearest_pos - off_pos
    line_norm = float(np.linalg.norm(to_hvt))
    if line_norm > 1e-9:
        los_unit = to_hvt / line_norm
        nearest_lateral = float(np.linalg.norm(np.cross(los_unit, to_def)))
        nearest_along = float(np.dot(to_def, los_unit))
    else:
        nearest_lateral = float("nan")
        nearest_along = float("nan")
    nearest_def_hvt_dist = float(np.linalg.norm(nearest_pos - hvt_pos))
    return {
        "nearest_def_id": nearest_id,
        "nearest_def_dist_m": nearest_dist,
        "nearest_def_hvt_dist_m": nearest_def_hvt_dist,
        "off_hvt_def_angle_deg": angle_between_deg(to_hvt, to_def),
        "hvt_angular_separation_deg": angle_between_deg(off_pos - hvt_pos, nearest_pos - hvt_pos),
        "nearest_def_lateral_sep_m": nearest_lateral,
        "nearest_def_along_los_m": nearest_along,
        "terminal_dominance_margin_m": nearest_def_hvt_dist - dist_hvt,
    }


def terminal_start_time(rows) -> float | None:
    for row in rows:
        if int(float(row.get("terminal_phase", 0))) > 0:
            return float(row["time_s"])
    return None


def shade_terminal(axis, rows, color: str = "#009E73"):
    start = terminal_start_time(rows)
    if start is not None and rows:
        axis.axvspan(start, float(rows[-1]["time_s"]), color=color, alpha=0.08, lw=0)


def row_at_time(rows, target_time: float):
    if not rows:
        return None
    return min(rows, key=lambda row: abs(float(row["time_s"]) - target_time))


def rows_in_window(rows, center_time: float, before: float, after: float):
    lo = center_time - before
    hi = center_time + after
    selected = [row for row in rows if lo <= float(row["time_s"]) <= hi]
    if selected:
        return selected
    return rows[-min(len(rows), 120):]


def breach_time(rows_hit):
    start = terminal_start_time(rows_hit)
    if start is not None:
        return start
    finite = [row for row in rows_hit if np.isfinite(float(row.get("terminal_dominance_margin_m", "nan")))]
    if finite:
        return float(min(finite, key=lambda row: abs(float(row["terminal_dominance_margin_m"]))) ["time_s"])
    return float(rows_hit[-1]["time_s"]) if rows_hit else 0.0


def closest_defender_rows(tele_seed, rows_hit, center_time: float):
    breach_row = row_at_time(rows_hit, center_time)
    def_id = int(float(breach_row.get("nearest_def_id", -1))) if breach_row is not None else -1
    if def_id < 0:
        candidates = [r for r in tele_seed if r["team"] == "defense"]
        if not candidates:
            return -1, []
        hrow = breach_row or rows_hit[-1]
        hx, hy, hz = float(hrow["x_m"]), float(hrow["y_m"]), float(hrow["z_m"])
        nearest = min(candidates, key=lambda r: (float(r["x_m"]) - hx) ** 2 + (float(r["y_m"]) - hy) ** 2 + (float(r["z_m"]) - hz) ** 2)
        def_id = int(nearest["agent"])
    return def_id, [r for r in tele_seed if r["team"] == "defense" and int(r["agent"]) == def_id]


def annotate_time_markers(axis, rows, times, color: str, marker: str, prefix: str):
    for t_item in times:
        row = row_at_time(rows, t_item)
        if row is None:
            continue
        x = float(row["x_m"])
        y = float(row["y_m"])
        axis.scatter([x], [y], s=28, marker=marker, color=color, edgecolor="k", linewidth=0.25, zorder=6)
        axis.text(x, y, f"{prefix}{t_item:.1f}s", fontsize=5.7, color=color,
                  ha="left", va="bottom", clip_on=True)


def set_3d_view(ax, rows, pad: float = 80.0):
    xs = farr(rows, "x_m")
    ys = farr(rows, "y_m")
    zs = farr(rows, "z_m")
    finite = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(zs)
    if not np.any(finite):
        return
    xs, ys, zs = xs[finite], ys[finite], zs[finite]
    ax.set_xlim(float(np.min(xs) - pad), float(np.max(xs) + pad))
    ax.set_ylim(float(np.min(ys) - pad), float(np.max(ys) + pad))
    z_min = min(float(np.min(zs) - pad * 0.35), -40.0)
    z_max = max(float(np.max(zs) + pad * 0.35), 360.0)
    ax.set_zlim(z_min, z_max)
    ax.set_box_aspect((2.6, 1.05, 0.72))
    ax.grid(True, alpha=0.28)
    ax.xaxis.pane.set_alpha(0.03)
    ax.yaxis.pane.set_alpha(0.03)
    ax.zaxis.pane.set_alpha(0.03)


def add_direction_arrows(ax, rows, color: str, length: float, count: int = 3):
    if len(rows) < 3:
        return
    indices = np.linspace(0, len(rows) - 2, count, dtype=int)
    for idx in indices:
        p0 = np.array([float(rows[idx]["x_m"]), float(rows[idx]["y_m"]), float(rows[idx]["z_m"])], dtype=np.float64)
        p1 = np.array([float(rows[min(idx + 2, len(rows) - 1)]["x_m"]), float(rows[min(idx + 2, len(rows) - 1)]["y_m"]), float(rows[min(idx + 2, len(rows) - 1)]["z_m"])], dtype=np.float64)
        direction = p1 - p0
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-9:
            continue
        direction = direction / norm * length
        ax.quiver(p0[0], p0[1], p0[2], direction[0], direction[1], direction[2],
                  color=color, linewidth=0.8, arrow_length_ratio=0.32, alpha=0.95)


def draw_hit_sphere(ax, radius: float = 5.0):
    u = np.linspace(0, 2 * np.pi, 32)
    v = np.linspace(0, np.pi, 12)
    x = 1200 + radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(x, y, z, color="0.15", linewidth=0.35, alpha=0.55)


def add_2d_arrows(axis, xs: np.ndarray, ys: np.ndarray, color: str, count: int = 3):
    if len(xs) < 3:
        return
    indices = np.linspace(0, len(xs) - 2, count, dtype=int)
    for idx in indices:
        end_idx = min(idx + max(2, len(xs) // 30), len(xs) - 1)
        axis.annotate(
            "",
            xy=(xs[end_idx], ys[end_idx]),
            xytext=(xs[idx], ys[idx]),
            arrowprops={"arrowstyle": "-|>", "lw": 0.75, "color": color, "shrinkA": 0, "shrinkB": 0},
        )


def set_ieee_style():
    plt.rcParams.update({
        "font.family": "DejaVu Serif",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "lines.linewidth": 1.25,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.4,
        "figure.dpi": 180,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save_fig(fig, out_dir: Path, name: str):
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def plot_figures(out_dir: Path, summary: dict):
    set_ieee_style()
    telemetry = load_rows(out_dir / "telemetry.csv")
    metrics = load_rows(out_dir / "metrics.csv")
    if not telemetry or not metrics:
        return []
    best_ep = None
    for ep in summary["episodes"]:
        if ep["success"]:
            best_ep = ep
            break
    if best_ep is None:
        best_ep = min(summary["episodes"], key=lambda item: item["best_min_dist_m"])
    seed = int(best_ep["seed"])
    hit_agent = int(best_ep["hit_indices"][0]) if best_ep["hit_indices"] else int(np.argmin(best_ep["min_dist_per_agent_m"]))
    tele_seed = [r for r in telemetry if int(r["seed"]) == seed]
    met_seed = [r for r in metrics if int(r["seed"]) == seed]
    figures = []

    # Direction-aware 3D trajectory with terminal top-view and altitude insets.
    fig = plt.figure(figsize=(7.16, 3.42), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=[1.28, 1.0], height_ratios=[1.0, 0.86])
    ax = fig.add_subplot(grid[:, 0], projection="3d")
    ax_xy = fig.add_subplot(grid[0, 1])
    ax_xz = fig.add_subplot(grid[1, 1])
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    ax.scatter([1200], [0], [0], marker="*", s=70, color="k", label="HVT", depthshade=False)
    ax.plot([-1200, 1200], [0, 0], [0, 0], linestyle=":", color="0.55", linewidth=0.75)
    draw_hit_sphere(ax, radius=5.0)
    ax_xy.scatter([1200], [0], marker="*", s=58, color="k", label="HVT", zorder=6)
    ax_xy.add_patch(plt.Circle((1200, 0), 5.0, fill=False, edgecolor="0.15", linestyle=":", linewidth=0.75))
    ax_xz.scatter([1200], [0], marker="*", s=54, color="k", zorder=6)
    ax_xz.axhline(0.0, color="0.55", linestyle=":", linewidth=0.75)
    for agent_id in range(4):
        rows = [r for r in tele_seed if r["team"] == "offense" and int(r["agent"]) == agent_id]
        if not rows:
            continue
        color = colors[agent_id]
        is_strike = agent_id == hit_agent
        label = f"A{agent_id}" + (" strike" if is_strike else "")
        width = 2.1 if is_strike else 1.05
        alpha = 0.98 if is_strike else 0.78
        ax.plot(farr(rows, "x_m"), farr(rows, "y_m"), farr(rows, "z_m"), color=color, lw=width, alpha=alpha, label=label)
        ax.scatter([float(rows[0]["x_m"])], [float(rows[0]["y_m"])], [float(rows[0]["z_m"])],
                   marker="^", s=30, color=color, edgecolor="k", linewidth=0.25, depthshade=False)
        terminal_rows = [r for r in rows if int(float(r["terminal_phase"])) > 0]
        if terminal_rows:
            ax.scatter([float(terminal_rows[0]["x_m"])], [float(terminal_rows[0]["y_m"])], [float(terminal_rows[0]["z_m"])],
                       marker="s", s=22, color=color, edgecolor="k", linewidth=0.25, depthshade=False)
        hit_rows = [r for r in rows if int(float(r["hit_hvt"])) > 0]
        end_row = hit_rows[0] if hit_rows else rows[-1]
        ax.scatter([float(end_row["x_m"])], [float(end_row["y_m"])], [float(end_row["z_m"])],
                   marker="*" if hit_rows else "o", s=64 if hit_rows else 22, color=color,
                   edgecolor="k", linewidth=0.3, depthshade=False)
        add_direction_arrows(ax, rows, color, length=115.0, count=3)

        rows_near = [r for r in rows if float(r["dist_hvt_m"]) <= 650.0 or int(float(r["terminal_phase"])) > 0]
        if not rows_near:
            rows_near = rows[-min(len(rows), 80):]
        x_near = farr(rows_near, "x_m")
        y_near = farr(rows_near, "y_m")
        z_near = farr(rows_near, "z_m")
        ax_xy.plot(x_near, y_near, color=color, lw=width, alpha=alpha, label=label)
        ax_xy.scatter([x_near[0]], [y_near[0]], marker="^", s=22, color=color, edgecolor="k", linewidth=0.25, zorder=5)
        ax_xy.scatter([float(end_row["x_m"])], [float(end_row["y_m"])], marker="*" if hit_rows else "o",
                      s=52 if hit_rows else 18, color=color, edgecolor="k", linewidth=0.3, zorder=6)
        add_2d_arrows(ax_xy, x_near, y_near, color, count=2)
        ax_xz.plot(x_near, z_near, color=color, lw=width, alpha=alpha)
        ax_xz.scatter([x_near[0]], [z_near[0]], marker="^", s=20, color=color, edgecolor="k", linewidth=0.25, zorder=5)
        ax_xz.scatter([float(end_row["x_m"])], [float(end_row["z_m"])], marker="*" if hit_rows else "o",
                      s=48 if hit_rows else 18, color=color, edgecolor="k", linewidth=0.3, zorder=6)
        add_2d_arrows(ax_xz, x_near, z_near, color, count=2)
    for agent_id in range(4):
        rows = [r for r in tele_seed if r["team"] == "defense" and int(r["agent"]) == agent_id]
        if not rows:
            continue
        ax.plot(farr(rows, "x_m"), farr(rows, "y_m"), farr(rows, "z_m"), "--", color="0.34", lw=0.85, alpha=0.62,
                label="Defenders" if agent_id == 0 else None)
        ax.scatter([float(rows[0]["x_m"])], [float(rows[0]["y_m"])], [float(rows[0]["z_m"])],
                   marker="x", s=24, color="0.25", depthshade=False)
        add_direction_arrows(ax, rows, "0.34", length=85.0, count=2)
        rows_near = [r for r in rows if float(r["dist_hvt_m"]) <= 700.0]
        if rows_near:
            x_near = farr(rows_near, "x_m")
            y_near = farr(rows_near, "y_m")
            z_near = farr(rows_near, "z_m")
            ax_xy.plot(x_near, y_near, "--", color="0.34", lw=0.85, alpha=0.62,
                       label="Defenders" if agent_id == 0 else None)
            ax_xz.plot(x_near, z_near, "--", color="0.34", lw=0.85, alpha=0.62)
            add_2d_arrows(ax_xy, x_near, y_near, "0.34", count=2)
            add_2d_arrows(ax_xz, x_near, z_near, "0.34", count=2)
    set_3d_view(ax, tele_seed, pad=95.0)
    ax.set_proj_type("ortho")
    ax.view_init(elev=22, azim=-57)
    ax.set_xlabel("X (m)", labelpad=-2)
    ax.set_ylabel("Y (m)", labelpad=-2)
    ax.set_zlabel("Z (m)", labelpad=-4)
    ax.tick_params(labelsize=6.3, pad=0)
    ax.set_title("(a) Full 3-D engagement", pad=2)
    ax_xy.set_xlim(640, 1235)
    ax_xy.set_ylim(-330, 330)
    ax_xy.set_aspect("equal", adjustable="box")
    ax_xy.set_ylabel("Y (m)")
    ax_xy.set_title("(b) Terminal top view", pad=2)
    ax_xz.set_xlim(640, 1235)
    ax_xz.set_ylim(-30, 360)
    ax_xz.set_xlabel("X (m)")
    ax_xz.set_ylabel("Z (m)")
    ax_xz.set_title("(c) Terminal altitude profile", pad=2)
    for axis in (ax_xy, ax_xz):
        axis.grid(True, alpha=0.32)
        axis.tick_params(labelsize=6.5)
    ax.legend(loc="upper left", bbox_to_anchor=(-0.02, 1.02), ncol=2, frameon=False,
              handlelength=1.55, columnspacing=0.85, fontsize=6.6)
    save_fig(fig, out_dir, "fig01_trajectory_3d")
    figures.append("fig01_trajectory_3d")

    # Distance and phase process.
    fig, ax = plt.subplots(figsize=(3.45, 2.25))
    t = farr(met_seed, "time_s")
    ax.plot(t, farr(met_seed, "min_off_hvt_dist_m"), label="Min attacker-HVT range", color="#0072B2")
    ax.plot(t, farr(met_seed, "nearest_def_hvt_dist_m"), label="Nearest defender-HVT range", color="#D55E00")
    ax.axhline(5.0, color="k", linestyle=":", label="Hit radius")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Range (m)")
    ax.grid(True, alpha=0.35)
    ax2 = ax.twinx()
    ax2.step(t, farr(met_seed, "terminal_pn_guided_agents"), where="post", color="#009E73", alpha=0.65, label="PN-guided agents")
    ax2.set_ylabel("Guided count")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, frameon=False, loc="upper right")
    ax.set_title("Phase Boundary and Hit Process")
    save_fig(fig, out_dir, "fig02_distance_phase")
    figures.append("fig02_distance_phase")

    rows_hit = [r for r in tele_seed if r["team"] == "offense" and int(r["agent"]) == hit_agent]
    th = farr(rows_hit, "time_s")
    # Kinematics and acceleration.
    fig, axes = plt.subplots(3, 1, figsize=(3.45, 3.35), sharex=True)
    axes[0].plot(th, farr(rows_hit, "speed_mps"), color="#0072B2")
    axes[0].set_ylabel("Speed (m/s)")
    axes[1].plot(th, farr(rows_hit, "heading_error_deg"), color="#D55E00")
    axes[1].set_ylabel("Heading error (deg)")
    axes[2].plot(th, farr(rows_hit, "an_pitch_g"), label="Pitch normal accel", color="#009E73")
    axes[2].plot(th, farr(rows_hit, "an_yaw_g"), label="Yaw normal accel", color="#CC79A7")
    axes[2].set_ylabel("Accel. (g)")
    axes[2].set_xlabel("Time (s)")
    for axis in axes:
        shade_terminal(axis, rows_hit)
        axis.grid(True, alpha=0.35)
    axes[2].legend(frameon=False, loc="best")
    axes[0].set_title(f"Strike Agent A{hit_agent}: Kinematics and Acceleration")
    save_fig(fig, out_dir, "fig03_speed_heading_accel")
    figures.append("fig03_speed_heading_accel")

    # LOS and terminal PN commands.
    fig, axes = plt.subplots(3, 1, figsize=(3.45, 3.35), sharex=True)
    axes[0].plot(th, farr(rows_hit, "obs_los_az"), label="Azimuth LOS rate", color="#0072B2")
    axes[0].plot(th, farr(rows_hit, "obs_los_el"), label="Elevation LOS rate", color="#D55E00")
    axes[0].set_ylabel("Normalized LOS rate")
    axes[1].plot(th, farr(rows_hit, "action_pitch"), label="Executed pitch", color="#009E73")
    axes[1].plot(th, farr(rows_hit, "pn_cmd_pitch"), "--", label="PN pitch command", color="#009E73")
    axes[1].plot(th, farr(rows_hit, "action_yaw"), label="Executed yaw", color="#CC79A7")
    axes[1].plot(th, farr(rows_hit, "pn_cmd_yaw"), "--", label="PN yaw command", color="#CC79A7")
    axes[1].set_ylabel("Action")
    axes[2].plot(th, farr(rows_hit, "dist_hvt_m"), color="k")
    axes[2].axhline(5.0, linestyle=":", color="0.25")
    axes[2].set_ylabel("Range (m)")
    axes[2].set_xlabel("Time (s)")
    for axis in axes:
        shade_terminal(axis, rows_hit)
        axis.grid(True, alpha=0.35)
    axes[0].legend(frameon=False, loc="best")
    axes[1].legend(frameon=False, loc="best", ncol=2)
    axes[0].set_title("LOS-Rate Terminal Guidance")
    save_fig(fig, out_dir, "fig04_los_guidance")
    figures.append("fig04_los_guidance")

    # Game-theoretic prior metrics.
    fig, axes = plt.subplots(3, 1, figsize=(3.45, 3.45), sharex=True)
    axes[0].plot(t, farr(met_seed, "Phi_decoy"), label="Decoy potential", color="#0072B2")
    axes[0].plot(t, farr(met_seed, "N_eff"), label="Effective penetration", color="#D55E00")
    axes[0].set_ylabel("Potential / score")
    axes[1].plot(th, farr(rows_hit, "role_decoy"), label="Decoy role", color="#0072B2")
    axes[1].plot(th, farr(rows_hit, "role_penetrate"), label="Penetration role", color="#D55E00")
    axes[1].plot(th, farr(rows_hit, "lock_pressure"), label="Lock pressure", color="#009E73")
    axes[1].set_ylabel("Role / pressure")
    axes[2].plot(t, farr(met_seed, "Gamma_mean"), label="Tracking mismatch", color="#CC79A7")
    axes[2].plot(t, farr(met_seed, "Xi_mean"), label="Escape trigger", color="#009E73")
    axes[2].set_ylabel("Prior metric")
    axes[2].set_xlabel("Time (s)")
    for axis in axes:
        axis.grid(True, alpha=0.35)
        axis.legend(frameon=False, loc="best")
    axes[0].set_title("Game-Theoretic Prior Process")
    save_fig(fig, out_dir, "fig05_game_metrics")
    figures.append("fig05_game_metrics")

    # Breach geometry: angular separation, defender clearance, and role pressure.
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.08), sharex=True, constrained_layout=True)
    ax_range, ax_clear, ax_prior, ax_pressure = axes.ravel()
    ax_range.plot(th, farr(rows_hit, "dist_hvt_m"), label="Strike-HVT", color="#0072B2")
    ax_range.plot(th, farr(rows_hit, "nearest_def_hvt_dist_m"), label="Nearest defender-HVT", color="#D55E00")
    ax_range.axhline(5.0, linestyle=":", color="0.25", label="5 m hit radius")
    ax_range.set_yscale("log")
    ax_range.set_ylabel("Range (m)")
    ax_range.set_title("(a) Terminal race to HVT", pad=2)
    ax_range.legend(frameon=False, loc="lower left", fontsize=6.5)

    ax_clear.plot(th, farr(rows_hit, "nearest_def_dist_m"), label="Nearest defender", color="#0072B2")
    ax_clear.plot(th, farr(rows_hit, "nearest_def_lateral_sep_m"), label="Lateral bypass", color="#D55E00")
    ax_clear.set_ylabel("Distance (m)")
    ax_clear_angle = ax_clear.twinx()
    ax_clear_angle.plot(th, farr(rows_hit, "hvt_angular_separation_deg"), label="HVT-view separation", color="#CC79A7")
    ax_clear_angle.plot(th, farr(rows_hit, "off_hvt_def_angle_deg"), "--", label="Off-HVT-def angle", color="#009E73")
    ax_clear_angle.set_ylabel("Angle (deg)")
    lines, labels = ax_clear.get_legend_handles_labels()
    lines2, labels2 = ax_clear_angle.get_legend_handles_labels()
    ax_clear.legend(lines + lines2, labels + labels2, frameon=False, loc="upper right", fontsize=6.2)
    ax_clear.set_title("(b) Clearance and angular separation", pad=2)

    ax_prior.plot(th, farr(rows_hit, "P_pen"), label="Penetration prior", color="#0072B2")
    ax_prior.plot(th, farr(rows_hit, "P_hit"), label="Hit prior", color="#D55E00")
    ax_prior.plot(th, farr(rows_hit, "E_esc"), label="Escape prior", color="#009E73")
    ax_prior.plot(th, farr(rows_hit, "role_penetrate"), "--", label="Penetration role", color="#CC79A7")
    ax_prior.set_ylabel("Prior / role")
    ax_prior.set_xlabel("Time (s)")
    ax_prior.set_ylim(-0.05, 1.08)
    ax_prior.legend(frameon=False, loc="best", fontsize=6.4, ncol=2)
    ax_prior.set_title("(c) Game-theoretic role process", pad=2)

    ax_pressure.plot(th, farr(rows_hit, "lock_pressure"), label="Lock pressure", color="#009E73")
    ax_pressure.plot(th, farr(rows_hit, "locked_by_count"), label="Locked-by count", color="#0072B2")
    ax_pressure.set_ylabel("Pressure / count")
    ax_pressure_adv = ax_pressure.twinx()
    ax_pressure_adv.plot(th, farr(rows_hit, "terminal_dominance_margin_m"), label="Terminal advantage", color="#D55E00")
    ax_pressure_adv.axhline(0.0, linestyle="--", color="0.45", linewidth=0.75)
    ax_pressure_adv.set_ylabel("Advantage (m)")
    lines, labels = ax_pressure.get_legend_handles_labels()
    lines2, labels2 = ax_pressure_adv.get_legend_handles_labels()
    ax_pressure.legend(lines + lines2, labels + labels2, frameon=False, loc="best", fontsize=6.4)
    ax_pressure.set_xlabel("Time (s)")
    ax_pressure.set_title("(d) Pressure sharing and terminal advantage", pad=2)
    for axis in (ax_range, ax_clear, ax_prior, ax_pressure):
        shade_terminal(axis, rows_hit)
        axis.grid(True, alpha=0.32)
        axis.tick_params(labelsize=6.5)
    save_fig(fig, out_dir, "fig06_breach_geometry")
    figures.append("fig06_breach_geometry")

    # Breach-instant local time sequence: independent zoom around the geometry switch.
    t_breach = breach_time(rows_hit)
    hit_time = float(rows_hit[-1]["time_s"])
    def_id, rows_def = closest_defender_rows(tele_seed, rows_hit, t_breach)
    local_after = max(8.0, min(16.0, hit_time - t_breach + 1.5))
    hit_window = rows_in_window(rows_hit, t_breach, before=8.0, after=local_after)
    def_window = rows_in_window(rows_def, t_breach, before=8.0, after=local_after) if rows_def else []
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.72), constrained_layout=True)
    ax_local, ax_state = axes
    x_hit = farr(hit_window, "x_m")
    y_hit = farr(hit_window, "y_m")
    ax_local.plot(x_hit, y_hit, color="#0072B2", lw=2.0, label=f"A{hit_agent} strike")
    add_2d_arrows(ax_local, x_hit, y_hit, "#0072B2", count=4)
    if def_window:
        x_def = farr(def_window, "x_m")
        y_def = farr(def_window, "y_m")
        ax_local.plot(x_def, y_def, "--", color="#D55E00", lw=1.55, label=f"D{def_id} nearest")
        add_2d_arrows(ax_local, x_def, y_def, "#D55E00", count=4)
    ax_local.scatter([1200], [0], marker="*", s=70, color="k", label="HVT", zorder=7)
    ax_local.add_patch(plt.Circle((1200, 0), 5.0, fill=False, edgecolor="0.15", linestyle=":", linewidth=0.75))
    marker_times = [max(float(hit_window[0]["time_s"]), t_breach - 6.0), t_breach,
                    min(float(hit_window[-1]["time_s"]), t_breach + 4.0),
                    min(float(hit_window[-1]["time_s"]), hit_time)]
    marker_times = sorted(set(round(t_item, 1) for t_item in marker_times))
    annotate_time_markers(ax_local, hit_window, marker_times, "#0072B2", "o", "")
    if def_window:
        annotate_time_markers(ax_local, def_window, marker_times, "#D55E00", "s", "")
    hit_breach_row = row_at_time(rows_hit, t_breach)
    def_breach_row = row_at_time(rows_def, t_breach) if rows_def else None
    if hit_breach_row is not None:
        ax_local.plot([1200, float(hit_breach_row["x_m"])], [0, float(hit_breach_row["y_m"])],
                      color="#0072B2", linestyle=":", lw=0.9)
    if def_breach_row is not None:
        ax_local.plot([1200, float(def_breach_row["x_m"])], [0, float(def_breach_row["y_m"])],
                      color="#D55E00", linestyle=":", lw=0.9)
    all_x = list(x_hit) + ([float(r["x_m"]) for r in def_window] if def_window else []) + [1200.0]
    all_y = list(y_hit) + ([float(r["y_m"]) for r in def_window] if def_window else []) + [0.0]
    pad = 42.0
    ax_local.set_xlim(min(all_x) - pad, max(all_x) + pad)
    ax_local.set_ylim(min(all_y) - pad, max(all_y) + pad)
    ax_local.set_aspect("equal", adjustable="box")
    ax_local.grid(True, alpha=0.32)
    ax_local.set_xlabel("X (m)")
    ax_local.set_ylabel("Y (m)")
    ax_local.set_title("(a) Time-stamped breach geometry", pad=2)
    ax_local.legend(frameon=False, loc="best", fontsize=6.2)

    tw = farr(hit_window, "time_s")
    ax_state.plot(tw, farr(hit_window, "terminal_dominance_margin_m"), color="#0072B2", label="Terminal advantage")
    ax_state.axhline(0.0, linestyle="--", color="0.45", linewidth=0.8)
    ax_state.axvline(t_breach, linestyle=":", color="k", linewidth=0.85, label="Breach instant")
    ax_state.set_ylabel("Advantage (m)")
    ax_state_angle = ax_state.twinx()
    ax_state_angle.plot(tw, farr(hit_window, "hvt_angular_separation_deg"), color="#D55E00", label="HVT-view angle")
    ax_state_angle.plot(tw, farr(hit_window, "off_hvt_def_angle_deg"), "--", color="#009E73", label="Off-HVT-def angle")
    ax_state_angle.set_ylabel("Angle (deg)")
    ax_state.set_xlabel("Time (s)")
    lines, labels = ax_state.get_legend_handles_labels()
    lines2, labels2 = ax_state_angle.get_legend_handles_labels()
    ax_state.legend(lines + lines2, labels + labels2, frameon=False, loc="best", fontsize=6.2)
    ax_state.grid(True, alpha=0.32)
    ax_state.set_title("(b) Breach criterion and angle change", pad=2)
    save_fig(fig, out_dir, "fig07_breach_moment_detail")
    figures.append("fig07_breach_moment_detail")

    # Strike aircraft versus interceptor range: closure before breach and opening after pass-through.
    fig, axes = plt.subplots(2, 1, figsize=(3.45, 3.25), sharex=True, constrained_layout=True)
    ax_dist, ax_margin = axes
    nearest_range = farr(rows_hit, "nearest_def_dist_m")
    full_time = farr(rows_hit, "time_s")
    finite_range = np.isfinite(nearest_range)
    if np.any(finite_range):
        finite_indices = np.where(finite_range)[0]
        closest_idx = finite_indices[int(np.argmin(nearest_range[finite_range]))]
        closest_time = float(full_time[closest_idx])
    else:
        closest_time = t_breach
    ax_dist.plot(full_time, nearest_range, color="#0072B2", label="A-D range")
    ax_dist.axvline(closest_time, color="#D55E00", linestyle="--", linewidth=0.85, label="Closest approach")
    ax_dist.axvline(t_breach, color="k", linestyle=":", linewidth=0.85, label="Breach")
    ax_dist.axvline(hit_time, color="#009E73", linestyle="-.", linewidth=0.85, label="HVT hit")
    ax_dist.set_ylabel("Range (m)")
    ax_dist.grid(True, alpha=0.32)
    ax_dist.legend(frameon=False, loc="best", fontsize=6.2, ncol=2)
    ax_dist.set_title(f"Strike Agent A{hit_agent} vs. Nearest Interceptor", pad=2)

    if len(full_time) > 2:
        opening_rate = np.gradient(nearest_range, full_time)
    else:
        opening_rate = np.zeros_like(nearest_range)
    ax_margin.plot(full_time, farr(rows_hit, "terminal_dominance_margin_m"), color="#CC79A7", label="HVT race margin")
    ax_margin.axhline(0.0, color="0.45", linestyle="--", linewidth=0.75)
    ax_rate = ax_margin.twinx()
    ax_rate.plot(full_time, opening_rate, color="#009E73", alpha=0.72, label="Range opening rate")
    ax_rate.axhline(0.0, color="0.75", linestyle=":", linewidth=0.7)
    ax_margin.axvline(t_breach, color="k", linestyle=":", linewidth=0.85)
    ax_margin.axvline(hit_time, color="#009E73", linestyle="-.", linewidth=0.85)
    ax_margin.set_ylabel("Margin (m)")
    ax_rate.set_ylabel("dRange/dt (m/s)")
    ax_margin.set_xlabel("Time (s)")
    lines, labels = ax_margin.get_legend_handles_labels()
    lines2, labels2 = ax_rate.get_legend_handles_labels()
    ax_margin.legend(lines + lines2, labels + labels2, frameon=False, loc="best", fontsize=6.2)
    ax_margin.grid(True, alpha=0.32)
    save_fig(fig, out_dir, "fig08_strike_interceptor_range")
    figures.append("fig08_strike_interceptor_range")
    return figures


def main():
    args = parse_args()
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    _, _, scenario_info = scenario_spec(args.scenario_case)
    scenario_label = args.case_label.strip() or scenario_info["label"]
    model_dir = (PROJECT_ROOT / args.model_dir).resolve() if not os.path.isabs(args.model_dir) else Path(args.model_dir)
    if not model_complete(model_dir):
        raise SystemExit(f"Incomplete model directory: {model_dir}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = args.tag.strip() or "latest"
    out_dir = PROJECT_ROOT / args.out_root / f"{timestamp}_{tag}_{args.scenario_case}"
    out_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("FOV_REWARD_PROFILE", "v68strictpnfix")
    os.environ.setdefault("FOV_OBS_PHASE_MASK", args.obs_mask)
    os.environ.setdefault("FOV_TERMINAL_GUIDANCE", args.terminal_guidance)
    os.environ.setdefault("FOV_TERMINAL_PN_GAIN", str(args.pn_gain))
    os.environ.setdefault("FOV_TERMINAL_PN_MAX_ACTION", str(args.pn_max_action))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_env = make_env(args)
    policies = load_policies(str(model_dir), base_env, device, hidden_size=HIDDEN, layer_N=LAYER_N)

    all_telemetry = []
    all_metrics = []
    episodes = []
    for seed in seeds:
        episode_summary, telemetry, metrics = collect_episode(args, policies, device, seed, out_dir)
        episodes.append(episode_summary)
        all_telemetry.extend(telemetry)
        all_metrics.extend(metrics)

    write_csv(out_dir / "telemetry.csv", all_telemetry)
    write_csv(out_dir / "metrics.csv", all_metrics)
    summary = {
        "timestamp": timestamp,
        "tag": tag,
        "model_dir": args.model_dir,
        "reward_profile": os.environ.get("FOV_REWARD_PROFILE", ""),
        "obs_mask": args.obs_mask,
        "terminal_guidance": args.terminal_guidance,
        "terminal_pn_gain": args.pn_gain,
        "terminal_pn_max_action": args.pn_max_action,
        "scenario_case": args.scenario_case,
        "scenario_label": scenario_label,
        "scenario_info": scenario_info,
        "seeds": seeds,
        "episodes": episodes,
        "success_episodes": int(sum(1 for ep in episodes if ep["success"])),
        "total_hits": int(sum(ep["hit_count"] for ep in episodes)),
        "hit_rate": float(np.mean([1.0 if ep["success"] else 0.0 for ep in episodes])),
        "best_min_dist_m": float(min(ep["best_min_dist_m"] for ep in episodes)),
    }
    figures = [] if args.no_plots else plot_figures(out_dir, summary)
    summary["figures"] = figures
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    latest_eval_link = PROJECT_ROOT / args.out_root / "latest_eval"
    if latest_eval_link.exists() or latest_eval_link.is_symlink():
        latest_eval_link.unlink()
    latest_eval_link.symlink_to(out_dir.resolve())
    if summary["success_episodes"] > 0:
        latest_success_link = PROJECT_ROOT / args.out_root / "latest_success"
        if latest_success_link.exists() or latest_success_link.is_symlink():
            latest_success_link.unlink()
        latest_success_link.symlink_to(out_dir.resolve())
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
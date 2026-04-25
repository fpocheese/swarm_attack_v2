"""Dump key TB scalars (last N steps + summary stats) for V38 diagnosis."""
import os, glob, sys
from tensorboard.backend.event_processing import event_accumulator

ROOT = sys.argv[1] if len(sys.argv) > 1 else "outputs/results/fov_penetration/mappo/v38_inertial_accel/run1/logs"

WANT = [
    "env/success", "env/hit_count", "env/n_escapes_total", "env/n_escaped_agents",
    "env/offensive_alive", "env/defensive_alive",
    "env/escape_reward", "env/cone_cost", "env/avg_cone_cost_per_agent",
    "env/two_stage_score",
    "env/Gamma_mean", "env/Gamma_max", "env/Xi_mean", "env/Xi_max",
    "env/Z_tilde_mean", "env/Z_tilde_max", "env/Z_ij_mean", "env/Z_ij_max",
    "env/hit_hvt", "env/near_trigger_count",
    "agent0/policy_loss", "agent0/value_loss", "agent0/dist_entropy",
    "agent0/actor_grad_norm", "agent0/ratio",
    "eval_average_episode_rewards", "eval_max_episode_rewards",
    "train_episode_rewards",
]

def find_evt(root, tag):
    """tag like env/success → look for events under root/env/success/**"""
    pat = os.path.join(root, tag, "**", "events.out.tfevents.*")
    files = glob.glob(pat, recursive=True)
    return files

def load_scalar(files):
    if not files:
        return None
    ea = event_accumulator.EventAccumulator(files[0], size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    if not tags:
        return None
    return ea.Scalars(tags[0])

def stats(events):
    if not events:
        return None
    vals = [e.value for e in events]
    n = len(vals)
    return {
        "n": n,
        "first": vals[0],
        "last": vals[-1],
        "min": min(vals),
        "max": max(vals),
        "mean": sum(vals) / n,
        "last5_mean": sum(vals[-5:]) / max(1, len(vals[-5:])),
    }

print(f"Reading from: {ROOT}\n")
print(f"{'TAG':45s} {'n':>4s} {'first':>10s} {'last':>10s} {'min':>10s} {'max':>10s} {'mean':>10s} {'last5':>10s}")
print("-"*120)
for tag in WANT:
    f = find_evt(ROOT, tag)
    ev = load_scalar(f)
    s = stats(ev)
    if s is None:
        print(f"{tag:45s}  (missing)")
    else:
        print(f"{tag:45s} {s['n']:4d} {s['first']:10.3f} {s['last']:10.3f} {s['min']:10.3f} {s['max']:10.3f} {s['mean']:10.3f} {s['last5_mean']:10.3f}")

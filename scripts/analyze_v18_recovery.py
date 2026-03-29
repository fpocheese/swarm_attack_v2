#!/usr/bin/env python
import argparse
import json
import os
import re

import matplotlib.pyplot as plt
import pandas as pd


def parse_training_log(train_log_path: str) -> pd.DataFrame:
    update_re = re.compile(r"updates\s+(\d+)/(\d+)\s+episodes, total num timesteps\s+(\d+)/(\d+)")
    step_reward_re = re.compile(r"average_step_rewards is\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")
    ep_rew_cost_re = re.compile(
        r"some episodes done, average rewards:\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?),\s+average costs:\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
    )
    eval_re = re.compile(r"eval_average_episode_rewards is\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")

    rows = []
    current = None

    with open(train_log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = update_re.search(line)
            if m:
                if current is not None:
                    rows.append(current)
                current = {
                    "update": int(m.group(1)),
                    "total_updates": int(m.group(2)),
                    "timesteps": int(m.group(3)),
                    "target_timesteps": int(m.group(4)),
                    "average_step_rewards": None,
                    "average_rewards": None,
                    "average_costs": None,
                    "eval_average_episode_rewards": None,
                }
                continue

            if current is None:
                continue

            m = step_reward_re.search(line)
            if m:
                current["average_step_rewards"] = float(m.group(1))
                continue

            m = ep_rew_cost_re.search(line)
            if m:
                current["average_rewards"] = float(m.group(1))
                current["average_costs"] = float(m.group(2))
                continue

            m = eval_re.search(line)
            if m:
                current["eval_average_episode_rewards"] = float(m.group(1))

    if current is not None:
        rows.append(current)

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No training entries parsed from: {train_log_path}")
    return df


def parse_eval_log(eval_log_path: str) -> dict:
    metrics = {}
    metric_patterns = {
        "policy": re.compile(r"Results \(([^,]+),\s*(\d+) eps\)"),
        "success_rate": re.compile(r"Success rate:\s+([\d.]+)%"),
        "attacker_killed_rate": re.compile(r"Attacker killed:\s+([\d.]+)%"),
        "timeout_rate": re.compile(r"Timeout:\s+([\d.]+)%"),
        "avg_reward": re.compile(r"Avg reward:\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"),
        "avg_cost": re.compile(r"Avg cost:\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"),
        "avg_length": re.compile(r"Avg length:\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"),
        "avg_escorts_alive": re.compile(r"Avg escorts alive:\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"),
        "avg_intc_alive": re.compile(r"Avg intc alive:\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"),
        "avg_escort_kills": re.compile(r"Avg escort kills:\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"),
    }

    with open(eval_log_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    m = metric_patterns["policy"].search(text)
    if m:
        metrics["policy"] = m.group(1).strip()
        metrics["episodes"] = int(m.group(2))

    for key, pattern in metric_patterns.items():
        if key == "policy":
            continue
        m = pattern.search(text)
        if m:
            val = float(m.group(1))
            if key.endswith("_rate"):
                val = val / 100.0
            metrics[key] = val

    if not metrics:
        raise RuntimeError(f"No eval metrics parsed from: {eval_log_path}")
    return metrics


def make_plots(df_train: pd.DataFrame, trained: dict, random_baseline: dict, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    axes[0, 0].plot(df_train["update"], df_train["average_rewards"], label="train avg reward")
    axes[0, 0].set_title("Training Average Reward")
    axes[0, 0].set_xlabel("Update")
    axes[0, 0].set_ylabel("Reward")
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(df_train["update"], df_train["average_costs"], color="tab:orange", label="train avg cost")
    axes[0, 1].set_title("Training Average Cost")
    axes[0, 1].set_xlabel("Update")
    axes[0, 1].set_ylabel("Cost")
    axes[0, 1].grid(alpha=0.3)

    eval_points = df_train.dropna(subset=["eval_average_episode_rewards"])
    axes[1, 0].plot(
        eval_points["update"],
        eval_points["eval_average_episode_rewards"],
        color="tab:green",
        marker="o",
        linewidth=1.5,
        markersize=3,
    )
    axes[1, 0].set_title("Periodic Eval Avg Episode Reward")
    axes[1, 0].set_xlabel("Update")
    axes[1, 0].set_ylabel("Eval Reward")
    axes[1, 0].grid(alpha=0.3)

    labels = ["Trained", "Random"]
    bar_vals_reward = [trained.get("avg_reward", 0.0), random_baseline.get("avg_reward", 0.0)]
    bar_vals_cost = [trained.get("avg_cost", 0.0), random_baseline.get("avg_cost", 0.0)]

    x = [0, 1]
    width = 0.35
    axes[1, 1].bar([i - width / 2 for i in x], bar_vals_reward, width=width, label="Avg reward")
    axes[1, 1].bar([i + width / 2 for i in x], bar_vals_cost, width=width, label="Avg cost")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(labels)
    axes[1, 1].set_title("Trained vs Random (40 eps)")
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "v18_recovery_analysis.png"), dpi=180)
    plt.close(fig)


def write_summary(df_train: pd.DataFrame, trained: dict, random_baseline: dict, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    last = df_train.iloc[-1]
    progress_ratio = float(last["timesteps"]) / float(last["target_timesteps"]) if float(last["target_timesteps"]) > 0 else 0.0

    summary = {
        "training_progress": {
            "last_update": int(last["update"]),
            "total_updates": int(last["total_updates"]),
            "timesteps_done": int(last["timesteps"]),
            "timesteps_target": int(last["target_timesteps"]),
            "progress_ratio": progress_ratio,
            "latest_average_rewards": float(last["average_rewards"]) if pd.notna(last["average_rewards"]) else None,
            "latest_average_costs": float(last["average_costs"]) if pd.notna(last["average_costs"]) else None,
            "latest_eval_average_episode_rewards": float(last["eval_average_episode_rewards"]) if pd.notna(last["eval_average_episode_rewards"]) else None,
        },
        "evaluation_trained": trained,
        "evaluation_random_baseline": random_baseline,
        "comparison": {
            "avg_reward_gain": float(trained.get("avg_reward", 0.0) - random_baseline.get("avg_reward", 0.0)),
            "avg_cost_delta": float(trained.get("avg_cost", 0.0) - random_baseline.get("avg_cost", 0.0)),
            "avg_length_gain": float(trained.get("avg_length", 0.0) - random_baseline.get("avg_length", 0.0)),
            "timeout_rate_delta": float(trained.get("timeout_rate", 0.0) - random_baseline.get("timeout_rate", 0.0)),
        },
    }

    with open(os.path.join(out_dir, "v18_recovery_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    lines = [
        "# V18 Recovery Analysis",
        "",
        "## Training Progress",
        f"- Last update: {summary['training_progress']['last_update']} / {summary['training_progress']['total_updates']}",
        f"- Timesteps: {summary['training_progress']['timesteps_done']} / {summary['training_progress']['timesteps_target']} ({summary['training_progress']['progress_ratio']:.1%})",
        f"- Latest train avg reward: {summary['training_progress']['latest_average_rewards']:.2f}",
        f"- Latest train avg cost: {summary['training_progress']['latest_average_costs']:.2f}",
        "",
        "## Evaluation (40 episodes)",
        f"- Trained avg reward: {trained.get('avg_reward', float('nan')):.2f}",
        f"- Trained avg cost: {trained.get('avg_cost', float('nan')):.2f}",
        f"- Trained timeout rate: {trained.get('timeout_rate', float('nan')):.1%}",
        f"- Random avg reward: {random_baseline.get('avg_reward', float('nan')):.2f}",
        f"- Random avg cost: {random_baseline.get('avg_cost', float('nan')):.2f}",
        f"- Reward gain (trained-random): {summary['comparison']['avg_reward_gain']:.2f}",
        "",
        "## Plot",
        "- v18_recovery_analysis.png",
    ]

    with open(os.path.join(out_dir, "v18_recovery_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze v18 recovery status and create plots")
    parser.add_argument("--train_log", required=True, help="Path to v18 training log")
    parser.add_argument("--trained_eval_log", required=True, help="Path to trained policy eval log")
    parser.add_argument("--random_eval_log", required=True, help="Path to random policy eval log")
    parser.add_argument("--out_dir", required=True, help="Output directory for analysis files")
    args = parser.parse_args()

    df_train = parse_training_log(args.train_log)
    trained = parse_eval_log(args.trained_eval_log)
    random_baseline = parse_eval_log(args.random_eval_log)

    os.makedirs(args.out_dir, exist_ok=True)
    df_train.to_csv(os.path.join(args.out_dir, "v18_train_progress.csv"), index=False)
    pd.DataFrame([trained, random_baseline]).to_csv(os.path.join(args.out_dir, "v18_eval_comparison.csv"), index=False)

    make_plots(df_train, trained, random_baseline, args.out_dir)
    write_summary(df_train, trained, random_baseline, args.out_dir)

    print("Analysis complete.")
    print(f"Output dir: {args.out_dir}")


if __name__ == "__main__":
    main()

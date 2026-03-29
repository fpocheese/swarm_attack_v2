#!/usr/bin/env python
"""持续监控训练进度 — 读取日志文件并输出关键指标"""
import sys, os, re, time, glob

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")

def parse_log(logfile):
    data = {"train": [], "eval": [], "fps": [], "last_step": 0, "total_steps": 10000000}
    with open(logfile, 'r') as f:
        for line in f:
            m = re.search(r'updates (\d+)/(\d+) episodes.*timesteps (\d+)/(\d+), FPS (\d+)', line)
            if m:
                data["last_step"] = int(m.group(3))
                data["total_steps"] = int(m.group(4))
                data["fps"].append(int(m.group(5)))
            m = re.search(r'average rewards: ([-\d.]+)', line)
            if m:
                data["train"].append(float(m.group(1)))
            m = re.search(r'eval_average_episode_rewards is ([-\d.eE+]+)', line)
            if m:
                try:
                    data["eval"].append(float(m.group(1).rstrip('.')))
                except ValueError:
                    pass
    return data

def main():
    patterns = ["v11c_safescale_*.log", "v11b_balanced_*.log", "v11_anticrash_*.log", "v10_balanced_*.log", "v10_optimized_*.log", "v10_train_*.log"]
    all_logs = []
    for p in patterns:
        all_logs.extend(glob.glob(os.path.join(LOG_DIR, p)))
    if not all_logs:
        print("No training logs found!")
        return
    logfile = max(all_logs, key=os.path.getmtime)
    print(f"Log: {os.path.basename(logfile)}")
    print("=" * 70)
    data = parse_log(logfile)
    if not data["train"]:
        print("No training data yet - still in warmup")
        return
    progress = data["last_step"] / data["total_steps"] * 100
    avg_fps = sum(data["fps"]) / len(data["fps"]) if data["fps"] else 0
    eta_h = (data["total_steps"] - data["last_step"]) / avg_fps / 3600 if avg_fps > 0 else 999
    print(f"Progress: {data['last_step']:>10,}/{data['total_steps']:>10,} ({progress:.1f}%)")
    print(f"FPS: {avg_fps:.0f}  |  ETA: {eta_h:.1f}h  |  Episodes: {len(data['train'])}")
    print(f"\nTrain Rewards (recent 10):")
    for i, r in enumerate(data["train"][-10:]):
        idx = len(data["train"]) - 10 + i if len(data["train"]) > 10 else i
        bar = "#" * max(0, int((r + 200) / 30))
        print(f"  Ep {idx:3d}: {r:>10.1f}  {bar}")
    if data["eval"]:
        print(f"\nEval Rewards:")
        for i, r in enumerate(data["eval"][-5:]):
            print(f"  Eval {i}: {r:.1f}")
    print("\n--- Diagnosis ---")
    latest = data["train"][-1]
    if latest > 1160:
        print("GOOD: Exceeds zero-action baseline (1160)")
    elif latest > 0:
        print("OK: Positive reward, below baseline (1160)")
    else:
        print("EARLY: Negative reward, needs more training")

if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""V36 Training Monitor - auto-refreshes every N seconds"""
import os, sys, time, datetime
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

BASE = os.path.join(os.path.dirname(__file__), '..', 
       'outputs/results/fov_penetration/mappo/v36_reward_scale_fix/run1/logs')
BASE = os.path.abspath(BASE)

METRICS = {
    'step_rwd':    'agent0/average_step_rewards/agent0/average_step_rewards',
    'train_rwd':   'train_episode_rewards/aver_rewards',
    'eval_avg':    'eval_average_episode_rewards/eval_average_episode_rewards',
    'eval_max':    'eval_max_episode_rewards/eval_max_episode_rewards',
    'hit_count':   'env/hit_count/env/hit_count',
    'success':     'env/success/env/success',
    'off_alive':   'env/offensive_alive/env/offensive_alive',
    'def_alive':   'env/defensive_alive/env/defensive_alive',
}

STEP_PER_UPDATE = 640000  # 80 threads * 8000 steps

def read_metric(name, subdir):
    full = os.path.join(BASE, subdir)
    if not os.path.exists(full):
        return []
    ea = EventAccumulator(full)
    ea.Reload()
    tags = ea.Tags().get('scalars', [])
    if not tags:
        return []
    return [(v.step // STEP_PER_UPDATE, v.value) for v in ea.Scalars(tags[0])]

def report():
    now = datetime.datetime.now().strftime('%H:%M:%S')
    print(f"\n{'='*60}")
    print(f"  V36 Training Monitor  |  {now}")
    print(f"{'='*60}")
    
    data = {}
    for name, subdir in METRICS.items():
        data[name] = read_metric(name, subdir)
    
    n_updates = len(data.get('step_rwd', []))
    total_steps = n_updates * STEP_PER_UPDATE / 1e6
    print(f"  Updates: {n_updates}  |  Steps: {total_steps:.1f}M / 200M")
    print()
    
    # Step rewards trend
    sr = data.get('step_rwd', [])
    if sr:
        vals = ' → '.join([f'{v:.2f}' for _, v in sr[-8:]])
        print(f"  step_reward:  {vals}")
    
    # Train rewards
    tr = data.get('train_rwd', [])
    if tr:
        vals = ' → '.join([f'{v:.0f}' for _, v in tr[-8:]])
        print(f"  train_reward: {vals}")
    
    # Eval rewards
    ev = data.get('eval_avg', [])
    if ev:
        vals = ' → '.join([f'u{u}={v:.0f}' for u, v in ev])
        print(f"  eval_reward:  {vals}")
    
    em = data.get('eval_max', [])
    if em:
        vals = ' → '.join([f'u{u}={v:.0f}' for u, v in em])
        print(f"  eval_max:     {vals}")
    
    # Hit/success
    hits = data.get('hit_count', [])
    succ = data.get('success', [])
    if hits:
        latest_hit = hits[-1][1]
        latest_succ = succ[-1][1] if succ else 0
        print(f"  hit_count:    {latest_hit:.2f}  |  success: {latest_succ:.2f}")
    
    # Alive counts
    oa = data.get('off_alive', [])
    da = data.get('def_alive', [])
    if oa:
        print(f"  off_alive:    {oa[-1][1]:.1f}  |  def_alive: {da[-1][1]:.1f}")
    
    print(f"{'='*60}")

if __name__ == '__main__':
    interval = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    report()
    if interval > 0:
        print(f"\n  Auto-refresh every {interval}s. Ctrl+C to stop.\n")
        while True:
            time.sleep(interval)
            report()

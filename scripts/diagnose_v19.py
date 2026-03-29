"""
V19 深度诊断脚本 - 使用eval脚本相同的模型加载方式
跑5个episode，记录每步详细数据，输出诊断图
"""
import sys, os, json
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "MACPO", "MACPO"))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter

from envs.fov_penetration import FOVPenetrationEnv

MODEL_DIR = os.path.join(PROJECT_ROOT, 'outputs/results/fov_penetration/macpo/v19_balanced_reward/run1/models')
OUT_DIR = os.path.join(PROJECT_ROOT, 'outputs/results/v19_diagnosis')
os.makedirs(OUT_DIR, exist_ok=True)

HIDDEN_SIZE = 256
LAYER_N = 3
N_EPS = 5
SEEDS = [42, 43, 44, 45, 46]

# ===== 加载模型 (same as eval script) =====
device = torch.device("cpu")
env = FOVPenetrationEnv()

from macpo.algorithms.r_mappo.algorithm.MACPPOPolicy import MACPPOPolicy
from macpo.config import get_config as macpo_get_config

parser = macpo_get_config()
all_args = parser.parse_known_args([])[0]
all_args.algorithm_name = "macpo"
all_args.hidden_size = HIDDEN_SIZE
all_args.layer_N = LAYER_N

policies = []
for agent_id in range(env.n_agents):
    po = MACPPOPolicy(all_args,
                      env.observation_space[agent_id],
                      env.share_observation_space[agent_id],
                      env.action_space[agent_id],
                      device=device)
    actor_path = os.path.join(MODEL_DIR, f"actor_agent{agent_id}.pt")
    state_dict = torch.load(actor_path, map_location=device)
    po.actor.load_state_dict(state_dict)
    po.actor.eval()
    policies.append(po)
    print(f"Loaded actor agent{agent_id}")

def get_actions(obs):
    actions = []
    for agent_id, po in enumerate(policies):
        obs_t = torch.FloatTensor(obs[agent_id]).unsqueeze(0).to(device)
        rnn = torch.zeros(1, 1, HIDDEN_SIZE).to(device)
        masks = torch.ones(1, 1).to(device)
        with torch.no_grad():
            action, _, _ = po.actor(obs_t, rnn, masks, deterministic=True)
        actions.append(action.cpu().numpy().flatten())
    return actions

n_agents = env.n_agents
config = env.config

# ===== 跑episodes =====
all_stats = []

for ep_idx, seed in enumerate(SEEDS[:N_EPS]):
    env.seed(seed)
    obs, _, _ = env.reset()
    
    ep_data = {
        'dists': [[] for _ in range(n_agents)],
        'z_pos': [[] for _ in range(n_agents)],
        'speeds': [[] for _ in range(n_agents)],
        'gammas': [[] for _ in range(n_agents)],
        'alive': [[] for _ in range(n_agents)],
        'actions': [[] for _ in range(n_agents)],
        'step_rewards': [],
        'step_costs': [],
        'xy_traj': [[] for _ in range(n_agents)],
        'def_xy': [[] for _ in range(config['n_defensive'])],
    }
    
    total_rewards = np.zeros(n_agents)
    total_costs = np.zeros(n_agents)
    death_reasons = ['alive'] * n_agents
    death_steps = [None] * n_agents
    hit_hvt = [False] * n_agents
    min_dist_to_hvt = [float('inf')] * n_agents
    
    for step in range(env.max_steps):
        # 记录状态
        for i, off in enumerate(env.offensives):
            d = off.distance_to(env.hvt.x, env.hvt.y, env.hvt.z) if off.alive else float('nan')
            ep_data['dists'][i].append(d)
            ep_data['z_pos'][i].append(off.z if off.alive else float('nan'))
            ep_data['speeds'][i].append(off.v if off.alive else float('nan'))
            ep_data['gammas'][i].append(np.degrees(off.gamma) if off.alive else float('nan'))
            ep_data['alive'][i].append(off.alive)
            ep_data['xy_traj'][i].append((off.x, off.y))
            if off.alive and not np.isnan(d) and d < min_dist_to_hvt[i]:
                min_dist_to_hvt[i] = d
        for j in range(config['n_defensive']):
            df = env.defensives[j]
            ep_data['def_xy'][j].append((df.x, df.y))
        
        # 执行动作
        actions = get_actions(obs)
        for i in range(n_agents):
            ep_data['actions'][i].append(actions[i].tolist())
        
        obs, _, rewards, costs, dones, infos, _ = env.step(actions)
        
        r_vals = [rewards[i][0] if isinstance(rewards[i], (list, np.ndarray)) else rewards[i] for i in range(n_agents)]
        c_vals = [costs[i][0] if isinstance(costs[i], (list, np.ndarray)) else costs[i] for i in range(n_agents)]
        ep_data['step_rewards'].append(sum(r_vals))
        ep_data['step_costs'].append(sum(c_vals))
        total_rewards += np.array(r_vals)
        total_costs += np.array(c_vals)
        
        # 检查死亡/命中
        for i, off in enumerate(env.offensives):
            if not off.alive and death_reasons[i] == 'alive':
                death_reasons[i] = 'killed'
                death_steps[i] = step
            if hasattr(off, 'hit_hvt') and off.hit_hvt and not hit_hvt[i]:
                hit_hvt[i] = True
                death_reasons[i] = 'hit_hvt'
                death_steps[i] = step
        
        if any(dones):
            break
    
    ep_length = step + 1
    success = any(hit_hvt)
    info = infos[0] if infos else {}
    done_reason = info.get('done_reason', 'timeout' if ep_length >= env.max_steps else 'unknown')
    
    stats = {
        'seed': seed,
        'ep_length': ep_length,
        'success': success,
        'done_reason': done_reason,
        'total_reward': float(sum(total_rewards)),
        'total_cost': float(sum(total_costs)),
        'min_dists': [float(d) for d in min_dist_to_hvt],
        'team_min_dist': float(min(min_dist_to_hvt)),
        'death_reasons': death_reasons,
        'death_steps': death_steps,
        'hit_hvt': hit_hvt,
    }
    all_stats.append(stats)
    
    print(f"\n=== Episode {ep_idx} (seed={seed}) ===")
    print(f"  Length: {ep_length}, Done: {done_reason}, Success: {success}")
    print(f"  Total reward: {stats['total_reward']:.1f}, Total cost: {stats['total_cost']:.1f}")
    print(f"  Team min dist to HVT: {stats['team_min_dist']:.0f}m")
    print(f"  Individual min dists: {[f'{d:.0f}' for d in min_dist_to_hvt]}")
    print(f"  Death reasons: {death_reasons}, steps: {death_steps}")
    
    # 动作分析
    for i in range(n_agents):
        acts = np.array(ep_data['actions'][i])
        if len(acts) > 0:
            print(f"  Agent{i} action: mean={np.round(acts.mean(0),3)}, std={np.round(acts.std(0),3)}")
    
    # ===== 诊断图 =====
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle(f'V19 Ep{ep_idx} seed={seed} len={ep_length} reason={done_reason} r={stats["total_reward"]:.0f}', fontsize=13)
    
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']
    
    ax = axes[0, 0]
    for i in range(n_agents):
        ax.plot(ep_data['dists'][i], color=colors[i], label=f'Agent{i}', alpha=0.8)
    ax.set_title('Distance to HVT')
    ax.set_ylabel('Distance (m)')
    ax.axhline(y=500, color='green', linestyle='--', alpha=0.5, label='hit_range')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    for i in range(n_agents):
        ax.plot(ep_data['z_pos'][i], color=colors[i], label=f'Agent{i}', alpha=0.8)
    ax.set_title('Altitude')
    ax.axhline(y=100, color='red', linestyle='--', alpha=0.5, label='z_min')
    ax.axhline(y=1200, color='orange', linestyle='--', alpha=0.5, label='z_high')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    cum_r = np.cumsum(ep_data['step_rewards'])
    cum_c = np.cumsum(ep_data['step_costs'])
    ax.plot(cum_r, 'b-', label='Cum Reward')
    ax.plot(cum_c, 'r-', label='Cum Cost')
    ax.set_title('Cumulative Reward vs Cost')
    ax.legend(); ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    for i in range(n_agents):
        ax.plot(ep_data['speeds'][i], color=colors[i], label=f'Agent{i}', alpha=0.8)
    ax.set_title('Speed')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    
    ax = axes[2, 0]
    for i in range(n_agents):
        ax.plot(ep_data['gammas'][i], color=colors[i], label=f'Agent{i}', alpha=0.8)
    ax.set_title('Flight Path Angle (gamma)')
    ax.axhline(y=-5, color='red', linestyle='--', alpha=0.5)
    ax.axhline(y=12, color='orange', linestyle='--', alpha=0.5)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    
    ax = axes[2, 1]
    for i in range(n_agents):
        traj = ep_data['xy_traj'][i]
        xs, ys = [p[0] for p in traj], [p[1] for p in traj]
        ax.plot(xs, ys, color=colors[i], label=f'Off{i}', alpha=0.7)
        ax.scatter(xs[0], ys[0], color=colors[i], marker='o', s=40, zorder=5)
        if death_steps[i]:
            di = min(death_steps[i], len(xs)-1)
            ax.scatter(xs[di], ys[di], color=colors[i], marker='x', s=60, zorder=5)
    for j in range(config['n_defensive']):
        dtraj = ep_data['def_xy'][j]
        ax.plot([p[0] for p in dtraj], [p[1] for p in dtraj], 'gray', alpha=0.4, linestyle='--')
    hvt_pos = config['hvt_position']
    ax.scatter(hvt_pos[0], hvt_pos[1], color='gold', marker='*', s=200, zorder=10, label='HVT')
    ax.set_title('XY Trajectories'); ax.set_aspect('equal')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, f'diagnosis_ep{ep_idx}_seed{seed}.png')
    plt.savefig(fig_path, dpi=150); plt.close()
    print(f"  Saved: {fig_path}")

# ===== 汇总 =====
print("\n" + "="*60)
print("V19 DIAGNOSIS SUMMARY")
print("="*60)
n_success = sum(1 for s in all_stats if s['success'])
n_timeout = sum(1 for s in all_stats if s.get('done_reason') == 'timeout')
avg_reward = np.mean([s['total_reward'] for s in all_stats])
avg_cost = np.mean([s['total_cost'] for s in all_stats])
avg_min_dist = np.mean([s['team_min_dist'] for s in all_stats])

print(f"Success: {n_success}/{N_EPS} ({100*n_success/N_EPS:.0f}%)")
print(f"Timeout: {n_timeout}/{N_EPS} ({100*n_timeout/N_EPS:.0f}%)")
print(f"Avg reward: {avg_reward:.0f}")
print(f"Avg cost: {avg_cost:.0f}")
print(f"Avg team min dist to HVT: {avg_min_dist:.0f}m")

all_death = []
for s in all_stats:
    all_death.extend(s['death_reasons'])
print(f"Death reasons: {dict(Counter(all_death))}")

for s in all_stats:
    print(f"  seed={s['seed']}: len={s['ep_length']}, r={s['total_reward']:.0f}, "
          f"c={s['total_cost']:.0f}, min_d={s['team_min_dist']:.0f}m, "
          f"done={s['done_reason']}, deaths={s['death_reasons']}")

with open(os.path.join(OUT_DIR, 'v19_diagnosis_summary.json'), 'w') as f:
    json.dump({'stats': all_stats, 'summary': {
        'success_rate': n_success/N_EPS, 'timeout_rate': n_timeout/N_EPS,
        'avg_reward': float(avg_reward), 'avg_cost': float(avg_cost),
        'avg_min_dist': float(avg_min_dist),
    }}, f, indent=2)

print(f"\nAll outputs saved to {OUT_DIR}/")

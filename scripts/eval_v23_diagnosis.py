#!/usr/bin/env python
"""
V23 诊断评估脚本
=====================
运行 N 个 episode，逐步记录所有关键指标，
绘制详细分析图表，找出突防失败原因。
"""
import sys, os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "MACPO", "MACPO"))

import argparse
import numpy as np
import torch
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from datetime import datetime

from envs.fov_penetration import FOVPenetrationEnv


# ── 模型加载 ──────────────────────────────────────────────
def load_actors(model_dir, n_agents, obs_dim, act_dim, hidden_size, layer_N, device):
    """加载 MAPPO actor 网络"""
    from macpo.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy
    from macpo.config import get_config
    
    # 用 get_config 生成完整 args，然后覆盖关键字段
    parser = get_config()
    args = parser.parse_args([
        '--algorithm_name', 'mappo',
        '--hidden_size', str(hidden_size),
        '--layer_N', str(layer_N),
        '--use_ReLU',
        '--use_feature_normalization',
        '--use_orthogonal',
        '--gain', '0.01',
        '--use_recurrent_policy',
        '--recurrent_N', '1',
        '--data_chunk_length', '10',
        '--lr', '3e-4',
        '--critic_lr', '3e-4',
        '--entropy_coef', '0.02',
        '--max_grad_norm', '10.0',
    ])
    
    # 构造 obs/act space
    try:
        import gymnasium as gym
    except ImportError:
        import gym
    obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
    act_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32)
    
    policies = []
    for i in range(n_agents):
        policy = R_MAPPOPolicy(args, obs_space, obs_space, act_space, device=device)
        actor_path = os.path.join(model_dir, f"actor_agent{i}.pt")
        if os.path.exists(actor_path):
            state_dict = torch.load(actor_path, map_location=device)
            policy.actor.load_state_dict(state_dict)
            print(f"  Loaded actor_agent{i}.pt")
        else:
            print(f"  WARNING: {actor_path} not found, using random policy")
        policy.actor.eval()
        policies.append(policy)
    
    return policies


def get_actions(policies, obs, rnn_states, masks, device, deterministic=True):
    """从 policies 获取动作 — 匹配 MACPO R_Actor 的 forward 签名
    R_Actor.forward(obs, rnn_states, masks, ...) 
    - obs:        (1, obs_dim)
    - rnn_states: (1, recurrent_N, hidden)
    - masks:      (1, 1)
    """
    n_agents = len(policies)
    actions = []
    new_rnn_states = []
    
    for i in range(n_agents):
        obs_i = np.array(obs[i], dtype=np.float32).flatten()
        obs_t = torch.FloatTensor(obs_i).unsqueeze(0).to(device)          # (1, obs_dim)
        
        rnn_t = torch.FloatTensor(rnn_states[i]).to(device)               # (1, recurrent_N, hidden)
        if rnn_t.dim() == 2:
            rnn_t = rnn_t.unsqueeze(0)                                     # ensure (1, recN, hid)
        
        mask_t = torch.FloatTensor([[masks[i]]]).to(device)                # (1, 1)
        
        with torch.no_grad():
            action, _, new_rnn = policies[i].actor(obs_t, rnn_t, mask_t, deterministic=deterministic)
        
        actions.append(action.cpu().numpy().flatten())
        new_rnn_states.append(new_rnn.cpu().numpy().squeeze(0))            # (recN, hid)
    
    return np.array(actions), new_rnn_states


# ── 单 episode 诊断运行 ──────────────────────────────────
def run_diagnostic_episode(env, policies, device, deterministic=True):
    """运行 1 个 episode，记录逐步指标"""
    reset_result = env.reset()
    # reset returns (obs, share_obs, avail) — obs shape (n_agents, obs_dim)
    obs_all = reset_result[0]  # (n_agents, obs_dim)
    n_agents = env.n_agents
    
    # RNN states: per agent (1, recurrent_N, hidden_size)
    recurrent_N = 1
    hidden_size = 256  # match training
    rnn_states = [np.zeros((1, recurrent_N, hidden_size), dtype=np.float32) for _ in range(n_agents)]
    masks = [1.0] * n_agents  # scalar per agent
    
    # 把 obs_all 切成 per-agent list
    obs = [obs_all[i] for i in range(n_agents)]
    
    # 逐步记录
    step_data = defaultdict(list)
    agent_trajectories = {i: {"x": [], "y": [], "z": [], "v": []} for i in range(n_agents)}
    
    done = False
    step = 0
    total_reward = np.zeros(n_agents)
    
    while not done:
        actions, rnn_states = get_actions(policies, obs, rnn_states, masks, device, deterministic)
        
        # 记录动作
        for i in range(n_agents):
            step_data[f"action_agent{i}_nx"].append(float(actions[i][0]))
            step_data[f"action_agent{i}_ny"].append(float(actions[i][1]))
            step_data[f"action_agent{i}_nz"].append(float(actions[i][2]))
        
        # step returns: obs, share_obs, rewards, costs, dones, infos, avail
        step_result = env.step(actions)
        obs_all_new = step_result[0]   # list of (obs_dim,) per agent
        rewards_raw = step_result[2]   # rewards
        costs_raw = step_result[3]     # costs
        dones_raw = step_result[4]     # dones
        infos_raw = step_result[5]     # infos
        
        # Parse obs
        if isinstance(obs_all_new, np.ndarray) and obs_all_new.ndim == 2:
            obs = [obs_all_new[i] for i in range(n_agents)]
        elif isinstance(obs_all_new, list):
            obs = obs_all_new
        else:
            obs = [np.array(obs_all_new[i]) for i in range(n_agents)]
        
        # Parse rewards — may be list of arrays or single array
        if isinstance(rewards_raw, (list, tuple)):
            # Could be per-agent reward arrays or scalar list
            step_rewards = np.zeros(n_agents)
            for i in range(min(n_agents, len(rewards_raw))):
                r = np.array(rewards_raw[i]).flatten()
                step_rewards[i] = float(r[0]) if len(r) > 0 else 0.0
        elif isinstance(rewards_raw, np.ndarray):
            step_rewards = rewards_raw.flatten()[:n_agents]
        else:
            step_rewards = np.zeros(n_agents)
        total_reward += step_rewards
        
        # Parse infos
        if isinstance(infos_raw, list) and len(infos_raw) > 0:
            info = infos_raw[0] if isinstance(infos_raw[0], dict) else {}
        elif isinstance(infos_raw, dict):
            info = infos_raw
        else:
            info = {}
        
        # Parse dones
        if isinstance(dones_raw, (list, tuple)):
            done_vals = [float(np.array(d).flatten()[0]) if hasattr(d, '__len__') else float(d) for d in dones_raw]
            done = all(d > 0.5 for d in done_vals)
        elif isinstance(dones_raw, np.ndarray):
            done = bool(dones_raw.all()) if dones_raw.ndim > 0 else bool(dones_raw)
        else:
            done = bool(dones_raw)
        
        # 标量指标
        for key in ['offensive_alive', 'defensive_alive', 'hit_count', 'n_escapes_total',
                     'n_escaped_agents', 'n_detected_now', 'avg_exposure_rate',
                     'penetration_success_score_team', 'n_locked_defenders',
                     'terminal_miss_distance_min', 'two_stage_score']:
            if key in info:
                step_data[key].append(float(info[key]))
        
        # per-agent 指标
        if 'penetration_success_score_per_agent' in info:
            for i, v in enumerate(info['penetration_success_score_per_agent']):
                step_data[f"pen_score_agent{i}"].append(float(v))
        if 'attack_gate_reward_per_agent' in info:
            for i, v in enumerate(info['attack_gate_reward_per_agent']):
                step_data[f"attack_gate_agent{i}"].append(float(v))
        
        # 奖励分解 (如果 reward_info 可用)
        step_data["step_reward_mean"].append(float(np.mean(step_rewards)))
        step_data["step_reward_sum"].append(float(np.sum(step_rewards)))
        
        # AP info
        for key in ['curriculum_mult', 'attack_reward_mean', 'attack_reward_sum',
                     'hvt_rho_mean', 'hvt_closing_speed_mean', 'hvt_omega_los_mean',
                     'attack_gate_mean']:
            if key in info:
                step_data[key].append(float(info[key]))
        
        # 记录进攻方位置 — entities 直接在 env 上 (env.offensives / env.defensives)
        for i in range(n_agents):
            try:
                off = env.offensives[i] if hasattr(env, 'offensives') and i < len(env.offensives) else None
                if off is not None:
                    agent_trajectories[i]["x"].append(float(off.x))
                    agent_trajectories[i]["y"].append(float(off.y))
                    agent_trajectories[i]["z"].append(float(off.z))
                    agent_trajectories[i]["v"].append(float(off.v))
            except Exception:
                pass
        
        step += 1
        
        # 采样防御方位置
        if hasattr(env, 'defensives') and step % 10 == 0:
            try:
                for j, d in enumerate(env.defensives):
                    step_data[f"def{j}_x"].append(float(d.x))
                    step_data[f"def{j}_y"].append(float(d.y))
                    step_data[f"def{j}_z"].append(float(d.z))
            except Exception:
                pass
    
    # episode-end info — use last info from loop
    end_info = info
    
    # 计算每个进攻方到 HVT 的最终距离
    hvt_pos = np.array(env.config.get("hvt_position", [1200, 0, 0]))
    final_dists = []
    min_dists = []
    for i in range(n_agents):
        if agent_trajectories[i]["x"]:
            traj_x = np.array(agent_trajectories[i]["x"])
            traj_y = np.array(agent_trajectories[i]["y"])
            traj_z = np.array(agent_trajectories[i]["z"])
            dists = np.sqrt((traj_x - hvt_pos[0])**2 + (traj_y - hvt_pos[1])**2 + (traj_z - hvt_pos[2])**2)
            final_dists.append(dists[-1])
            min_dists.append(dists.min())
        else:
            final_dists.append(float('inf'))
            min_dists.append(float('inf'))
    
    episode_summary = {
        "steps": step,
        "total_reward": total_reward.tolist(),
        "mean_reward": float(np.mean(total_reward)),
        "done_reason": end_info.get("done_reason", "unknown"),
        "success": bool(end_info.get("success", False)),
        "hit_count": int(end_info.get("hit_count", 0)),
        "n_escapes_total": int(end_info.get("n_escapes_total", 0)),
        "n_escaped_agents": int(end_info.get("n_escaped_agents", 0)),
        "offensive_alive_end": int(end_info.get("offensive_alive", 0)),
        "defensive_alive_end": int(end_info.get("defensive_alive", 0)),
        "final_dists_to_hvt": final_dists,
        "min_dists_to_hvt": min_dists,
        "terminal_miss_distance_min": float(end_info.get("terminal_miss_distance_min", -1)),
    }
    
    return episode_summary, step_data, agent_trajectories


# ── 绘图函数 ──────────────────────────────────────────────
def plot_episode_details(ep_idx, step_data, trajectories, summary, save_dir, config):
    """为单个 episode 绘制详细图表"""
    fig, axes = plt.subplots(4, 3, figsize=(24, 20))
    fig.suptitle(f"Episode {ep_idx} — {summary['done_reason']} | "
                 f"Reward={summary['mean_reward']:.0f} | "
                 f"Steps={summary['steps']} | "
                 f"Hits={summary['hit_count']} | "
                 f"Escapes={summary['n_escapes_total']}",
                 fontsize=14, fontweight='bold')
    
    n_agents = len(trajectories)
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']
    
    # (0,0) 进攻方 XY 轨迹 + 防御方
    ax = axes[0, 0]
    hvt = config.get("hvt_position", [1200, 0, 0])
    ax.plot(hvt[0], hvt[1], '*', color='gold', markersize=15, zorder=10, label='HVT')
    for i in range(n_agents):
        if trajectories[i]["x"]:
            ax.plot(trajectories[i]["x"], trajectories[i]["y"], '-', 
                    color=colors[i % len(colors)], alpha=0.7, label=f'Off{i}')
            ax.plot(trajectories[i]["x"][-1], trajectories[i]["y"][-1], 'x', 
                    color=colors[i % len(colors)], markersize=8)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('XY Trajectories')
    ax.legend(fontsize=8)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # (0,1) 进攻方 XZ 轨迹
    ax = axes[0, 1]
    for i in range(n_agents):
        if trajectories[i]["x"]:
            ax.plot(trajectories[i]["x"], trajectories[i]["z"], '-', 
                    color=colors[i % len(colors)], alpha=0.7, label=f'Off{i}')
    ax.axhline(y=200, color='orange', linestyle='--', alpha=0.5, label='z_low=200m')
    ax.axhline(y=600, color='red', linestyle='--', alpha=0.5, label='z_high=600m')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')
    ax.set_title('XZ Side View')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # (0,2) 到 HVT 距离随时间变化
    ax = axes[0, 2]
    for i in range(n_agents):
        if trajectories[i]["x"]:
            dists = np.sqrt((np.array(trajectories[i]["x"]) - hvt[0])**2 + 
                           (np.array(trajectories[i]["y"]) - hvt[1])**2 +
                           (np.array(trajectories[i]["z"]) - hvt[2])**2)
            t = np.arange(len(dists)) * 0.01  # dt=0.01
            ax.plot(t, dists, '-', color=colors[i % len(colors)], label=f'Off{i}')
    ax.axhline(y=5, color='green', linestyle='--', alpha=0.8, label='hit_range=5m')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Distance to HVT (m)')
    ax.set_title('Distance to HVT')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # (1,0) 存活数量
    ax = axes[1, 0]
    if 'offensive_alive' in step_data:
        t = np.arange(len(step_data['offensive_alive'])) * 0.01
        ax.plot(t, step_data['offensive_alive'], 'r-', label='Offensive alive', linewidth=2)
    if 'defensive_alive' in step_data:
        t = np.arange(len(step_data['defensive_alive'])) * 0.01
        ax.plot(t, step_data['defensive_alive'], 'b-', label='Defensive alive', linewidth=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Count')
    ax.set_title('Alive Count')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # (1,1) 逐步奖励
    ax = axes[1, 1]
    if 'step_reward_mean' in step_data:
        t = np.arange(len(step_data['step_reward_mean'])) * 0.01
        r = np.array(step_data['step_reward_mean'])
        # 滑动平均
        window = min(100, len(r) // 4)
        if window > 1:
            r_smooth = np.convolve(r, np.ones(window)/window, mode='valid')
            t_smooth = t[:len(r_smooth)]
            ax.plot(t_smooth, r_smooth, 'b-', label=f'Mean reward (MA{window})', linewidth=1.5)
        ax.plot(t, r, 'b-', alpha=0.15, linewidth=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Mean Step Reward')
    ax.set_title('Step Rewards')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # (1,2) 累积奖励
    ax = axes[1, 2]
    if 'step_reward_mean' in step_data:
        t = np.arange(len(step_data['step_reward_mean'])) * 0.01
        cum_r = np.cumsum(step_data['step_reward_mean'])
        ax.plot(t, cum_r, 'g-', linewidth=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Cumulative Mean Reward')
    ax.set_title('Cumulative Reward')
    ax.grid(True, alpha=0.3)
    
    # (2,0) 动作分布 - ny (横向机动)
    ax = axes[2, 0]
    for i in range(n_agents):
        key = f"action_agent{i}_ny"
        if key in step_data:
            t = np.arange(len(step_data[key])) * 0.01
            ax.plot(t, step_data[key], '-', color=colors[i % len(colors)], alpha=0.3, linewidth=0.5)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='max=1')
    ax.axhline(y=-1.0, color='gray', linestyle='--', alpha=0.5, label='min=-1')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('action_ny (raw)')
    ax.set_title('Lateral Maneuver Actions (ny)')
    ax.grid(True, alpha=0.3)
    
    # (2,1) 速度
    ax = axes[2, 1]
    for i in range(n_agents):
        if trajectories[i]["v"]:
            t = np.arange(len(trajectories[i]["v"])) * 0.01
            ax.plot(t, trajectories[i]["v"], '-', color=colors[i % len(colors)], 
                    alpha=0.7, label=f'Off{i}')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Speed (m/s)')
    ax.set_title('Agent Speeds')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # (2,2) penetration score
    ax = axes[2, 2]
    for i in range(n_agents):
        key = f"pen_score_agent{i}"
        if key in step_data:
            t = np.arange(len(step_data[key])) * 0.01
            ax.plot(t, step_data[key], '-', color=colors[i % len(colors)], alpha=0.7, label=f'Off{i}')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Penetration Score')
    ax.set_title('Penetration Success Score')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # (3,0) 逃逸和检测
    ax = axes[3, 0]
    if 'n_escapes_total' in step_data:
        t = np.arange(len(step_data['n_escapes_total'])) * 0.01
        ax.plot(t, step_data['n_escapes_total'], 'g-', linewidth=2, label='Total escapes')
    if 'n_detected_now' in step_data:
        t2 = np.arange(len(step_data['n_detected_now'])) * 0.01
        ax.plot(t2, step_data['n_detected_now'], 'r--', linewidth=1.5, label='Currently detected')
    if 'n_locked_defenders' in step_data:
        t3 = np.arange(len(step_data['n_locked_defenders'])) * 0.01
        ax.plot(t3, step_data['n_locked_defenders'], 'b:', linewidth=1.5, label='Locked defenders')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Count')
    ax.set_title('Detection & Escape Events')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # (3,1) 高度分布
    ax = axes[3, 1]
    for i in range(n_agents):
        if trajectories[i]["z"]:
            ax.hist(trajectories[i]["z"], bins=30, alpha=0.5, color=colors[i % len(colors)], label=f'Off{i}')
    ax.axvline(x=200, color='orange', linestyle='--', label='z_low=200')
    ax.axvline(x=600, color='red', linestyle='--', label='z_high=600')
    ax.set_xlabel('Altitude (m)')
    ax.set_ylabel('Frequency')
    ax.set_title('Altitude Distribution')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # (3,2) terminal miss distance
    ax = axes[3, 2]
    if 'terminal_miss_distance_min' in step_data:
        t = np.arange(len(step_data['terminal_miss_distance_min'])) * 0.01
        ax.plot(t, step_data['terminal_miss_distance_min'], 'r-', linewidth=1.5)
    ax.axhline(y=5, color='green', linestyle='--', alpha=0.8, label='kill_range=5m')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Min Miss Distance (m)')
    ax.set_title('Terminal Miss Distance (closest approach)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(os.path.join(save_dir, f"episode_{ep_idx}_detail.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_summary(all_summaries, save_dir):
    """绘制所有 episode 的汇总对比图"""
    n_eps = len(all_summaries)
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f"V23 Evaluation Summary — {n_eps} Episodes", fontsize=14, fontweight='bold')
    
    ep_ids = list(range(n_eps))
    
    # (0,0) Episode rewards
    rewards = [s['mean_reward'] for s in all_summaries]
    ax = axes[0, 0]
    bars = ax.bar(ep_ids, rewards, color=['green' if s['success'] else 'red' for s in all_summaries])
    ax.set_xlabel('Episode')
    ax.set_ylabel('Mean Reward')
    ax.set_title('Episode Mean Reward')
    ax.grid(True, alpha=0.3)
    
    # (0,1) Episode duration
    durations = [s['steps'] * 0.01 for s in all_summaries]
    ax = axes[0, 1]
    ax.bar(ep_ids, durations, color=['green' if s['success'] else 'orange' for s in all_summaries])
    ax.set_xlabel('Episode')
    ax.set_ylabel('Duration (s)')
    ax.set_title('Episode Duration')
    ax.grid(True, alpha=0.3)
    
    # (0,2) Done reasons pie chart
    reasons = [s['done_reason'] for s in all_summaries]
    from collections import Counter
    reason_counts = Counter(reasons)
    ax = axes[0, 2]
    ax.pie(reason_counts.values(), labels=reason_counts.keys(), autopct='%1.0f%%',
           colors=['green' if 'success' in k else 'red' if 'kill' in k else 'orange' 
                   for k in reason_counts.keys()])
    ax.set_title('Episode End Reasons')
    
    # (1,0) Min dist to HVT per agent per episode
    ax = axes[1, 0]
    n_agents = len(all_summaries[0]['min_dists_to_hvt'])
    width = 0.8 / n_agents
    for i in range(n_agents):
        agent_min_dists = [s['min_dists_to_hvt'][i] for s in all_summaries]
        x_pos = [e + i * width for e in ep_ids]
        ax.bar(x_pos, agent_min_dists, width=width, label=f'Off{i}', alpha=0.8)
    ax.axhline(y=5, color='green', linestyle='--', label='hit_range=5m')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Min Distance to HVT (m)')
    ax.set_title('Closest Approach to HVT')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # (1,1) Alive at end
    ax = axes[1, 1]
    off_alive = [s['offensive_alive_end'] for s in all_summaries]
    def_alive = [s['defensive_alive_end'] for s in all_summaries]
    x = np.arange(n_eps)
    ax.bar(x - 0.2, off_alive, 0.35, label='Offensive alive', color='red', alpha=0.7)
    ax.bar(x + 0.2, def_alive, 0.35, label='Defensive alive', color='blue', alpha=0.7)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Count')
    ax.set_title('Alive at Episode End')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # (1,2) Escapes
    escapes = [s['n_escapes_total'] for s in all_summaries]
    escaped_agents = [s['n_escaped_agents'] for s in all_summaries]
    ax = axes[1, 2]
    ax.bar(x - 0.2, escapes, 0.35, label='Total escapes', color='green', alpha=0.7)
    ax.bar(x + 0.2, escaped_agents, 0.35, label='Escaped agents', color='teal', alpha=0.7)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Count')
    ax.set_title('Escape Events')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(os.path.join(save_dir, "summary.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_action_distribution(all_step_data, save_dir):
    """绘制所有 episode 的动作分布"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Action Distribution Across All Episodes", fontsize=14)
    
    for dim_idx, dim_name in enumerate(['nx', 'ny', 'nz']):
        ax = axes[dim_idx]
        all_vals = []
        for ep_data in all_step_data:
            for i in range(4):
                key = f"action_agent{i}_{dim_name}"
                if key in ep_data:
                    all_vals.extend(ep_data[key])
        if all_vals:
            ax.hist(all_vals, bins=50, density=True, alpha=0.7, color='steelblue')
            ax.axvline(x=np.mean(all_vals), color='red', linestyle='--', label=f'mean={np.mean(all_vals):.3f}')
            ax.axvline(x=0, color='gray', linestyle='-', alpha=0.5)
        ax.set_xlabel(f'action_{dim_name}')
        ax.set_ylabel('Density')
        ax.set_title(f'{dim_name} Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "action_distribution.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)


def print_analysis(all_summaries, all_step_data):
    """打印分析结论"""
    print("\n" + "="*80)
    print("                    V23 DIAGNOSTIC ANALYSIS REPORT")
    print("="*80)
    
    n_eps = len(all_summaries)
    success_count = sum(1 for s in all_summaries if s['success'])
    print(f"\n[基本统计]")
    print(f"  总 episodes: {n_eps}")
    print(f"  成功突防: {success_count}/{n_eps} ({100*success_count/n_eps:.0f}%)")
    
    reasons = [s['done_reason'] for s in all_summaries]
    from collections import Counter
    reason_counts = Counter(reasons)
    print(f"  结束原因分布: {dict(reason_counts)}")
    
    mean_rewards = [s['mean_reward'] for s in all_summaries]
    print(f"  平均奖励: {np.mean(mean_rewards):.1f} ± {np.std(mean_rewards):.1f}")
    
    durations = [s['steps'] * 0.01 for s in all_summaries]
    print(f"  平均时长: {np.mean(durations):.1f}s ± {np.std(durations):.1f}s")
    
    print(f"\n[突防距离分析]")
    for i in range(4):
        min_ds = [s['min_dists_to_hvt'][i] for s in all_summaries]
        print(f"  Agent{i}: 最近到 HVT 距离 = {np.mean(min_ds):.1f}m ± {np.std(min_ds):.1f}m "
              f"(最小 {np.min(min_ds):.1f}m)")
    
    all_min = [min(s['min_dists_to_hvt']) for s in all_summaries]
    print(f"  全局最近: {np.mean(all_min):.1f}m ± {np.std(all_min):.1f}m (最小 {np.min(all_min):.1f}m)")
    
    print(f"\n[存活分析]")
    off_alive = [s['offensive_alive_end'] for s in all_summaries]
    def_alive = [s['defensive_alive_end'] for s in all_summaries]
    print(f"  终局进攻方存活: {np.mean(off_alive):.1f} ± {np.std(off_alive):.1f}")
    print(f"  终局防守方存活: {np.mean(def_alive):.1f} ± {np.std(def_alive):.1f}")
    
    print(f"\n[逃逸分析]")
    escapes = [s['n_escapes_total'] for s in all_summaries]
    escaped = [s['n_escaped_agents'] for s in all_summaries]
    print(f"  总逃逸次数: {np.mean(escapes):.1f} ± {np.std(escapes):.1f}")
    print(f"  逃逸过的 agent 数: {np.mean(escaped):.1f} ± {np.std(escaped):.1f}")
    
    print(f"\n[动作分析]")
    for dim in ['nx', 'ny', 'nz']:
        all_vals = []
        for ep_data in all_step_data:
            for i in range(4):
                key = f"action_agent{i}_{dim}"
                if key in ep_data:
                    all_vals.extend(ep_data[key])
        if all_vals:
            arr = np.array(all_vals)
            print(f"  {dim}: mean={arr.mean():.4f}, std={arr.std():.4f}, "
                  f"min={arr.min():.3f}, max={arr.max():.3f}, "
                  f"|>0.8|={100*(np.abs(arr) > 0.8).mean():.1f}%")
    
    print(f"\n[关键发现 & 诊断]")
    
    # 检查是否全部超时
    if reason_counts.get('timeout', 0) == n_eps:
        print("  ⚠ 所有 episode 都因超时结束 — 进攻方未能在 60s 内到达 HVT 或被消灭")
    
    # 检查是否全部被杀
    if reason_counts.get('all_killed', 0) > 0:
        pct = 100 * reason_counts['all_killed'] / n_eps
        print(f"  ⚠ {pct:.0f}% episode 进攻方全灭 — 拦截器命中率太高")

    # 检查距离是否接近过
    if np.min(all_min) > 100:
        print(f"  ⚠ 最近一次到 HVT 距离 {np.min(all_min):.0f}m >> 5m — 进攻方根本没接近 HVT")
        print("     可能原因: 进攻方在中途被拦截器消灭, 或飞行方向有问题")
    elif np.min(all_min) > 5:
        print(f"  △ 最近距离 {np.min(all_min):.1f}m > 5m — 差一点但未达 hit 阈值")
    
    # 检查动作是否利用充分
    for dim in ['ny']:
        all_vals = []
        for ep_data in all_step_data:
            for i in range(4):
                key = f"action_agent{i}_{dim}"
                if key in ep_data:
                    all_vals.extend(ep_data[key])
        if all_vals:
            arr = np.array(all_vals)
            if arr.std() < 0.1:
                print(f"  ⚠ {dim} 动作标准差仅 {arr.std():.4f} — 策略几乎不机动!")
    
    # 检查是否很快被杀
    for s in all_summaries:
        if s['done_reason'] == 'all_killed' and s['steps'] * 0.01 < 15:
            print(f"  ⚠ 有 episode 在 {s['steps']*0.01:.1f}s 内全灭 — 拦截太快")
            break
    
    print("\n" + "="*80)


# ── 主函数 ────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--n_episodes', type=int, default=10)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--layer_N', type=int, default=3)
    parser.add_argument('--deterministic', action='store_true', default=True)
    parser.add_argument('--stochastic', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default=None)
    args = parser.parse_args()
    
    if args.stochastic:
        args.deterministic = False
    
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 保存目录
    if args.save_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.save_dir = os.path.join(PROJECT_ROOT, "outputs", "results", f"v23_diagnosis_{ts}")
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"Results will be saved to: {args.save_dir}")
    
    # 创建环境
    ap_override = {"analytic_priors": {
        "enable_cone_cost": False,
        "enable_assignment_mismatch_reward": False,
        "enable_escape_reward": True,
        "enable_decoy_game": True,
        "enable_effective_penetration": True,
    }}
    env = FOVPenetrationEnv(config=ap_override, scenario="scenario_1")
    env.seed(args.seed)
    
    n_agents = env.n_agents
    obs_dim = env.observation_space[0].shape[0]
    act_dim = env.action_space[0].shape[0]
    
    print(f"Env: n_agents={n_agents}, obs_dim={obs_dim}, act_dim={act_dim}")
    print(f"Config: dt={env.config.get('dt')}, max_steps={env.config.get('max_steps')}, "
          f"map_size={env.config.get('map_size')}, hit_hvt_range={env.config.get('hit_hvt_range')}")
    print(f"Speeds: off_v_nom={env.config.get('offensive',{}).get('v_nominal')}, "
          f"def_v_nom={env.config.get('defensive',{}).get('v_nominal')}")
    
    # 加载模型
    print(f"\nLoading model from: {args.model_dir}")
    policies = load_actors(args.model_dir, n_agents, obs_dim, act_dim, 
                          args.hidden_size, args.layer_N, device)
    
    # 运行评估
    all_summaries = []
    all_step_data = []
    all_trajectories = []
    
    for ep in range(args.n_episodes):
        try:
            env.seed(args.seed + ep * 1000)
            print(f"\n--- Episode {ep} ---")
            summary, step_data, trajectories = run_diagnostic_episode(
                env, policies, device, deterministic=args.deterministic)
            
            print(f"  Done: {summary['done_reason']} | Steps: {summary['steps']} "
                  f"({summary['steps']*0.01:.1f}s) | Reward: {summary['mean_reward']:.0f}")
            print(f"  HVT hits: {summary['hit_count']} | Escapes: {summary['n_escapes_total']} "
                  f"| Off alive: {summary['offensive_alive_end']} | Def alive: {summary['defensive_alive_end']}")
            min_d = min(summary['min_dists_to_hvt'])
            print(f"  Min dist to HVT: {min_d:.1f}m | Per-agent: {[f'{d:.0f}m' for d in summary['min_dists_to_hvt']]}")
            
            all_summaries.append(summary)
            all_step_data.append(step_data)
            all_trajectories.append(trajectories)
            
            # 绘制单 episode 详图
            plot_episode_details(ep, step_data, trajectories, summary, args.save_dir, env.config)
        except Exception as e:
            import traceback
            print(f"  ⚠ Episode {ep} failed: {e}")
            traceback.print_exc()
    
    # 汇总绘图
    plot_summary(all_summaries, args.save_dir)
    plot_action_distribution(all_step_data, args.save_dir)
    
    # 打印分析
    print_analysis(all_summaries, all_step_data)
    
    # 保存 JSON 数据
    json_path = os.path.join(args.save_dir, "eval_results.json")
    with open(json_path, 'w') as f:
        json.dump({
            "summaries": all_summaries,
            "config": {
                "model_dir": args.model_dir,
                "n_episodes": args.n_episodes,
                "deterministic": args.deterministic,
                "seed": args.seed,
            }
        }, f, indent=2, default=str)
    
    print(f"\n✅ All results saved to: {args.save_dir}")
    print(f"   - episode_X_detail.png (per-episode plots)")
    print(f"   - summary.png (all-episode comparison)")
    print(f"   - action_distribution.png")
    print(f"   - eval_results.json")


if __name__ == "__main__":
    main()

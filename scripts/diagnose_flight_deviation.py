"""诊断 "飞歪" 问题: 记录每个agent的航向误差、动作输出、轨迹"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from envs.fov_penetration import FOVPenetrationEnv
from eval_v28_10episodes import load_policies

def _get_actions(policies, obs_tuple, device, hidden_size=256):
    """Fixed version: obs_tuple = (obs, share_obs, avail_acts), obs shape = (n_agents, obs_dim)"""
    obs_all = obs_tuple[0] if isinstance(obs_tuple, tuple) else obs_tuple
    actions = []
    for agent_id, po in enumerate(policies):
        o = np.array(obs_all[agent_id]).flatten()
        obs_t = torch.FloatTensor(o).unsqueeze(0).to(device)
        rnn = torch.zeros(1, 1, hidden_size).to(device)
        masks = torch.ones(1, 1).to(device)
        with torch.no_grad():
            action, _, _ = po.actor(obs_t, rnn, masks, deterministic=True)
        actions.append(action.cpu().numpy().flatten())
    return actions

def diagnose(model_dir, hidden_size=256, layer_N=3, n_steps=4000):
    env = FOVPenetrationEnv(scenario='scenario_1')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policies = load_policies(model_dir, env, device, hidden_size=hidden_size, layer_N=layer_N)
    
    obs, _, _ = env.reset()
    n_agents = len(obs)
    
    # 记录数据
    tracks = {i: {'x':[], 'y':[], 'z':[], 'heading':[], 'heading_err':[], 
                   'dist_to_hvt':[], 'actions':[], 'alive':[], 'locked':[]} 
              for i in range(n_agents)}
    
    hvt = env.hvt
    print(f"HVT position: ({hvt.x:.0f}, {hvt.y:.0f}, {hvt.z:.0f})")
    for i, off in enumerate(env.offensives):
        d = off.distance_to(hvt.x, hvt.y, hvt.z)
        print(f"Agent {i}: pos=({off.x:.0f},{off.y:.0f},{off.z:.0f}), heading={np.degrees(off.heading):.1f}°, dist={d:.0f}m")
    
    for step in range(n_steps):
        actions = _get_actions(policies, obs, device, hidden_size)
        obs, _, _, _, dones, _, _ = env.step(actions)
        
        for i, off in enumerate(env.offensives):
            dx = hvt.x - off.x
            dy = hvt.y - off.y
            bearing = np.arctan2(dy, dx)
            herr = bearing - off.heading
            herr = (herr + np.pi) % (2*np.pi) - np.pi
            d = off.distance_to(hvt.x, hvt.y, hvt.z)
            
            is_locked = bool(env.lock_on_map.get(i, []))
            
            tracks[i]['x'].append(off.x)
            tracks[i]['y'].append(off.y)
            tracks[i]['z'].append(off.z)
            tracks[i]['heading'].append(np.degrees(off.heading))
            tracks[i]['heading_err'].append(np.degrees(herr))
            tracks[i]['dist_to_hvt'].append(d)
            tracks[i]['actions'].append(actions[i].tolist() if hasattr(actions[i],'tolist') else list(actions[i]))
            tracks[i]['alive'].append(off.alive)
            tracks[i]['locked'].append(is_locked)
        
        if step % 500 == 0:
            print(f"\nStep {step}:")
            for i, off in enumerate(env.offensives):
                d = tracks[i]['dist_to_hvt'][-1]
                herr = tracks[i]['heading_err'][-1]
                act = tracks[i]['actions'][-1]
                locked = tracks[i]['locked'][-1]
                status = "DEAD" if not off.alive else ("LOCKED" if locked else "FREE")
                print(f"  Agent {i} [{status}]: dist={d:.0f}m, heading_err={herr:+.1f}°, actions=[{act[0]:.3f},{act[1]:.3f},{act[2]:.3f}]")
        
        if np.all(dones):
            print(f"\nEpisode ended at step {step}")
            break
    
    # 最终分析
    print("\n" + "="*70)
    print("FLIGHT DEVIATION ANALYSIS")
    print("="*70)
    for i in range(n_agents):
        t = tracks[i]
        alive_steps = sum(1 for a in t['alive'] if a)
        if alive_steps == 0:
            print(f"\nAgent {i}: Killed immediately")
            continue
        
        # 只分析存活期间的数据
        alive_herr = [h for h, a in zip(t['heading_err'], t['alive']) if a]
        alive_acts = [a for a, al in zip(t['actions'], t['alive']) if al]
        alive_dist = [d for d, a in zip(t['dist_to_hvt'], t['alive']) if a]
        locked_steps = sum(1 for l, a in zip(t['locked'], t['alive']) if l and a)
        free_steps = alive_steps - locked_steps
        
        # action[2] (mu) 统计 — 这是导致飞歪的关键
        mu_actions = [a[2] for a in alive_acts]
        mu_mean = np.mean(mu_actions)
        mu_std = np.std(mu_actions)
        
        # 只看FREE期间的mu偏差
        free_mu = [a[2] for a, l, al in zip(t['actions'], t['locked'], t['alive']) if al and not l]
        free_herr = [h for h, l, a in zip(t['heading_err'], t['locked'], t['alive']) if a and not l]
        
        min_dist = min(alive_dist)
        min_dist_step = alive_dist.index(min_dist)
        
        print(f"\nAgent {i}:")
        print(f"  Alive: {alive_steps} steps, Locked: {locked_steps} steps, Free: {free_steps} steps")
        print(f"  Min dist to HVT: {min_dist:.1f}m (at step {min_dist_step})")
        print(f"  Final dist: {alive_dist[-1]:.1f}m")
        print(f"  Heading error: mean={np.mean(alive_herr):+.1f}°, max={max(abs(h) for h in alive_herr):.1f}°")
        print(f"  Action[2] (mu): mean={mu_mean:+.4f}, std={mu_std:.4f}")
        if free_mu:
            print(f"  FREE period mu: mean={np.mean(free_mu):+.4f}, std={np.std(free_mu):.4f}")
            print(f"  FREE period heading_err: mean={np.mean(free_herr):+.1f}°, max_abs={max(abs(h) for h in free_herr):.1f}°")
            # mu导致的heading drift估算
            # psi_dot ≈ G * cos(pi/2 + mu_bias*pi) / v = G * (-sin(mu_bias*pi)) / v
            mu_bias = np.mean(free_mu)
            psi_dot_est = 9.81 * (-np.sin(mu_bias * np.pi)) / 45.0  # rad/s
            drift_per_sec = np.degrees(psi_dot_est)
            drift_total = drift_per_sec * (free_steps * 0.01)
            print(f"  Estimated drift from mu bias: {drift_per_sec:+.2f}°/s → total {drift_total:+.1f}° over {free_steps*0.01:.0f}s")
    
    env.close()
    return tracks

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--layer_N', type=int, default=3)
    parser.add_argument('--n_steps', type=int, default=6000)
    args = parser.parse_args()
    diagnose(args.model_dir, args.hidden_size, args.layer_N, args.n_steps)

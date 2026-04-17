#!/usr/bin/env python
"""
Test episode length requirements for V37.
Run one long episode and track when each agent reaches minimum distance to HVT.
"""
import sys, os, torch, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.fov_penetration import FOVPenetrationEnv, get_config
from eval_v28_10episodes import load_policies, get_actions

def test_episode_length(model_dir, max_test_steps=15000, hidden_size=256, layer_N=3):
    device = torch.device("cpu")
    config = get_config(scenario="scenario_1")
    
    # Override max_steps to test longer
    original_max = config["max_steps"]
    config["max_steps"] = max_test_steps
    
    env = FOVPenetrationEnv(config)
    policies = load_policies(model_dir, env, device, hidden_size=hidden_size, layer_N=layer_N)
    
    n_trials = 3
    print(f"Testing episode length: max_test_steps={max_test_steps} ({max_test_steps*0.01:.0f}s)")
    print(f"Original max_steps={original_max} ({original_max*0.01:.0f}s)")
    print(f"Theoretical straight-line: 5375 steps (53.7s)")
    print("="*70)
    
    for trial in range(n_trials):
        obs, _, _ = env.reset()
        
        # Track per-agent distances
        n_agents = env.n_agents
        min_dists = [float('inf')] * n_agents
        min_dist_steps = [0] * n_agents
        alive_at = [0] * n_agents  # last alive step
        team_min_dist = float('inf')
        team_min_step = 0
        done = False
        done_reason = "timeout"
        
        for step in range(max_test_steps):
            actions = get_actions(policies, obs, device, hidden_size)
            obs, _, rewards, _, dones, infos, _ = env.step(actions)
            
            # Track distances
            hvt_pos = np.array([env.hvt.x, env.hvt.y, env.hvt.z])
            for i, ac in enumerate(env.offensives):
                if ac.alive:
                    alive_at[i] = step
                    d = np.linalg.norm(np.array([ac.x, ac.y, ac.z]) - hvt_pos)
                    if d < min_dists[i]:
                        min_dists[i] = d
                        min_dist_steps[i] = step
                    if d < team_min_dist:
                        team_min_dist = d
                        team_min_step = step
            
            # Check done
            if np.all(dones):
                done = True
                # try to get reason from info
                for info in (infos if isinstance(infos, list) else [infos]):
                    if isinstance(info, dict) and 'done_reason' in info:
                        done_reason = info['done_reason']
                        break
                break
        
        actual_steps = step + 1
        print(f"\nTrial {trial+1}/{n_trials}: {actual_steps} steps ({actual_steps*0.01:.1f}s), reason={done_reason}")
        print(f"  Team best: {team_min_dist:.0f}m at step {team_min_step} ({team_min_step*0.01:.1f}s)")
        
        for i in range(n_agents):
            status = "ALIVE" if env.offensives[i].alive else f"KILLED@{alive_at[i]}"
            print(f"  Agent {i} [{status}]: min_dist={min_dists[i]:.0f}m at step {min_dist_steps[i]} ({min_dist_steps[i]*0.01:.1f}s)")
        
        # Analysis
        print(f"\n  Step analysis:")
        for threshold in [8000, 10000, 12000]:
            sufficient = team_min_step < threshold
            print(f"    max_steps={threshold} ({threshold*0.01:.0f}s): {'✓ sufficient' if sufficient else '✗ NOT enough'} (best at step {team_min_step})")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--max_steps", type=int, default=15000)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--layer_N", type=int, default=3)
    args = parser.parse_args()
    
    test_episode_length(args.model_dir, args.max_steps, args.hidden_size, args.layer_N)

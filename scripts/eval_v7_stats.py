"""V7模型多seed快速统计"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'third_party', 'MACPO', 'MACPO'))

import numpy as np
import torch
from envs.fov_penetration.fov_penetration_env import FOVPenetrationEnv
from macpo.config import get_config as get_macpo_config
from macpo.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_dir = os.path.join(project_root, 'outputs', 'results', 'fov_penetration',
                         'macpo', 'v7_long_train', 'run1', 'models')

env = FOVPenetrationEnv(scenario='scenario_1')

# Load policies (same method as diagnose_episode.py)
parser = get_macpo_config()
args = parser.parse_known_args([
    '--algorithm_name', 'macpo',
    '--hidden_size', '256',
    '--layer_N', '3',
    '--lr', '3e-4',
    '--critic_lr', '3e-4',
])[0]

obs_space = env.observation_space[0]
share_obs_space = env.share_observation_space[0]
act_space = env.action_space[0]

policies = []
for i in range(4):
    policy = R_MAPPOPolicy(args, obs_space, share_obs_space, act_space, device=torch.device('cpu'))
    state_dict = torch.load(os.path.join(model_dir, f'actor_agent{i}.pt'), map_location='cpu')
    policy.actor.load_state_dict(state_dict)
    policy.actor.eval()
    policies.append(policy)

print(f"Model: {model_dir}")
print(f"Running 20 episodes with different seeds...\n")

results = []
for seed in range(20):
    env.seed(seed)
    obs_list, share_obs, _ = env.reset()
    rnn_states = [np.zeros((1, 1, 256), dtype=np.float32) for _ in range(4)]
    masks = [np.ones((1, 1), dtype=np.float32) for _ in range(4)]
    ep_reward = [0.0] * 4
    
    for step in range(500):
        actions = []
        for i in range(4):
            obs_np = np.array(obs_list[i]).reshape(1, 1, -1)
            with torch.no_grad():
                action, _, rnn_out = policies[i].actor(obs_np, rnn_states[i], masks[i], deterministic=True)
            actions.append(action.squeeze().numpy())
            rnn_states[i] = rnn_out.numpy()
        
        obs_list, share_obs, rewards, costs, dones, infos, avails = env.step(actions)
        for i in range(4):
            ep_reward[i] += rewards[i][0]
        if dones[0]:
            break
    
    final_dists = []
    alive_count = 0
    min_z = [9999.0] * 4
    for idx, off in enumerate(env.offensives):
        d = off.distance_to(env.hvt.x, env.hvt.y, env.hvt.z)
        final_dists.append(d)
        if off.alive:
            alive_count += 1
    
    hit_count = sum(1 for off in env.offensives if off.hit_hvt)
    avg_reward = np.mean(ep_reward)
    min_dist = min(final_dists)
    
    results.append({
        'seed': seed, 'reward': avg_reward, 'alive': alive_count,
        'hits': hit_count, 'min_dist': min_dist, 'steps': step+1,
        'dists': final_dists
    })
    
    status = "HIT!" if hit_count > 0 else f"alive={alive_count}"
    print(f"  Seed {seed:2d}: reward={avg_reward:8.1f}, {status}, min_dist={min_dist:.0f}m, "
          f"dists=[{','.join(f'{d:.0f}' for d in final_dists)}]")

print(f"\n=== Summary (20 episodes) ===")
avg_r = np.mean([r['reward'] for r in results])
avg_alive = np.mean([r['alive'] for r in results])
avg_min_dist = np.mean([r['min_dist'] for r in results])
hit_eps = sum(1 for r in results if r['hits'] > 0)
approach_eps = sum(1 for r in results if r['min_dist'] < 4000)
close_eps = sum(1 for r in results if r['min_dist'] < 2000)
print(f"Avg reward:    {avg_r:.1f}")
print(f"Avg alive:     {avg_alive:.1f}/4")
print(f"Avg min_dist:  {avg_min_dist:.0f}m")
print(f"HVT hits:      {hit_eps}/20 episodes")
print(f"dist<4000m:    {approach_eps}/20 episodes")
print(f"dist<2000m:    {close_eps}/20 episodes")

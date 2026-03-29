import sys
sys.path.insert(0, '.')
from envs.fov_penetration.config import get_config
from envs.fov_penetration.fov_penetration_env import FOVPenetrationEnv
import numpy as np

config = get_config(scenario='scenario_1')
env = FOVPenetrationEnv(config)
obs = env.reset()
import numpy as np
obs_arr = np.array(obs[0]) if isinstance(obs, tuple) else np.array(obs)
print(f'Env created OK, n_agents={env.n_agents}, obs_shape={obs_arr.shape}')

total_r = 0
total_c = 0
for step in range(10):
    actions = np.random.uniform(-1, 1, (env.n_agents, 3))
    obs, share_obs, rewards, costs, dones, infos, avail = env.step(actions)
    r_flat = [r for sublist in rewards for r in (sublist if isinstance(sublist, list) else [sublist])]
    c_flat = [c for sublist in costs for c in (sublist if isinstance(sublist, list) else [sublist])]
    total_r += sum(r_flat)
    total_c += sum(c_flat)
print(f'10 steps OK. total_reward={total_r:.2f}, total_cost={total_c:.2f}')

mb = config['reward'].get('milestone_bonuses', {})
print(f'milestone_bonuses: {mb}')
print(f'approach_hvt_coef: {config["reward"]["approach_hvt_coef"]}')
print(f'fov_exposure cost: {config["cost"]["fov_exposure"]}')
print(f'danger_zone cost: {config["cost"]["danger_zone"]}')
print(f'action_scale: {config["offensive"]["action_scale"]}')
print('ALL V19 SMOKE TEST PASSED')

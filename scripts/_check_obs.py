import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.fov_penetration import FOVPenetrationEnv
import numpy as np

env = FOVPenetrationEnv(config={"analytic_priors": {
    "enable_cone_cost": False,
    "enable_assignment_mismatch_reward": False,
    "enable_escape_reward": True,
    "enable_decoy_game": True,
    "enable_effective_penetration": True,
}}, scenario="scenario_1")

obs = env.reset()
print("type:", type(obs))
if isinstance(obs, (tuple, list)):
    print("len:", len(obs))
    for i, o in enumerate(obs):
        o = np.array(o)
        print(f"  [{i}] shape={o.shape}")
else:
    obs = np.array(obs)
    print("shape:", obs.shape)

# Also test step
actions = [env.action_space[i].sample() for i in range(env.n_agents)]
result = env.step(actions)
print("\nstep returns:", len(result), "items")
print("obs type:", type(result[0]))
if isinstance(result[0], (tuple, list)):
    print("obs len:", len(result[0]))
    for i, o in enumerate(result[0]):
        o = np.array(o)
        print(f"  [{i}] shape={o.shape}")
else:
    r0 = np.array(result[0])
    print("obs shape:", r0.shape)
print("rewards:", result[1])
print("dones:", result[2])
print("infos type:", type(result[3]))

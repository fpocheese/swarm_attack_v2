"""测试随机策略的reward，与零动作策略对比"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from envs.fov_penetration.fov_penetration_env import FOVPenetrationEnv

env = FOVPenetrationEnv({"scenario": "scenario_1"})

# 1. 随机动作
total_random = []
for ep in range(10):
    env.reset()
    ep_reward = [0.0] * 4
    for step in range(500):
        actions = [np.random.uniform(-1, 1, size=3) for _ in range(4)]
        result = env.step(actions)
        for i in range(4):
            ep_reward[i] += result[2][i][0]
        if result[4][0]:
            break
    total_random.append(np.mean(ep_reward))

# 2. 零动作（直飞）
total_zero = []
for ep in range(10):
    env.reset()
    ep_reward = [0.0] * 4
    for step in range(500):
        actions = [np.zeros(3) for _ in range(4)]
        result = env.step(actions)
        for i in range(4):
            ep_reward[i] += result[2][i][0]
        if result[4][0]:
            break
    total_zero.append(np.mean(ep_reward))

# 3. 直飞+轻微随机
total_slight = []
for ep in range(10):
    env.reset()
    ep_reward = [0.0] * 4
    for step in range(500):
        actions = [np.random.uniform(-0.1, 0.1, size=3) for _ in range(4)]
        result = env.step(actions)
        for i in range(4):
            ep_reward[i] += result[2][i][0]
        if result[4][0]:
            break
    total_slight.append(np.mean(ep_reward))

print(f"随机动作:     avg_reward = {np.mean(total_random):.1f} ± {np.std(total_random):.1f}")
print(f"零动作(直飞): avg_reward = {np.mean(total_zero):.1f} ± {np.std(total_zero):.1f}")
print(f"微随机:       avg_reward = {np.mean(total_slight):.1f} ± {np.std(total_slight):.1f}")

#!/usr/bin/env python
"""评估v11_01 FOV-Escape模型"""
import sys, os, numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "MACPO", "MACPO"))

import torch
from gym.spaces import Box
from envs.fov_penetration.fov_penetration_env import FOVPenetrationEnv
from macpo.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor

MODEL_DIR = os.path.join(PROJECT_ROOT, "outputs/results/fov_penetration/macpo/v11_01_fov_escape/run1/models")
OBS_DIM = 96
HIDDEN = 256
LAYER_N = 3
N_EPISODES = 20
EP_LEN = 1200

class Args:
    hidden_size = HIDDEN
    layer_N = LAYER_N
    use_feature_normalization = True
    use_recurrent_policy = False
    recurrent_N = 1
    use_policy_active_masks = True
    use_naive_recurrent_policy = False
    gain = 0.01
    stacked_frames = 1
    use_stacked_frames = False
    use_orthogonal = True
    use_ReLU = True
    algorithm_name = 'macpo'
    std_x_coef = 1.0
    std_y_coef = 0.5

obs_space = Box(low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32)
act_space = Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

actors = []
for i in range(4):
    actor = R_Actor(Args(), obs_space, act_space, 'cpu')
    sd = torch.load(os.path.join(MODEL_DIR, "actor_agent{}.pt".format(i)), map_location='cpu')
    actor.load_state_dict(sd)
    actor.eval()
    actors.append(actor)
print("V11_01 models loaded from:", MODEL_DIR)

results = []
for seed in range(N_EPISODES):
    env = FOVPenetrationEnv({'scenario': 'scenario_1'})
    env.seed(seed * 100)
    obs, _, _ = env.reset()
    ep_rew = np.zeros(4)
    rnn_states = [np.zeros((1,1,HIDDEN), dtype=np.float32) for _ in range(4)]
    masks = [np.ones((1,1), dtype=np.float32) for _ in range(4)]
    
    for step in range(EP_LEN):
        actions = []
        with torch.no_grad():
            for aid in range(4):
                ot = torch.FloatTensor(obs[aid]).unsqueeze(0)
                rt = torch.FloatTensor(rnn_states[aid])
                mt = torch.FloatTensor(masks[aid])
                act_out, _, rn = actors[aid](ot, rt, mt, deterministic=True)
                actions.append(act_out.cpu().numpy().flatten())
                rnn_states[aid] = rn.cpu().numpy()
        
        result = env.step(np.array(actions))
        obs, _, rewards, costs, dones, infos, _ = result
        ep_rew += np.array(rewards).flatten()
        if np.all(dones):
            break
    
    n_hit = sum(o.hit_hvt for o in env.offensives)
    off_alive = sum(1 for o in env.offensives if o.alive)
    reason = infos[0]['done_reason']
    results.append({
        'hit': n_hit, 'alive': off_alive,
        'rew': np.mean(ep_rew), 'reason': reason,
        'steps': env.current_step
    })
    print("  ep{:2d}: hit={}, alive={}/4, rew={:.0f}, steps={}, reason={}".format(
        seed, n_hit, off_alive, np.mean(ep_rew), env.current_step, reason))

print()
print("=" * 60)
print("  V11_01 FOV-Escape (51/416 = 12.3%) 评估结果")
print("=" * 60)
hit_rate = sum(1 for r in results if r['hit'] > 0) / N_EPISODES * 100
total_hits = sum(r['hit'] for r in results)
avg_rew = np.mean([r['rew'] for r in results])
std_rew = np.std([r['rew'] for r in results])
avg_alive = np.mean([r['alive'] for r in results])
avg_steps = np.mean([r['steps'] for r in results])
reasons = {}
for r in results:
    reasons[r['reason']] = reasons.get(r['reason'], 0) + 1
print("  突防成功率: {:.1f}% ({}/{})".format(hit_rate, sum(1 for r in results if r['hit']>0), N_EPISODES))
print("  总命中数:   {}/{} ({:.1f}%)".format(total_hits, N_EPISODES*4, total_hits/(N_EPISODES*4)*100))
print("  平均奖励:   {:.1f} +/- {:.1f}".format(avg_rew, std_rew))
print("  平均存活:   {:.1f}/4".format(avg_alive))
print("  平均步数:   {:.0f}/{}".format(avg_steps, EP_LEN))
print("  终止原因:   {}".format(reasons))
print("=" * 60)

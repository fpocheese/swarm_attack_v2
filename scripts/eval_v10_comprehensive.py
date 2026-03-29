#!/usr/bin/env python
"""
V10 综合评估脚本 — 训练模型 vs 基准对比
Usage: python scripts/eval_v10_comprehensive.py --model_dir <path>
"""
import sys, os, argparse
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "MACPO", "MACPO"))

import torch
from envs.fov_penetration.fov_penetration_env import FOVPenetrationEnv


def eval_zero_action(n_episodes=50, verbose=False):
    """零动作基准"""
    results = []
    for seed in range(n_episodes):
        env = FOVPenetrationEnv({'scenario': 'scenario_1'})
        env.seed(seed * 100)
        obs, share_obs, avail = env.reset()
        ep_rew = np.zeros(4)
        for step in range(1500):
            result = env.step(np.zeros((4, 3)))
            ep_rew += np.array(result[2]).flatten()
            if np.all(result[4]):
                break
        n_hit = sum(o.hit_hvt for o in env.offensives)
        off_alive = sum(1 for o in env.offensives if o.alive)
        def_killed = sum(1 for d in env.defensives if not d.alive)
        results.append({
            'reward': np.mean(ep_rew),
            'hit': n_hit > 0,
            'n_hit': n_hit,
            'off_alive': off_alive,
            'def_killed': def_killed,
            'steps': env.current_step,
            'reason': result[5][0]['done_reason'],
        })
        if verbose:
            print(f"  [zero] seed={seed:2d}: hit={n_hit}, rew={np.mean(ep_rew):.0f}, "
                  f"reason={result[5][0]['done_reason']}")
    return results


def eval_random_action(n_episodes=50, verbose=False):
    """随机动作基准"""
    results = []
    for seed in range(n_episodes):
        env = FOVPenetrationEnv({'scenario': 'scenario_1'})
        env.seed(seed * 100)
        rng = np.random.RandomState(seed)
        obs, share_obs, avail = env.reset()
        ep_rew = np.zeros(4)
        for step in range(1500):
            actions = rng.uniform(-1, 1, (4, 3)).astype(np.float32)
            result = env.step(actions)
            ep_rew += np.array(result[2]).flatten()
            if np.all(result[4]):
                break
        n_hit = sum(o.hit_hvt for o in env.offensives)
        off_alive = sum(1 for o in env.offensives if o.alive)
        def_killed = sum(1 for d in env.defensives if not d.alive)
        results.append({
            'reward': np.mean(ep_rew),
            'hit': n_hit > 0,
            'n_hit': n_hit,
            'off_alive': off_alive,
            'def_killed': def_killed,
            'steps': env.current_step,
            'reason': result[5][0]['done_reason'],
        })
        if verbose:
            print(f"  [rand] seed={seed:2d}: hit={n_hit}, rew={np.mean(ep_rew):.0f}, "
                  f"reason={result[5][0]['done_reason']}")
    return results


def load_trained_policy(model_dir, hidden_size=256, layer_N=3, device='cpu'):
    """加载训练好的MACPO策略"""
    from macpo.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor
    from gym.spaces import Box
    
    obs_space = Box(low=-np.inf, high=np.inf, shape=(84,), dtype=np.float32)
    share_obs_space = Box(low=-np.inf, high=np.inf, shape=(76,), dtype=np.float32)
    act_space = Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
    
    actors = []
    for agent_id in range(4):
        actor_path = os.path.join(model_dir, f"actor_agent{agent_id}.pt")
        if not os.path.exists(actor_path):
            print(f"  WARNING: {actor_path} not found!")
            return None
        
        # Create a simple args object
        class Args:
            pass
        args = Args()
        args.hidden_size = hidden_size
        args.layer_N = layer_N
        args.use_feature_normalization = True
        args.use_recurrent_policy = False
        args.recurrent_N = 1
        args.use_policy_active_masks = True
        args.use_naive_recurrent_policy = False
        args.gain = 0.01
        args.stacked_frames = 1
        args.use_stacked_frames = False
        
        actor = R_Actor(args, obs_space, act_space, device)
        state_dict = torch.load(actor_path, map_location=device)
        actor.load_state_dict(state_dict)
        actor.eval()
        actors.append(actor)
    
    return actors


def eval_trained_policy(model_dir, n_episodes=50, hidden_size=256, layer_N=3, 
                        verbose=False, device='cpu'):
    """评估训练好的模型"""
    actors = load_trained_policy(model_dir, hidden_size, layer_N, device)
    if actors is None:
        print("Failed to load policy!")
        return []
    
    results = []
    rnn_states = [np.zeros((1, 1, hidden_size), dtype=np.float32) for _ in range(4)]
    masks = [np.ones((1, 1), dtype=np.float32) for _ in range(4)]
    
    for seed in range(n_episodes):
        env = FOVPenetrationEnv({'scenario': 'scenario_1'})
        env.seed(seed * 100)
        obs, share_obs, avail = env.reset()
        ep_rew = np.zeros(4)
        
        # Reset RNN states
        for i in range(4):
            rnn_states[i] = np.zeros((1, 1, hidden_size), dtype=np.float32)
            masks[i] = np.ones((1, 1), dtype=np.float32)
        
        for step in range(1500):
            actions = []
            with torch.no_grad():
                for agent_id in range(4):
                    obs_tensor = torch.FloatTensor(obs[agent_id]).unsqueeze(0).to(device)
                    rnn_tensor = torch.FloatTensor(rnn_states[agent_id]).to(device)
                    mask_tensor = torch.FloatTensor(masks[agent_id]).to(device)
                    
                    action, temp_rnn = actors[agent_id].act(
                        obs_tensor, rnn_tensor, mask_tensor, deterministic=True)
                    actions.append(action.cpu().numpy().flatten())
                    rnn_states[agent_id] = temp_rnn.cpu().numpy()
            
            actions = np.array(actions)
            result = env.step(actions)
            obs, share_obs, rewards, costs, dones, infos, avail = result
            ep_rew += np.array(rewards).flatten()
            
            if np.all(dones):
                break
        
        n_hit = sum(o.hit_hvt for o in env.offensives)
        off_alive = sum(1 for o in env.offensives if o.alive)
        def_killed = sum(1 for d in env.defensives if not d.alive)
        results.append({
            'reward': np.mean(ep_rew),
            'hit': n_hit > 0,
            'n_hit': n_hit,
            'off_alive': off_alive,
            'def_killed': def_killed,
            'steps': env.current_step,
            'reason': infos[0]['done_reason'],
        })
        if verbose:
            print(f"  [trained] seed={seed:2d}: hit={n_hit}, off_alive={off_alive}, "
                  f"def_killed={def_killed}, rew={np.mean(ep_rew):.0f}, "
                  f"reason={infos[0]['done_reason']}, steps={env.current_step}")
    
    return results


def print_comparison(zero_results, random_results, trained_results=None):
    """打印三种策略的对比"""
    def summarize(results, name):
        if not results:
            return
        n = len(results)
        hit_rate = sum(r['hit'] for r in results) / n * 100
        n_hit_total = sum(r['n_hit'] for r in results)
        avg_rew = np.mean([r['reward'] for r in results])
        std_rew = np.std([r['reward'] for r in results])
        avg_alive = np.mean([r['off_alive'] for r in results])
        avg_def_killed = np.mean([r['def_killed'] for r in results])
        avg_steps = np.mean([r['steps'] for r in results])
        reasons = {}
        for r in results:
            reasons[r['reason']] = reasons.get(r['reason'], 0) + 1
        
        print(f"\n  {name} ({n} episodes)")
        print(f"  {'─'*50}")
        print(f"  突防成功率:   {hit_rate:.1f}% ({sum(r['hit'] for r in results)}/{n})")
        print(f"  总命中数:     {n_hit_total}/{n*4} ({n_hit_total/(n*4)*100:.1f}%)")
        print(f"  平均奖励:     {avg_rew:.1f} ± {std_rew:.1f}")
        print(f"  平均存活:     {avg_alive:.1f}/4 进攻方")
        print(f"  平均击杀:     {avg_def_killed:.1f}/4 防御方")
        print(f"  平均步数:     {avg_steps:.0f}/1500")
        print(f"  终止原因:     {reasons}")
    
    print(f"\n{'='*60}")
    print(f"  V10 综合评估对比")
    print(f"{'='*60}")
    
    summarize(zero_results, "🔵 零动作基准 (直飞)")
    summarize(random_results, "🟡 随机动作基准")
    if trained_results:
        summarize(trained_results, "🟢 训练模型")
        
        # 改进分析
        zero_hit = sum(r['hit'] for r in zero_results) / len(zero_results) * 100
        trained_hit = sum(r['hit'] for r in trained_results) / len(trained_results) * 100
        zero_rew = np.mean([r['reward'] for r in zero_results])
        trained_rew = np.mean([r['reward'] for r in trained_results])
        
        print(f"\n  --- 改进分析 ---")
        print(f"  突防率提升: {zero_hit:.1f}% → {trained_hit:.1f}% ({trained_hit-zero_hit:+.1f}%)")
        print(f"  奖励提升:   {zero_rew:.0f} → {trained_rew:.0f} ({trained_rew-zero_rew:+.0f})")
        
        if trained_hit > zero_hit + 10:
            print(f"  ✅ 训练模型显著优于基准!")
        elif trained_hit > zero_hit:
            print(f"  🟡 训练模型略优于基准, 仍可改进")
        else:
            print(f"  ⚠️  训练模型未优于基准, 需要调参!")
    
    print(f"{'='*60}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default=None,
                        help='Path to saved model directory')
    parser.add_argument('--n_episodes', type=int, default=50)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--layer_N', type=int, default=3)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--skip_baselines', action='store_true')
    args = parser.parse_args()
    
    print("Running baselines...")
    zero_results = [] if args.skip_baselines else eval_zero_action(args.n_episodes, args.verbose)
    random_results = [] if args.skip_baselines else eval_random_action(args.n_episodes, args.verbose)
    
    trained_results = None
    if args.model_dir:
        if os.path.exists(args.model_dir):
            print(f"\nLoading trained model from: {args.model_dir}")
            trained_results = eval_trained_policy(
                args.model_dir, args.n_episodes, args.hidden_size, args.layer_N, args.verbose)
        else:
            print(f"Model dir not found: {args.model_dir}")
            # Try to find the latest model
            results_dir = os.path.join(PROJECT_ROOT, "outputs", "results", "fov_penetration", "macpo")
            if os.path.exists(results_dir):
                print(f"Available experiments: {os.listdir(results_dir)}")
    
    print_comparison(zero_results, random_results, trained_results)

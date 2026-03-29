#!/usr/bin/env python
"""
渲染训练后策略 Episode
========================
加载训练好的模型并生成可视化动画
"""
import sys
import os
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "MACPO", "MACPO"))

import numpy as np
import torch

from envs.fov_penetration import FOVPenetrationEnv
from envs.fov_penetration.render import render_episode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to trained model directory (containing actor_agent*.pt)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None,
                        help="Output file path (.gif or .mp4)")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--hidden_size", type=int, default=128,
                        help="Hidden size of actor network (must match trained model)")
    parser.add_argument("--layer_N", type=int, default=2,
                        help="Number of hidden layers (must match trained model)")
    args = parser.parse_args()
    
    device = torch.device("cpu")
    env = FOVPenetrationEnv()
    env.seed(args.seed)
    
    # 加载模型
    from macpo.algorithms.r_mappo.algorithm.MACPPOPolicy import MACPPOPolicy
    from macpo.config import get_config
    
    config_parser = get_config()
    all_args = config_parser.parse_known_args([])[0]
    all_args.algorithm_name = "macpo"
    all_args.hidden_size = args.hidden_size
    all_args.layer_N = args.layer_N
    
    policies = []
    for agent_id in range(env.n_agents):
        po = MACPPOPolicy(all_args,
                          env.observation_space[agent_id],
                          env.share_observation_space[agent_id],
                          env.action_space[agent_id],
                          device=device)
        
        actor_path = os.path.join(args.model_dir, f"actor_agent{agent_id}.pt")
        if os.path.exists(actor_path):
            state_dict = torch.load(actor_path, map_location=device)
            po.actor.load_state_dict(state_dict)
            print(f"Loaded actor for agent {agent_id}")
        else:
            print(f"Warning: {actor_path} not found, using random weights")
        
        po.actor.eval()
        policies.append(po)
    
    def policy_fn(obs_list):
        actions = []
        for agent_id, po in enumerate(policies):
            obs_tensor = torch.FloatTensor(obs_list[agent_id]).unsqueeze(0).to(device)
            rnn_states = torch.zeros(1, 1, args.hidden_size).to(device)
            masks = torch.ones(1, 1).to(device)
            
            with torch.no_grad():
                action, _, _ = po.actor(obs_tensor, rnn_states, masks, deterministic=True)
            actions.append(action.cpu().numpy().flatten())
        return actions
    
    # 确定输出路径
    if args.output is None:
        os.makedirs(os.path.join(PROJECT_ROOT, "outputs", "gifs"), exist_ok=True)
        output_path = os.path.join(PROJECT_ROOT, "outputs", "gifs", "trained_policy.gif")
    else:
        output_path = args.output
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Rendering trained policy to {output_path}...")
    render_episode(env, policy_fn=policy_fn, save_path=output_path, fps=args.fps)
    print("Done!")


if __name__ == "__main__":
    main()

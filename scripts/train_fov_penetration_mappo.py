#!/usr/bin/env python
"""
FOV Penetration MAPPO 训练脚本
=====================================
三维同构集群, 纯奖励优化 (无 cost 约束)
基于 train_fov_penetration_macpo.py, 替换为 MAPPO runner
"""
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "MACPO", "MACPO"))

import wandb
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch

from macpo.config import get_config
from macpo.envs.env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv
from envs.fov_penetration import FOVPenetrationEnv
from scripts.phase_obs_wrapper import PhaseMaskedFOVWrapper
from scripts.terminal_pn_action_wrapper import TerminalPNActionWrapper


class PatchedShareDummyVecEnv(ShareDummyVecEnv):
    def __init__(self, env_fns):
        super().__init__(env_fns)
        self.n_agents = self.envs[0].n_agents


def _get_ap_override(ap_config_name: str) -> dict:
    """Map ablation config name to analytic_priors override dict."""
    if ap_config_name == "none":
        return {"analytic_priors": {
            "enable_cone_cost": False,
            "enable_assignment_mismatch_reward": False,
            "enable_escape_reward": False,
            "enable_decoy_game": False,
            "enable_effective_penetration": False,
        }}
    elif ap_config_name == "v22":
        return {"analytic_priors": {
            "enable_cone_cost": False,      # MAPPO 不用 cost, cone_cost 无意义
            "enable_assignment_mismatch_reward": False,
            "enable_escape_reward": True,
            "enable_decoy_game": True,
            "enable_effective_penetration": True,
        }}
    elif ap_config_name == "v22_full":
        return {"analytic_priors": {
            "enable_cone_cost": True,       # 仍计算但仅进 info, 不影响策略
            "enable_assignment_mismatch_reward": False,
            "enable_escape_reward": True,
            "enable_decoy_game": True,
            "enable_effective_penetration": True,
        }}
    elif ap_config_name == "v28":
        # V28: 全部 AP 模块启用 — obs 和 reward 都需要
        return {"analytic_priors": {
            "enable_cone_cost": True,
            "enable_assignment_mismatch_reward": False,
            "enable_escape_reward": True,
            "enable_decoy_game": True,
            "enable_effective_penetration": True,
            "enable_hvt_guidance": True,
            "enable_attack_gate_reward": False,  # V28 奖励函数内部处理
        }}
    else:
        return {}


def make_train_env(all_args):
    ap_override = _get_ap_override(getattr(all_args, 'ap_config', 'v22'))
    obs_phase_mask = getattr(all_args, 'obs_phase_mask', 'none')
    terminal_guidance = getattr(all_args, 'terminal_guidance', 'none')
    terminal_pn_gain = getattr(all_args, 'terminal_pn_gain', 3.0)
    terminal_pn_max_action = getattr(all_args, 'terminal_pn_max_action', 0.8)

    def get_env_fn(rank):
        def init_env():
            env = FOVPenetrationEnv(config=ap_override, scenario=all_args.scenario)
            if obs_phase_mask != 'none':
                env = PhaseMaskedFOVWrapper(env, mode=obs_phase_mask)
            if terminal_guidance == 'pn_los':
                env = TerminalPNActionWrapper(env, gain=terminal_pn_gain,
                                              max_action=terminal_pn_max_action)
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env

    if all_args.n_rollout_threads == 1:
        return PatchedShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
    ap_override = _get_ap_override(getattr(all_args, 'ap_config', 'v22'))
    obs_phase_mask = getattr(all_args, 'obs_phase_mask', 'none')
    terminal_guidance = getattr(all_args, 'terminal_guidance', 'none')
    terminal_pn_gain = getattr(all_args, 'terminal_pn_gain', 3.0)
    terminal_pn_max_action = getattr(all_args, 'terminal_pn_max_action', 0.8)

    def get_env_fn(rank):
        def init_env():
            env = FOVPenetrationEnv(config=ap_override, scenario=all_args.scenario)
            if obs_phase_mask != 'none':
                env = PhaseMaskedFOVWrapper(env, mode=obs_phase_mask)
            if terminal_guidance == 'pn_los':
                env = TerminalPNActionWrapper(env, gain=terminal_pn_gain,
                                              max_action=terminal_pn_max_action)
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env
        return init_env

    if all_args.n_eval_rollout_threads == 1:
        return PatchedShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def parse_args(args, parser):
    parser.add_argument('--scenario', type=str, default='scenario_1',
                        choices=['scenario_1', 'scenario_2', 'scenario_3'],
                        help='scenario_1=4v4, scenario_2=4v6, scenario_3=6v4')
    parser.add_argument("--use_single_network", action='store_true', default=False)
    parser.add_argument('--ap_config', type=str, default='v28',
                        choices=['none', 'v22', 'v22_full', 'v28'],
                        help='Analytic priors config: '
                             'none=no AP, v22=decoy+escape+eff_pen, '
                             'v22_full=v22+cone_cost, v28=all AP modules')
    parser.add_argument('--obs_phase_mask', type=str,
                        default=os.environ.get('FOV_OBS_PHASE_MASK', 'none').strip().lower(),
                        choices=['none', 'v60_phase', 'v65_strict_los'],
                        help='Policy observation masking mode. v60_phase is split-phase; v65_strict_los fully hides HVT attack guidance in penetration and hides opponent/team cues in terminal.')
    parser.add_argument('--terminal_guidance', type=str,
                        default=os.environ.get('FOV_TERMINAL_GUIDANCE', 'none').strip().lower(),
                        choices=['none', 'pn_los'],
                        help='Optional policy-layer terminal guidance. pn_los replaces terminal pitch/yaw with commands from HVT LOS-rate obs[5:7].')
    parser.add_argument('--terminal_pn_gain', type=float,
                        default=float(os.environ.get('FOV_TERMINAL_PN_GAIN', '3.0')),
                        help='Gain for --terminal_guidance pn_los.')
    parser.add_argument('--terminal_pn_max_action', type=float,
                        default=float(os.environ.get('FOV_TERMINAL_PN_MAX_ACTION', '0.8')),
                        help='Absolute pitch/yaw action limit for --terminal_guidance pn_los.')
    all_args = parser.parse_known_args(args)[0]
    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    # MAPPO: always use separated policy, no cost constraint
    if all_args.algorithm_name == "mappo":
        all_args.share_policy = False
    else:
        raise NotImplementedError(f"This script supports mappo only, got {all_args.algorithm_name}")

    if all_args.cuda and torch.cuda.is_available():
        print("Using GPU...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("Using CPU...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    run_dir = Path(os.path.join(PROJECT_ROOT, "outputs", "results")) / \
              "fov_penetration" / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    if all_args.use_wandb:
        run = wandb.init(config=all_args,
                         project="fov_penetration",
                         entity=all_args.user_name,
                         notes=socket.gethostname(),
                         name=f"{all_args.algorithm_name}_{all_args.experiment_name}_seed{all_args.seed}",
                         dir=str(run_dir),
                         job_type="training",
                         reinit=True)
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if
                         str(folder.name).startswith('run')] if run_dir.exists() else []
        curr_run = 'run%i' % (max(exst_run_nums) + 1) if exst_run_nums else 'run1'
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    setproctitle.setproctitle(
        f"mappo-fov-{all_args.experiment_name}@{all_args.user_name}")

    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    num_agents = envs.n_agents

    print(f"=== FOV Penetration MAPPO Training ===")
    print(f"Scenario: {all_args.scenario}")
    print(f"Analytic Priors: {getattr(all_args, 'ap_config', 'v22')}")
    print(f"Num agents (trained): {num_agents}")
    print(f"Obs dim: {envs.observation_space[0].shape}")
    print(f"Share obs dim: {envs.share_observation_space[0].shape}")
    print(f"Action space: {envs.action_space[0]}")
    print(f"Obs phase mask: {getattr(all_args, 'obs_phase_mask', 'none')}")
    print(f"Terminal guidance: {getattr(all_args, 'terminal_guidance', 'none')}")
    if getattr(all_args, 'terminal_guidance', 'none') == 'pn_los':
        print(f"Terminal PN gain/max_action: {all_args.terminal_pn_gain}/{all_args.terminal_pn_max_action}")
    print(f"Episode length: {all_args.episode_length}")
    print(f"Num env steps: {all_args.num_env_steps}")
    print(f"Hidden size: {all_args.hidden_size}")
    print(f"LR: {all_args.lr}")

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }

    # Use MAPPO runner (no cost constraint)
    from macpo.runner.separated.mujoco_runner import MujocoRunner as Runner
    runner = Runner(config)
    runner.run()

    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
        runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])

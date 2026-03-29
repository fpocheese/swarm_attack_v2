#!/usr/bin/env python
"""
FOV Penetration MACPO 训练脚本 V3
=====================================
三维同构集群, 支持 --scenario 工况切换
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
        }}
    elif ap_config_name == "cone":
        return {"analytic_priors": {
            "enable_cone_cost": True,
            "enable_assignment_mismatch_reward": False,
            "enable_escape_reward": False,
        }}
    elif ap_config_name == "cone_mismatch":
        return {"analytic_priors": {
            "enable_cone_cost": True,
            "enable_assignment_mismatch_reward": True,
            "enable_escape_reward": False,
        }}
    elif ap_config_name == "full":
        return {"analytic_priors": {
            "enable_cone_cost": True,
            "enable_assignment_mismatch_reward": True,
            "enable_escape_reward": True,
        }}
    else:
        return {}


def make_train_env(all_args):
    ap_override = _get_ap_override(getattr(all_args, 'ap_config', 'full'))

    def get_env_fn(rank):
        def init_env():
            env = FOVPenetrationEnv(config=ap_override, scenario=all_args.scenario)
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env

    if all_args.n_rollout_threads == 1:
        return PatchedShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
    ap_override = _get_ap_override(getattr(all_args, 'ap_config', 'full'))

    def get_env_fn(rank):
        def init_env():
            env = FOVPenetrationEnv(config=ap_override, scenario=all_args.scenario)
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
    parser.add_argument('--ap_config', type=str, default='full',
                        choices=['none', 'cone', 'cone_mismatch', 'full'],
                        help='Analytic priors ablation config: '
                             'none=baseline, cone=cone_cost only, '
                             'cone_mismatch=cone+mismatch, full=all three')
    all_args = parser.parse_known_args(args)[0]
    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    if all_args.algorithm_name == "macpo":
        all_args.share_policy = False
    else:
        raise NotImplementedError(f"Only macpo supported, got {all_args.algorithm_name}")

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
        f"macpo-fov-{all_args.experiment_name}@{all_args.user_name}")

    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    num_agents = envs.n_agents

    print(f"=== FOV Penetration MACPO Training V3 ===")
    print(f"Scenario: {all_args.scenario}")
    print(f"Analytic Priors: {getattr(all_args, 'ap_config', 'full')}")
    print(f"Num agents (trained): {num_agents}")
    print(f"Obs dim: {envs.observation_space[0].shape}")
    print(f"Share obs dim: {envs.share_observation_space[0].shape}")
    print(f"Action space: {envs.action_space[0]}")
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

    from macpo.runner.separated.mujoco_runner_macpo import MujocoRunner as Runner
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

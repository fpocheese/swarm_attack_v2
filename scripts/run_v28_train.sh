#!/bin/bash
# =====================================================
# V28 MAPPO Training — 新观测(45维) + 新4层奖励 + 终端奖励
# =====================================================
# 变更相对 V25:
#   1. 观测空间: 108→45维 (Self:10, HVT:5, Top2Threats:18, Team:6, Priors:5, Time:1)
#   2. 奖励函数: 4层结构 (task + game + escape − risk) + 终端奖励
#   3. Cost全部并入reward, 使用纯MAPPO (不需要Lagrangian)
#   4. episode_length=6000 (dt=0.01, 60s总时长)
#   5. 全部AP模块启用 (cone_cost, escape, decoy, eff_pen, hvt_guidance)
# =====================================================

cd "$(dirname "$0")/.."

conda run --no-capture-output -n rlgpu python -u scripts/train_fov_penetration_mappo.py \
    --algorithm_name mappo \
    --experiment_name v28_mappo_new_obs_reward \
    --scenario scenario_1 \
    --ap_config v28 \
    --seed 42 \
    --cuda \
    --n_rollout_threads 80 \
    --n_training_threads 8 \
    --n_eval_rollout_threads 20 \
    --num_env_steps 200000000 \
    --episode_length 6000 \
    --ppo_epoch 5 \
    --use_value_active_masks \
    --hidden_size 256 \
    --layer_N 3 \
    --lr 3e-4 \
    --critic_lr 3e-4 \
    --entropy_coef 0.02 \
    --gamma 0.99 \
    --gae_lambda 0.95 \
    --clip_param 0.2 \
    --value_loss_coef 1.0 \
    --num_mini_batch 40 \
    --max_grad_norm 10.0 \
    --use_max_grad_norm \
    --use_eval \
    --eval_interval 5 \
    --eval_episodes 20 \
    --log_interval 1 \
    --save_interval 10 \
    --use_linear_lr_decay \
    --user_name fov_team \
    --use_ReLU \
    --use_feature_normalization \
    --use_orthogonal \
    --gain 0.01 \
    --data_chunk_length 10 \
    --use_recurrent_policy \
    2>&1 | tee outputs/v28_mappo_train.log

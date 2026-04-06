#!/bin/bash
# =====================================================
# V22 MAPPO Training — dt=0.05, hit_range=3m, CPA enabled
# =====================================================
# 纯 MAPPO (无 cost 约束), 用于验证去掉 MACPO 约束后收敛性
#
# dt=0.05  =>  step_size = 50*0.05 = 2.5m < 3m (hit OK)
# max_steps=2400  =>  total_time = 2400*0.05 = 120s (同原来)
# episode_length=2400  =>  与 max_steps 匹配
# =====================================================

cd "$(dirname "$0")/.."

conda run --no-capture-output -n rlgpu python -u scripts/train_fov_penetration_mappo.py \
    --algorithm_name mappo \
    --experiment_name v22_mappo_dt05_3m \
    --scenario scenario_1 \
    --ap_config v22 \
    --seed 42 \
    --cuda \
    --n_rollout_threads 20 \
    --n_training_threads 4 \
    --n_eval_rollout_threads 4 \
    --num_env_steps 10000000 \
    --episode_length 1200 \
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
    --num_mini_batch 20 \
    --max_grad_norm 10.0 \
    --use_max_grad_norm \
    --use_eval \
    --eval_interval 5 \
    --eval_episodes 8 \
    --log_interval 1 \
    --save_interval 10 \
    --use_linear_lr_decay \
    --user_name fov_team \
    --use_ReLU \
    --use_feature_normalization \
    --use_orthogonal \
    --gain 0.01 \
    --n_training_threads 4 \
    --data_chunk_length 10 \
    --use_recurrent_policy \
    2>&1 | tee outputs/v22_mappo_train.log

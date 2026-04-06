#!/bin/bash
# =====================================================
# V24 MAPPO Training — Decoy/Collision Model
# =====================================================
# 变更相对 V23:
#   action_scale=1.0     (was 0.5, 进攻方全机动)
#   off_v_nominal=60     (was 50), off_v_max=80 (was 70)
#   def_v_nominal=65     (was 70), def_v_max=85 (was 90)
#   init_dist=2500       (was 5000, 修正归一化基准)
#   FOV escape 机制完全移除 (无锁定/交战半径/脱锁)
#   击杀逻辑: 纯碰撞双杀 CPA<5m
#   新增 decoy 诱饵牺牲奖励:
#     decoy_sacrifice_bonus=800
#     decoy_attract_coef=2.0
#     decoy_front_bonus_coef=1.5
#   观测: overload_saturation → is_chasing_me,
#          escaped_flag → n_chasers
# =====================================================

cd "$(dirname "$0")/.."

conda run --no-capture-output -n rlgpu python -u scripts/train_fov_penetration_mappo.py \
    --algorithm_name mappo \
    --experiment_name v24_mappo_decoy_collision \
    --scenario scenario_1 \
    --ap_config v22 \
    --seed 42 \
    --cuda \
    --n_rollout_threads 80 \
    --n_training_threads 8 \
    --n_eval_rollout_threads 20 \
    --num_env_steps 100000000 \
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
    2>&1 | tee outputs/v24_mappo_train.log

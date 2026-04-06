#!/bin/bash
# =====================================================
# V25 MAPPO Training — No Extrapolation + Fixed Decoy Rewards
# =====================================================
# 变更相对 V24:
#   1. 拦截器制导: 取消FOV外位置外推，直接用上次已知位置
#      → 信息不更新时拦截器朝旧位置飞，滞后更大，有利进攻方突防
#   2. 奖励函数修复:
#      - 合并mutual_kill和decoy_sacrifice(避免重复奖励)
#      - 持续诱饵奖励: 通过defensive_policies精确判断追击目标
#      - front_mult逻辑修正: 离HVT最远的agent做诱饵价值最大
#   3. 观测空间不变(108维，已支持诱饵策略)
# =====================================================

cd "$(dirname "$0")/.."

conda run --no-capture-output -n rlgpu python -u scripts/train_fov_penetration_mappo.py \
    --algorithm_name mappo \
    --experiment_name v25_mappo_no_extrap_decoy \
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
    2>&1 | tee outputs/v25_mappo_train.log

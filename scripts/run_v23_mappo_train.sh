#!/bin/bash
# =====================================================
# V23 MAPPO Training — dt=0.01s, hit_range=5m, LOS rate obs
# =====================================================
# 变更相对 V22:
#   dt=0.01s   => 步长 = 50*0.01 = 0.5m (极小步, 防止飞过目标)
#   max_steps=12000 => 总时长 = 12000*0.01 = 120s (同 V22)
#   hit_hvt_range=5m, collision_kill_range=5m (对称命中阈值)
#   z_min=0m (低空突防不再因高度不足死亡)
#   防御方速度: v_nominal=70, v_max=90 (降低，避免尾追过强)
#   观测空间: 新增对 HVT 的视线角速度 (LOS rate, 2维)
#   击杀逻辑: 简化为纯 CPA 碰撞双杀 (脱靶量<5m 即命中)
#
# episode_length=1200 => 每次 rollout = 1200*0.01 = 12s
# num_env_steps=50000000 => 约 50M 步 (dt=0.01 需要更多步)
# =====================================================

cd "$(dirname "$0")/.."

conda run --no-capture-output -n rlgpu python -u scripts/train_fov_penetration_mappo.py \
    --algorithm_name mappo \
    --experiment_name v23_mappo_dt01_5m_losrate \
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
    2>&1 | tee outputs/v23_mappo_train.log
